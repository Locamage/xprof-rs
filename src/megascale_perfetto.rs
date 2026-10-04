use crate::memory_viewer::std_sort;
use crate::xplane::{Field, Plane, Value as Raw, fields, slice};
use rustc_hash::FxHashMap as HashMap;
use std::collections::{BTreeMap, VecDeque};
use std::io::Write;

const TPU_PLANE: &str = "/device:TPU:";
const MEGASCALE_PLANE: &str = "/device:CUSTOM:Megascale Trace";
const TPU_LINES: [&str; 4] = ["Steps", "XLA Modules", "XLA Ops", "XLA TraceMe"];
const SKIPPED_STATS: [&str; 13] = ["_c", "_ct", "flops", "flow", "_a", "_r", "kernel_details", "_p", "_pt", "program_id", "symbol_id", "source", "source_stack"];
const RENDEZVOUS_PREFIX: &str = "_xla_host_transfer_rendezvous=\"";
const MAX_RATE_DELTA_GBPS: f64 = 200.0;
const TINY_PS: i64 = 1000;
const HIDDEN_DESCRIPTION: &str = "If you would like to see them, please add &group_tiny_events=false to the URL.";
const SEQUENCE_ID: u64 = 1;
const INCREMENTAL_STATE_CLEARED: u64 = 1;
const NEEDS_INCREMENTAL_STATE: u64 = 2;
const SLICE_BEGIN: u64 = 1;
const SLICE_END: u64 = 2;
const INSTANT: u64 = 3;
const COUNTER: u64 = 4;
const OS_UNIX: u8 = 3;
const TRANSFERS: [&str; 4] = ["send", "recv", "send-done", "recv-done"];
const COUNTERS: [(&str, &str, &str); 6] = [
    ("Outstanding Bytes RX", "bytes", "outstanding_bytes"),
    ("Outstanding Bytes TX", "bytes", "outstanding_bytes"),
    ("Bandwidth RX (Gbps)", "Gbps", "bandwidth"),
    ("Bandwidth TX (Gbps)", "Gbps", "bandwidth"),
    ("Inflight Collectives", "count", "inflight_collectives"),
    ("Inflight Collective Payload", "bytes", "inflight_collective_payload"),
];
const NETWORK_COUNTERS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    Int(i64),
    Uint(u64),
    Double(f64),
    Text(u32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Arg {
    key: u32,
    value: Value,
}

#[derive(Clone, Default, Debug, PartialEq)]
struct Event {
    name: u32,
    ts: i64,
    dur: i64,
    args: Vec<Arg>,
    run_id: i64,
    flows: Vec<(i64, bool)>,
}

#[derive(Default, Debug)]
struct Track {
    name: String,
    events: Vec<Event>,
}

#[derive(Clone, Copy)]
enum Sample {
    Int(i64),
    Double(f64),
}

#[derive(Default)]
struct Strings {
    list: Vec<String>,
    index: HashMap<String, u32>,
}

#[derive(Default)]
struct Trace {
    strings: Strings,
    tpu: BTreeMap<i64, Vec<Track>>,
    megascale: BTreeMap<i64, Vec<Track>>,
    counters: [Vec<(i64, Sample)>; 6],
}

#[derive(Default, PartialEq, PartialOrd)]
struct GraphKey {
    short_name: String,
    device_id: i64,
    iteration: i64,
}

impl Strings {
    fn intern(&mut self, text: &str) -> u32 {
        if let Some(&id) = self.index.get(text) {
            return id;
        }
        self.list.push(text.to_string());
        self.index.insert(text.to_string(), self.list.len() as u32 - 1);
        self.list.len() as u32 - 1
    }

    fn get(&self, id: u32) -> &str {
        &self.list[id as usize]
    }
}

fn digits(text: &str) -> usize {
    text.bytes().take_while(u8::is_ascii_digit).count()
}

fn transfer(name: &str) -> Option<&'static str> {
    TRANSFERS.into_iter().find(|prefix| name.strip_prefix(prefix).is_some_and(|rest| rest.strip_prefix('.').unwrap_or(rest).bytes().all(|byte| byte.is_ascii_digit())))
}

fn hlo_id(name: &str) -> i64 {
    let Some((head, tail)) = name.split_once('.') else { return 0 };
    let valid = !head.is_empty() && head.bytes().all(|byte| byte.is_ascii_alphabetic() || byte == b'-') && !tail.is_empty() && digits(tail) == tail.len();
    if valid { tail.parse().unwrap_or(0) } else { 0 }
}

fn graph_class(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-')
}

fn graph_keys(key: &str) -> impl Iterator<Item = (usize, &str, &str)> {
    key.match_indices("device_").filter_map(|(start, _)| {
        let rest = &key[start + "device_".len()..];
        let count = digits(rest);
        Some((start, &rest[..count], rest[count..].strip_prefix("_gid_").filter(|_| count > 0)?))
    })
}

fn graph_name(name: &str) -> Option<(&str, &str)> {
    let (0, device, tail) = graph_keys(name).next()? else { return None };
    let trailing = tail.len() - tail.bytes().rev().take_while(u8::is_ascii_digit).count();
    let matched = tail.bytes().all(graph_class) && trailing >= 2 && trailing < tail.len() && tail.as_bytes()[trailing - 1] == b'_';
    matched.then_some((device, tail))
}

fn graph_rendezvous(key: &str) -> &str {
    let rendezvous = graph_keys(key).find_map(|(_, _, tail)| {
        let run = &tail[..tail.bytes().take_while(|&byte| graph_class(byte)).count()];
        let bytes = run.as_bytes();
        let split = (1..bytes.len().saturating_sub(1)).rev().find(|&index| bytes[index] == b'_' && bytes[index + 1].is_ascii_digit())?;
        Some(&run[..split + 1 + digits(&run[split + 1..])])
    });
    rendezvous.unwrap_or("")
}

fn parse_graph_key(key: &str) -> GraphKey {
    let mut info = GraphKey::default();
    let Some((0, device, tail)) = graph_keys(key).next() else { return info };
    let Some(dollar) = tail.find('$').filter(|&dollar| dollar > 0) else { return info };
    let after = &tail[dollar + 1..];
    let Some(marker) = after.match_indices("^i").map(|(index, _)| index).filter(|&index| after[index + 2..].starts_with(|c: char| c.is_ascii_digit())).last() else { return info };
    let Ok(device_id) = device.parse() else { return info };
    info.device_id = device_id;
    info.short_name = tail[..dollar].to_string();
    let iteration = &after[marker + 2..];
    if let Ok(value) = iteration[..digits(iteration)].parse() {
        info.iteration = value;
    }
    info
}

fn buffer_sizes(text: &str) -> Vec<(i64, i64, i64)> {
    let bytes = text.as_bytes();
    let matched = |start: usize| -> Option<(usize, [&str; 3])> {
        let mut position = start + 1;
        let number = |position: &mut usize| {
            let count = digits(&text[*position..]);
            *position += count;
            (count > 0).then(|| &text[*position - count..*position])
        };
        let source = number(&mut position)?;
        position = text[position..].starts_with("->").then_some(position + 2)?;
        let destination = number(&mut position)?;
        position = text[position..].starts_with("d|$c").then_some(position + 4)?;
        number(&mut position)?;
        position = text[position..].starts_with('=').then_some(position + 1)?;
        let size = number(&mut position)?;
        Some((position, [source, destination, size]))
    };
    let (mut out, mut position) = (Vec::new(), 0);
    while let Some((start, (end, found))) = (position..bytes.len()).filter(|&start| bytes[start] == b's').find_map(|start| Some((start, matched(start)?))) {
        let Ok(parsed) = found.iter().map(|value| value.parse::<i64>()).collect::<Result<Vec<_>, _>>() else { break };
        out.push((parsed[0], parsed[1], parsed[2]));
        position = end.max(start + 1);
    }
    out
}

fn raw_stats(raw: &[u8], field: u32) -> Vec<(usize, Raw<'_>)> {
    fields(raw)
        .filter_map(|(tag, stat)| match (tag == field, stat) {
            (true, Field::Bytes(_, body)) => {
                let (mut id, mut value) = (0, None);
                for (tag, field) in fields(body) {
                    match (tag, field) {
                        (1, Field::Num(number)) => id = number as usize,
                        (2, Field::Num(bits)) => value = Some(Raw::Double(f64::from_bits(bits))),
                        (3, Field::Num(number)) => value = Some(Raw::Uint(number)),
                        (4, Field::Num(number)) => value = Some(Raw::Int(number as i64)),
                        (5, Field::Bytes(_, text)) => value = Some(Raw::Str(text)),
                        (6, Field::Bytes(_, data)) => value = Some(Raw::Bytes(data)),
                        (7, Field::Num(reference)) => value = Some(Raw::Ref(reference)),
                        _ => {}
                    }
                }
                Some((id, value?))
            }
            _ => None,
        })
        .collect()
}

fn stat_name(plane: &Plane, id: usize) -> &str {
    plane.stat_names.get(id).map_or("", |name| name)
}

fn add_args(plane: &Plane, raw: &[u8], field: u32, strings: &mut Strings, args: &mut Vec<Arg>) {
    for (id, value) in raw_stats(raw, field) {
        let name = stat_name(plane, id);
        if SKIPPED_STATS.contains(&name) {
            continue;
        }
        let value = match value {
            Raw::Int(number) => Value::Int(number),
            Raw::Uint(number) => Value::Uint(number),
            Raw::Double(number) => Value::Double(number),
            Raw::Str(text) => Value::Text(strings.intern(&crate::xplane::lossy(text))),
            Raw::Bytes(_) => Value::Text(strings.intern("")),
            Raw::Ref(reference) => Value::Text(strings.intern(stat_name(plane, reference as usize))),
        };
        args.push(Arg { key: strings.intern(name), value });
    }
}

type Deltas<T> = Vec<(i64, T)>;
type Prefixes = Vec<Option<(u32, Vec<Arg>)>>;

fn event(plane: &Plane, map: &[u8], prefixes: &mut Prefixes, ev: &crate::xplane::Ev, ts: i64, strings: &mut Strings) -> Event {
    let (name, mut args) = match plane.meta.get(ev.meta as usize) {
        Some(found) => prefixes[ev.meta as usize]
            .get_or_insert_with(|| {
                let mut args = Vec::new();
                let name = if found.display.is_empty() {
                    strings.intern(&found.name)
                } else {
                    let long_name = strings.intern(&found.long_name(map));
                    args.push(Arg { key: strings.intern("long_name"), value: Value::Text(long_name) });
                    strings.intern(&found.display)
                };
                add_args(plane, slice(map, found.raw), 5, strings, &mut args);
                (name, args)
            })
            .clone(),
        None => (strings.intern(""), Vec::new()),
    };
    add_args(plane, slice(map, ev.raw), 4, strings, &mut args);
    Event { name, ts, dur: ev.dur as i64, args, run_id: -1, flows: Vec::new() }
}

fn arg_text<'a>(event: &Event, strings: &'a Strings, key: &str) -> Option<&'a str> {
    event.args.iter().find(|arg| strings.get(arg.key) == key && matches!(arg.value, Value::Text(_))).map(|arg| if let Value::Text(id) = arg.value { strings.get(id) } else { "" })
}

fn arg_int(event: &Event, strings: &Strings, key: &str) -> Option<i64> {
    event.args.iter().find_map(|arg| if let (true, Value::Int(value)) = (strings.get(arg.key) == key, arg.value) { Some(value) } else { None })
}

fn load(planes: &[Plane], map: &[u8]) -> Trace {
    let mut trace = Trace::default();
    let origin_ps = crate::xplane::origin_ns(planes).wrapping_mul(1000);
    let mut tpu_ids = Vec::new();
    for plane in planes {
        let Some(tpu) = plane.name.strip_prefix(TPU_PLANE).filter(|id| !id.is_empty() && digits(id) == id.len()).and_then(|id| id.parse::<i64>().ok()) else { continue };
        tpu_ids.push(tpu);
        let (mut tracks, mut prefixes) = (Vec::new(), vec![None; plane.meta.len()]);
        for line in plane.lines.iter().filter(|line| TPU_LINES.contains(&line.name.as_str())) {
            let events = line.events.iter().map(|ev| event(plane, map, &mut prefixes, ev, (ev.ts as i64).wrapping_add(origin_ps), &mut trace.strings)).collect();
            tracks.push(Track { name: line.name.clone(), events });
        }
        trace.tpu.entry(tpu).or_default().extend(tracks);
    }
    tpu_ids.sort_unstable();
    let mut raw_fragments: BTreeMap<i64, Vec<Track>> = BTreeMap::new();
    for plane in planes.iter().filter(|plane| plane.name == MEGASCALE_PLANE) {
        let (mut positions, mut prefixes): (HashMap<String, (i64, usize)>, Prefixes) = (HashMap::default(), vec![None; plane.meta.len()]);
        for line in &plane.lines {
            for ev in &line.events {
                let key = raw_stats(slice(map, ev.raw), 4).into_iter().rfind(|(id, _)| stat_name(plane, *id) == "graph_key").map_or(String::new(), |(_, value)| match value {
                    Raw::Str(text) => crate::xplane::lossy(text).into_owned(),
                    Raw::Ref(reference) => stat_name(plane, reference as usize).to_string(),
                    _ => String::new(),
                });
                if key.is_empty() {
                    continue;
                }
                let Some(device) = graph_keys(&key).next().map(|(_, device, _)| device.parse().unwrap_or(-1)).filter(|&device| device != -1) else { continue };
                let (device, index) = *positions.entry(key.clone()).or_insert_with(|| {
                    let tracks = raw_fragments.entry(device).or_default();
                    tracks.push(Track { name: key.split('$').next().unwrap_or("").to_string(), events: Vec::new() });
                    (device, tracks.len() - 1)
                });
                let built = event(plane, map, &mut prefixes, ev, (ev.ts as i64).wrapping_add(origin_ps), &mut trace.strings);
                raw_fragments.get_mut(&device).unwrap()[index].events.push(built);
            }
        }
    }
    if raw_fragments.is_empty() {
        return trace;
    }
    let devices: Vec<i64> = raw_fragments.keys().copied().collect();
    let targets: HashMap<i64, i64> = devices.iter().zip(&tpu_ids).map(|(&device, &tpu)| (device, tpu)).collect();
    for (device, mut tracks) in raw_fragments {
        let strings = &trace.strings;
        let key_of = |track: &Track| parse_graph_key(arg_text(&track.events[0], strings, "graph_key").unwrap_or(""));
        tracks.sort_by(|a, b| key_of(a).partial_cmp(&key_of(b)).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(&tpu) = targets.get(&device) {
            trace.megascale.insert(tpu, tracks);
        }
    }
    trace
}

fn upper_bound(runs: &[(i64, i64)], ts: i64) -> usize {
    let (mut first, mut length) = (0, runs.len());
    while length > 0 {
        let half = length >> 1;
        if ts < runs[first + half].0 {
            length = half;
        } else {
            first += half + 1;
            length -= half + 1;
        }
    }
    first
}

fn sort_track(track: &mut Track) {
    let mut order: Vec<usize> = (0..track.events.len()).collect();
    let events = &track.events;
    std_sort(&mut order, &|a, b| if events[a].ts == events[b].ts { events[a].dur > events[b].dur } else { events[a].ts < events[b].ts });
    let mut taken = std::mem::take(&mut track.events);
    track.events = order.into_iter().map(|index| std::mem::take(&mut taken[index])).collect();
}

fn assign_run_ids(trace: &mut Trace) {
    let run_id = trace.strings.intern("run_id");
    let mut runs: HashMap<i64, Vec<(i64, i64)>> = HashMap::default();
    for (&tpu, tracks) in &mut trace.tpu {
        for track in tracks.iter_mut().filter(|track| track.name.contains("XLA Modules")) {
            let list = runs.entry(tpu).or_default();
            let mut counter = 1i64;
            for event in &mut track.events {
                let found = event.args.iter().find_map(|arg| if let (true, Value::Uint(value)) = (arg.key == run_id, arg.value) { Some(value as i64) } else { None });
                let id = found.unwrap_or_else(|| {
                    counter += 1;
                    event.args.push(Arg { key: run_id, value: Value::Int(counter - 1) });
                    counter - 1
                });
                event.run_id = id;
                list.push((event.ts, id));
            }
        }
    }
    for fragments in [&mut trace.tpu, &mut trace.megascale] {
        for (tpu, tracks) in fragments.iter_mut() {
            let Some(list) = runs.get(tpu).filter(|list| !list.is_empty()) else { continue };
            for track in tracks.iter_mut().filter(|track| !track.name.contains("XLA Modules")) {
                for event in &mut track.events {
                    let position = upper_bound(list, event.ts);
                    if position > 0 {
                        event.run_id = list[position - 1].1;
                        event.args.push(Arg { key: run_id, value: Value::Int(event.run_id) });
                    }
                }
            }
        }
    }
}

fn group_tiny_events(trace: &mut Trace) {
    let (description_key, description) = (trace.strings.intern("description"), trace.strings.intern(HIDDEN_DESCRIPTION));
    let hidden_key = trace.strings.intern("hidden_events");
    let Trace { strings, tpu, .. } = trace;
    for track in tpu.values_mut().flatten().filter(|track| track.name.contains("XLA Ops") || track.name.contains("XLA TraceMe")) {
        let events = std::mem::take(&mut track.events);
        let mut read = 0;
        while read < events.len() {
            let start = &events[read];
            let tiny = |event: &Event| event.dur < TINY_PS && transfer(strings.get(event.name)).is_none();
            let (mut next, mut end) = (read + 1, start.ts + start.dur);
            while tiny(start) && events.get(next).is_some_and(|current| tiny(current) && current.ts / 1000 == start.ts / 1000 && current.run_id == start.run_id) {
                end = end.max(events[next].ts + events[next].dur);
                next += 1;
            }
            if next - read >= 2 {
                let hidden = events[read..next].iter().map(|event| strings.get(event.name)).collect::<Vec<_>>().join(", ");
                let hidden = strings.intern(&hidden);
                let summary_name = strings.intern(&format!("{} events are hidden", next - read));
                track.events.push(Event {
                    name: summary_name,
                    ts: start.ts,
                    dur: end - start.ts,
                    run_id: start.run_id,
                    args: vec![Arg { key: description_key, value: Value::Text(description) }, Arg { key: hidden_key, value: Value::Text(hidden) }],
                    flows: Vec::new(),
                });
            } else {
                track.events.push(start.clone());
            }
            read = next;
        }
    }
}

fn mark_last_dma_events(trace: &mut Trace) {
    let last = trace.strings.intern("is_last_instance");
    let Trace { strings, megascale, .. } = trace;
    for track in megascale.values_mut().flatten() {
        let executions: Vec<usize> = (0..track.events.len()).filter(|&index| graph_name(strings.get(track.events[index].name)).is_some()).collect();
        for index in executions {
            let end = track.events[index].ts + track.events[index].dur;
            let (mut h2d, mut d2h) = (None, None);
            for later in index + 1..track.events.len() {
                if track.events[later].ts >= end {
                    break;
                }
                match strings.get(track.events[later].name) {
                    "HostToDevice END" => h2d = Some(later),
                    "DeviceToHost END" => d2h = Some(later),
                    _ => {}
                }
            }
            for found in [h2d, d2h].into_iter().flatten() {
                track.events[found].args.push(Arg { key: last, value: Value::Int(1) });
            }
        }
    }
}

type Queue = HashMap<String, VecDeque<i64>>;

struct Flows {
    next: i64,
    flow_in: u32,
    flow_out: u32,
}

impl Flows {
    fn start(&mut self, producer: &mut Event) -> i64 {
        let id = self.next;
        self.next += 1;
        producer.flows.push((id, false));
        producer.args.push(Arg { key: self.flow_out, value: Value::Int(id) });
        id
    }

    fn push(&mut self, queue: &mut Queue, key: &str, producer: &mut Event) {
        let id = self.start(producer);
        queue.entry(key.to_string()).or_default().push_back(id);
    }

    fn pop(&mut self, queue: &mut Queue, key: &str, consumer: &mut Event) {
        if let Some(id) = queue.get_mut(key).and_then(VecDeque::pop_front) {
            consumer.args.push(Arg { key: self.flow_in, value: Value::Int(id) });
            consumer.flows.push((id, true));
        }
    }
}

fn xla_ops(tracks: &mut [Track]) -> impl Iterator<Item = &mut Event> {
    tracks.iter_mut().filter(|track| track.name.contains("XLA Ops")).flat_map(|track| track.events.iter_mut()).filter(|event| event.run_id != -1)
}

fn resolve_flows(trace: &mut Trace) {
    let mut flows = Flows { next: 1, flow_in: trace.strings.intern("flow_in"), flow_out: trace.strings.intern("flow_out") };
    let Trace { strings, tpu, megascale, .. } = trace;
    let mut send_rendezvous: HashMap<String, String> = HashMap::default();
    let [mut send_to_send_done, mut send_to_d2h, mut recv_to_h2d, mut d2h_to_send_done] = std::array::from_fn(|_| Queue::default());
    let key = |tpu: i64, run: i64, rendezvous: &str| format!("{tpu}_{run}_{rendezvous}");
    for (&device, tracks) in tpu.iter_mut() {
        for event in xla_ops(tracks) {
            let name = strings.get(event.name);
            let sending = match transfer(name) {
                Some("send") => true,
                Some("recv") => false,
                _ => continue,
            };
            let long_name = arg_text(event, strings, "long_name").unwrap_or("");
            let rendezvous = long_name.split_once(RENDEZVOUS_PREFIX).and_then(|(_, rest)| rest.split_once('"')).map_or("", |(rendezvous, _)| rendezvous);
            if rendezvous.is_empty() {
                continue;
            }
            let queue_key = key(device, event.run_id, rendezvous);
            if sending {
                send_rendezvous.insert(key(device, event.run_id, &hlo_id(name).to_string()), rendezvous.to_string());
                flows.push(&mut send_to_send_done, &queue_key, event);
                flows.start(event);
                flows.push(&mut send_to_d2h, &queue_key, event);
            } else {
                flows.start(event);
                flows.push(&mut recv_to_h2d, &queue_key, event);
            }
        }
    }
    for (&device, tracks) in megascale.iter_mut() {
        for event in tracks.iter_mut().flat_map(|track| track.events.iter_mut()).filter(|event| event.run_id != -1) {
            let (d2h, start) = match strings.get(event.name) {
                "DeviceToHost START" => (true, true),
                "DeviceToHost END" => (true, false),
                "HostToDevice START" => (false, true),
                "HostToDevice END" => (false, false),
                _ => continue,
            };
            let rendezvous = graph_rendezvous(arg_text(event, strings, "graph_key").unwrap_or(""));
            if rendezvous.is_empty() {
                continue;
            }
            let queue_key = key(device, event.run_id, rendezvous);
            if start {
                if arg_int(event, strings, "action_index") == Some(0) {
                    flows.pop(if d2h { &mut send_to_d2h } else { &mut recv_to_h2d }, &queue_key, event);
                }
            } else if arg_int(event, strings, "is_last_instance") == Some(1) {
                if d2h {
                    flows.push(&mut d2h_to_send_done, &queue_key, event);
                } else {
                    event.dur = (event.dur - 1000).max(0);
                    flows.start(event);
                }
            }
        }
    }
    for (&device, tracks) in tpu.iter_mut() {
        for event in xla_ops(tracks).filter(|event| transfer(strings.get(event.name)) == Some("send-done")) {
            let hlo_key = key(device, event.run_id, &hlo_id(strings.get(event.name)).to_string());
            let Some(rendezvous) = send_rendezvous.get(&hlo_key).filter(|rendezvous| !rendezvous.is_empty()) else { continue };
            let queue_key = key(device, event.run_id, rendezvous);
            flows.pop(&mut send_to_send_done, &queue_key, event);
            flows.pop(&mut d2h_to_send_done, &queue_key, event);
        }
    }
}

fn counter<T: Copy + Default + std::ops::AddAssign + PartialOrd>(mut deltas: Vec<(i64, T)>, sample: fn(T) -> Sample) -> Vec<(i64, Sample)> {
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut current = T::default();
    let mut points = Vec::new();
    for (index, &(ts, delta)) in deltas.iter().enumerate() {
        current += delta;
        if deltas.get(index + 1).is_none_or(|next| next.0 != ts) {
            points.push((ts, sample(current)));
        }
    }
    points
}

fn add_global_counters(trace: &mut Trace) {
    let [mut rx, mut tx, mut count, mut bytes]: [Deltas<i64>; 4] = Default::default();
    let [mut rx_bandwidth, mut tx_bandwidth]: [Deltas<f64>; 2] = Default::default();
    let strings = &trace.strings;
    for event in trace.megascale.values().flatten().flat_map(|track| &track.events) {
        let name = strings.get(event.name);
        let end = event.ts + event.dur;
        if graph_name(name).is_some() {
            count.extend([(event.ts, 1), (end, -1)]);
            if let Some(size) = arg_int(event, strings, "input_size") {
                bytes.extend([(event.ts, size), (end, -size)]);
            }
            continue;
        }
        let (receive, window_ps) = match name {
            "NetworkReceive END" => {
                match event.args.iter().find_map(|arg| if let (true, Value::Uint(value)) = (strings.get(arg.key) == "network_transport_latency_us", arg.value) { Some(value) } else { None }) {
                    Some(latency) => (true, latency as i64),
                    None => continue,
                }
            }
            "NetworkSend END" => match arg_int(event, strings, "action_duration_ns") {
                Some(duration) => (false, duration),
                None => continue,
            },
            _ => continue,
        };
        let Some(sizes) = arg_text(event, strings, "buffer_sizes") else { continue };
        for (source, destination, size) in buffer_sizes(sizes) {
            if source == destination {
                continue;
            }
            let start = end - window_ps * if receive { 1_000_000 } else { 1000 };
            let (amounts, rates) = if receive { (&mut rx, &mut rx_bandwidth) } else { (&mut tx, &mut tx_bandwidth) };
            amounts.extend([(start, size), (end, -size)]);
            if window_ps > 0 {
                let rate = size as f64 * 8.0 / if receive { window_ps as f64 * 1000.0 } else { window_ps as f64 };
                if rate <= MAX_RATE_DELTA_GBPS {
                    rates.extend([(start, rate), (end, -rate)]);
                }
            }
        }
    }
    trace.counters =
        [counter(rx, Sample::Int), counter(tx, Sample::Int), counter(rx_bandwidth, Sample::Double), counter(tx_bandwidth, Sample::Double), counter(count, Sample::Int), counter(bytes, Sample::Int)];
}

fn rename_track(track: &mut Track) {
    if let Some((device, part)) = graph_name(&track.name).filter(|(device, _)| device.parse::<u64>().is_ok()) {
        track.name = format!("{part} ({})", device.parse::<u64>().unwrap());
        return;
    }
    let renamed = match track.name.as_str() {
        "Steps" => "1. Steps",
        "XLA Modules" => "2. XLA Modules",
        "XLA Ops" => "3. XLA Ops",
        "XLA TraceMe" => "4. XLA TraceMe",
        _ => return,
    };
    track.name = renamed.into();
}

fn process(trace: &mut Trace) {
    trace.tpu.values_mut().chain(trace.megascale.values_mut()).flatten().for_each(sort_track);
    assign_run_ids(trace);
    group_tiny_events(trace);
    mark_last_dma_events(trace);
    resolve_flows(trace);
    add_global_counters(trace);
    trace.tpu.values_mut().chain(trace.megascale.values_mut()).flatten().for_each(rename_track);
}

fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn number(out: &mut Vec<u8>, field: u64, value: u64) {
    varint(out, field << 3);
    varint(out, value);
}

fn fixed(out: &mut Vec<u8>, field: u64, bits: u64) {
    varint(out, field << 3 | 1);
    out.extend_from_slice(&bits.to_le_bytes());
}

fn bytes(out: &mut Vec<u8>, field: u64, body: &[u8]) {
    varint(out, field << 3 | 2);
    varint(out, body.len() as u64);
    out.extend_from_slice(body);
}

#[derive(Default)]
struct Packet {
    timestamp: Option<u64>,
    event: Vec<u8>,
    interned: [Vec<u8>; 3],
    descriptor: Vec<u8>,
}

struct Writer<'a> {
    trace: &'a Trace,
    out: Vec<u8>,
    next_uuid: u64,
    next_iid: u64,
    interned: [HashMap<u32, u64>; 3],
}

impl Writer<'_> {
    fn emit(&mut self, packet: &Packet) {
        let mut body = Vec::new();
        if let Some(timestamp) = packet.timestamp {
            number(&mut body, 8, timestamp);
        }
        number(&mut body, 10, SEQUENCE_ID);
        if !packet.event.is_empty() {
            bytes(&mut body, 11, &packet.event);
        }
        if packet.interned.iter().any(|entries| !entries.is_empty()) {
            bytes(&mut body, 12, &packet.interned.concat());
        }
        number(&mut body, 13, if self.out.is_empty() { INCREMENTAL_STATE_CLEARED | NEEDS_INCREMENTAL_STATE } else { NEEDS_INCREMENTAL_STATE });
        if !packet.descriptor.is_empty() {
            bytes(&mut body, 60, &packet.descriptor);
        }
        bytes(&mut self.out, 1, &body);
    }

    fn descriptor(&mut self, name: &str, parent: u64, counter: Option<(&str, &str)>) -> u64 {
        let uuid = self.next_uuid;
        self.next_uuid += 1;
        let mut descriptor = Vec::new();
        number(&mut descriptor, 1, uuid);
        bytes(&mut descriptor, 2, name.as_bytes());
        if parent != 0 {
            number(&mut descriptor, 5, parent);
        }
        if let Some((unit, share)) = counter {
            let mut body = Vec::new();
            bytes(&mut body, 6, unit.as_bytes());
            if !share.is_empty() {
                bytes(&mut body, 7, share.as_bytes());
            }
            bytes(&mut descriptor, 8, &body);
        }
        self.emit(&Packet { descriptor, ..Default::default() });
        uuid
    }

    fn intern(&mut self, which: usize, id: u32, packet: &mut Packet) -> u64 {
        if let Some(&iid) = self.interned[which].get(&id) {
            return iid;
        }
        let iid = self.next_iid;
        self.next_iid += 1;
        self.interned[which].insert(id, iid);
        let mut entry = Vec::new();
        number(&mut entry, 1, iid);
        bytes(&mut entry, 2, self.trace.strings.get(id).as_bytes());
        bytes(&mut packet.interned[which], [2, 3, 29][which], &entry);
        iid
    }

    fn value(&mut self, value: Value, packet: &mut Packet) -> Vec<u8> {
        let mut out = Vec::new();
        match value {
            Value::Int(number_value) => number(&mut out, 4, number_value as u64),
            Value::Uint(number_value) => number(&mut out, 3, number_value),
            Value::Double(double) => fixed(&mut out, 5, double.to_bits()),
            Value::Text(id) => {
                let iid = self.intern(2, id, packet);
                number(&mut out, 17, iid);
            }
        }
        out
    }

    fn event(&mut self, event: &Event, track: u64) {
        let instant = event.dur == 0;
        let mut packet = Packet { timestamp: Some((event.ts / 1000) as u64), ..Default::default() };
        let name = self.intern(0, event.name, &mut packet);
        let mut keys: Vec<u32> = Vec::new();
        for arg in &event.args {
            if !keys.contains(&arg.key) {
                keys.push(arg.key);
            }
        }
        let mut body = Vec::new();
        for key in keys {
            let mut annotation = Vec::new();
            let iid = self.intern(1, key, &mut packet);
            number(&mut annotation, 1, iid);
            let encoded: Vec<Vec<u8>> = event.args.iter().filter(|arg| arg.key == key).map(|arg| self.value(arg.value, &mut packet)).collect();
            match &encoded[..] {
                [single] => annotation.extend(single),
                many => many.iter().for_each(|value| bytes(&mut annotation, 12, value)),
            }
            bytes(&mut body, 4, &annotation);
        }
        number(&mut body, 9, if instant { INSTANT } else { SLICE_BEGIN });
        number(&mut body, 10, name);
        number(&mut body, 11, track);
        for &(id, sink) in &event.flows {
            if sink || instant {
                fixed(&mut body, 47, id as u64);
            }
        }
        packet.event = body;
        self.emit(&packet);
        if instant {
            return;
        }
        let mut end = Vec::new();
        number(&mut end, 9, SLICE_END);
        number(&mut end, 11, track);
        for &(id, _) in event.flows.iter().filter(|(_, sink)| !sink) {
            fixed(&mut end, 47, id as u64);
        }
        self.emit(&Packet { timestamp: Some(((event.ts + event.dur) / 1000) as u64), event: end, ..Default::default() });
    }

    fn track(&mut self, track: &Track, parent: u64) {
        let uuid = self.descriptor(&track.name, parent, None);
        for event in &track.events {
            self.event(event, uuid);
        }
    }

    fn counter(&mut self, (name, unit, share): (&str, &str, &str), points: &[(i64, Sample)], parent: u64) {
        let uuid = self.descriptor(name, parent, Some((unit, share)));
        for &(ts, value) in points {
            let mut body = Vec::new();
            number(&mut body, 9, COUNTER);
            number(&mut body, 11, uuid);
            match value {
                Sample::Int(value) => number(&mut body, 30, value as u64),
                Sample::Double(value) => fixed(&mut body, 44, value.to_bits()),
            }
            self.emit(&Packet { timestamp: Some((ts / 1000) as u64), event: body, ..Default::default() });
        }
    }

    fn write(mut self) -> Vec<u8> {
        let trace = self.trace;
        if trace.counters.iter().any(|points| !points.is_empty()) {
            let global = self.descriptor("1. Global Counters", 0, None);
            for (name, range) in [("Network", 0..NETWORK_COUNTERS), ("Megascale", NETWORK_COUNTERS..COUNTERS.len())] {
                if trace.counters[range.clone()].iter().any(|points| !points.is_empty()) {
                    let parent = self.descriptor(name, global, None);
                    for index in range.filter(|&index| !trace.counters[index].is_empty()) {
                        self.counter(COUNTERS[index], &trace.counters[index], parent);
                    }
                }
            }
        }
        let devices: std::collections::BTreeSet<i64> = trace.tpu.keys().chain(trace.megascale.keys()).copied().collect();
        if !devices.is_empty() {
            let tpus = self.descriptor("2. TPUs", 0, None);
            for device in devices {
                let uuid = self.descriptor(&format!("/device:TPU:{device}"), tpus, None);
                for track in trace.tpu.get(&device).into_iter().flatten() {
                    self.track(track, uuid);
                }
                if let Some(tracks) = trace.megascale.get(&device) {
                    let parent = self.descriptor("Megascale", uuid, None);
                    tracks.iter().for_each(|track| self.track(track, parent));
                }
            }
        }
        self.out
    }
}

pub fn render(map: &[u8]) -> Option<Vec<u8>> {
    let planes = crate::xplane::parse(map).ok()?;
    let mut trace = load(&planes, map);
    process(&mut trace);
    let proto = Writer { trace: &trace, out: Vec::new(), next_uuid: 1, next_iid: 1, interned: Default::default() }.write();
    let mut encoder = flate2::GzBuilder::new().operating_system(OS_UNIX).write(Vec::new(), flate2::Compression::new(1));
    encoder.write_all(&proto).ok()?;
    encoder.finish().ok()
}
