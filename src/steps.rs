use crate::derive::is_tensor_core;
use crate::framework_op_stats::{is_jax_op_type, is_tf_op_name, is_tf_op_type, parse_tf_op};
use crate::group::is_sparse_core;
use crate::input_pipeline_analyzer::{TC_IDLE, tpu_step_details};
use crate::opstats::{Builder, Db, EventReader, IDLE, Metrics, Template, safe_divide};
use crate::roofline::accumulate;
use crate::xplane::{Ev, Field, NONE_GROUP, Own, Plane, Value, fields, slice, stats};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};

pub const SPARSE_CORE_START: u32 = 1_000_000;
pub const CPU_ONLY: u8 = 1;
pub const GPU: u8 = 2;
pub const TPU: u8 = 3;
pub const UNKNOWN_TIME: u32 = 0;
pub const HOST_COMPUTE: u32 = 10;
pub const HOST_COMPILE: u32 = 60;
pub const HOST_TO_HOST: u32 = 70;
pub const HOST_TO_DEVICE: u32 = 80;
pub const HOST_PREPARE: u32 = 90;
pub const DEVICE_COLLECTIVES: u32 = 100;
pub const HOST_WAIT_INPUT: u32 = 110;
pub const DEVICE_TO_DEVICE: u32 = 120;
pub const DEVICE_TO_HOST: u32 = 130;
pub const DEVICE_COMPUTE_32: u32 = 140;
pub const DEVICE_COMPUTE_16: u32 = 150;
pub const DEVICE_WAIT_DEVICE: u32 = 160;
pub const DEVICE_WAIT_HOST: u32 = 170;
const LAST_EVENT_TYPE: usize = 170;
const DERIVED_MIN: i32 = 0xdeadbeef_u32 as i32;
const DERIVED_MAX: i32 = DERIVED_MIN + 389;
const TPU_PREFIX: &str = "/device:TPU:";
const HOST_PLANE: &str = "/host:CPU";
const OFF_DUTY: [&str; 8] = ["infeed", "outfeed", "host send", "host send-done", "host recv", "host recv-done", "async-done", "megacore fusion"];
const PROGRAM_LINES: [&str; 3] = ["XLA Ops", "TensorFlow Ops", "Sparse Core Ops"];
const EAGER_WRAPPERS: [&str; 4] = ["EagerExecute", "EagerLocalExecute", "EagerKernelExecute", "FunctionRun"];
const DEVICE_STATS: [&str; 10] = ["group_id", "device_offset_ps", "device_duration_ps", "all_reduce_unique_id", "program_id", "symbol_id", "flops", "model_flops", "uses_ici", "hlo_category"];
const GPU_MEMCPY: [(&str, u32); 3] = [("MemcpyHToD", HOST_TO_DEVICE), ("MemcpyDToH", DEVICE_TO_HOST), ("MemcpyDToD", DEVICE_TO_DEVICE)];
const PROGRAM: usize = 4;
const SYMBOL: usize = 5;
const FLOPS: usize = 6;
const CATEGORY: usize = 9;

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Span {
    pub begin: u64,
    pub duration: u64,
}

impl Span {
    pub fn end(self) -> u64 {
        self.begin.wrapping_add(self.duration)
    }

    pub(crate) fn includes(self, other: Span) -> bool {
        self.begin <= other.begin && other.end() <= self.end()
    }
}

pub struct Marker {
    device: bool,
    core: Option<u32>,
    pub span: Span,
}

type Collective = (u64, u64, u64);

#[derive(Default)]
pub struct Details {
    pub markers: Vec<Marker>,
    pub events: Vec<(u32, Span)>,
    collectives: BTreeMap<u32, Vec<Collective>>,
    name: String,
    pub cores: BTreeMap<u32, (BTreeMap<String, u64>, u64)>,
}

pub type StepEvents = HashMap<i64, Details>;

impl Details {
    fn combine(&mut self, other: Details) {
        self.markers.extend(other.markers);
        self.events.extend(other.events);
        for (core, collectives) in other.collectives {
            self.collectives.entry(core).or_insert(collectives);
        }
        self.cores.extend(other.cores);
        if self.name.is_empty() {
            self.name = other.name;
        }
    }

    fn longest(&self, keep: impl Fn(&Marker) -> bool) -> Span {
        self.markers.iter().filter(|marker| keep(marker)).fold(Span::default(), |best, marker| if marker.span.duration > best.duration { marker.span } else { best })
    }
}

fn intersect<T>(mut source: HashMap<i64, T>, destination: &mut HashMap<i64, T>, merge: impl Fn(&mut T, T)) {
    if destination.is_empty() {
        *destination = source;
        return;
    }
    destination.retain(|step, _| source.contains_key(step));
    for (step, value) in destination.iter_mut() {
        merge(value, source.remove(step).unwrap());
    }
}

#[derive(Clone)]
pub enum Breakdown {
    Categories(BTreeMap<String, u64>),
    Types(BTreeMap<u32, u64>),
}

#[derive(Clone)]
pub struct StepInfo {
    pub name: String,
    pub begin: u64,
    pub duration: u64,
    pub breakdown: Breakdown,
}

#[derive(Clone, Default)]
pub struct StepRecord {
    pub num: u32,
    pub cores: BTreeMap<u32, StepInfo>,
    pub collectives: BTreeMap<u32, Vec<Collective>>,
}

#[derive(Clone, Default)]
pub struct Core {
    pub hostname: String,
    pub ordinal: u32,
    pub chip: u32,
    pub sparse: bool,
}

#[derive(Default)]
pub struct Extra {
    pub hostnames: Vec<String>,
    pub tasks: i32,
    pub empty_intersect: bool,
    pub device_type: String,
    pub core_count: i32,
    pub hardware: u8,
    pub training: bool,
    pub busy_ps: [u64; 2],
    pub idle_ps: [u64; 2],
    pub steps: Vec<StepRecord>,
    pub cores: BTreeMap<u32, Core>,
    pub mxu: f64,
    pub hbm: f64,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub host_input: Option<HashMap<i64, u64>>,
    pub infeed_enqueue: (u64, u64),
    pub megacore: bool,
    pub merged_vmem: bool,
    pub program_steps: Vec<(i64, [Metrics; 2])>,
    pub program_total: [Metrics; 2],
    pub precision: [u64; 2],
}

#[derive(Default)]
struct StepPrograms {
    markers: Vec<u64>,
    cores: Vec<(u64, [Metrics; 2], u64)>,
}

type Scanned<'a> = ([Option<i64>; 9], Option<Cow<'a, str>>);
type Op = ([Option<i64>; 4], [bool; 2]);
type Intervals = Vec<(u64, u64)>;
type TimedEvent<'a> = (&'a Ev, (u64, u64));
type Tracker = (Intervals, Option<(u64, u64)>);
type CategoryTimes = (HashMap<(i64, i64), u32>, BTreeMap<String, u64>, u64);

pub struct Device {
    pub events: StepEvents,
    programs: HashMap<i64, StepPrograms>,
    active: [Intervals; 2],
    total: Option<(u64, u64)>,
    core: Option<Core>,
}

fn scan<'a>(plane: &'a Plane, raw: &'a [u8], field: u32, ids: &[Option<usize>; 10]) -> Scanned<'a> {
    let mut out: Scanned = Default::default();
    for stat in stats(raw, field, |_| true) {
        match ids.iter().position(|&id| id == Some(stat.id)) {
            Some(CATEGORY) => {
                out.1 = Some(match stat.value {
                    Value::Str(bytes) => crate::xplane::lossy(bytes),
                    Value::Ref(id) => Cow::Borrowed(plane.stat_names.get(id as usize).map_or("", |name| &**name)),
                    _ => Cow::Borrowed(""),
                })
            }
            Some(index) => out.0[index] = stat.value.int().or(out.0[index]),
            None => {}
        }
    }
    out
}

fn nest<T>(items: impl Iterator<Item = (Span, T)>, mut finish: impl FnMut(T, Span, u64)) {
    let mut stack: Vec<(Span, u64, T)> = Vec::new();
    for (span, item) in items {
        while stack.last().is_some_and(|top| !top.0.includes(span)) {
            let (span, children, item) = stack.pop().unwrap();
            finish(item, span, span.duration.saturating_sub(children));
        }
        if let Some(top) = stack.last_mut() {
            top.1 = top.1.saturating_add(span.duration);
        }
        stack.push((span, 0, item));
    }
    for (span, children, item) in stack.into_iter().rev() {
        finish(item, span, span.duration.saturating_sub(children));
    }
}

fn step_programs(plane: &Plane, map: &[u8], templates: &[Template]) -> HashMap<i64, StepPrograms> {
    let mut markers: HashMap<i64, Vec<u64>> = HashMap::new();
    let mut ops: HashMap<i64, Vec<TimedEvent>> = HashMap::new();
    let reader = EventReader::new(plane);
    for line in &plane.lines {
        if line.name == "Steps" {
            markers.clear();
            for (index, event) in line.events.iter().enumerate().filter(|(_, event)| event.group != NONE_GROUP) {
                let merged =
                    line.steps.get(&index).and_then(|step| step.stats.iter().find(|(key, _)| *key == "device_duration_ps").filter(|_| step.stats.iter().any(|(key, _)| *key == "device_offset_ps")));
                let duration = merged.map_or_else(|| reader.span(map, event).1, |&(_, duration)| duration as u64);
                markers.entry(event.group).or_default().push(duration);
            }
        } else if PROGRAM_LINES.contains(&line.name.as_str()) {
            ops.clear();
            let events: Vec<(Span, &Ev)> = line
                .events
                .par_iter()
                .filter(|event| event.group != NONE_GROUP)
                .map(|event| {
                    let (begin, duration) = reader.span(map, event);
                    (Span { begin, duration }, event)
                })
                .collect();
            nest(events.into_iter(), |event, span, self_time| ops.entry(event.group).or_default().push((event, (span.duration, self_time))));
        }
    }
    ops.into_par_iter()
        .filter_map(|(group, events)| {
            let markers = markers.get(&group)?.clone();
            let mut builder = Builder::new(templates);
            events.into_iter().for_each(|(event, times)| builder.add(event, &reader.read(map, event), times, false));
            let (total_op_time_ps, (sums, infeed_outfeed)) = builder.program();
            Some((group, StepPrograms { markers, cores: vec![(total_op_time_ps, sums, infeed_outfeed)] }))
        })
        .collect()
}

pub fn device_plane(plane: &Plane, raw_plane: &[u8], map: &[u8], templates: &[Template], origin: u64, hostname: &str) -> Device {
    let ids = DEVICE_STATS.map(|name| plane.id(name));
    let (tensor, sparse) = (is_tensor_core(&plane.name), is_sparse_core(&plane.name));
    let step_core = if sparse { SPARSE_CORE_START } else { 0 } + plane.id as u32;
    let metas: Vec<Scanned> = plane.meta.iter().map(|meta| scan(plane, slice(map, meta.raw), 5, &ids)).collect();
    let span_of = |offset: Option<i64>, duration: Option<i64>, event: &Ev| match (offset, duration) {
        (Some(offset), Some(duration)) => Span { begin: offset as u64, duration: duration as u64 },
        _ => Span { begin: event.ts.wrapping_add(origin), duration: event.dur },
    };
    let mut device = Device { events: HashMap::new(), programs: HashMap::new(), active: Default::default(), total: None, core: None };
    let mut markers: StepEvents = HashMap::new();
    for line in &plane.lines {
        let name = line.name.as_str();
        let step_line = line.id as i32 == DERIVED_MIN || (tensor && name == "Steps") || (sparse && name == "Sparse Core Steps");
        let derived = (DERIVED_MIN..=DERIVED_MAX).contains(&(line.id as i32));
        let op_line = !step_line && !derived && ((tensor && matches!(name, "XLA Ops" | "Framework Ops" | "Sparse Core Ops")) || (sparse && name == "Sparse Core Ops"));
        if matches!(name, "XLA Ops" | "Sparse Core Ops" | "XLA Modules" | "Sparse Core Modules") {
            for event in &line.events {
                let (begin, end) = (event.ts.wrapping_add(origin), (event.ts + event.dur).wrapping_add(origin));
                if begin != 0 || end != begin {
                    device.total = Some(device.total.map_or((begin, end), |(low, high)| (low.min(begin), high.max(end))));
                }
                if name == "Sparse Core Ops" {
                    device.active.iter_mut().for_each(|intervals| intervals.push((begin, end)));
                }
            }
        }
        if name != "XLA Ops" && !op_line && !step_line {
            continue;
        }
        let ops: Vec<Op> = line
            .events
            .par_iter()
            .map(|event| {
                let (own, own_category) = scan(plane, slice(map, event.raw), 4, &ids);
                let (meta, meta_category) = &metas[event.meta as usize];
                let category = meta_category.as_deref().or(own_category.as_deref());
                let off = category.is_some_and(|category| OFF_DUTY.contains(&category));
                let custom_off = category == Some("custom-call")
                    && own_category.as_deref().or(meta_category.as_deref()) == Some("custom-call")
                    && (FLOPS..CATEGORY).all(|index| own[index].or(meta[index]).unwrap_or(0) <= 0);
                (own[..4].try_into().unwrap(), [!off, !(off || custom_off)])
            })
            .collect();
        if name == "XLA Ops" {
            for (event, (_, active)) in line.events.iter().zip(&ops) {
                for (intervals, _) in device.active.iter_mut().zip(active).filter(|(_, keep)| **keep) {
                    intervals.push((event.ts.wrapping_add(origin), (event.ts + event.dur).wrapping_add(origin)));
                }
            }
        }
        let grouped = line.events.iter().zip(&ops).enumerate().filter_map(|(index, (event, &(own, _)))| Some((index, event, own, if event.group != NONE_GROUP { event.group } else { own[0]? })));
        if step_line {
            markers = HashMap::new();
            for (index, event, [_, offset, duration, _], group) in grouped {
                let overrides = line.steps.get(&index);
                let stat = |key: &str, own: Option<i64>| overrides.and_then(|step| step.stats.iter().find(|(name, _)| *name == key).map(|(_, value)| *value)).or(own);
                let span = span_of(stat("device_offset_ps", offset), stat("device_duration_ps", duration), event);
                markers.entry(group).or_default().markers.push(Marker { device: true, core: Some(step_core), span });
            }
        } else if op_line {
            let mut result: StepEvents = HashMap::new();
            let mut dbs: HashMap<i64, CategoryTimes> = HashMap::new();
            let nested = grouped.map(|(_, event, [_, offset, duration, all_reduce], group)| {
                if let Some(unique) = all_reduce {
                    let start = offset.map_or(event.ts.wrapping_add(origin), |value| value as u64);
                    let end = duration.map_or((event.ts + event.dur).wrapping_add(origin), |value| start.wrapping_add(value as u64));
                    result.entry(group).or_default().collectives.entry(step_core).or_default().push((unique as u64, start, end));
                }
                (span_of(offset, duration, event), (group, event.meta))
            });
            nest(nested, |(group, meta_index), _, self_time| {
                let meta = &metas[meta_index as usize].0;
                let db = dbs.entry(group).or_default();
                let (Some(program), Some(symbol)) = (meta[PROGRAM], meta[SYMBOL].filter(|&symbol| symbol != 0)) else { return };
                let first = *db.0.entry((program, symbol)).or_insert(meta_index);
                let category = metas[first as usize].1.as_deref().unwrap_or("");
                match db.1.get_mut(category) {
                    Some(time) => *time += self_time,
                    None => _ = db.1.insert(category.to_string(), self_time),
                }
                db.2 += self_time;
            });
            for (group, (_, categories, total)) in dbs {
                result.entry(group).or_default().cores.insert(step_core, (categories, total));
            }
            device.events = result;
        }
    }
    if !device.events.is_empty() {
        intersect(markers, &mut device.events, Details::combine);
    }
    device.core = plane.id("core_details").and_then(|id| match stats(raw_plane, 6, |stat| stat == id).next()?.value {
        Value::Bytes(bytes) => {
            let mut core = Core { hostname: hostname.to_string(), sparse, ..Default::default() };
            for (tag, field) in fields(bytes) {
                match (tag, field) {
                    (2, Field::Num(value)) => core.ordinal = value as u32,
                    (4, Field::Num(value)) => core.chip = value as u32,
                    _ => {}
                }
            }
            Some(core)
        }
        _ => None,
    });
    if tensor {
        device.programs = step_programs(plane, map, templates);
    }
    device
}

pub fn gpu_device(plane: &Plane, map: &[u8], origin: u64) -> StepEvents {
    let groups = crate::gpu::groups(plane, map);
    let lines = || plane.lines.iter().enumerate().filter(|(_, line)| !crate::derive::is_derived(line.id));
    let mut order: Vec<(u64, u64, i64)> = lines()
        .flat_map(|(line_index, line)| line.events.iter().enumerate().filter_map(|(index, event)| Some((event.ts + origin, event.dur, groups[line_index][index]?))).collect::<Vec<_>>())
        .collect();
    order.sort_by_key(|&(begin, duration, _)| (begin, duration));
    let mut markers: StepEvents = HashMap::new();
    let mut spans: Vec<(i64, u64, u64)> = Vec::new();
    for (begin, duration, group) in order {
        match spans.last_mut() {
            Some(last) if last.0 == group => last.2 = last.2.max(begin + duration),
            _ => spans.push((group, begin, begin + duration)),
        }
    }
    for (group, begin, end) in spans {
        markers.entry(group).or_default().markers.push(Marker { device: true, core: Some(plane.id as u32), span: Span { begin, duration: end - begin } });
    }
    let (scanner, shapes) = (crate::gpu::Scanner::new(plane, map), plane.id("tensor_shapes"));
    let kinds: Vec<(u32, bool)> = (0..plane.meta.len() as u32)
        .map(|meta| {
            let name = crate::gpu::event_name(plane, map, meta);
            let op = parse_tf_op(&name);
            let memcpy = GPU_MEMCPY.iter().find(|(kind, _)| op.kind == *kind && !name.contains(':')).map(|memcpy| memcpy.1);
            let collective = name.len() >= 4 && name.as_bytes()[..4].eq_ignore_ascii_case(b"nccl");
            (memcpy.unwrap_or(if collective { DEVICE_COLLECTIVES } else { UNKNOWN_TIME }), name.contains("half") || name.contains("fp16"))
        })
        .collect();
    let mut events: StepEvents = HashMap::new();
    for (line_index, line) in lines() {
        let mut stream: StepEvents = HashMap::new();
        for (index, event) in line.events.iter().enumerate() {
            let Some(group) = groups[line_index][index].filter(|&group| group >= 0 && scanner.scan(event).int(crate::gpu::CORRELATION).is_some()) else { continue };
            let kind = match kinds[event.meta as usize] {
                (UNKNOWN_TIME, half) => {
                    let shape = shapes.and_then(|id| stats(slice(map, event.raw), 4, |stat| stat == id).next()).map_or(Cow::Borrowed(""), |stat| crate::gpu::text(plane, &stat.value));
                    if (shape.is_empty() && half) || shape.contains("half") { DEVICE_COMPUTE_16 } else { DEVICE_COMPUTE_32 }
                }
                (kind, _) => kind,
            };
            let span = Span { begin: event.ts + origin, duration: event.dur };
            let details = stream.entry(group).or_default();
            details.events.push((kind, span));
            if kind == DEVICE_COLLECTIVES {
                let end_offset = (event.ts + origin).wrapping_sub((line.timestamp_ns as u64).wrapping_mul(1000)) + event.dur;
                details.collectives.entry(plane.id as u32).or_default().push((0, span.begin, end_offset));
            }
        }
        for (group, details) in stream {
            events.entry(group).or_default().combine(details);
        }
    }
    if !events.is_empty() {
        intersect(markers, &mut events, Details::combine);
    }
    events
}

fn merged_active(mut intervals: Vec<(u64, u64)>) -> u64 {
    intervals.par_sort_unstable_by_key(|&(begin, end)| (begin, std::cmp::Reverse(end)));
    let Some(&first) = intervals.first() else { return 0 };
    let (sum, (start, stop)) = intervals[1..]
        .iter()
        .fold((0u64, first), |(sum, (start, stop)), &(begin, end)| if begin <= stop { (sum, (start, stop.max(end))) } else { (sum.wrapping_add(stop.wrapping_sub(start)), (begin, end)) });
    sum.wrapping_add(stop.wrapping_sub(start))
}

fn nested(data: &[u8], wanted: u32) -> impl Iterator<Item = &[u8]> {
    fields(data).filter_map(move |(tag, field)| if let (true, Field::Bytes(_, body)) = (tag == wanted, field) { Some(body) } else { None })
}

pub fn host_steps(plane: &Plane, map: &[u8], origin: u64) -> StepEvents {
    let id = |name| plane.id(name);
    let (group_id, step_name, stage_name, consumer_type, consumer_id, producer_type, producer_id) = (id("group_id"), id("step_name"), id("_ipl_stage_name"), id("_ct"), id("_c"), id("_pt"), id("_p"));
    let find = |event: &Ev, id: Option<usize>| stats(slice(map, event.raw), 4, |stat| Some(stat) == id).next().map(|stat| stat.value);
    let number = |event: &Ev, id: Option<usize>| find(event, id).and_then(|value| value.int());
    let parents: HashSet<(i64, i64)> = if producer_type.is_some() && producer_id.is_some() {
        plane.lines.iter().flat_map(|line| &line.events).filter_map(|event| number(event, producer_type).zip(number(event, producer_id))).collect()
    } else {
        HashSet::new()
    };
    let per_line: Vec<StepEvents> = plane
        .lines
        .par_iter()
        .map(|line| {
            let mut result: StepEvents = HashMap::new();
            let mut stack: Vec<(bool, bool, Span)> = Vec::new();
            for (index, event) in line.events.iter().enumerate() {
                let name = &*plane.meta[event.meta as usize].name;
                let text = |id| find(event, id).map(|value| plane.text(&value)).unwrap_or_default();
                let step = line.steps.get(&index).map(|step| step.name.clone()).filter(|name| !name.is_empty()).unwrap_or_else(|| text(step_name));
                let async_parent = number(event, consumer_type).zip(number(event, consumer_id)).is_some_and(|key| parents.contains(&key));
                let span = Span { begin: event.ts.wrapping_add(origin), duration: event.dur };
                while stack.last().is_some_and(|top| !top.2.includes(span)) {
                    stack.pop();
                }
                let current = (!text(stage_name).is_empty(), async_parent || stack.last().is_some_and(|top| top.1 || top.0), span);
                stack.push(current);
                let group = if event.group != NONE_GROUP { event.group } else { number(event, group_id).unwrap_or(-1) };
                if group < 0 {
                    continue;
                }
                let explicit = (name.starts_with("train") || name.starts_with("test") || name.starts_with("TraceContext")) && !name.contains('/');
                let details = result.entry(group).or_default();
                if explicit || !step.is_empty() {
                    details.markers.push(Marker { device: false, core: None, span });
                } else if !EAGER_WRAPPERS.iter().any(|wrapper| name.starts_with(wrapper)) {
                    let op = parse_tf_op(name);
                    let memcpy = |kind: &str| op.kind == kind && !name.contains(':');
                    let kind = if op.kind.starts_with("InfeedEnqueue") || memcpy("MemcpyHToD") {
                        HOST_TO_DEVICE
                    } else if memcpy("MemcpyHToH") {
                        HOST_TO_HOST
                    } else if name.len() >= 15 && name.as_bytes()[..15].eq_ignore_ascii_case(b"IteratorGetNext") || (current.0 && !current.1) {
                        HOST_WAIT_INPUT
                    } else {
                        HOST_COMPUTE
                    };
                    details.events.push((kind, span));
                }
                if !step.is_empty() {
                    details.name = step;
                }
            }
            result
        })
        .collect();
    let mut combined: StepEvents = HashMap::new();
    for (step, details) in per_line.into_iter().flatten() {
        combined.entry(step).or_default().combine(details);
    }
    combined
}

pub fn non_overlapped(events: &[(u32, Span)]) -> Vec<(u32, Span)> {
    let mut boundaries: Vec<(u64, u32, bool)> = events.iter().flat_map(|&(kind, span)| [(span.begin, kind, true), (span.end(), kind, false)]).collect();
    boundaries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| if a.2 == b.2 { b.1.cmp(&a.1) } else { a.2.cmp(&b.2) }));
    let (mut counts, mut current, mut result) = (vec![0i64; LAST_EVENT_TYPE + 1], UNKNOWN_TIME, Vec::new());
    for window in boundaries.windows(2) {
        let (_, kind, start) = window[0];
        if start {
            counts[kind as usize] += 1;
            current = current.max(kind);
        } else {
            counts[kind as usize] -= 1;
            if kind == current && counts[kind as usize] == 0 {
                current = (0..kind as usize).rev().find(|&index| counts[index] > 0).map_or(UNKNOWN_TIME, |index| index as u32);
            }
        }
        let (begin, end) = (window[0].0, window[1].0);
        result.push((current, Span { begin, duration: end.saturating_sub(begin) }));
    }
    result
}

fn step_db(events: &StepEvents, has_device: bool) -> Vec<StepRecord> {
    let mut numbers: Vec<i64> = events.keys().copied().collect();
    numbers.sort_unstable();
    numbers
        .into_iter()
        .filter_map(|number| {
            let details = &events[&number];
            let (host, device) = (details.longest(|marker| !marker.device), details.longest(|marker| marker.device));
            let step_time = if (device.begin == 0 && device.duration == 0) || host.includes(device) { host } else { device };
            let info = |span: Span, breakdown| StepInfo { name: details.name.clone(), begin: span.begin, duration: span.duration, breakdown };
            let cores: BTreeMap<u32, StepInfo> = if !details.cores.is_empty() {
                let cores = details.cores.iter().filter(|_| step_time.duration != 0);
                cores
                    .map(|(&core, (categories, total_op))| {
                        let span = if core >= SPARSE_CORE_START { details.longest(|marker| marker.core == Some(core)) } else { step_time };
                        let mut breakdown = categories.clone();
                        *breakdown.entry(IDLE.to_string()).or_default() += (*total_op).max(span.duration) - total_op;
                        (core, info(span, Breakdown::Categories(breakdown)))
                    })
                    .collect()
            } else {
                let mut types: BTreeMap<u32, u64> = BTreeMap::new();
                for &(kind, span) in &details.events {
                    *types.entry(kind).or_default() +=
                        if step_time.begin <= span.end() && span.begin <= step_time.end() { step_time.end().min(span.end()) - step_time.begin.max(span.begin) } else { 0 };
                }
                let covered: u64 = types.values().sum();
                if covered < step_time.duration {
                    *types.entry(UNKNOWN_TIME).or_default() += step_time.duration - covered;
                }
                let well_formed = if has_device { types.contains_key(&DEVICE_COMPUTE_16) || types.contains_key(&DEVICE_COMPUTE_32) } else { types.contains_key(&HOST_COMPUTE) };
                if !well_formed {
                    return None;
                }
                BTreeMap::from([(1, info(step_time, Breakdown::Types(types)))])
            };
            let collectives = if cores.is_empty() { BTreeMap::new() } else { details.collectives.clone() };
            Some(StepRecord { num: number as u32, cores, collectives })
        })
        .collect()
}

fn is_training(planes: &[Plane], map: &[u8]) -> bool {
    let modules = crate::hlo::protos(planes, map);
    let instructions: Vec<&[u8]> = modules.iter().flat_map(|(_, proto)| nested(proto, 1)).flat_map(|module| nested(module, 3)).flat_map(|computation| nested(computation, 2)).collect();
    instructions.par_iter().flat_map_iter(|instruction| nested(instruction, 7)).any(|metadata| {
        let text = |wanted: u32| nested(metadata, wanted).last().map(|bytes| crate::xplane::lossy(bytes).into_owned()).unwrap_or_default();
        let (kind, name) = (text(1), text(2));
        if is_tf_op_type(&kind) && is_tf_op_name(&name) {
            let scopes: Vec<&str> = name.split('/').collect();
            scopes[..scopes.len() - 1].iter().any(|scope| scope.strip_prefix("gradient").is_some_and(|rest| rest == "_tape" || rest.starts_with('s')))
        } else if (!name.is_empty() && is_jax_op_type(&kind) && name.rsplit('/').next().unwrap().contains(kind.as_str())) || kind.is_empty() {
            name.split('/').any(|scope| scope.starts_with("transpose("))
        } else {
            false
        }
    })
}

fn add_metrics(total: &mut [Metrics; 2], part: &[Metrics; 2], times: [u64; 2]) {
    for ((sum, part), time) in total.iter_mut().zip(part).zip(times) {
        accumulate(sum, part);
        sum.time_ps = sum.time_ps.wrapping_add(time);
    }
}

pub fn extra(planes: &[Plane], map: &[u8], templates: &[Vec<Template>]) -> Extra {
    let texts = |tag: u32| {
        let mut seen = HashSet::new();
        nested(map, tag).map(|bytes| crate::xplane::lossy(bytes).into_owned()).filter(|text| seen.insert(text.clone())).collect::<Vec<String>>()
    };
    let raw_planes: Vec<&[u8]> = nested(map, 1).collect();
    let hostname = texts(4).into_iter().next().unwrap_or_else(|| "localhost".into());
    let origin = (crate::xplane::origin_ns(planes) as u64).wrapping_mul(1000);
    let mut extra = Extra { errors: texts(2), warnings: texts(3), hostnames: vec![hostname.clone()], tasks: 1, device_type: "CPU".into(), hardware: CPU_ONLY, ..Default::default() };
    let tensor_cores: Vec<&Plane> = planes.iter().filter(|plane| is_tensor_core(&plane.name)).collect();
    let devices: Vec<(usize, &Plane)> = planes.iter().enumerate().filter(|(_, plane)| plane.name.starts_with(TPU_PREFIX)).collect();
    if let Some(first) = tensor_cores.first() {
        let flag = |name: &str| devices[0].1.own.iter().any(|(key, value)| &**key == name && matches!(value, Own::Int(value) if *value != 0));
        extra.device_type = first.own_text("device_type_string").unwrap_or_default().to_string();
        extra.core_count = tensor_cores.len() as i32;
        extra.hardware = TPU;
        extra.megacore = flag("has_megacore");
        extra.merged_vmem = flag("has_merged_vmem");
    }
    if let Some(first) = planes.iter().find(|plane| plane.name.starts_with(crate::gpu::PREFIX)) {
        let model = crate::gpu::model_name(first);
        extra.device_type = if model.is_empty() { "GPU".into() } else { model };
        extra.core_count = planes.iter().filter(|plane| plane.name.starts_with(crate::gpu::PREFIX)).count() as i32;
        extra.hardware = GPU;
    }
    let gpus = crate::gpu::devices(planes);
    if !gpus.is_empty() {
        extra.cores.insert(crate::gpu::CORE, Core { hostname: hostname.clone(), ..Default::default() });
    }
    let host = planes.iter().find(|plane| plane.name == HOST_PLANE);
    let ((outputs, gpu_events), (training, host_events)) = rayon::join(
        || {
            rayon::join(
                || devices.par_iter().map(|&(index, plane)| device_plane(plane, raw_planes.get(index).copied().unwrap_or(&[]), map, &templates[index], origin, &hostname)).collect::<Vec<Device>>(),
                || gpus.par_iter().map(|plane| gpu_device(plane, map, origin)).collect::<Vec<StepEvents>>(),
            )
        },
        || rayon::join(|| is_training(planes, map), || host.map(|plane| host_steps(plane, map, origin))),
    );
    extra.training = training;
    let mut step_events: StepEvents = HashMap::new();
    let mut chips: BTreeMap<(bool, u32), [Tracker; 2]> = BTreeMap::new();
    let mut programs: HashMap<i64, StepPrograms> = HashMap::new();
    for (index, ((_, plane), output)) in devices.iter().zip(outputs).enumerate() {
        if !output.events.is_empty() {
            intersect(output.events, &mut step_events, Details::combine);
        }
        if programs.is_empty() || !output.programs.is_empty() {
            intersect(output.programs, &mut programs, |step, other| {
                step.markers.extend(other.markers);
                step.cores.extend(other.cores);
            });
        }
        let key = output.core.as_ref().map_or((true, index as u32), |core| (false, core.chip));
        for (tracker, active) in chips.entry(key).or_default().iter_mut().zip(output.active) {
            tracker.0.extend(active);
            tracker.1 = match (tracker.1, output.total) {
                (Some((begin, end)), Some((other_begin, other_end))) => Some((begin.min(other_begin), end.max(other_end))),
                (left, right) => left.or(right),
            };
        }
        if let Some(core) = output.core {
            extra.cores.insert(plane.id as u32, core);
        }
    }
    let tpu = extra.hardware == TPU;
    for (index, (active, total)) in chips.into_values().flatten().enumerate().filter(|_| tpu) {
        let active_ps = merged_active(active);
        extra.busy_ps[index % 2] += active_ps;
        extra.idle_ps[index % 2] += total.map_or(0, |(begin, end)| end - begin).wrapping_sub(active_ps);
    }
    let mut sequence: Vec<(i64, [Metrics; 2])> = programs
        .into_iter()
        .map(|(group, step)| {
            let duration = step.markers.iter().copied().max().unwrap_or(0);
            let mut sums: [Metrics; 2] = Default::default();
            for (total_op, core, infeed_outfeed) in step.cores.iter().filter(|_| duration != 0) {
                let total = (*total_op).max(duration);
                add_metrics(&mut sums, core, [total, total.wrapping_sub(*infeed_outfeed)]);
            }
            (group, sums)
        })
        .collect();
    sequence.sort_unstable_by_key(|(group, _)| *group);
    for (_, sums) in &sequence {
        add_metrics(&mut extra.program_total, sums, [sums[0].time_ps, sums[1].time_ps]);
    }
    extra.program_steps = sequence;
    if let Some(plane) = host {
        extra.mxu = plane.own_double("matrix_unit_utilization_percent");
        extra.hbm = plane.own_double("hbm_utilization_percent");
    }
    if !gpus.is_empty() {
        let mut combined: StepEvents = HashMap::new();
        for (step, details) in gpu_events.into_iter().flatten() {
            combined.entry(step).or_default().combine(details);
        }
        let nonoverlapped: StepEvents = combined.into_iter().map(|(step, details)| (step, Details { events: non_overlapped(&details.events), ..details })).collect();
        for (kind, span) in nonoverlapped.values().flat_map(|details| &details.events) {
            match *kind {
                DEVICE_COMPUTE_32 => extra.precision[0] += span.duration,
                DEVICE_COMPUTE_16 => extra.precision[1] += span.duration,
                _ => {}
            }
        }
        extra.steps = step_db(&nonoverlapped, true);
    } else if devices.is_empty() {
        extra.cores.insert(1, Core { hostname, ..Default::default() });
        let nonoverlapped: StepEvents = host_events.unwrap_or_default().into_iter().map(|(step, details)| (step, Details { events: non_overlapped(&details.events), ..details })).collect();
        extra.steps = step_db(&nonoverlapped, false);
    } else {
        extra.steps = step_db(&step_events, true);
        let input = |details: &Details| details.events.iter().filter(|(kind, _)| *kind == HOST_WAIT_INPUT || *kind == HOST_TO_DEVICE).map(|(_, span)| span.duration).sum();
        extra.host_input = host_events.map(|events| events.iter().map(|(step, details)| (*step, input(details))).collect());
    }
    extra
}

pub fn fix(extra: &mut Extra, device_db: &Db) {
    let Some(host_input) = extra.host_input.take() else { return };
    if device_db.metrics.iter().any(|metrics| metrics.category == "infeed" || metrics.category == "sparsecorev0 infeed") {
        return;
    }
    for index in 0..extra.steps.len() {
        let total_input = host_input.get(&(extra.steps[index].num as i64)).copied().unwrap_or(0);
        if total_input == 0 {
            continue;
        }
        let idle_ms = tpu_step_details(&extra.steps[index], &extra.cores).fields[TC_IDLE];
        let ratio = safe_divide(total_input as f64 / 1e9, idle_ms).min(1.0);
        for (_, info) in extra.steps[index].cores.iter_mut().filter(|(core, _)| **core < SPARSE_CORE_START) {
            if let Breakdown::Categories(categories) = &mut info.breakdown {
                let idle = categories.get(IDLE).copied().unwrap_or(0);
                let infeed = (idle as f64 * ratio) as u64;
                categories.insert("infeed".into(), infeed);
                categories.insert(IDLE.into(), idle - infeed);
            }
        }
    }
}

fn alignment(subordinate: &[Span], chief: &[Span]) -> (u32, u32, u32) {
    if subordinate.is_empty() || chief.is_empty() {
        return (0, 0, 0);
    }
    let base = subordinate.len() as i64 - 1;
    let mut similarity = vec![0u64; subordinate.len() + chief.len() - 1];
    let (mut sub, mut main) = (0usize, 0usize);
    while sub < subordinate.len() && main < chief.len() {
        let (left, right) = (subordinate[sub], chief[main]);
        if left.duration == 0 || right.duration == 0 {
            (sub, main) = (sub + usize::from(left.duration == 0), main + usize::from(left.duration != 0));
            continue;
        }
        let (start, stop) = (left.begin.max(right.begin), left.end().min(right.end()));
        similarity[(main as i64 - sub as i64 + base) as usize] += stop.saturating_sub(start);
        (sub, main) = (sub + usize::from(left.end() <= right.end()), main + usize::from(left.end() >= right.end()));
    }
    let offsets = (0..chief.len() as i64).chain((1..subordinate.len() as i64).map(|k| -k));
    let offset = offsets.min_by_key(|offset| std::cmp::Reverse(similarity[(offset + base) as usize])).unwrap_or(0);
    let (begin_sub, begin_chief) = if offset >= 0 { (0, offset as u32) } else { ((-offset) as u32, 0) };
    (begin_sub, begin_chief, (subordinate.len() as u32 - begin_sub).min(chief.len() as u32 - begin_chief))
}

fn add_program(total: &mut [Metrics; 2], part: &[Metrics; 2]) {
    for (total, part) in total.iter_mut().zip(part) {
        accumulate(total, part);
        total.time_ps = total.time_ps.wrapping_add(part.time_ps);
    }
}

pub fn combine(all: &[&Extra]) -> Extra {
    let has_device = |extra: &Extra| extra.device_type.contains("GPU") || (extra.device_type != "CPU" && extra.device_type.contains("TPU"));
    let no_accelerator = !all.iter().any(|extra| has_device(extra));
    let workers: Vec<usize> = (0..all.len()).filter(|&host| has_device(all[host]) || no_accelerator).collect();
    let step_span = |record: &StepRecord| {
        let begin = record.cores.values().map(|info| info.begin).min().unwrap_or(u64::MAX);
        let end = record.cores.values().map(|info| info.begin + info.duration).max().unwrap_or(0);
        match record.cores.len() {
            1 => Span { begin, duration: end.saturating_sub(begin) },
            2.. if begin < end => Span { begin, duration: end - begin },
            _ => Span::default(),
        }
    };
    let spans: Vec<Vec<Span>> = workers.iter().map(|&host| all[host].steps.iter().map(step_span).collect()).collect();
    let extent = |spans: &[Span]| {
        let active = spans.iter().filter(|span| span.duration > 0);
        active.clone().map(|span| span.end()).max().unwrap_or(0).saturating_sub(active.map(|span| span.begin).min().unwrap_or(u64::MAX))
    };
    let mut combined = Extra::default();
    let (mut first_steps, mut steps) = (vec![0u32; all.len()], 0u32);
    if let Some(chief) = (0..spans.len()).min_by_key(|&index| extent(&spans[index])).filter(|&chief| !spans[chief].is_empty()) {
        let alignments: Vec<(u32, u32, u32)> =
            spans.iter().enumerate().map(|(index, host_spans)| if index == chief { (0, 0, host_spans.len() as u32) } else { alignment(host_spans, &spans[chief]) }).collect();
        let begin = alignments.iter().map(|alignment| alignment.1).max().unwrap_or(0);
        let end = alignments.iter().map(|alignment| alignment.1 + alignment.2).min().unwrap_or(u32::MAX);
        combined.empty_intersect = begin > end;
        if begin <= end {
            steps = end - begin;
            for (index, alignment) in alignments.iter().enumerate() {
                first_steps[workers[index]] = alignment.0 + (begin - alignment.1);
            }
        }
    }
    combined.steps = (0..steps).map(|num| StepRecord { num, ..Default::default() }).collect();
    for (host, extra) in all.iter().enumerate() {
        let global = |core: &u32| host as u32 * 1000 + core;
        for (index, record) in combined.steps.iter_mut().enumerate().filter(|_| workers.contains(&host)) {
            let source = &extra.steps[first_steps[host] as usize + index];
            source.cores.iter().for_each(|(core, info)| _ = record.cores.entry(global(core)).or_insert_with(|| info.clone()));
            source.collectives.iter().for_each(|(core, reduces)| _ = record.collectives.entry(global(core)).or_insert_with(|| reduces.clone()));
        }
        extra.cores.iter().for_each(|(core, details)| _ = combined.cores.entry(global(core)).or_insert_with(|| details.clone()));
        for name in &extra.hostnames {
            if !combined.hostnames.contains(name) {
                combined.hostnames.push(name.clone());
            }
        }
        if extra.device_type != "CPU" && extra.device_type != "Device" {
            combined.device_type.clone_from(&extra.device_type);
            combined.core_count += extra.core_count;
        } else if combined.device_type.is_empty() {
            combined.device_type.clone_from(&extra.device_type);
        }
        combined.busy_ps = std::array::from_fn(|index| combined.busy_ps[index] + extra.busy_ps[index]);
        combined.idle_ps = std::array::from_fn(|index| combined.idle_ps[index] + extra.idle_ps[index]);
        combined.hardware = combined.hardware.max(extra.hardware);
        combined.training = extra.training;
        combined.errors.extend(extra.errors.iter().cloned());
        combined.warnings.extend(extra.warnings.iter().cloned());
        combined.tasks += extra.tasks;
        combined.infeed_enqueue = (combined.infeed_enqueue.0 + extra.infeed_enqueue.0, combined.infeed_enqueue.1 + extra.infeed_enqueue.1);
        combined.mxu += extra.mxu;
        combined.hbm += extra.hbm;
        combined.precision = std::array::from_fn(|index| combined.precision[index] + extra.precision[index]);
    }
    combined.program_steps = (0..steps as usize)
        .map(|index| {
            let mut sums: [Metrics; 2] = Default::default();
            for &host in &workers {
                if let Some((_, part)) = all[host].program_steps.get(first_steps[host] as usize + index) {
                    add_program(&mut sums, part);
                }
            }
            (combined.steps[index].num as i64, sums)
        })
        .collect();
    for (_, sums) in &combined.program_steps {
        add_program(&mut combined.program_total, sums);
    }
    combined.mxu /= all.len() as f64;
    combined.hbm /= all.len() as f64;
    combined
}

#[cfg(test)]
#[path = "tests/inline/steps.rs"]
mod tests;
