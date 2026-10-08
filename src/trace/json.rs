use crate::trace::{DERIVED_META, Device, Event, FLOW_END, FLOW_MID, FLOW_START, NONE_FLOW, NONE_RESOURCE, Trace};
use crate::xplane::{Meta, NONE_GROUP, Plane, Value, slice, stats};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

pub const HOST_PID_STRIDE: u32 = 1000;
const IEEE_LIMIT: u64 = 1 << 53;
const INTERN_THRESHOLD: usize = 16;
const FRAME_CHUNK: usize = 2048;
const FRAME_OPEN: &str = "\u{1}";
const FRAME_CLOSE: &str = "\u{2}";
const LONG_NAME_LIMIT: usize = 10_000;
const HASH_MUL: u64 = 0xc6a4a7935bd1e995;
const HASH_SEED: u64 = 0xc70f6907;
pub const CONTEXT_TYPES: &str =
    "||tf_exec|tfrt_exec|batch_sched|PjRt|as_batch_sched|tfrt_rt|tpu_embed|gpu_launch|batcher|tpu_stream|tpu_launch|pathways_exec|pjrt_library_call|threadpool_event|jax_serving|sparsecore_offload";

pub struct View<'a> {
    pub trace: &'a Trace,
    pub map: &'a [u8],
    pub planes: &'a [Plane],
    pub events: Vec<u32>,
}

type Extra<'a> = (&'a Plane, &'a [u8], &'a mut Vec<String>);

fn push_quoted(out: &mut String, text: &str) {
    if text.bytes().all(|byte| byte >= 0x20 && !matches!(byte, b'"' | b'\\' | b'<' | b'>' | b'&' | 0xe2)) {
        out.push('"');
        out.push_str(text);
        out.push('"');
    } else {
        out.push_str(&serde_json::to_string(text).unwrap().replace('<', "\\u003c").replace('>', "\\u003e").replace('&', "\\u0026").replace('\u{2028}', "\\u2028").replace('\u{2029}', "\\u2029"));
    }
}

pub fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    push_quoted(&mut out, text);
    out
}

fn number(out: &mut String, value: u64) {
    out.push_str(itoa::Buffer::new().format(value));
}

/// Writes `%.17g` of `ps / 1e6` as glibc does. Below 100 ps, `%g` uses an exponent, and `snprintf` writes the text.
pub fn micros(out: &mut String, ps: u64) {
    let value = ps as f64 / 1e6;
    if ps < 100 {
        return out.push_str(&crate::hlo::general(value, 17));
    }
    let bits = value.to_bits();
    let (mantissa, shift) = (u128::from(bits & ((1 << 52) - 1) | 1 << 52), 1075 - (bits >> 52) as u32);
    const POWERS: [u128; 21] = {
        let mut powers = [1; 21];
        let mut index = 1;
        while index < 21 {
            powers[index] = powers[index - 1] * 10;
            index += 1;
        }
        powers
    };
    let mut exponent = ps.ilog10() as i32 - 6;
    let digits = loop {
        let scaled = mantissa * POWERS[(16 - exponent) as usize];
        let (mut digits, rest, half) = (scaled >> shift, scaled & ((1 << shift) - 1), 1u128 << (shift - 1));
        digits += u128::from(rest > half || rest == half && digits & 1 == 1);
        match digits {
            ..10_000_000_000_000_000 => exponent -= 1,
            100_000_000_000_000_000.. => exponent += 1,
            _ => break digits as u64,
        }
    };
    let mut buffer = itoa::Buffer::new();
    let text = buffer.format(digits);
    let (whole, fraction) = if exponent < 0 { ("0", text) } else { text.split_at(exponent as usize + 1) };
    out.push_str(whole);
    let fraction = fraction.trim_end_matches('0');
    if !fraction.is_empty() {
        out.push('.');
        (exponent + 1..0).for_each(|_| out.push('0'));
        out.push_str(fraction);
    }
}

pub fn double(value: f64) -> String {
    match value {
        value if value.is_nan() => "\"NaN\"".to_string(),
        value if value.is_infinite() => format!("\"{}Infinity\"", if value < 0.0 { "-" } else { "" }),
        value => value.to_string(),
    }
}

fn long_text(text: &str) -> String {
    if text.len() > LONG_NAME_LIMIT { format!("{}...<truncated>", String::from_utf8_lossy(&text.as_bytes()[..LONG_NAME_LIMIT])) } else { text.to_string() }
}

fn fingerprint(text: &str) -> u64 {
    let shift = |value: u64| value ^ (value >> 47);
    let bytes = [b"@@", text.as_bytes()].concat();
    let mut hash = HASH_SEED ^ (bytes.len() as u64).wrapping_mul(HASH_MUL);
    let (words, tail) = bytes.as_chunks::<8>();
    for word in words {
        hash = (hash ^ shift(u64::from_le_bytes(*word).wrapping_mul(HASH_MUL)).wrapping_mul(HASH_MUL)).wrapping_mul(HASH_MUL);
    }
    if !tail.is_empty() {
        hash = (hash ^ tail.iter().rev().fold(0u64, |acc, &byte| acc << 8 | u64::from(byte))).wrapping_mul(HASH_MUL);
    }
    shift(shift(hash).wrapping_mul(HASH_MUL))
}

fn put(texts: &mut FxHashMap<u64, String>, text: String) {
    if text.len() > INTERN_THRESHOLD {
        texts.entry(fingerprint(&text)).or_insert(text);
    }
}

fn add_texts(texts: &mut FxHashMap<u64, String>, planes: &[Plane], map: &[u8], events: &[Event], long_names: &HashMap<u32, Box<str>>) {
    let framed = |plane: &Plane| plane.stat_names.iter().any(|name| matches!(&**name, "long_name" | "hlo_text"));
    let strings = |plane: &Plane, raw: &[u8], field: u32| -> Vec<String> {
        let named = |id: usize| plane.stat_names.get(id).is_some_and(|name| matches!(&**name, "long_name" | "hlo_text"));
        stats(raw, field, |_| true).filter(|stat| matches!(stat.value, Value::Str(_) | Value::Ref(_)) && named(stat.id)).map(|stat| long_text(&plane.text_cow(&stat.value))).collect()
    };
    let used: Vec<Vec<AtomicBool>> = planes.iter().map(|plane| plane.meta.iter().map(|_| AtomicBool::new(false)).collect()).collect();
    events.par_iter().filter(|event| event.meta != DERIVED_META && event.ts != u64::MAX).for_each(|event| used[event.plane as usize][event.meta as usize].store(true, Relaxed));
    long_names.values().for_each(|long| put(texts, long_text(long)));
    let metas: Vec<(&Plane, &Meta, bool)> =
        planes.iter().zip(&used).flat_map(|(plane, flags)| plane.meta.iter().zip(flags).filter(|(_, used)| used.load(Relaxed)).map(move |(meta, _)| (plane, meta, framed(plane)))).collect();
    let found = metas
        .par_iter()
        .fold(FxHashMap::default, |mut found, &(plane, meta, framed)| {
            if !meta.display.is_empty() {
                put(&mut found, long_text(&meta.long_name(map)));
            }
            if framed {
                strings(plane, slice(map, meta.raw), 5).into_iter().for_each(|text| put(&mut found, text));
            }
            found
        })
        .reduce(FxHashMap::default, |mut left, right| {
            left.extend(right);
            left
        });
    texts.extend(found);
    let framed: Vec<bool> = planes.iter().map(framed).collect();
    let events: Vec<String> = events
        .par_iter()
        .filter(|event| event.meta != DERIVED_META && event.ts != u64::MAX && framed[event.plane as usize])
        .flat_map_iter(|event| strings(&planes[event.plane as usize], slice(map, event.raw), 4))
        .collect();
    events.into_iter().for_each(|text| put(texts, text));
}

fn table<'a>(sources: impl IntoIterator<Item = (&'a [Plane], &'a [u8], &'a [Event], &'a HashMap<u32, Box<str>>)>) -> Vec<String> {
    let mut texts = FxHashMap::default();
    sources.into_iter().for_each(|(planes, map, events, long_names)| add_texts(&mut texts, planes, map, events, long_names));
    let mut sorted: Vec<(u64, String)> = texts.into_iter().collect();
    sorted.sort_unstable_by_key(|entry| entry.0);
    sorted.into_iter().map(|entry| entry.1).collect()
}

fn frames_json(frames: &[String]) -> Vec<String> {
    let chunk = |(chunk, frames): (usize, &[String])| {
        let mut out = String::new();
        for (index, frame) in (chunk * FRAME_CHUNK..).zip(frames) {
            write!(out, "{}\"{}\":{{\"name\":", if index > 0 { "," } else { "" }, index + 1).unwrap();
            push_quoted(&mut out, frame);
            out.push('}');
        }
        out
    };
    frames.par_chunks(FRAME_CHUNK).enumerate().map(chunk).collect()
}

fn stat_double(name: &str, v: f64) -> String {
    if (name.ends_with("(util %)") || name.ends_with(" (MB/sec)")) && v.is_finite() { format!("{v:.2}") } else { double(v) }
}

pub fn stack_frames(planes: &[Plane], map: &[u8], events: &[Event], long_names: &HashMap<u32, Box<str>>) -> String {
    frames_json(&table([(planes, map, events, long_names)])).concat()
}

fn full_args(trace: &Trace, plane: &Plane, event: &Event, map: &[u8], frames: &mut Vec<String>) -> (Vec<String>, Option<usize>) {
    let (mut args, mut frame) = (Vec::new(), None);
    let mut text_arg = |args: &mut Vec<String>, name: &str, text: String| {
        if matches!(name, "long_name" | "hlo_text") && text.len() > INTERN_THRESHOLD && frame.is_none() {
            frames.push(text);
            frame = Some(frames.len());
        } else {
            args.push(format!("{}:{}", quoted(name), quoted(&text)));
        }
    };
    if event.meta == DERIVED_META {
        if let Some(long) = trace.long_names.get(&event.name) {
            text_arg(&mut args, "long_name", long_text(long));
        }
        if event.raw.0 == u32::MAX {
            args.extend(trace.args[event.raw.1 as usize].iter().cloned());
        }
        return (args, frame);
    }
    let meta = &plane.meta[event.meta as usize];
    if !meta.display.is_empty() {
        text_arg(&mut args, "long_name", long_text(&meta.long_name(map)));
    }
    let step = trace.steps.get(&event.raw);
    let mut used = Vec::new();
    for (field, name, stat) in plane.named_stats(map, event.meta, event.raw) {
        if let Some(eager) = event.eager.filter(|_| field == 4 && &**name == "is_eager") {
            args.push(format!("\"is_eager\":{}", u8::from(eager)));
            used.push("is_eager");
            continue;
        }
        let over = step.and_then(|step| step.stats.iter().find(|(key, _)| key == &&**name));
        let number = match stat.value {
            Value::Int(v) if !plane.consumes_number(stat.id) => Some(i128::from(over.map_or(v, |over| over.1))),
            Value::Uint(v) if !plane.consumes_number(stat.id) => Some(over.map_or(i128::from(v), |over| i128::from(over.1))),
            _ => None,
        };
        if over.is_some() {
            used.push(&**name);
        }
        if let Some(v) = number {
            args.push(format!("{}:{}", quoted(name), if v.unsigned_abs() > IEEE_LIMIT as u128 { format!("\"{v}\"") } else { v.to_string() }));
        } else if let Value::Double(v) = stat.value {
            args.push(format!("{}:{}", quoted(name), stat_double(name, v)));
        } else if matches!(stat.value, Value::Str(_) | Value::Ref(_)) {
            let text = if &**name == "step_name" { step.map_or_else(|| plane.text(&stat.value), |step| step.name.clone()) } else { plane.text(&stat.value) };
            if &**name == "step_name" {
                used.push("step_name");
            }
            text_arg(&mut args, name, text);
        }
    }
    if let Some(step) = step {
        for (key, value) in step.stats.iter().filter(|(key, _)| !used.contains(key)) {
            args.push(format!("\"{key}\":{value}"));
        }
        if !used.contains(&"step_name") {
            args.push(format!("\"step_name\":{}", quoted(&step.name)));
        }
    }
    if let Some(eager) = event.eager.filter(|_| !used.contains(&"is_eager")) {
        args.push(format!("\"is_eager\":{}", u8::from(eager)));
    }
    (args, frame)
}

pub fn ordered(views: &[View]) -> Vec<(u32, u32)> {
    let mut out = vec![(0, 0); views.iter().map(|view| view.events.len()).sum()];
    let mut base = 0;
    for (host, view) in views.iter().enumerate() {
        let (events, tracks) = (&view.trace.events, view.trace.tracks);
        let (mut first, mut starts) = (vec![u32::MAX; tracks], vec![0; tracks]);
        for &index in &view.events {
            let track = events[index as usize].track as usize;
            if first[track] == u32::MAX {
                first[track] = index;
            }
            starts[track] += 1;
        }
        let mut order: Vec<usize> = (0..tracks).filter(|&track| starts[track] > 0).collect();
        order.sort_by_key(|&track| {
            let event = &events[first[track] as usize];
            (event.device, event.resource != NONE_RESOURCE, if event.resource == NONE_RESOURCE { &*view.trace.names[event.name as usize] } else { "" }, event.resource)
        });
        for track in order {
            (starts[track], base) = (base, base + starts[track]);
        }
        for &index in &view.events {
            let track = events[index as usize].track as usize;
            out[starts[track]] = (host as u32, index);
            starts[track] += 1;
        }
    }
    out
}

pub fn devices<'a>(views: &'a [View]) -> Vec<(u32, &'a Device)> {
    let mut devices: Vec<(u32, &Device)> =
        views.iter().enumerate().flat_map(|(host, view)| view.trace.devices.iter().map(move |(id, device)| (id + (host as u32 + 1) * HOST_PID_STRIDE, device))).collect();
    devices.sort_by_key(|(pid, _)| *pid);
    devices
}

pub fn counter_values(plane: &Plane, event: &Event, map: &[u8]) -> (Option<Box<str>>, Vec<String>) {
    let (mut first, mut values) = (None, Vec::new());
    for (_, name, stat) in plane.named_stats(map, event.meta, event.raw) {
        let value = match stat.value {
            Value::Int(v) => v.to_string(),
            Value::Uint(v) => v.to_string(),
            Value::Double(v) => stat_double(name, v),
            Value::Str(_) | Value::Ref(_) => quoted(&plane.text_cow(&stat.value)),
            Value::Bytes(_) => continue,
        };
        first.get_or_insert_with(|| name.clone());
        values.push(value);
    }
    (first, values)
}

pub fn write_event(out: &mut String, trace: &Trace, event: &Event, pid: u32, extra: Option<Extra>, forced_entry: Option<u8>, marked: bool) {
    let (entry, category) = (forced_entry.unwrap_or(event.flow_entry), CONTEXT_TYPES.split('|').nth(event.flow_cat as usize).unwrap_or(""));
    out.push_str("{\"pid\":");
    number(out, pid as u64);
    if event.resource != NONE_RESOURCE {
        out.push_str(",\"tid\":");
        number(out, event.resource as u64);
    }
    out.push_str(",\"name\":");
    push_quoted(out, &trace.names[event.name as usize]);
    out.push_str(",\"ts\":");
    micros(out, if forced_entry.is_some() { event.ts + event.dur } else { event.ts });
    if event.resource == NONE_RESOURCE {
        out.push_str(",\"id\":");
        number(out, trace.flow_ids[event.flow as usize]);
        write!(out, ",\"cat\":\"{category}\",\"ph\":\"{}\"", if matches!(entry, FLOW_START | FLOW_MID) { "b" } else { "e" }).unwrap();
    } else {
        out.push_str(",\"dur\":");
        micros(out, event.dur.max(1));
        if event.flow != NONE_FLOW {
            out.push_str(",\"bind_id\":");
            number(out, trace.flow_ids[event.flow as usize]);
            if !category.is_empty() {
                write!(out, ",\"cat\":\"{category}\"").unwrap();
            }
            out.push_str(match entry {
                FLOW_START => ",\"flow_out\":true",
                FLOW_MID => ",\"flow_in\":true,\"flow_out\":true",
                FLOW_END => ",\"flow_in\":true",
                _ => "",
            });
        }
        out.push_str(",\"ph\":\"X\"");
    }
    let mut frame = None;
    if forced_entry.is_none() || event.group != NONE_GROUP {
        out.push_str(",\"args\":{");
        let start = out.len();
        if event.group != NONE_GROUP {
            write!(out, "\"group_id\":{}", event.group).unwrap();
        }
        match extra.filter(|_| forced_entry.is_none()) {
            None if forced_entry.is_none() => write!(out, "{}\"uid\":{}", if out.len() > start { "," } else { "" }, event.serial).unwrap(),
            None => {}
            Some((plane, map, frames)) => {
                let (args, stack) = full_args(trace, plane, event, map, frames);
                frame = stack;
                for arg in args {
                    out.push_str(if out.len() > start { "," } else { "" });
                    out.push_str(&arg);
                }
            }
        }
        out.push('}');
    }
    if let Some(frame) = frame {
        let (open, close) = if marked { (FRAME_OPEN, FRAME_CLOSE) } else { ("", "") };
        write!(out, ",\"sf\":{open}{frame}{close}").unwrap();
    }
    if event.serial > 0 {
        out.push_str(",\"z\":");
        number(out, event.serial as u64);
    }
    out.push('}');
    if event.resource == NONE_RESOURCE && entry == FLOW_MID && forced_entry.is_none() {
        out.push(',');
        write_event(out, trace, event, pid, None, Some(FLOW_END), marked);
    }
}

pub fn render(views: &[View], full_dma: bool, detail: bool) -> Vec<u8> {
    let total: usize = views.iter().map(|view| view.events.len()).sum();
    let (min_ps, max_ps) = (views.iter().map(|view| view.trace.min_ps).min().unwrap_or(0), views.iter().map(|view| view.trace.max_ps).max().unwrap_or(0));
    let seconds = |ps: u64| format!("{}", format!("{:.5e}", ps as f64 / 1e9).parse::<f64>().unwrap());
    let mut out = String::with_capacity(total * 140 + 4096);
    out.push_str("{\"displayTimeUnit\":\"ns\",\"metadata\":{\"highres-ticks\":true}, \"codeLink\":\"\",\"useNewBackend\": false,");
    let tpu = views.iter().any(|view| !view.trace.tpu_devices.is_empty());
    write!(out, "\"details\":[{{\"name\":\"mpmd_pipeline_view\",\"value\":false}}{}],", if tpu { format!(",{{\"name\":\"full_dma\",\"value\":{full_dma}}}") } else { String::new() }).unwrap();
    write!(out, "\"returnedEventsSize\":{total},\"filteredByVisibility\":true,\"fullTimespan\":[{},{}],", seconds(min_ps), seconds(max_ps)).unwrap();
    let devices = devices(views);
    let mut body = String::new();
    for (pid, device) in &devices {
        body.push_str(if body.is_empty() { "" } else { "," });
        write!(body, "{{\"args\":{{\"name\":{}}},\"name\":\"process_name\",\"ph\":\"M\",\"pid\":{pid},\"thread_count\":{}}},{{\"args\":{{\"sort_index\":{pid}}},\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":{pid}}}", quoted(&device.name), device.resources.len()).unwrap();
        for (resource, name) in &device.resources {
            write!(body, ",{{\"args\":{{\"name\":{}}},\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":{pid},\"tid\":{resource}}},{{\"args\":{{\"sort_index\":{resource}}},\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":{pid},\"tid\":{resource}}}", quoted(name)).unwrap();
        }
    }
    let ordered = ordered(views);
    let reused = views.len() == 1 && !detail;
    let mut frames = if detail || reused { Vec::new() } else { table(views.iter().map(|view| (view.planes, view.map, &view.trace.events[..], &view.trace.long_names))) };
    let is_counter = |&(host, index): &(u32, u32)| {
        let event = &views[host as usize].trace.events[index as usize];
        event.resource == NONE_RESOURCE && event.flow == NONE_FLOW
    };
    let counters = ordered.iter().filter(|entry| is_counter(entry)).count();
    let counter_key = |&(host, index): &(u32, u32)| {
        let (view, event) = (&views[host as usize], &views[host as usize].trace.events[index as usize]);
        (host, event.device, &view.trace.names[event.name as usize])
    };
    let write_chunk = |chunk: &[(u32, u32)], mut frames: Option<&mut Vec<String>>| {
        let mut text = String::with_capacity(chunk.len() * 150);
        let mut open: Option<(u32, u32, &Box<str>)> = None;
        for (position, entry) in chunk.iter().enumerate() {
            let (host, index) = *entry;
            let (view, event) = (&views[host as usize], &views[host as usize].trace.events[index as usize]);
            let pid = event.device + (host + 1) * HOST_PID_STRIDE;
            if is_counter(entry) {
                let (first, values) = counter_values(&view.planes[event.plane as usize], event, view.map);
                if open != Some(counter_key(entry)) {
                    text.push_str(if open.is_some() {
                        "]},{"
                    } else if position > 0 {
                        ",{"
                    } else {
                        "{"
                    });
                    write!(text, "\"pid\":{pid},\"name\":{},\"ph\":\"C\"", quoted(&view.trace.names[event.name as usize])).unwrap();
                    if let Some(first) = first {
                        write!(text, ",\"event_stats\":{}", quoted(&first)).unwrap();
                    }
                    text.push_str(",\"entries\":[");
                    open = Some(counter_key(entry));
                } else if !values.is_empty() && !text.ends_with('[') {
                    text.push(',');
                }
                if !values.is_empty() {
                    text.push('[');
                    text.push_str(&crate::hlo::general(event.ts as f64 / 1e6, 17));
                    values.iter().for_each(|value| write!(text, ",{value}").unwrap());
                    text.push(']');
                }
                continue;
            }
            if open.take().is_some() {
                text.push_str("]}");
            }
            text.push_str(if position > 0 { "," } else { "" });
            write_event(&mut text, view.trace, event, pid, frames.as_deref_mut().map(|frames| (&view.planes[event.plane as usize], view.map, frames)), None, detail);
        }
        if open.is_some() {
            text.push_str("]}");
        }
        text
    };
    let mut bounds = vec![0];
    while *bounds.last().unwrap() < ordered.len() {
        let mut end = (bounds.last().unwrap() + 4096).min(ordered.len());
        while end < ordered.len() && is_counter(&ordered[end]) && counter_key(&ordered[end - 1]) == counter_key(&ordered[end]) && is_counter(&ordered[end - 1]) {
            end += 1;
        }
        bounds.push(end);
    }
    let pieces: Vec<&[(u32, u32)]> = bounds.windows(2).map(|window| &ordered[window[0]..window[1]]).collect();
    let chunks: Vec<String> = if detail {
        let parts: Vec<(String, Vec<String>)> = pieces
            .par_iter()
            .map(|chunk| {
                let mut found = Vec::new();
                (write_chunk(chunk, Some(&mut found)), found)
            })
            .collect();
        let offsets: Vec<usize> = parts.iter().scan(0, |total, (_, found)| Some(std::mem::replace(total, *total + found.len()))).collect();
        let renumbered = parts.par_iter().zip(&offsets).map(|((text, _), &offset)| renumber(text, offset)).collect();
        frames = parts.into_iter().flat_map(|(_, found)| found).collect();
        renumbered
    } else {
        pieces.par_iter().map(|chunk| write_chunk(chunk, None)).collect()
    };
    out.push_str("\"stackFrames\":{");
    let listed: Vec<String> = if reused { Vec::new() } else { frames_json(&frames) };
    let middle = format!("}},\"traceEvents\":[{body}");
    let tail = format!("], \"showCounterMessage\": \"\" ,\"totalCounterEvents\":{counters}}}");
    let mut pieces: Vec<&[u8]> = vec![out.as_bytes()];
    if reused {
        pieces.push(views[0].trace.stack_frames.as_bytes());
    }
    pieces.extend(listed.iter().map(String::as_bytes));
    pieces.push(middle.as_bytes());
    for (index, chunk) in chunks.iter().filter(|chunk| !chunk.is_empty()).enumerate() {
        if index > 0 || !body.is_empty() {
            pieces.push(b",");
        }
        pieces.push(chunk.as_bytes());
    }
    pieces.push(tail.as_bytes());
    concat(&pieces)
}

fn renumber(text: &str, offset: usize) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let mut rest = text;
    while let Some(open) = rest.find(FRAME_OPEN) {
        let (before, marked) = rest.split_at(open);
        let close = marked.find(FRAME_CLOSE).unwrap_or(marked.len() - 1);
        out.push_str(before);
        out.push_str(&(marked[1..close].parse::<usize>().unwrap_or(0) + offset).to_string());
        rest = &marked[close + 1..];
    }
    out.push_str(rest);
    out
}

fn concat(pieces: &[&[u8]]) -> Vec<u8> {
    let total = pieces.iter().map(|piece| piece.len()).sum();
    let mut out = Vec::<u8>::with_capacity(total);
    let mut rest = out.spare_capacity_mut();
    let mut targets = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let (head, tail) = rest.split_at_mut(piece.len());
        targets.push(head);
        rest = tail;
    }
    targets.into_par_iter().zip(pieces).for_each(|(target, piece)| {
        target.write_copy_of_slice(piece);
    });
    unsafe { out.set_len(total) };
    out
}
