use crate::hlo::general;
use crate::run_tools::{python_string, python_string_into};
use crate::table::repr;
use crate::xplane::{INTERNAL_STATS, NONE_GROUP, Plane, Value, slice, stats};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write;

const HOST_THREADS: &str = "/host:CPU";
const HOST_DEVICE_ID: i64 = 701;
const PLANE_PREFIXES: [&str; 3] = ["/device:GPU:", "/device:TPU:", "/device:CUSTOM:"];
const ASYNC_OPS_LINE: &str = "Async XLA Ops";
const MAX_EVENTS: usize = 5_000_000;
const EVENTS_PER_CHUNK: usize = 16_384;
const BARRIER: &str = "barrier-cores";

type Devices = BTreeMap<i64, (String, BTreeMap<u32, String>)>;

struct Event {
    device: i64,
    resource: u32,
    name: String,
    ts: u64,
    dur: u64,
    args: BTreeMap<String, String>,
}

fn convert(device: i64, plane: &Plane, map: &[u8], only: Option<&str>, trimmed: bool) -> Vec<Event> {
    // A scan for one name needs no other event, unless the event limit can cut the list. Only a step name can change a name.
    let skip_others = only.filter(|_| trimmed && plane.stat_names.iter().all(|name| &**name != "step_name"));
    let lines = plane.lines.par_iter().filter(|line| line.name != ASYNC_OPS_LINE);
    lines
        .flat_map(|line| {
            let derived = !line.labels.is_empty();
            let ordinal = crate::group::ordinal(plane, line);
            line.events.par_iter().enumerate().filter_map(move |(index, event)| {
                let meta = (!derived).then(|| &plane.meta[event.meta as usize]);
                if meta.is_some_and(|meta| meta.internal) {
                    return None;
                }
                if let Some(only) = skip_others.filter(|_| !line.steps.contains_key(&index)) {
                    let name = match meta {
                        None => &*line.labels[event.meta as usize],
                        Some(meta) if meta.display.is_empty() => &*meta.name,
                        Some(meta) => &*meta.display,
                    };
                    if name != only {
                        return None;
                    }
                }
                let all = only.is_none();
                let mut args = BTreeMap::new();
                let mut name = match meta {
                    None => line.labels[event.meta as usize].to_string(),
                    Some(meta) if meta.display.is_empty() => meta.name.to_string(),
                    Some(meta) => meta.display.to_string(),
                };
                if let Some(meta) = meta {
                    if all && !meta.display.is_empty() {
                        args.insert("long_name".into(), meta.long_name(map).into_owned());
                    }
                    for (raw, field) in [(slice(map, meta.raw), 5), (slice(map, event.raw), 4)] {
                        for stat in stats(raw, field, |_| true) {
                            let Some(stat_name) = plane.stat_names.get(stat.id).filter(|stat_name| !INTERNAL_STATS.contains(&&***stat_name)) else { continue };
                            let is_step = &**stat_name == "step_name";
                            if !all && !is_step {
                                continue;
                            }
                            let text = match &stat.value {
                                Value::Int(value) => value.to_string(),
                                Value::Uint(value) => value.to_string(),
                                Value::Double(value) => general(*value, 6),
                                Value::Str(_) | Value::Ref(_) => plane.text(&stat.value),
                                Value::Bytes(_) => "<opaque bytes>".into(),
                            };
                            if is_step {
                                name.clone_from(&text);
                            }
                            if all {
                                args.insert(stat_name.to_string(), text);
                            }
                        }
                    }
                    if all {
                        let links = plane.links(event.meta, slice(map, event.raw), ordinal);
                        if let Some(group) = Some(event.group).filter(|&group| group != NONE_GROUP).or(links.group) {
                            args.insert("group_id".into(), group.to_string());
                        }
                        if let Some(flow) = links.flow {
                            args.insert("flow".into(), flow.to_string());
                        }
                        if let Some(eager) = event.eager {
                            args.insert("is_eager".into(), u8::from(eager).to_string());
                        }
                    }
                } else if all {
                    if let Some(long) = line.longs.get(event.meta as usize).filter(|long| !long.is_empty()) {
                        args.insert("long_name".into(), long.to_string());
                    }
                    for fragment in line.args.get(&index).into_iter().flatten() {
                        if let Ok(serde_json::Value::Object(object)) = serde_json::from_str::<serde_json::Value>(&format!("{{{fragment}}}")) {
                            for (key, value) in object {
                                args.insert(
                                    key,
                                    if let Some(text) = value.as_str() {
                                        text.to_string()
                                    } else if value.is_f64() {
                                        general(value.as_f64().unwrap_or(0.0), 6)
                                    } else {
                                        value.to_string()
                                    },
                                );
                            }
                        }
                    }
                }
                if all && derived && event.group != NONE_GROUP {
                    args.insert("group_id".into(), event.group.to_string());
                }
                if let Some(step) = line.steps.get(&index) {
                    name.clone_from(&step.name);
                    if all {
                        args.insert("step_name".into(), step.name.clone());
                        step.stats.iter().for_each(|(key, value)| _ = args.insert((*key).into(), value.to_string()));
                    }
                }
                if only.is_some_and(|only| only != name) {
                    name.clear();
                }
                let ts = event.ts + (plane.origin_ns as u64).wrapping_mul(1000);
                Some(Event { device, resource: line.resource_id(), name, ts, dur: event.dur, args })
            })
        })
        .collect()
}

fn collect(planes: &[Plane], map: &[u8], only: Option<&str>) -> (Devices, Vec<Event>) {
    let mut devices = Devices::new();
    let host = planes.iter().find(|plane| plane.name == HOST_THREADS).map(|host| (HOST_DEVICE_ID, host));
    let prefixed = PLANE_PREFIXES.iter().map(|prefix| planes.iter().filter(|plane| plane.name.starts_with(prefix)).collect::<Vec<_>>()).find(|found| !found.is_empty()).unwrap_or_default();
    let jobs: Vec<(i64, &Plane)> = host.into_iter().chain(prefixed.into_iter().map(|plane| (1 + plane.id, plane))).collect();
    for &(device, plane) in &jobs {
        let resources = devices.entry(device).or_default();
        resources.0.clone_from(&plane.name);
        for line in plane.lines.iter().filter(|line| plane.name == HOST_THREADS || !line.events.is_empty()) {
            resources.1.insert(line.resource_id(), if line.display_name.is_empty() { line.name.clone() } else { line.display_name.clone() });
        }
    }
    let trimmed = jobs.iter().map(|(_, plane)| plane.lines.iter().map(|line| line.events.len()).sum::<usize>()).sum::<usize>() <= MAX_EVENTS;
    let parts: Vec<Vec<Event>> = jobs.par_iter().map(|&(device, plane)| convert(device, plane, map, only, trimmed)).collect();
    let mut events: Vec<Event> = parts.into_iter().flatten().collect();
    if events.len() > MAX_EVENTS {
        events.sort_by_key(|event| event.ts);
        events.truncate(MAX_EVENTS);
    }
    (devices, events)
}

pub fn barrier_durations(planes: &[Plane], map: &[u8]) -> Vec<f64> {
    collect(planes, map, Some(BARRIER)).1.iter().filter(|event| event.name == BARRIER).map(|event| event.dur as f64 / 1e6).collect()
}

pub fn render(planes: &[Plane], map: &[u8]) -> String {
    let (devices, events) = collect(planes, map, None);
    let mut out = String::from("{\"displayTimeUnit\":\"ns\",\"metadata\":{\"highres-ticks\":true},\n\"traceEvents\":[\n");
    for (device, (name, resources)) in &devices {
        if !name.is_empty() {
            writeln!(out, "{{\"args\": {{\"name\": {}}}, \"name\": \"process_name\", \"ph\": \"M\", \"pid\": {device}}},", python_string(name)).unwrap();
        }
        writeln!(out, "{{\"args\": {{\"sort_index\": {device}}}, \"name\": \"process_sort_index\", \"ph\": \"M\", \"pid\": {device}}},").unwrap();
        for (resource, name) in resources {
            if !name.is_empty() {
                writeln!(out, "{{\"args\": {{\"name\": {}}}, \"name\": \"thread_name\", \"ph\": \"M\", \"pid\": {device}, \"tid\": {resource}}},", python_string(name)).unwrap();
            }
            writeln!(out, "{{\"args\": {{\"sort_index\": {resource}}}, \"name\": \"thread_sort_index\", \"ph\": \"M\", \"pid\": {device}, \"tid\": {resource}}},").unwrap();
        }
    }
    let lines: Vec<String> = events
        .par_chunks(EVENTS_PER_CHUNK)
        .map(|chunk| {
            let mut part = String::new();
            for event in chunk {
                part.push('{');
                if !event.args.is_empty() {
                    part.push_str("\"args\": {");
                    for (index, (key, value)) in event.args.iter().enumerate() {
                        part.push_str(if index > 0 { ", " } else { "" });
                        python_string_into(&mut part, key);
                        part.push_str(": ");
                        python_string_into(&mut part, value);
                    }
                    part.push_str("}, ");
                }
                if event.dur != 0 {
                    write!(part, "\"dur\": {}, ", repr(event.dur as f64 / 1e6)).unwrap();
                }
                part.push_str("\"name\": ");
                python_string_into(&mut part, &event.name);
                write!(part, ", \"ph\": {}, \"pid\": {}, ", if event.dur != 0 { "\"X\"" } else { "\"i\"" }, event.device).unwrap();
                if event.dur == 0 {
                    part.push_str("\"s\": \"t\", ");
                }
                writeln!(part, "\"tid\": {}, \"ts\": {}}},", event.resource, repr(event.ts as f64 / 1e6)).unwrap();
            }
            part
        })
        .collect();
    out.reserve(lines.iter().map(String::len).sum::<usize>() + 8);
    lines.iter().for_each(|part| out.push_str(part));
    out.push_str("{}]}\n");
    crate::release(events);
    out
}
