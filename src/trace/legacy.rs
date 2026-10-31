use crate::hlo::general;
use crate::server::run_tools::{python_string, python_string_into};
use crate::tools::table::repr_into;
use crate::xplane::{Ev, Line, NONE_GROUP, Plane, Value, slice};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write;

const HOST_THREADS: &str = "/host:CPU";
const HOST_DEVICE_ID: i64 = 701;
const PLANE_PREFIXES: [&str; 3] = ["/device:GPU:", "/device:TPU:", "/device:CUSTOM:"];
const ASYNC_OPS_LINE: &str = "Async XLA Ops";
const MAX_EVENTS: usize = 5_000_000;
const EVENTS_PER_CHUNK: usize = 16_384;
const BARRIER: &str = "barrier-cores";
const SAMPLES: usize = 1 << 18;
/// The sampled place of `limit` is at most this far from its true place, except with a very small probability: the standard deviation is at most 256.
const SAMPLE_MARGIN: usize = 2048;

type Devices = BTreeMap<i64, (String, BTreeMap<u32, String>)>;

struct Event {
    device: i64,
    resource: u32,
    name: String,
    ts: u64,
    dur: u64,
    /// The args in the order of their keys, with one value for each key.
    args: Vec<(String, String)>,
}

fn stat_text(plane: &Plane, value: &Value) -> String {
    match value {
        Value::Int(value) => value.to_string(),
        Value::Uint(value) => value.to_string(),
        Value::Double(value) => general(*value, 6),
        Value::Str(_) | Value::Ref(_) => plane.text(value),
        Value::Bytes(_) => "<opaque bytes>".into(),
    }
}

/// The events up to the time `last`.
fn convert(device: i64, plane: &Plane, map: &[u8], last: u64) -> Vec<Event> {
    let lines = plane
        .lines
        .par_iter()
        .filter(|line| line.name != ASYNC_OPS_LINE);
    lines
        .flat_map(|line| {
            let derived = !line.labels.is_empty();
            let ordinal = crate::xplane::group::ordinal(plane, line);
            line.events
                .par_iter()
                .enumerate()
                .filter_map(move |(index, event)| {
                    let meta = (!derived).then(|| &plane.meta[event.meta as usize]);
                    if meta.is_some_and(|meta| meta.internal) || time(plane, event) > last {
                        return None;
                    }
                    let mut name = label(plane, line, event).to_string();
                    let mut args = Vec::new();
                    if let Some(meta) = meta {
                        if !meta.display.is_empty() {
                            set(
                                &mut args,
                                "long_name".into(),
                                meta.long_name(map).into_owned(),
                            );
                        }
                        for (_, stat_name, stat) in plane.named_stats(map, event.meta, event.raw) {
                            let text = stat_text(plane, &stat.value);
                            if &**stat_name == "step_name" {
                                name.clone_from(&text);
                            }
                            set(&mut args, stat_name.to_string(), text);
                        }
                        let links = plane.links(event.meta, slice(map, event.raw), ordinal);
                        if let Some(group) = Some(event.group)
                            .filter(|&group| group != NONE_GROUP)
                            .or(links.group)
                        {
                            set(&mut args, "group_id".into(), group.to_string());
                        }
                        if let Some(flow) = links.flow {
                            set(&mut args, "flow".into(), flow.to_string());
                        }
                        if let Some(eager) = event.eager {
                            set(&mut args, "is_eager".into(), u8::from(eager).to_string());
                        }
                    } else {
                        if let Some(long) = line
                            .longs
                            .get(event.meta as usize)
                            .filter(|long| !long.is_empty())
                        {
                            set(&mut args, "long_name".into(), long.to_string());
                        }
                        for fragment in line.args.get(&index).into_iter().flatten() {
                            if let Ok(serde_json::Value::Object(object)) =
                                serde_json::from_str::<serde_json::Value>(&format!(
                                    "{{{fragment}}}"
                                ))
                            {
                                for (key, value) in object {
                                    set(
                                        &mut args,
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
                    if derived && event.group != NONE_GROUP {
                        set(&mut args, "group_id".into(), event.group.to_string());
                    }
                    if let Some(step) = line.steps.get(&index) {
                        name.clone_from(&step.name);
                        set(&mut args, "step_name".into(), step.name.clone());
                        step.stats.iter().for_each(|(key, value)| {
                            set(&mut args, (*key).into(), value.to_string());
                        });
                    }
                    args.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    Some(Event {
                        device,
                        resource: line.resource_id(),
                        name,
                        ts: time(plane, event),
                        dur: event.dur,
                        args,
                    })
                })
        })
        .collect()
}

/// Sets the value of `key`, as an insert into a map does.
fn set(args: &mut Vec<(String, String)>, key: String, value: String) {
    match args.iter_mut().find(|(known, _)| *known == key) {
        Some(slot) => slot.1 = value,
        None => args.push((key, value)),
    }
}

fn label<'a>(plane: &'a Plane, line: &'a Line, event: &Ev) -> &'a str {
    if !line.labels.is_empty() {
        return &line.labels[event.meta as usize];
    }
    let meta = &plane.meta[event.meta as usize];
    if meta.display.is_empty() {
        &meta.name
    } else {
        &meta.display
    }
}

fn time(plane: &Plane, event: &Ev) -> u64 {
    event.ts + (plane.origin_ns as u64).wrapping_mul(1000)
}

fn kept(plane: &Plane, line: &Line, event: &Ev) -> bool {
    !line.labels.is_empty() || !plane.meta[event.meta as usize].internal
}

/// The lines of the trace, in the order of its events.
fn lines<'a>(jobs: &[(i64, &'a Plane)]) -> Vec<(&'a Plane, &'a Line)> {
    jobs.iter()
        .flat_map(|&(_, plane)| {
            plane
                .lines
                .iter()
                .filter(|line| line.name != ASYNC_OPS_LINE)
                .map(move |line| (plane, line))
        })
        .collect()
}

/// The time of the last event that a trace of `limit` events keeps, and the number of the kept events at that time. There is none when the trace keeps all events.
/// The `select` of the event times. A sorted sample of the times gives a range near the place of `limit`, and one pass finds the cut when it is in that range.
fn cut(lines: &[(&Plane, &Line)], limit: usize) -> Option<(u64, usize)> {
    let values = || {
        lines.par_iter().flat_map_iter(|&(plane, line)| {
            line.events
                .iter()
                .filter(move |event| kept(plane, line, event))
                .map(move |event| time(plane, event))
        })
    };
    let total: usize = lines.iter().map(|(_, line)| line.events.len()).sum();
    if total <= limit {
        return None;
    }
    // The sample has the events at each `step` place of all the lines, so each time in the sample stands for `step` events.
    let (step, mut start, mut sample) = ((total / SAMPLES).max(1), 0, Vec::new());
    for &(plane, line) in lines {
        sample.extend(
            line.events
                .iter()
                .skip((step - start % step) % step)
                .step_by(step)
                .filter(|event| kept(plane, line, event))
                .map(|event| time(plane, event)),
        );
        start += line.events.len();
    }
    sample.sort_unstable();
    let place = limit / step;
    let low = place
        .checked_sub(SAMPLE_MARGIN)
        .and_then(|index| sample.get(index))
        .map_or(0, |&low| low);
    let high = sample
        .get(place + SAMPLE_MARGIN)
        .map_or(u64::MAX, |&high| high);
    let (below, mut inside, above) = values()
        .fold(
            || (0, Vec::new(), 0),
            |(below, mut inside, above), value| {
                if value < low {
                    (below + 1, inside, above)
                } else if value > high {
                    (below, inside, above + 1)
                } else {
                    inside.push(value);
                    (below, inside, above)
                }
            },
        )
        .reduce(
            || (0, Vec::new(), 0),
            |(below, mut inside, above), (other_below, other_inside, other_above)| {
                inside.extend(other_inside);
                (below + other_below, inside, above + other_above)
            },
        );
    if below + inside.len() + above <= limit {
        return None;
    }
    if below < limit && limit <= below + inside.len() {
        let (smaller, &mut found, _) = inside.select_nth_unstable(limit - 1 - below);
        return Some((
            found,
            limit - below - smaller.iter().filter(|&&value| value < found).count(),
        ));
    }
    select(values, limit)
}

fn jobs(planes: &[Plane]) -> Vec<(i64, &Plane)> {
    let host = planes
        .iter()
        .find(|plane| plane.name == HOST_THREADS)
        .map(|host| (HOST_DEVICE_ID, host));
    let prefixed = PLANE_PREFIXES
        .iter()
        .map(|prefix| {
            planes
                .iter()
                .filter(|plane| plane.name.starts_with(prefix))
                .collect::<Vec<_>>()
        })
        .find(|found| !found.is_empty())
        .unwrap_or_default();
    host.into_iter()
        .chain(prefixed.into_iter().map(|plane| (1 + plane.id, plane)))
        .collect()
}

fn collect(planes: &[Plane], map: &[u8], limit: usize) -> (Devices, Vec<Event>) {
    let mut devices = Devices::new();
    let jobs = jobs(planes);
    for &(device, plane) in &jobs {
        let resources = devices.entry(device).or_default();
        resources.0.clone_from(&plane.name);
        for line in plane
            .lines
            .iter()
            .filter(|line| plane.name == HOST_THREADS || !line.events.is_empty())
        {
            resources.1.insert(
                line.resource_id(),
                if line.display_name.is_empty() {
                    line.name.clone()
                } else {
                    line.display_name.clone()
                },
            );
        }
    }
    let cut = cut(&lines(&jobs), limit);
    let parts: Vec<Vec<Event>> = jobs
        .par_iter()
        .map(|&(device, plane)| convert(device, plane, map, cut.map_or(u64::MAX, |cut| cut.0)))
        .collect();
    let mut events: Vec<Event> = parts.into_iter().flatten().collect();
    if let Some((last, quota)) = cut {
        let mut ties = 0;
        events.retain(|event| {
            ties += usize::from(event.ts == last);
            event.ts != last || ties <= quota
        });
        events.sort_by_key(|event| event.ts);
    }
    (devices, events)
}

/// The value at the last place in the first `limit` sorted values, and the number of values equal to it in those places. There is no such value when all values fit.
fn select<I: ParallelIterator<Item = u64>>(
    values: impl Fn() -> I,
    limit: usize,
) -> Option<(u64, usize)> {
    let (mut found, mut rank) = (0u64, limit.checked_sub(1)?);
    for shift in [48, 32, 16, 0] {
        let high = |value: u64| value.checked_shr(shift + 16).unwrap_or(0);
        let add = |mut counts: Vec<usize>, other: Vec<usize>| {
            counts
                .iter_mut()
                .zip(other)
                .for_each(|(count, other)| *count += other);
            counts
        };
        let counts = values()
            .filter(|&value| high(value) == high(found))
            .fold(
                || vec![0; 1 << 16],
                |mut counts, value| {
                    counts[(value >> shift) as usize & 0xffff] += 1;
                    counts
                },
            )
            .reduce(|| vec![0; 1 << 16], add);
        if shift == 48 && counts.iter().sum::<usize>() <= limit {
            return None;
        }
        let mut bucket = 0;
        while rank >= counts[bucket] {
            rank -= counts[bucket];
            bucket += 1;
        }
        found |= (bucket as u64) << shift;
    }
    Some((found, rank + 1))
}

pub fn barrier_durations(planes: &[Plane], map: &[u8]) -> Vec<f64> {
    barriers_within(planes, map, MAX_EVENTS)
}

/// The durations of the barrier events in the trace that `render` gives with `limit` events, without the events of the trace.
pub fn barriers_within(planes: &[Plane], map: &[u8], limit: usize) -> Vec<f64> {
    let lines = lines(&jobs(planes));
    let cut = cut(&lines, limit);
    let found: Vec<_> = lines
        .par_iter()
        .map(|&(plane, line)| {
            let named = plane.stat_names.iter().any(|name| &**name == "step_name");
            let (mut barriers, mut ties) = (Vec::new(), 0);
            for (index, event) in line
                .events
                .iter()
                .enumerate()
                .filter(|(_, event)| kept(plane, line, event))
            {
                let mut name = Cow::Borrowed(label(plane, line, event));
                if named && line.labels.is_empty() {
                    for (_, _, stat) in plane
                        .named_stats(map, event.meta, event.raw)
                        .filter(|(_, stat_name, _)| &***stat_name == "step_name")
                    {
                        name = Cow::Owned(stat_text(plane, &stat.value));
                    }
                }
                if let Some(step) = line.steps.get(&index) {
                    name = Cow::Borrowed(&step.name);
                }
                let ts = time(plane, event);
                if name == BARRIER {
                    barriers.push((ts, event.dur, ties));
                }
                ties += usize::from(cut.is_some_and(|(cut, _)| ts == cut));
            }
            (barriers, ties)
        })
        .collect();
    let mut before = 0;
    let mut durations: Vec<(u64, u64)> = Vec::new();
    for (barriers, ties) in found {
        durations.extend(
            barriers
                .into_iter()
                .filter(|&(ts, _, tie)| {
                    cut.is_none_or(|(cut, quota)| ts < cut || ts == cut && before + tie < quota)
                })
                .map(|(ts, dur, _)| (ts, dur)),
        );
        before += ties;
    }
    if cut.is_some() {
        durations.sort_by_key(|event| event.0);
    }
    durations.iter().map(|event| event.1 as f64 / 1e6).collect()
}

pub fn render(planes: &[Plane], map: &[u8]) -> String {
    render_within(planes, map, MAX_EVENTS)
}

/// The trace of `render`. It frees the planes before it writes the text, because the text uses only the events that it takes from them.
pub fn render_owned(planes: Vec<Plane>, map: &[u8]) -> String {
    let collected = collect(&planes, map, MAX_EVENTS);
    crate::release(planes);
    text(collected)
}

/// The trace with the first `limit` events in time.
pub fn render_within(planes: &[Plane], map: &[u8], limit: usize) -> String {
    text(collect(planes, map, limit))
}

fn text((devices, events): (Devices, Vec<Event>)) -> String {
    let mut out = String::from(
        "{\"displayTimeUnit\":\"ns\",\"metadata\":{\"highres-ticks\":true},\n\"traceEvents\":[\n",
    );
    for (device, (name, resources)) in &devices {
        if !name.is_empty() {
            writeln!(out, "{{\"args\": {{\"name\": {}}}, \"name\": \"process_name\", \"ph\": \"M\", \"pid\": {device}}},", python_string(name)).expect("writing to a string cannot fail");
        }
        writeln!(out, "{{\"args\": {{\"sort_index\": {device}}}, \"name\": \"process_sort_index\", \"ph\": \"M\", \"pid\": {device}}},").expect("writing to a string cannot fail");
        for (resource, name) in resources {
            if !name.is_empty() {
                writeln!(out, "{{\"args\": {{\"name\": {}}}, \"name\": \"thread_name\", \"ph\": \"M\", \"pid\": {device}, \"tid\": {resource}}},", python_string(name)).expect("writing to a string cannot fail");
            }
            writeln!(out, "{{\"args\": {{\"sort_index\": {resource}}}, \"name\": \"thread_sort_index\", \"ph\": \"M\", \"pid\": {device}, \"tid\": {resource}}},").expect("writing to a string cannot fail");
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
                    part.push_str("\"dur\": ");
                    repr_into(&mut part, event.dur as f64 / 1e6);
                    part.push_str(", ");
                }
                part.push_str("\"name\": ");
                python_string_into(&mut part, &event.name);
                write!(
                    part,
                    ", \"ph\": {}, \"pid\": {}, ",
                    if event.dur != 0 { "\"X\"" } else { "\"i\"" },
                    event.device
                )
                .expect("writing to a string cannot fail");
                if event.dur == 0 {
                    part.push_str("\"s\": \"t\", ");
                }
                write!(part, "\"tid\": {}, \"ts\": ", event.resource)
                    .expect("writing to a string cannot fail");
                repr_into(&mut part, event.ts as f64 / 1e6);
                part.push_str("},\n");
            }
            part
        })
        .collect();
    crate::release(events);
    let pieces: Vec<&[u8]> = [out.as_bytes()]
        .into_iter()
        .chain(lines.iter().map(String::as_bytes))
        .chain([&b"{}]}\n"[..]])
        .collect();
    String::from_utf8(crate::concat(&pieces)).expect("The pieces are text")
}

#[cfg(test)]
#[path = "../tests/inline/trace/legacy.rs"]
mod tests;
