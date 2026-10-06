use crate::tools::table::{Cell, Table, number, string};
use crate::tools::utilization::{Device, Metric, compute};
use crate::xplane::{Field, Value, fields, nested, stats, varint};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::{Value as Json, json};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt::Write;

const TPU_PREFIX: &str = "/device:TPU:";
const GPU_PREFIX: &str = "/device:GPU:";
const PERF_COLUMNS: &str = r#"{"cols":[{"id":"Host","label":"Host","type":"string"},{"id":"Chip","label":"Chip","type":"number"},{"id":"Kernel","label":"Kernel","type":"string"},{"id":"Sample","label":"Sample","type":"number"},{"id":"Counter","label":"Counter","type":"string"},{"id":"Value","label":"Value (Hex)","type":"number"},{"id":"Description","label":"Description","type":"string"},{"id":"Set","label":"Set","type":"string"}],"rows":["#;
const UTILIZATION_COLUMNS: [(&str, &str, &str); 8] = [
    ("host", "number", "Host"),
    ("device", "number", "Device"),
    ("sample", "number", "Sample"),
    ("node", "number", "Node"),
    ("name", "string", "Name"),
    ("achieved", "number", "Achieved"),
    ("peak", "number", "Peak"),
    ("unit", "string", "Unit"),
];
const HOST_PLANES: [&str; 8] = ["/host:CPU", "Host CPUs", "/host:tfstreamz", "/host:metadata", "Syscalls", "/host:python-tracer", "/host:CUPTI", "/host:__ScopeRangeCallStack__"];
const DEVICE_PLANES: [&str; 3] = ["/device", "#Chip", "/custom:"];
const KERNEL_LINES: [&str; 4] = ["PALLAS", "XLA OPS", "Pallas", "XLA Ops"];
const DEFAULT_KERNEL: &str = "default_kernel";
const FIELDS_PER_CHUNK: usize = 4096;

pub struct Plane<'a> {
    pub name: String,
    bytes: &'a [u8],
    pub stat_names: FxHashMap<u64, &'a [u8]>,
    pub metadata: FxHashMap<u64, (&'a [u8], &'a [u8])>,
    pub lines: Vec<&'a [u8]>,
}

struct Line<'a> {
    id: i64,
    name: String,
    bytes: &'a [u8],
}

pub struct Event<'a> {
    pub meta: u64,
    pub offset_ps: u64,
    pub duration_ps: u64,
    pub raw: &'a [u8],
}

#[derive(Clone, Copy)]
enum Message {
    Space,
    Plane,
    Line,
    Event,
    Stat,
    EventEntry,
    StatEntry,
    EventMetadata,
    StatMetadata,
}

enum Slot {
    Message(Message),
    Text,
    Packed,
    Other,
}

fn slot(message: Message, field: u64) -> Slot {
    match (message, field) {
        (Message::Space, 1) => Slot::Message(Message::Plane),
        (Message::Space, 2..=4) | (Message::Plane, 2) | (Message::Line, 2 | 11) | (Message::Stat, 5) | (Message::EventMetadata, 2 | 4) | (Message::StatMetadata, 2 | 3) => Slot::Text,
        (Message::Plane, 3) => Slot::Message(Message::Line),
        (Message::Plane, 4) => Slot::Message(Message::EventEntry),
        (Message::Plane, 5) => Slot::Message(Message::StatEntry),
        (Message::Plane, 6) | (Message::Event, 4) | (Message::EventMetadata, 5) => Slot::Message(Message::Stat),
        (Message::Line, 4) => Slot::Message(Message::Event),
        (Message::EventEntry, 2) => Slot::Message(Message::EventMetadata),
        (Message::StatEntry, 2) => Slot::Message(Message::StatMetadata),
        (Message::EventMetadata, 6) => Slot::Packed,
        _ => Slot::Other,
    }
}

pub fn valid_space(buf: &[u8]) -> bool {
    valid(buf, Message::Space, true, true)
}

/// The same check as `valid_space` without the events, for a parse that checks the events.
pub fn valid_besides_events(buf: &[u8]) -> bool {
    valid(buf, Message::Space, true, false)
}

/// With `split`, a line checks only its own fields, and then parts of about `SPLIT` bytes check all of their fields in parallel.
fn valid(buf: &[u8], message: Message, split: bool, events: bool) -> bool {
    const SPLIT: usize = 1 << 16;
    let split = split && matches!(message, Message::Line);
    let (mut pos, mut start, mut children) = (0, 0, Vec::new());
    while pos < buf.len() {
        let Some(key) = varint(buf, &mut pos) else { return false };
        let field = key >> 3;
        if field == 0 || field >= 1 << 29 {
            return false;
        }
        let complete = match key & 7 {
            0 => varint(buf, &mut pos).is_some(),
            1 | 5 => {
                pos += if key & 7 == 1 { 8 } else { 4 };
                pos <= buf.len()
            }
            2 => {
                let Some(length) = varint(buf, &mut pos).filter(|&length| length <= (buf.len() - pos) as u64) else { return false };
                let body = &buf[pos..pos + length as usize];
                pos += length as usize;
                match slot(message, field) {
                    Slot::Message(child) if matches!(message, Message::Space | Message::Plane) => {
                        children.push((body, child, true));
                        true
                    }
                    Slot::Message(_) if split => true,
                    Slot::Message(Message::Event) => !events || valid_event(body),
                    Slot::Message(child) => valid(body, child, false, events),
                    Slot::Text => std::str::from_utf8(body).is_ok(),
                    Slot::Packed => {
                        let mut at = 0;
                        while at < body.len() && varint(body, &mut at).is_some() {}
                        at == body.len()
                    }
                    Slot::Other => true,
                }
            }
            _ => false,
        };
        if !complete {
            return false;
        }
        if split && pos - start >= SPLIT {
            children.push((&buf[start..pos], message, false));
            start = pos;
        }
    }
    if split {
        children.push((&buf[start..], message, false));
    }
    children.par_iter().all(|&(body, child, split)| valid(body, child, split, events))
}

/// The same check as `valid` for an event and its stats.
fn valid_event(buf: &[u8]) -> bool {
    walk(buf, |field, body| field != 4 || valid_stat(body))
}

/// The same check as `valid` for a stat.
pub fn valid_stat(buf: &[u8]) -> bool {
    walk(buf, |field, body| field != 5 || std::str::from_utf8(body).is_ok())
}

/// Checks the keys and the lengths of the fields. `body` checks each field that has a length.
#[inline(always)]
fn walk(buf: &[u8], mut body: impl FnMut(u64, &[u8]) -> bool) -> bool {
    let mut pos = 0;
    while pos < buf.len() {
        let Some(key) = varint(buf, &mut pos) else { return false };
        let field = key >> 3;
        if field == 0 || field >= 1 << 29 {
            return false;
        }
        let complete = match key & 7 {
            0 => varint(buf, &mut pos).is_some(),
            1 | 5 => {
                pos += if key & 7 == 1 { 8 } else { 4 };
                pos <= buf.len()
            }
            2 => {
                let Some(length) = varint(buf, &mut pos).filter(|&length| length <= (buf.len() - pos) as u64) else { return false };
                pos += length as usize;
                body(field, &buf[pos - length as usize..pos])
            }
            _ => false,
        };
        if !complete {
            return false;
        }
    }
    true
}

fn unsigned(value: &Value) -> u64 {
    value.int().map_or(0, |value| value as u64)
}

fn counter_value(value: &Value) -> u64 {
    match value {
        Value::Double(double) if *double != 0.0 => *double as u64,
        value => unsigned(value),
    }
}

pub fn first<'a, const N: usize>(raw: &'a [u8], field: u32, ids: [Option<u64>; N]) -> [Option<Value<'a>>; N] {
    let mut found: [Option<Value<'a>>; N] = std::array::from_fn(|_| None);
    if ids.iter().all(Option::is_none) {
        return found;
    }
    for stat in stats(raw, field, |id| ids.contains(&Some(id as u64))) {
        if let Some(slot) = found.iter_mut().zip(ids).find_map(|(slot, id)| (slot.is_none() && id == Some(stat.id as u64)).then_some(slot)) {
            *slot = Some(stat.value);
        }
    }
    found
}

fn entry(bytes: &[u8]) -> (u64, &[u8]) {
    let (mut key, mut value) = (0, &[][..]);
    for (tag, field) in fields(bytes) {
        match (tag, field) {
            (1, Field::Num(number)) => key = number,
            (2, Field::Bytes(_, body)) => value = body,
            _ => {}
        }
    }
    (key, value)
}

pub fn planes(map: &[u8], keep: impl Fn(&[u8]) -> bool) -> Vec<Plane<'_>> {
    let named = |bytes| nested(bytes, 2).last().unwrap_or(&[]);
    let bodies: Vec<&[u8]> = nested(map, 1).filter(|bytes| keep(named(bytes))).collect();
    bodies
        .into_par_iter()
        .map(|bytes| {
            let mut plane = Plane { name: String::new(), bytes, stat_names: FxHashMap::default(), metadata: FxHashMap::default(), lines: Vec::new() };
            for (tag, field) in fields(bytes) {
                match (tag, field) {
                    (2, Field::Bytes(_, name)) => plane.name = String::from_utf8_lossy(name).into_owned(),
                    (3, Field::Bytes(_, line)) => plane.lines.push(line),
                    (4, Field::Bytes(_, item)) => {
                        let (key, value) = entry(item);
                        plane.metadata.insert(key, (named(value), value));
                    }
                    (5, Field::Bytes(_, item)) => {
                        let (key, value) = entry(item);
                        plane.stat_names.insert(key, named(value));
                    }
                    _ => {}
                }
            }
            plane
        })
        .collect()
}

fn line(bytes: &[u8]) -> Line<'_> {
    let (id, name) = entry(bytes);
    Line { id: id as i64, name: String::from_utf8_lossy(name).into_owned(), bytes }
}

fn chunks(bytes: &[u8]) -> Vec<&[u8]> {
    let (mut pieces, mut start, mut count, mut items) = (Vec::new(), 0, 0, fields(bytes));
    while items.next().is_some() {
        count += 1;
        if count == FIELDS_PER_CHUNK {
            pieces.push(&bytes[start..items.pos]);
            (start, count) = (items.pos, 0);
        }
    }
    if start < bytes.len() {
        pieces.push(&bytes[start..]);
    }
    pieces
}

pub fn events(bytes: &[u8]) -> impl Iterator<Item = Event<'_>> {
    nested(bytes, 4).map(|raw| {
        let mut event = Event { meta: 0, offset_ps: 0, duration_ps: 0, raw };
        for (tag, field) in fields(raw) {
            match (tag, field) {
                (1, Field::Num(meta)) => event.meta = meta,
                (2, Field::Num(offset)) => event.offset_ps = offset,
                (3, Field::Num(duration)) => event.duration_ps = duration,
                (5, Field::Num(_)) => event.offset_ps = 0,
                _ => {}
            }
        }
        event
    })
}

impl<'a> Plane<'a> {
    pub fn stat_id(&self, name: &str) -> Option<u64> {
        self.stat_names.iter().filter(|(_, stat)| **stat == name.as_bytes()).map(|(&id, _)| id).min()
    }

    fn text(&self, value: &Value) -> String {
        match value {
            Value::Str(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            Value::Ref(id) => self.stat_names.get(id).map(|name| String::from_utf8_lossy(name).into_owned()).unwrap_or_default(),
            _ => String::new(),
        }
    }

    fn own(&self, name: &[u8]) -> Option<Value<'a>> {
        stats(self.bytes, 6, |_| true).filter(|stat| self.stat_names.get(&(stat.id as u64)).copied().unwrap_or(&[]) == name).last().map(|stat| stat.value)
    }

    fn own_id(&self, name: &[u8]) -> i64 {
        self.own(name).map_or(-1, |value| unsigned(&value) as i64)
    }

    fn is_tensor_core(&self) -> bool {
        self.name.strip_prefix(TPU_PREFIX).is_some_and(|core| core.bytes().all(|byte| byte.is_ascii_digit()))
    }

    fn device_type(&self) -> String {
        self.own(b"device_type_string").map(|value| self.text(&value)).unwrap_or_default()
    }

    fn sampler(&self) -> Sampler {
        let mut sampler = Sampler { ids: [self.stat_id("performance_counter_id"), self.stat_id("counter_value")], metadata: FxHashMap::default() };
        if sampler.ids.iter().any(Option::is_some) {
            sampler.metadata = self.metadata.par_iter().map(|(&key, (_, raw))| (key, sampler.read(raw, 5))).filter(|(_, found)| found.iter().any(Option::is_some)).collect();
        }
        sampler
    }
}

struct Sampler {
    ids: [Option<u64>; 2],
    metadata: FxHashMap<u64, [Option<u64>; 2]>,
}

impl Sampler {
    fn read(&self, raw: &[u8], field: u32) -> [Option<u64>; 2] {
        let [id, value] = first(raw, field, self.ids);
        [id.map(|id| unsigned(&id)), value.map(|value| counter_value(&value))]
    }

    fn counters(&self, line: &Line, sort: bool, accumulate: bool) -> (FxHashMap<u64, u64>, f64) {
        let pieces: Vec<(Vec<_>, f64)> = chunks(line.bytes)
            .into_par_iter()
            .map(|piece| {
                let (mut samples, mut longest) = (Vec::new(), 0.0f64);
                for event in events(piece) {
                    longest = longest.max(event.duration_ps as f64 / 1e3 / 1000.0);
                    let [mut id, mut value] = self.read(event.raw, 4);
                    if let Some([meta_id, meta_value]) = self.metadata.get(&event.meta) {
                        id = id.or(*meta_id);
                        value = value.or(*meta_value);
                    }
                    let counter = id.filter(|&id| id != 0).unwrap_or(event.meta);
                    if let (Some(value), true) = (value, counter != 0) {
                        samples.push(((event.offset_ps, Reverse(event.duration_ps)), counter, value));
                    }
                }
                (samples, longest)
            })
            .collect();
        let longest = pieces.iter().fold(0.0f64, |longest, piece| longest.max(piece.1));
        let mut samples: Vec<_> = pieces.into_iter().flat_map(|piece| piece.0).collect();
        if sort {
            samples.sort_by_key(|sample| sample.0);
        }
        let mut counters = FxHashMap::<u64, u64>::default();
        for (_, counter, value) in samples {
            let slot = counters.entry(counter).or_insert(0);
            *slot = if accumulate { slot.wrapping_add(value) } else { value };
        }
        (counters, longest)
    }
}

fn sorted_samples(planes: &[Plane]) -> bool {
    let device = |plane: &&Plane| HOST_PLANES.iter().chain(&DEVICE_PLANES).any(|prefix| plane.name.starts_with(prefix));
    !planes.iter().filter(device).all(|plane| plane.stat_id("group_id").is_some())
}

fn perf_rows(plane: &Plane, host: &str) -> Vec<String> {
    let chip = Some(plane.own_id(b"global_chip_id")).filter(|&chip| chip != -1).unwrap_or_else(|| plane.own_id(b"device_id"));
    let ids = [plane.stat_id("counter_value"), plane.stat_id("performance_counter_description"), plane.stat_id("performance_counter_sets")];
    if chip == -1 || ids[0].is_none() {
        return Vec::new();
    }
    plane
        .lines
        .par_iter()
        .flat_map(|bytes| {
            let line = line(bytes);
            let mut prefix = format!("{{\"c\":[{{\"v\":{}}},{{\"v\":", serde_json::to_string(host).unwrap());
            number(&mut prefix, chip as f64);
            write!(prefix, "}},{{\"v\":{}}},{{\"v\":", serde_json::to_string(&line.name).unwrap()).unwrap();
            number(&mut prefix, line.id as f64);
            prefix.push_str("},{\"v\":");
            chunks(line.bytes)
                .into_par_iter()
                .map(|piece| {
                    let (mut out, mut names) = (String::new(), FxHashMap::default());
                    for event in events(piece) {
                        let [Some(value), description, set] = first(event.raw, 4, ids) else { continue };
                        let value = unsigned(&value);
                        if !out.is_empty() {
                            out.push(',');
                        }
                        out.push_str(&prefix);
                        let name = names.entry(event.meta).or_insert_with(|| plane.metadata.get(&event.meta).map_or_else(String::new, |(name, _)| String::from_utf8_lossy(name).to_ascii_lowercase()));
                        string(&mut out, name);
                        write!(out, "}},{{\"f\":\"0x{value:x}\",\"v\":").unwrap();
                        number(&mut out, value as f64);
                        for text in [description, set] {
                            out.push_str("},{\"v\":");
                            string(&mut out, &text.map(|value| plane.text(&value)).unwrap_or_default());
                        }
                        out.push_str("}]}");
                    }
                    out
                })
                .collect::<Vec<_>>()
        })
        .filter(|rows| !rows.is_empty())
        .collect()
}

pub fn perf_counters(hosts: &[(String, &[u8])]) -> String {
    let rows: Vec<String> = hosts
        .iter()
        .flat_map(|(host, map)| {
            let planes = planes(map, |name| name.starts_with(TPU_PREFIX.as_bytes()) || name.starts_with(GPU_PREFIX.as_bytes()));
            planes.par_iter().flat_map(|plane| perf_rows(plane, host)).collect::<Vec<_>>()
        })
        .collect();
    format!("{PERF_COLUMNS}{}]}}", rows.join(","))
}

pub fn utilization_viewer(map: &[u8]) -> String {
    let planes = planes(map, |_| true);
    let sorted = sorted_samples(&planes);
    let mut table = Table::new(&UTILIZATION_COLUMNS);
    table.rows = planes
        .par_iter()
        .filter(|plane| plane.name.starts_with(TPU_PREFIX))
        .flat_map(|plane| {
            let (device_id, sampler) = (plane.own_id(b"device_id"), plane.sampler());
            let Some(device) = Device::from_type(&plane.device_type()).filter(|_| device_id != -1 && sampler.ids[1].is_some()) else { return Vec::new() };
            let sort = sorted && plane.is_tensor_core();
            plane
                .lines
                .par_iter()
                .flat_map_iter(|bytes| {
                    let line = line(bytes);
                    let (counters, _) = sampler.counters(&line, sort, false);
                    let metrics = if counters.is_empty() { Vec::new() } else { compute(&counters, device) };
                    metrics.into_iter().map(move |metric| {
                        let ids = [0.0, device_id as i32 as u64 as f64, line.id as i32 as u64 as f64, metric.node as f64];
                        let mut row: Vec<Cell> = ids.into_iter().map(Cell::Number).collect();
                        row.extend([Cell::Text(metric.name), Cell::Number(metric.achieved), Cell::Number(metric.peak), Cell::Text(metric.unit.into())]);
                        row
                    })
                })
                .collect()
        })
        .collect();
    table.json()
}

fn percent(value: f64, total: f64) -> f64 {
    (value / total * 10000.0).round() / 100.0
}

fn summarize(name: &str, duration_us: f64, metrics: &[Metric]) -> Json {
    let (mut mxu, mut mxu_peak, mut precisions) = (0.0, 0.0, [0.0f64; 4]);
    let mut other: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for metric in metrics {
        match metric.name.as_str() {
            "Avg MXU Busy" => {
                mxu += metric.achieved;
                mxu_peak += metric.peak;
            }
            "MXU BF16" => precisions[0] += metric.achieved,
            "MXU I8" => precisions[1] += metric.achieved,
            "MXU I4" => precisions[2] += metric.achieved,
            "MXU E4M3 + E5M2" => precisions[3] += metric.achieved,
            name if metric.peak > 0.0 => {
                let name = if name.starts_with("HBM Rd+Wr") { "HBM Bandwidth Utilization" } else { name };
                let sums = other.entry(name.to_string()).or_default();
                sums.0 += metric.achieved;
                sums.1 += metric.peak;
            }
            _ => {}
        }
    }
    let total = precisions[0] + precisions[1] + precisions[2] + precisions[3];
    let share = |value: f64| if total > 0.0 { percent(value, total) } else { 0.0 };
    let utilization = if mxu_peak > 0.0 { percent(mxu, mxu_peak) } else { 0.0 };
    let other: BTreeMap<String, f64> = other.into_iter().filter(|(_, (_, peak))| *peak > 0.0).map(|(name, (achieved, peak))| (name, percent(achieved, peak))).collect();
    json!({
        "duration_us": duration_us,
        "kernel_name": name,
        "mxu_cycles_breakdown": {"BF16": share(precisions[0]), "FP8": share(precisions[3]), "Int4": share(precisions[2]), "Int8": share(precisions[1])},
        "mxu_is_anomaly": utilization > 100.0,
        "mxu_utilization": utilization,
        "other_metrics": other,
    })
}

fn kernel_name(line_name: &str, fallback: &str) -> String {
    let name = match line_name.strip_prefix("counters_") {
        Some("0") => fallback,
        Some(suffix) => suffix,
        None if line_name == "_counters_" || line_name == "counters" => fallback,
        None => "",
    };
    if name.is_empty() { DEFAULT_KERNEL.to_string() } else { name.to_string() }
}

fn first_kernel(plane: &Plane, sort: bool) -> String {
    let named = |piece: &&[u8]| {
        let mut named = events(piece).filter_map(|event| plane.metadata.get(&event.meta).filter(|(name, _)| !name.is_empty()).map(|(name, _)| ((event.offset_ps, Reverse(event.duration_ps)), *name)));
        if sort { named.min_by_key(|found| found.0) } else { named.next() }
    };
    let lines = plane.lines.iter().map(|bytes| line(bytes)).filter(|line| KERNEL_LINES.contains(&line.name.as_str()));
    let found = lines.into_iter().find_map(|line| {
        let pieces = chunks(line.bytes);
        if sort { pieces.par_iter().filter_map(named).collect::<Vec<_>>().into_iter().min_by_key(|found| found.0) } else { pieces.par_iter().find_map_first(named) }
    });
    found.map(|(_, name)| String::from_utf8_lossy(name).into_owned()).unwrap_or_default()
}

pub struct Filter {
    pub kernel: String,
    pub duration_us: f64,
    pub force: bool,
    pub device: i64,
}

impl Default for Filter {
    fn default() -> Self {
        Self { kernel: String::new(), duration_us: 0.0, force: false, device: -1 }
    }
}

fn device_kernels(plane: &Plane, sorted: bool, filter: &Filter) -> Option<Json> {
    let (mut device_id, device_type) = (plane.own_id(b"device_id"), plane.device_type());
    if device_id == -1 {
        device_id = plane.name.strip_prefix(TPU_PREFIX).and_then(|rest| rest.trim_matches(|c: char| c.is_ascii_whitespace()).parse().ok()).unwrap_or(-1);
    }
    let device = Device::from_type(&device_type).filter(|_| device_id != -1 && (filter.device < 0 || device_id == filter.device))?;
    let sampler = plane.sampler();
    let kernels: Vec<Json> = if sampler.ids[1].is_none() {
        Vec::new()
    } else {
        let sort = sorted && plane.is_tensor_core();
        let fallback = first_kernel(plane, sort);
        plane
            .lines
            .par_iter()
            .filter_map(|bytes| {
                let line = line(bytes);
                let name = kernel_name(&line.name, &fallback);
                if !name.contains(filter.kernel.as_str()) {
                    return None;
                }
                let (mut counters, measured_us) = sampler.counters(&line, false, true);
                if counters.is_empty() {
                    return None;
                }
                let duration_us = if filter.duration_us > 0.0 { filter.duration_us } else { measured_us };
                let nominal = duration_us * (device.frequency_hz() / 1e6);
                if nominal > 0.0 {
                    for id in device.cycle_counters() {
                        let count = counters.entry(id).or_insert(0);
                        if *count == 0 || filter.force {
                            *count = nominal as u64;
                        }
                    }
                }
                Some(summarize(&name, duration_us, &compute(&counters, device)))
            })
            .collect()
    };
    Some(json!({"device_id": device_id, "device_type": device_type, "kernels": kernels}))
}

pub fn pretty(out: &mut String, value: &Json, depth: usize, quote: fn(&mut String, &str)) {
    let newline = |out: &mut String, depth: usize| {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', 2 * depth));
    };
    let items: Vec<(Option<&String>, &Json)> = match value {
        Json::Array(items) => items.iter().map(|item| (None, item)).collect(),
        Json::Object(entries) => entries.iter().map(|(key, item)| (Some(key), item)).collect(),
        Json::String(text) => return quote(out, text),
        Json::Number(value) if value.is_f64() => return number(out, value.as_f64().unwrap_or_default()),
        other => return out.push_str(&other.to_string()),
    };
    if items.is_empty() {
        return out.push_str(if value.is_array() { "[]" } else { "{}" });
    }
    out.push(if value.is_array() { '[' } else { '{' });
    for (index, (key, item)) in items.into_iter().enumerate() {
        out.push_str(if index > 0 { "," } else { "" });
        newline(out, depth + 1);
        if let Some(key) = key {
            quote(out, key);
            out.push_str(": ");
        }
        pretty(out, item, depth + 1, quote);
    }
    newline(out, depth);
    out.push(if value.is_array() { ']' } else { '}' });
}

pub fn kernel_utilization(map: &[u8], filter: &Filter) -> String {
    let planes = planes(map, |_| true);
    let sorted = sorted_samples(&planes);
    let devices: Vec<Json> = planes.par_iter().filter(|plane| plane.name.starts_with(TPU_PREFIX)).filter_map(|plane| device_kernels(plane, sorted, filter)).collect();
    let mut out = String::new();
    pretty(&mut out, &json!({"devices": devices, "status": "SUCCESS"}), 0, string);
    out
}

pub fn corrupt(paths: &[std::path::PathBuf]) -> bool {
    paths.iter().any(|path| crate::read_file(path).is_ok_and(|map| !valid_space(&map)))
}

pub fn serve(tag: &str, paths: &[std::path::PathBuf]) -> Option<String> {
    let maps: Vec<(String, Option<Vec<u8>>)> = paths.iter().map(|path| (crate::host_name(path), crate::read_file(path).ok().filter(|map| valid_space(map)))).collect();
    match (tag, maps.as_slice()) {
        ("perf_counters", [_, ..]) => Some(perf_counters(&maps.iter().filter_map(|(host, map)| Some((host.clone(), &map.as_ref()?[..]))).collect::<Vec<_>>())),
        ("utilization_viewer", [(_, Some(map))]) => Some(utilization_viewer(map)),
        ("kernel_utilization", [(_, Some(map))]) => Some(kernel_utilization(map, &Filter::default())),
        _ => None,
    }
}

#[cfg(test)]
#[path = "../tests/inline/tools/counters.rs"]
mod tests;
