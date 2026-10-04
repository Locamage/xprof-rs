use crate::framework_op_stats::{is_jax_op_type, is_tf_op_name, is_tf_op_type};
use crate::gpu;
use crate::hlo::xla::HloInstructionMeta;
use crate::xplane::{Ev, Line, NONE_GROUP, Plane, Value, slice, stats};
use prost::Message;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

pub fn is_tensor_core(name: &str) -> bool {
    name.strip_prefix("/device:TPU:").is_some_and(|core| !core.is_empty() && core.bytes().all(|byte| byte.is_ascii_digit()))
}

pub fn is_grouped(planes: &[Plane]) -> bool {
    planes.iter().filter(|plane| plane.name.starts_with("/device:") || plane.name.starts_with("/host:CPU")).all(|plane| plane.stat_names.iter().any(|stat| &**stat == "group_id"))
}

const MEMCPY: [&str; 4] = ["MemcpyHToD", "MemcpyDToH", "MemcpyDToD", "MemcpyHToH"];

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Category {
    Unknown,
    TensorFlow,
    Jax,
    TfData,
    Memcpy,
}

pub struct TfOp<'a> {
    pub category: Category,
    pub name: &'a str,
    pub kind: &'a str,
}

impl TfOp<'_> {
    pub fn event_name(&self) -> String {
        match self.category {
            Category::Unknown => self.name.trim_end_matches(|c: char| c.is_ascii_whitespace()).to_string(),
            Category::TfData => format!("Iterator::{}", self.name.rsplit("::").next().unwrap()),
            _ => self.kind.to_string(),
        }
    }

    pub fn scopes(&self) -> Vec<&str> {
        let mut scopes: Vec<&str> = if matches!(self.category, Category::TensorFlow | Category::Jax) { self.name.split('/').collect() } else { Vec::new() };
        scopes.pop();
        scopes
    }
}

pub fn tf_op(full: &str) -> TfOp<'_> {
    let op = |category: Category, name, kind| TfOp { category, name, kind };
    let Some((name, kind)) = full.split_once(':') else {
        return match MEMCPY.iter().find(|memcpy| full.len() >= memcpy.len() && full.as_bytes()[..memcpy.len()].eq_ignore_ascii_case(memcpy.as_bytes())) {
            Some(memcpy) => op(Category::Memcpy, full, memcpy),
            None => op(Category::Unknown, full, ""),
        };
    };
    if name == "Iterator" {
        return op(Category::TfData, full, "Dataset");
    }
    if is_tf_op_name(name) && is_tf_op_type(kind) {
        return op(Category::TensorFlow, name, kind);
    }
    let derived = if kind.is_empty() {
        let last = name.rsplit('/').next().unwrap();
        let suffix = last.rsplit('_').next().unwrap();
        if last.len() > suffix.len() && suffix.trim_ascii().parse::<i64>().is_ok() { &last[..last.len() - suffix.len() - 1] } else { last }
    } else {
        kind
    };
    if is_jax_op_type(derived) {
        op(Category::Jax, name, &derived[..derived.find('[').unwrap_or(derived.len())])
    } else if kind.is_empty() {
        op(Category::TensorFlow, name, derived)
    } else {
        op(Category::Unknown, full, "")
    }
}

const TF_OPS: usize = 0;
const SCOPES: usize = 1;
const SOURCES: usize = 2;
const HLO_OPS: usize = 2;
const MODULES: usize = 3;
const STREAM_SOURCES: usize = 4;
const TPU_LINES: [(i64, &str, &[usize]); 3] = [(3735928562, "Framework Ops", &[]), (3735928561, "Framework Name Scope", &[TF_OPS]), (3735928566, "Source code", &[])];
const STREAM_LINES: [(i64, &str, &[usize]); 5] =
    [(1, "Framework Ops", &[]), (0, "Framework Name Scope", &[TF_OPS]), (3, "XLA Ops", &[]), (2, "XLA Modules", &[SCOPES, HLO_OPS]), (5, "Source code", &[])];
const PS_PER_NS: u64 = 1000;
const HOST: &str = "/host:CPU";
const STEP_LINE: i64 = 0xdead_beef;
const LAUNCH_LINE: i64 = STEP_LINE + 1;
const DEVICE_DERIVED: i64 = STEP_LINE + 290;
const DEVICE_DERIVED_MAX: i64 = DEVICE_DERIVED + 99;
const PER_STREAM: i64 = 6;

type Launches = BTreeMap<i64, (u64, u64, i64, u64, u64)>;
type Symbols = HashMap<u64, HashMap<String, (String, String)>>;

pub fn is_derived(id: i64) -> bool {
    (STEP_LINE as i32..=DEVICE_DERIVED_MAX as i32).contains(&(id as i32))
}

#[derive(Default)]
struct Builder {
    id: i64,
    name: String,
    labels: Vec<Box<str>>,
    longs: Vec<Box<str>>,
    ids: FxHashMap<String, u32>,
    events: Vec<Ev>,
    args: HashMap<usize, Vec<(&'static str, String)>>,
    last: Vec<Option<(usize, Option<i64>)>>,
    dependents: &'static [usize],
}

struct Derived {
    lines: Vec<Builder>,
    gpu: bool,
}

/// The op, the source and whether the event is asynchronous.
type Tags = (String, String, u64);
/// The start, the duration, the line and the index of an event.
type Position = (u64, Reverse<u64>, usize, usize);

struct Parsed {
    scopes: Vec<u32>,
    op: Option<u32>,
    source: Option<u32>,
}

impl Derived {
    fn new(lines: &[(i64, &str, &'static [usize])], base: i64, suffix: &str, gpu: bool) -> Derived {
        let lines = lines.iter().map(|&(id, name, dependents)| Builder { id: base + id, name: format!("{name}{suffix}"), dependents, ..Default::default() }).collect();
        Derived { lines, gpu }
    }

    fn meta(&mut self, line: usize, key: &str, display: &str) -> u32 {
        let builder = &mut self.lines[line];
        if let Some(&id) = builder.ids.get(key) {
            return id;
        }
        let id = builder.labels.len() as u32;
        builder.ids.insert(key.into(), id);
        builder.labels.push(if display.is_empty() { key } else { display }.into());
        builder.longs.push(if display.is_empty() { "" } else { key }.into());
        id
    }

    fn expand_or_add(&mut self, line: usize, meta: u32, (start, end): (u64, u64), group: i64, scope: Option<i64>, level: usize) {
        let gpu = self.gpu;
        let builder = &mut self.lines[line];
        if builder.last.len() <= level {
            builder.last.resize(level + 1, None);
        }
        if let Some((index, last_scope)) = builder.last[level] {
            let event = &mut builder.events[index];
            let same_scope = scope.is_none() || last_scope.is_none() || scope == last_scope;
            if event.meta == meta && event.group == group && same_scope && (gpu || group != NONE_GROUP || event.ts.saturating_add(event.dur.saturating_mul(3)) >= start) {
                event.dur = (event.ts + event.dur).max(end) - event.ts;
                return;
            }
        }
        self.reset(line, level);
        let builder = &mut self.lines[line];
        builder.events.push(Ev { ts: start, dur: end - start, group, raw: (0, 0), meta, eager: None });
        builder.last[level] = Some((builder.events.len() - 1, scope));
    }

    fn expand_or_add_all(&mut self, line: usize, metas: &[u32], span: (u64, u64), group: i64, scopes: &[Option<i64>]) {
        if metas.is_empty() {
            return;
        }
        for (level, &meta) in metas.iter().enumerate() {
            self.expand_or_add(line, meta, span, group, scopes.get(level).copied().flatten(), level);
        }
        self.reset(line, metas.len());
    }

    fn reset(&mut self, line: usize, level: usize) {
        let builder = &mut self.lines[line];
        if let Some(top) = (level..builder.last.len()).take_while(|&index| builder.last[index].is_some()).last().filter(|&top| top > level) {
            let max_shrink = (builder.events[builder.last[top].unwrap().0].dur / PS_PER_NS) as i64 - 1;
            let (mut shrink, mut previous) = (0i64, None);
            for index in level..=top {
                let event = &mut builder.events[builder.last[index].unwrap().0];
                let span = (event.ts, event.dur);
                if shrink < max_shrink && previous == Some(span) {
                    shrink += 1;
                }
                previous = Some(span);
                event.dur = event.dur.saturating_sub(PS_PER_NS * shrink as u64);
            }
        }
        builder.last.iter_mut().skip(level).for_each(|last| *last = None);
        if level == 0 {
            for &dependent in builder.dependents {
                self.reset(dependent, 0);
            }
        }
    }

    fn set_stat(&mut self, line: usize, level: usize, key: &'static str, value: String) {
        let builder = &mut self.lines[line];
        let Some(Some((index, _))) = builder.last.get(level) else { return };
        let args = builder.args.entry(*index).or_default();
        match args.iter_mut().find(|(existing, _)| *existing == key) {
            Some(slot) => slot.1 = value,
            None => args.push((key, value)),
        }
    }

    fn parse(&mut self, tf_op: &str, source: &str) -> Parsed {
        let (mut scopes, mut op) = (Vec::new(), None);
        if !tf_op.is_empty() {
            let parsed = crate::derive::tf_op(tf_op);
            scopes = parsed.scopes().iter().map(|scope| self.meta(SCOPES, scope, "")).collect();
            op = Some(self.meta(TF_OPS, tf_op, &parsed.event_name()));
        }
        Parsed { scopes, op, source: (!source.is_empty()).then(|| self.meta(SOURCES, source, "")) }
    }

    fn apply(&mut self, parsed: &Parsed, span: (u64, u64), group: i64) {
        self.expand_or_add_all(SCOPES, &parsed.scopes, span, group, &[]);
        for (line, meta) in [(TF_OPS, parsed.op), (SOURCES, parsed.source)] {
            if let Some(meta) = meta {
                self.expand_or_add(line, meta, span, group, None, 0);
            }
        }
    }

    fn into_lines(self, sort: bool) -> Vec<Line> {
        self.lines
            .into_iter()
            .map(|mut builder| {
                if sort {
                    builder.events.sort_by_key(|event| (event.ts, Reverse(event.dur)));
                }
                let args = builder.args.into_iter().map(|(index, args)| (index, args.into_iter().map(|(key, value)| format!("\"{key}\":{value}")).collect())).collect();
                Line { id: builder.id, name: builder.name, labels: builder.labels, longs: builder.longs, events: builder.events, args, ..Default::default() }
            })
            .collect()
    }
}

pub fn derive(plane: &mut Plane, map: &[u8]) {
    let (tf_op_id, source_id, async_id) = (plane.id("tf_op"), plane.id("source"), plane.id("_a"));
    let wanted: Vec<usize> = [tf_op_id, source_id, async_id].into_iter().flatten().collect();
    let scan = |raw: &[u8], field: u32| {
        let (mut tf_op, mut source, mut is_async) = (String::new(), String::new(), 0u64);
        for stat in stats(raw, field, |id| wanted.contains(&id)) {
            match stat.value {
                Value::Str(_) | Value::Ref(_) if Some(stat.id) == tf_op_id => tf_op = plane.text(&stat.value),
                Value::Str(_) | Value::Ref(_) if Some(stat.id) == source_id => source = plane.text(&stat.value),
                Value::Int(_) | Value::Uint(_) if Some(stat.id) == async_id => is_async = stat.value.int().unwrap() as u64,
                _ => {}
            }
        }
        (tf_op, source, is_async)
    };
    let tags: Vec<Tags> = plane.meta.par_iter().map(|meta| scan(slice(map, meta.raw), 5)).collect();
    // The events to derive from, in time order, with their groups and the tags of their own stats.
    let mut order: Vec<(Position, i64, Option<Tags>)> = Vec::new();
    for (line_index, line) in plane.lines.iter().enumerate() {
        order.par_extend(line.events.par_iter().enumerate().filter_map(|(index, event)| {
            let raw = slice(map, event.raw);
            let own = stats(raw, 4, |id| wanted.contains(&id)).next().is_some();
            let (tf_op, source, _) = &tags[event.meta as usize];
            if tf_op.is_empty() && source.is_empty() && !own {
                return None;
            }
            let group = if event.group != NONE_GROUP { event.group } else { plane.group_of(event.meta, raw).unwrap_or(NONE_GROUP) };
            Some(((event.ts, Reverse(event.dur), line_index, index), group, own.then(|| scan(raw, 4))))
        }));
    }
    order.par_sort_unstable_by_key(|&(position, _, _)| position);
    let mut derived = Derived::new(&TPU_LINES, 0, "", false);
    // The same texts always give the same metadata. Thus the code parses each pair of texts one time.
    let mut cache: FxHashMap<(&str, &str), Parsed> = FxHashMap::default();
    for ((start, Reverse(dur), line_index, index), group, own) in &order {
        let (tf_op, source, is_async) = &tags[plane.lines[*line_index].events[*index].meta as usize];
        let texts = match own {
            Some((event_tf_op, event_source, event_async)) => {
                if (if *event_async != 0 { *event_async } else { *is_async }) != 0 {
                    continue;
                }
                (if event_tf_op.is_empty() { tf_op } else { event_tf_op }, if event_source.is_empty() { source } else { event_source })
            }
            None if *is_async == 0 => (tf_op, source),
            None => continue,
        };
        let parsed = cache.entry((texts.0.as_str(), texts.1.as_str())).or_insert_with(|| derived.parse(texts.0, texts.1));
        derived.apply(parsed, (*start, start + dur), *group);
    }
    derived.reset(SCOPES, 0);
    plane.lines.extend(derived.into_lines(true).into_iter().filter(|line| !line.events.is_empty()));
}

fn sorted(plane: &Plane, lines: &[usize]) -> Vec<(usize, usize)> {
    let mut order: Vec<(u64, Reverse<u64>, usize, usize)> =
        lines.iter().flat_map(|&line| plane.lines[line].events.iter().enumerate().map(move |(index, event)| (event.ts, Reverse(event.dur), line, index))).collect();
    order.sort_unstable();
    order.into_iter().map(|(_, _, line, index)| (line, index)).collect()
}

fn int_value(value: &Value) -> i64 {
    if let Value::Int(value) = value { *value } else { 0 }
}

fn group_of(map: &[u8], event: &Ev, group_id: Option<usize>) -> Option<i64> {
    if event.group != NONE_GROUP {
        return Some(event.group);
    }
    stats(slice(map, event.raw), 4, |id| Some(id) == group_id).next().map(|stat| int_value(&stat.value))
}

fn symbols(planes: &[Plane], map: &[u8]) -> Symbols {
    crate::hlo::protos(planes, map)
        .into_par_iter()
        .map(|(program, proto)| {
            let bodies: Vec<&[u8]> = crate::hlo::split(crate::hlo::last_bytes(proto, 1), 3).1.into_iter().flat_map(|computation| crate::hlo::split(computation, 2).1).collect();
            let decoded: Vec<(String, (String, String))> = bodies
                .par_iter()
                .map(|body| {
                    let inst = HloInstructionMeta::decode(*body).unwrap_or_default();
                    let metadata = inst.metadata.unwrap_or_default();
                    let source = if metadata.source_file.is_empty() { String::new() } else { format!("{}:{}", metadata.source_file.rsplit('/').next().unwrap(), metadata.source_line) };
                    (inst.name, (format!("{}:{}", metadata.op_name, metadata.op_type), source))
                })
                .collect();
            let mut found = HashMap::new();
            for (name, symbol) in decoded {
                found.entry(name).or_insert(symbol);
            }
            (program, found)
        })
        .collect()
}

fn annotate(plane: &Plane, map: &[u8], stream: usize, base: i64, symbols: &Symbols) -> Vec<Line> {
    let scanner = crate::gpu::Scanner::new(plane, map);
    let line = &plane.lines[stream];
    let mut derived = Derived::new(&STREAM_LINES, base, &format!(" - from #{}", line.id), true);
    let (mut key, mut parsed, mut levels, mut metas) = (String::new(), HashMap::new(), Vec::new(), Vec::new());
    for (_, index) in sorted(plane, &[stream]) {
        let event = &line.events[index];
        let found = scanner.scan(event);
        if found.text(gpu::KERNEL).is_empty() && !found.has(gpu::GRAPH_EXEC) {
            continue;
        }
        let group = if event.group != NONE_GROUP { event.group } else { found.signed(gpu::GROUP).unwrap_or(NONE_GROUP) };
        let (span, module, program, hlo_op) = ((event.ts, event.ts + event.dur), found.text(gpu::MODULE), found.program(), found.text(gpu::HLO_OP));
        let has_ops = found.has(gpu::HLO_OP);
        levels.clear();
        if !module.is_empty() || has_ops {
            levels.extend(found.signed(gpu::SCOPE).map(Some));
        }
        if !module.is_empty() {
            key.clear();
            match program {
                Some(program) => _ = write!(key, "{module}({program})"),
                None => key.push_str(&module),
            }
            let meta = derived.meta(MODULES, &key, "");
            derived.expand_or_add(MODULES, meta, span, group, levels.last().copied().flatten(), 0);
        }
        if has_ops {
            let symbol = program.and_then(|program| symbols.get(&program)).and_then(|module| module.get(hlo_op.rsplit("::").next().unwrap_or_default()));
            metas.clear();
            for op in hlo_op.split("::") {
                key.clear();
                match program {
                    Some(program) => _ = write!(key, "{program}/{op}"),
                    None => _ = write!(key, "{module}/{op}"),
                }
                metas.push(derived.meta(HLO_OPS, &key, op));
            }
            if levels.len() == metas.len() {
                levels.reverse();
            } else {
                levels.clear();
            }
            derived.expand_or_add_all(HLO_OPS, &metas, span, group, &levels);
            if let Some(graph) = found.unsigned(gpu::GRAPH).filter(|&graph| graph != 0) {
                derived.set_stat(HLO_OPS, metas.len() - 1, "cuda_graph_id", graph.to_string());
                if let Some(correlation) = found.int(gpu::CORRELATION) {
                    derived.set_stat(HLO_OPS, metas.len() - 1, "correlation_id", correlation.to_string());
                }
            }
            if let Some((tf_op, source)) = symbol {
                if !tf_op.is_empty() {
                    let parsed = parsed.entry(tf_op.as_str()).or_insert_with(|| derived.parse(tf_op, ""));
                    derived.apply(parsed, span, group);
                }
                if !source.is_empty() {
                    let meta = derived.meta(STREAM_SOURCES, source, "");
                    derived.expand_or_add(STREAM_SOURCES, meta, span, group, None, 0);
                }
            }
        } else if !found.text(gpu::TF_OP).is_empty() {
            let parsed = derived.parse(&found.text(gpu::TF_OP), "");
            derived.apply(&parsed, span, group);
        }
    }
    derived.into_lines(false)
}

fn steps(plane: &mut Plane, map: &[u8], names: &HashMap<i64, String>) {
    let group_id = plane.id("group_id");
    let lines: Vec<usize> = (0..plane.lines.len()).filter(|&line| !is_derived(plane.lines[line].id)).collect();
    let mut last = None;
    let mut derived = Derived::new(&[(STEP_LINE, "Steps", &[])], 0, "", true);
    let mut key = String::new();
    for (line, index) in sorted(plane, &lines) {
        let event = &mut plane.lines[line].events[index];
        match group_of(map, event, group_id) {
            Some(group) => last = Some(group),
            None => {
                let Some(group) = last else { continue };
                event.group = group;
            }
        }
        let group = last.unwrap();
        key.clear();
        _ = write!(key, "{group}");
        let meta = derived.meta(0, &key, names.get(&group).map_or("", String::as_str));
        derived.expand_or_add(0, meta, (event.ts, event.ts + event.dur), group, None, 0);
    }
    let mut line = derived.into_lines(false).pop().unwrap();
    line.longs.iter_mut().for_each(|long| *long = "".into());
    for (index, event) in line.events.iter().enumerate() {
        if let Some(name) = names.get(&event.group) {
            line.args.insert(index, vec![format!("\"step_name\":{}", crate::json::quoted(name))]);
        }
    }
    plane.lines.push(line);
}

pub fn derive_gpu(planes: &mut [Plane], map: &[u8], groups: Option<&HashMap<i64, String>>, trace: bool) {
    let gpus: Vec<usize> = (0..planes.len()).filter(|&index| planes[index].name.starts_with(gpu::PREFIX)).collect();
    if gpus.is_empty() {
        return;
    }
    let none = HashMap::new();
    let names = groups.unwrap_or(&none);
    let (symbols, launches) = if trace { rayon::join(|| symbols(planes, map), || launch_lines(planes, map, gpus.len(), names)) } else { Default::default() };
    planes.par_iter_mut().filter(|plane| plane.name.starts_with(gpu::PREFIX)).for_each(|plane| {
        if groups.is_some() || plane.id("group_id").is_some() {
            steps(plane, map, names);
        }
        if trace {
            annotate_streams(plane, map, &symbols);
        }
        plane.lines.retain(|line| !line.events.is_empty());
    });
    for (device, line) in launches {
        planes[gpus[device]].lines.push(line);
    }
}

fn annotate_streams(plane: &mut Plane, map: &[u8], symbols: &Symbols) {
    let mut streams: Vec<usize> = (0..plane.lines.len()).filter(|&line| !is_derived(plane.lines[line].id)).collect();
    streams.sort_by_key(|&line| (!plane.lines[line].name.contains("Compute"), Reverse(plane.lines[line].events.len())));
    let streams: Vec<(usize, i64)> = streams.into_iter().zip((DEVICE_DERIVED..).step_by(PER_STREAM as usize)).take_while(|&(_, base)| base + PER_STREAM <= DEVICE_DERIVED_MAX).collect();
    let derived: Vec<Vec<Line>> = streams.par_iter().map(|&(stream, base)| annotate(plane, map, stream, base, symbols)).collect();
    let mut ordered = Vec::new();
    for (&(stream, _), lines) in streams.iter().zip(&derived) {
        let mut ids: Vec<i64> = lines.iter().map(|line| line.id).collect();
        ids.sort_unstable();
        ordered.push(plane.lines[stream].id);
        ordered.extend(ids);
    }
    let display: HashMap<i64, i64> = ordered.into_iter().zip(1..).collect();
    plane.lines.extend(derived.into_iter().flatten());
    for line in &mut plane.lines {
        if let Some(&index) = display.get(&line.id) {
            line.display_id = index;
        }
    }
}

fn merge_launches(mut into: Vec<Launches>, from: Vec<Launches>) -> Vec<Launches> {
    for (into, from) in into.iter_mut().zip(from) {
        for (group, (begin, end, count, max, sum)) in from {
            let entry = into.entry(group).or_insert((begin, end, 0, 0, 0));
            *entry = (entry.0.min(begin), entry.1.max(end), entry.2 + count, entry.3.max(max), entry.4 + sum);
        }
    }
    into
}

fn launch_lines(planes: &[Plane], map: &[u8], devices: usize, names: &HashMap<i64, String>) -> Vec<(usize, Line)> {
    let Some(plane) = planes.iter().find(|plane| plane.name == HOST) else { return Vec::new() };
    let ids = ["device_id", "correlation_id", "group_id"].map(|name| plane.id(name));
    let runtime: Vec<bool> = plane.meta.iter().map(|meta| meta.full_name(map).starts_with("cu")).collect();
    let empty = || vec![Launches::new(); devices];
    let launches = plane
        .lines
        .par_iter()
        .filter(|line| !is_derived(line.id))
        .map(|line| {
            let mut launches = empty();
            for event in line.events.iter().filter(|event| !runtime[event.meta as usize]) {
                let (mut device, mut correlation, mut group) = (None, None, None);
                for stat in stats(slice(map, event.raw), 4, |id| ids.contains(&Some(id))) {
                    match ids.iter().position(|id| *id == Some(stat.id)).unwrap() {
                        0 => device = stat.value.int(),
                        1 => correlation = stat.value.int(),
                        _ => group = Some(int_value(&stat.value)),
                    }
                }
                let group = if event.group != NONE_GROUP { Some(event.group) } else { group };
                let (Some(device), Some(_), Some(group)) = (device, correlation, group) else { continue };
                let Some(device) = usize::try_from(device).ok().filter(|&device| device < devices) else { continue };
                let entry = launches[device].entry(group).or_insert((event.ts, event.ts + event.dur, 0, 0, 0));
                *entry = (entry.0.min(event.ts), entry.1.max(event.ts + event.dur), entry.2 + 1, entry.3.max(event.dur), entry.4 + event.dur);
            }
            launches
        })
        .reduce(empty, merge_launches);
    let mut lines = Vec::new();
    for (device, launches) in launches.into_iter().enumerate().filter(|(_, launches)| !launches.is_empty()) {
        let mut line = Line { id: LAUNCH_LINE, name: "Launch Stats".into(), ..Default::default() };
        for (group, (begin, end, count, max, sum)) in launches {
            let Some(name) = names.get(&group) else { continue };
            line.args.insert(
                line.events.len(),
                vec![
                    format!("\"num_launches\":{count}"),
                    format!("\"max_launch_time_us\":{}", crate::json::double(max as f64 / 1e6)),
                    format!("\"avg_launch_time_us\":{}", crate::json::double(sum as f64 / count as f64 / 1e6)),
                ],
            );
            line.events.push(Ev { ts: begin, dur: end - begin, group, raw: (0, 0), meta: line.labels.len() as u32, eager: None });
            line.labels.push(format!("Launch Stats for {name}").into());
            line.longs.push("".into());
        }
        lines.push((device, line));
    }
    lines
}
