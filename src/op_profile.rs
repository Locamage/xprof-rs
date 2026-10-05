use crate::input_pipeline_analyzer::{HARDWARE, hardware};
use crate::opstats::{GIBI_IN_GIGA, HBM, IDLE, Metrics, OpStats, READ, SPARSE_CORE, Source, WRITE, add, combine_memory, giga_to_gibi, pico_to_nano, safe_divide};
use crate::pbtext::json_string;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write;

const CHILDREN_PER_NODE: usize = 100;
const ROOT: usize = 0;
const PARALLEL_LEVELS: usize = 4;
const PROGRAM: usize = 0;
const CATEGORY: usize = 1;
const PROVENANCE: usize = 2;
const PROVENANCE_NODE: u8 = 0;
const CATEGORY_NODE: u8 = 1;
const DEDUPLICATED_NODE: u8 = 2;
const GROUPINGS: [(&str, &str, i32); 3] = [("by_program", "byProgram", 2), ("by_category", "byCategory", 1), ("by_provenance", "byProvenance", 100)];

#[derive(Clone, Default)]
struct Xla<'a> {
    expression: &'a str,
    provenance: &'a str,
    category: &'a str,
    layout: bool,
    program_id: u64,
    source: Option<&'a Source>,
}

#[derive(Clone, Default)]
struct Acc {
    time_ps: u64,
    self_time_ps: u64,
    normalized_time_ps: u64,
    occurrences: u64,
    flops_v2: f64,
    model_flops_v2: f64,
    memory: Vec<(u8, u64, u64)>,
}

#[derive(Clone, Default)]
struct Node<'a> {
    name: Cow<'a, str>,
    children: Vec<usize>,
    xla: Option<Xla<'a>>,
    num_children: i32,
    metrics: Acc,
    fused: Option<&'a Metrics>,
}

struct Builder<'a> {
    grouping: usize,
    nodes: Vec<Node<'a>>,
    programs: FxHashMap<u64, usize>,
    children: FxHashMap<(u8, usize, &'a str), usize>,
    names: &'a HashMap<u64, String>,
    peak_gigaflops: f64,
    peak_bandwidths: Vec<f64>,
    total_time_ps: u64,
}

fn is_fusion(category: &str) -> bool {
    category.ends_with(" fusion")
}

fn children_time(metrics: &Metrics) -> u64 {
    metrics.time_ps.wrapping_sub(metrics.self_time_ps)
}

fn update(child: &Metrics, parent: &mut Acc) {
    parent.time_ps += child.self_time_ps;
    add!(parent, child, normalized_time_ps, self_time_ps);
    if children_time(child) == 0 {
        add!(parent, child, flops_v2, model_flops_v2);
        combine_memory(&child.memory, &mut parent.memory);
    }
}

fn combine(source: &Metrics, destination: &mut Acc) {
    add!(destination, source, occurrences, time_ps, self_time_ps, normalized_time_ps, flops_v2, model_flops_v2);
    combine_memory(&source.memory, &mut destination.memory);
}

fn hlo_unescape(text: &str) -> Option<String> {
    let (mut out, mut characters) = (String::new(), text.chars());
    while let Some(character) = characters.next() {
        match character {
            '"' => return Some(out),
            '\\' => match characters.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                other => out.push(other),
            },
            other => out.push(other),
        }
    }
    None
}

fn kernel_metadata(expression: &str) -> String {
    let Some(start) = expression.find("kernel_metadata=").filter(|_| expression.contains("xprof_metadata")) else { return String::new() };
    let rest = &expression[start + "kernel_metadata=".len()..];
    let metadata: Option<serde_json::Value> = match rest.chars().next() {
        Some('"') => hlo_unescape(&rest[1..]).and_then(|value| serde_json::from_str(&value).ok()),
        Some('{') => serde_json::Deserializer::from_str(rest).into_iter().next().and_then(Result::ok),
        _ => None,
    };
    let xprof = match metadata.as_ref().and_then(|metadata| metadata.get("xprof_metadata")) {
        Some(serde_json::Value::String(text)) => serde_json::from_str(text).ok(),
        other => other.cloned(),
    };
    xprof.map(|xprof: serde_json::Value| xprof.to_string()).unwrap_or_default()
}

fn symbol(metrics: &Metrics) -> Xla<'_> {
    Xla { expression: &metrics.long_name, provenance: &metrics.provenance, category: &metrics.category, layout: false, program_id: metrics.module, source: metrics.source.as_ref() }
}

pub fn partial_sort(items: &mut [usize], k: usize, less: impl Fn(usize, usize) -> bool) {
    let adjust = |items: &mut [usize], mut hole: usize, length: usize, value: usize| {
        let top = hole;
        let mut second = hole;
        while second < length.saturating_sub(1) / 2 {
            second = 2 * (second + 1);
            if less(items[second], items[second - 1]) {
                second -= 1;
            }
            items[hole] = items[second];
            hole = second;
        }
        if length != 0 && length & 1 == 0 && second == (length - 2) / 2 {
            second = 2 * (second + 1);
            items[hole] = items[second - 1];
            hole = second - 1;
        }
        while hole > top {
            let parent = (hole - 1) / 2;
            if !less(items[parent], value) {
                break;
            }
            items[hole] = items[parent];
            hole = parent;
        }
        items[hole] = value;
    };
    if k >= 2 {
        let mut parent = (k - 2) / 2;
        loop {
            let value = items[parent];
            adjust(items, parent, k, value);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }
    for index in k..items.len() {
        if less(items[index], items[0]) {
            let value = items[index];
            items[index] = items[0];
            adjust(items, 0, k, value);
        }
    }
    for last in (1..k).rev() {
        let value = items[last];
        items[last] = items[0];
        adjust(items, 0, last, value);
    }
}

impl<'a> Builder<'a> {
    fn add_child(&mut self, parent: usize, name: Cow<'a, str>) -> usize {
        self.nodes.push(Node { name, ..Default::default() });
        let index = self.nodes.len() - 1;
        self.nodes[parent].children.push(index);
        index
    }

    fn add_leaf(&mut self, metrics: &'a Metrics, parent: usize) -> usize {
        let leaf = self.add_child(parent, Cow::Borrowed(&metrics.name));
        self.nodes[leaf].xla = Some(symbol(metrics));
        self.nodes[leaf].fused = Some(metrics);
        leaf
    }

    fn add(&mut self, metrics: &'a Metrics) {
        update(metrics, &mut self.nodes[ROOT].metrics);
        let idle = metrics.category == IDLE;
        let mut program = None;
        if !idle && self.grouping != CATEGORY {
            let next = self.nodes.len();
            let node = *self.programs.entry(metrics.module).or_insert(next);
            if node == next {
                let name = self.names.get(&metrics.module).map_or_else(|| "main".to_string(), |name| format!("{name}({})", metrics.module));
                self.add_child(ROOT, Cow::Owned(name));
            }
            update(metrics, &mut self.nodes[node].metrics);
            program = Some(node);
        }
        if children_time(metrics) > 0 && !metrics.children.metrics.iter().any(|child| child.core_type == SPARSE_CORE) {
            return;
        }
        let grouping = if idle {
            vec![self.add_leaf(metrics, ROOT)]
        } else {
            let mut parent = program.unwrap_or(ROOT);
            if self.grouping == PROVENANCE && !metrics.provenance.is_empty() {
                let parts: Vec<&str> = metrics.provenance.split('/').filter(|part| !part.is_empty()).collect();
                for (position, part) in parts.iter().enumerate() {
                    parent = self.child(PROVENANCE_NODE, parent, if position + 1 == parts.len() { part.split(':').next().unwrap() } else { part });
                    update(metrics, &mut self.nodes[parent].metrics);
                }
            }
            let category = self.child(CATEGORY_NODE, parent, &metrics.category);
            let deduplicated = self.child(DEDUPLICATED_NODE, category, if metrics.deduplicated_name.is_empty() { &metrics.name } else { &metrics.deduplicated_name });
            vec![category, deduplicated, self.add_leaf(metrics, deduplicated)]
        };
        for node in grouping {
            combine(metrics, &mut self.nodes[node].metrics);
        }
    }

    fn child(&mut self, kind: u8, parent: usize, name: &'a str) -> usize {
        let next = self.nodes.len();
        let node = *self.children.entry((kind, parent, name)).or_insert(next);
        if node == next {
            self.add_child(parent, Cow::Borrowed(name));
        }
        node
    }

    /// Adds the fused children of a node only when the node stays after the prune.
    fn sort_and_prune(&mut self, k: usize, level: i32, root: usize) {
        let mut pending = vec![(root, level)];
        while let Some((node, level)) = pending.pop() {
            self.prune(k, level, node);
            pending.extend(self.nodes[node].children.iter().rev().map(|&child| (child, level - 1)));
        }
    }

    fn prune(&mut self, k: usize, level: i32, node: usize) {
        for child in self.nodes[node].fused.take().map_or(&[][..], |metrics| &metrics.children.metrics) {
            let index = self.add_child(node, Cow::Borrowed(&child.name));
            self.nodes[index].xla = Some(symbol(child));
            self.nodes[index].fused = Some(child);
            combine(child, &mut self.nodes[index].metrics);
        }
        self.nodes[node].num_children = self.nodes[node].children.len() as i32;
        let mut children = std::mem::take(&mut self.nodes[node].children);
        let kept = if level > 0 { children.len() } else { k.min(children.len()) };
        if children.len() > 1 {
            let nodes = &self.nodes;
            if self.nodes[node].xla.as_ref().is_some_and(|xla| is_fusion(xla.category)) {
                partial_sort(&mut children, kept, |a, b| nodes[a].metrics.model_flops_v2 > nodes[b].metrics.model_flops_v2);
            } else {
                partial_sort(&mut children, kept, |a, b| nodes[a].metrics.time_ps as f64 > nodes[b].metrics.time_ps as f64);
            }
            children.truncate(kept);
        }
        self.nodes[node].children = children;
    }

    fn finalize_deduplicated(&mut self, root: usize) {
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            let children = &self.nodes[node].children;
            if children.is_empty() || self.nodes[node].xla.is_some() || children.iter().any(|&child| self.nodes[child].xla.is_none()) {
                pending.extend(children.iter().rev());
            } else if let [only] = children[..] {
                self.nodes[node] = std::mem::take(&mut self.nodes[only]);
            } else {
                self.deduplicate(node);
            }
        }
    }

    fn deduplicate(&mut self, node: usize) {
        let top = &self.nodes[self.nodes[node].children[0]];
        let top_xla = top.xla.clone().unwrap();
        let name = Cow::Owned(format!("{} and its duplicate(s)", top.name));
        let mut xla = Xla { expression: top_xla.expression, source: top_xla.source, program_id: top_xla.program_id, category: top_xla.category, ..Default::default() };
        if !is_fusion(xla.category) {
            xla.provenance = top_xla.provenance;
            xla.layout = true;
        }
        self.nodes[node].name = name;
        self.nodes[node].xla = Some(xla);
    }

    fn write(&self, out: &mut String, root: usize, parallel: usize) {
        self.open(out, root);
        let mut stack = vec![(root, 0, parallel)];
        while let Some(&mut (index, ref mut next, parallel)) = stack.last_mut() {
            let children = &self.nodes[index].children;
            if *next == 0 && parallel > 0 && children.len() > 1 {
                let parts: Vec<String> = children
                    .par_iter()
                    .map(|&child| {
                        let mut part = String::new();
                        self.write(&mut part, child, parallel - 1);
                        part
                    })
                    .collect();
                out.reserve(parts.iter().map(|part| part.len() + 1).sum());
                for (position, part) in parts.iter().enumerate() {
                    out.push_str(if position > 0 { "," } else { "" });
                    out.push_str(part);
                }
                *next = children.len();
            } else if let Some(&child) = children.get(*next) {
                out.push_str(if *next > 0 { "," } else { "" });
                *next += 1;
                self.open(out, child);
                stack.push((child, 0, if children.len() == 1 { parallel } else { 0 }));
            } else {
                self.close(out, index);
                stack.pop();
            }
        }
    }

    fn open(&self, out: &mut String, index: usize) {
        let (node, metrics) = (&self.nodes[index], &self.nodes[index].metrics);
        let time_ns = pico_to_nano(metrics.time_ps);
        let mut rate = safe_divide(metrics.flops_v2, time_ns);
        if metrics.normalized_time_ps != 0 {
            rate *= safe_divide(metrics.time_ps as f64, metrics.normalized_time_ps as f64);
        }
        let uncapped = safe_divide(rate, self.peak_gigaflops);
        let fraction = safe_divide(metrics.time_ps as f64, self.total_time_ps as f64);
        let bytes = |on_chip: bool, operation: u8| metrics.memory.iter().filter(|entry| entry.0 == operation && (entry.1 == HBM) != on_chip).map(|entry| entry.2).sum::<u64>() as f64;
        let gibibytes = [
            giga_to_gibi(safe_divide(bytes(false, READ), time_ns)) + giga_to_gibi(safe_divide(bytes(false, WRITE), time_ns)),
            giga_to_gibi(safe_divide(bytes(true, READ), time_ns)),
            giga_to_gibi(safe_divide(bytes(true, WRITE), time_ns)),
        ];
        let peak = |index: usize| self.peak_bandwidths.get(index).copied().unwrap_or(0.0);
        let average = safe_divide(metrics.time_ps as f64, metrics.occurrences as f64);
        let bandwidths: [f64; 3] = std::array::from_fn(|index| safe_divide(gibibytes[index], peak(index)).min(1.0));
        let fields: [(&str, &[f64]); 10] = [
            ("flops", &[uncapped.min(1.0) * fraction]),
            ("bandwidthUtils", &bandwidths),
            ("rawTime", &[metrics.time_ps as f64]),
            ("rawFlops", &[metrics.model_flops_v2]),
            ("rawBytesAccessedArray", &gibibytes.map(|gibibytes| gibibytes * GIBI_IN_GIGA * time_ns)),
            ("occurrences", &[metrics.occurrences as u32 as f64]),
            ("avgTimePs", &[average]),
            ("bf16Flops", &[metrics.flops_v2]),
            ("uncappedFlops", &[uncapped * fraction]),
            ("normalizedTimePs", &[metrics.normalized_time_ps as f64]),
        ];
        out.push_str("{\"name\":");
        json_string(out, &node.name);
        out.push_str(",\"metrics\":{");
        for (position, (key, values)) in fields.iter().enumerate() {
            write!(out, "{}\"{key}\":", if position > 0 { "," } else { "" }).unwrap();
            if let [value] = values {
                proto_double(out, *value);
            } else {
                out.push('[');
                for (position, value) in values.iter().enumerate() {
                    out.push_str(if position > 0 { "," } else { "" });
                    proto_double(out, *value);
                }
                out.push(']');
            }
        }
        out.push_str("},\"children\":[");
    }

    fn close(&self, out: &mut String, index: usize) {
        let node = &self.nodes[index];
        out.push(']');
        if let Some(xla) = &node.xla {
            out.push_str(",\"xla\":{\"op\":\"\",\"expression\":");
            json_string(out, xla.expression);
            out.push_str(",\"provenance\":");
            json_string(out, xla.provenance);
            out.push_str(",\"category\":");
            json_string(out, xla.category);
            if xla.layout {
                out.push_str(",\"layout\":{\"dimensions\":[]}");
            }
            write!(out, ",\"computationPrimitiveSize\":0,\"fingerprint\":\"0\",\"programId\":\"{}\",\"sourceInfo\":{{\"fileName\":", xla.program_id).unwrap();
            let (file, line, stack) = xla.source.map_or(("", 0, ""), |source| (&*source.file, source.line, &*source.stack));
            json_string(out, file);
            write!(out, ",\"lineNumber\":{line},\"stackFrame\":").unwrap();
            json_string(out, stack);
            out.push_str("},\"xprofKernelMetadata\":");
            json_string(out, &kernel_metadata(xla.expression));
            out.push('}');
        }
        write!(out, ",\"numChildren\":{}}}", node.num_children).unwrap();
    }
}

pub fn proto_double(out: &mut String, value: f64) {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        if value == 0.0 && value.is_sign_negative() {
            out.push('-');
        }
        out.push_str(itoa::Buffer::new().format(value as i64));
    } else if value.is_nan() {
        out.push_str("\"NaN\"");
    } else if value.is_infinite() {
        out.push_str(if value > 0.0 { "\"Infinity\"" } else { "\"-Infinity\"" });
    } else {
        crate::hlo::round_trip(out, value);
    }
}

fn tree(stats: &OpStats, grouping: usize, exclude_idle: bool) -> String {
    let db = &stats.db;
    let mut out = String::new();
    if db.metrics.is_empty() {
        out.push_str("{\"name\":\"\",\"children\":[],\"numChildren\":0}");
        return out;
    }
    let mut builder = Builder {
        grouping,
        nodes: Vec::with_capacity(3 * db.metrics.len()),
        programs: FxHashMap::default(),
        children: FxHashMap::with_capacity_and_hasher(2 * db.metrics.len(), Default::default()),
        names: &stats.programs,
        peak_gigaflops: stats.perf.peak_tera_flops * 1e3,
        peak_bandwidths: stats.perf.bandwidths.iter().map(|&bandwidth| giga_to_gibi(bandwidth)).collect(),
        total_time_ps: if exclude_idle { db.total_op_time_ps } else { db.total_time_ps },
    };
    builder.nodes.push(Node { name: Cow::Borrowed(GROUPINGS[grouping].0), ..Default::default() });
    for metrics in db.metrics.iter().filter(|metrics| !(metrics.name.starts_with("region") || exclude_idle && metrics.category == IDLE)) {
        builder.add(metrics);
    }
    builder.nodes[ROOT].metrics.time_ps = builder.total_time_ps;
    builder.sort_and_prune(CHILDREN_PER_NODE, GROUPINGS[grouping].2, ROOT);
    builder.sort_and_prune(CHILDREN_PER_NODE, GROUPINGS[grouping].2, ROOT);
    builder.finalize_deduplicated(ROOT);
    builder.write(&mut out, ROOT, PARALLEL_LEVELS);
    out
}

pub fn json(stats: &OpStats, group_by: Option<&str>) -> String {
    json_trees(stats, group_by, true)
}

/// Without `with_busy`, the output has no tree for the busy time. The command line tools use no such tree.
pub fn json_trees(stats: &OpStats, group_by: Option<&str>, with_busy: bool) -> String {
    let grouping = match group_by {
        Some("category") => CATEGORY,
        Some("provenance") => PROVENANCE,
        _ => PROGRAM,
    };
    let device = HARDWARE[hardware(&stats.extra.device_type) as usize];
    let key = GROUPINGS[grouping].1;
    let (all, busy) = rayon::join(|| tree(stats, grouping, false), || if with_busy { tree(stats, grouping, true) } else { String::new() });
    let mut out = String::with_capacity(all.len() + busy.len() + 256);
    if grouping == PROVENANCE {
        write!(out, "{{\"deviceType\":\"{device}\",\"{key}\":").unwrap();
        out.push_str(&all);
    } else {
        write!(out, "{{\"{key}\":").unwrap();
        out.push_str(&all);
        write!(out, ",\"deviceType\":\"{device}\"").unwrap();
    }
    if with_busy {
        write!(out, ",\"{key}ExcludeIdle\":").unwrap();
        out.push_str(&busy);
    }
    out.push_str(",\"aggDvfsTimeScaleMultiplier\":");
    proto_double(&mut out, safe_divide(stats.db.normalized_total_op_time_ps as f64, stats.db.total_op_time_ps as f64));
    out.push('}');
    out
}

#[cfg(test)]
#[path = "tests/inline/op_profile.rs"]
mod tests;
