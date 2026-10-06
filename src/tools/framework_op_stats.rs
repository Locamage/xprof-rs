use crate::tools::hlo_stats::roofline;
use crate::tools::opstats::{Db, IDLE, Metrics, OpStats, add, pico_to_micro, safe_divide};
use crate::tools::table::{Cell, Table};
use crate::xplane::derive::is_derived;
use crate::xplane::{Line, Plane, event_stats};
use arcstr::ArcStr;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::borrow::Cow;

const MAX_OPS: usize = 500;
const HOST_PLANE: &str = "/host:CPU";
const UNKNOWN: &str = "Unknown";
const MEMCPY: [&str; 4] = ["MemcpyHToD", "MemcpyDToH", "MemcpyDToD", "MemcpyHToH"];
const COLUMNS: [(&str, &str, &str); 19] = [
    ("rank", "number", "Rank"),
    ("host_or_device", "string", "Host/device"),
    ("type", "string", "Operation Type"),
    ("operation", "string", "Operation Name"),
    ("occurrences", "number", "#Occurrences"),
    ("total_time", "number", "Total time (us)"),
    ("avg_time", "number", "Avg. time (us)"),
    ("total_self_time", "number", "Total self-time (us)"),
    ("avg_self_time", "number", "Avg. self-time (us)"),
    ("device_total_self_time_percent", "number", "Total self-time on Device (%)"),
    ("device_cumulative_total_self_time_percent", "number", "Cumulative total-self time on Device (%)"),
    ("host_total_self_time_percent", "number", "Total self-time on Host (%)"),
    ("Host_cumulative_total_self_time_percent", "number", "Cumulative total-self time on Host (%)"),
    ("measured_flop_rate", "number", "Normalized FLOP Rate (FLOPs/s)"),
    ("model_flop_rate", "number", "Model FLOP Rate (GFLOP/s)"),
    ("measured_memory_bw", "number", "Measured Memory BW (GBytes/Sec)"),
    ("operational_intensity", "number", "Operational Intensity (FLOPs/Byte)"),
    ("bound_by", "string", "Bound by"),
    ("eager", "string", "Execution mode"),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TfOp {
    pub known: bool,
    pub name: String,
    pub kind: String,
    pub id: i64,
}

type Op = (ArcStr, ArcStr, i64);
type Activity = (u64, u32, Option<(Op, bool)>);

pub fn is_tf_op_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes.next().is_some_and(|first| first.is_ascii_alphanumeric() || first == b'.') && bytes.all(|byte| byte.is_ascii_alphanumeric() || b"_./>-".contains(&byte))
}

pub fn is_tf_op_type(kind: &str) -> bool {
    let mut bytes = kind.bytes();
    bytes.next().is_some_and(|first| first.is_ascii_uppercase() || first == b'_') && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub fn is_jax_op_type(kind: &str) -> bool {
    let bytes = kind.as_bytes();
    if !bytes.first().is_some_and(|first| first.is_ascii_lowercase() || *first == b'_') {
        return false;
    }
    let end = bytes.iter().position(|byte| !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_')).unwrap_or(bytes.len());
    let rest = &bytes[end..];
    rest.is_empty() || (rest.len() >= 2 && rest[0] == b'[' && rest[rest.len() - 1] == b']' && !rest.contains(&b'\n'))
}

fn derive_op_type(full: &str) -> &str {
    let name = full.rsplit('/').next().unwrap();
    let suffix = name.rsplit('_').next().unwrap();
    match (suffix.trim_ascii().parse::<i64>(), name.len() > suffix.len()) {
        (Ok(_), true) => &name[..name.len() - suffix.len() - 1],
        _ => name,
    }
}

pub fn jax_op_type<'a>(name: &'a str, kind: &'a str) -> Option<&'a str> {
    let derived = if kind.is_empty() { derive_op_type(name) } else { kind };
    if is_jax_op_type(derived) { Some(&derived[..derived.find('[').unwrap_or(derived.len())]) } else { kind.is_empty().then_some(derived) }
}

pub fn parse_tf_op(full: &str) -> TfOp {
    let op = |known: bool, name: &str, kind: &str| TfOp { known, name: name.to_string(), kind: kind.to_string(), id: 0 };
    let Some((name, kind)) = full.split_once(':') else {
        return match MEMCPY.iter().find(|memcpy| full.get(..memcpy.len()).is_some_and(|prefix| prefix.eq_ignore_ascii_case(memcpy))) {
            Some(memcpy) => op(true, full, memcpy),
            None => op(false, full, ""),
        };
    };
    if name == "Iterator" {
        op(true, full, "Dataset")
    } else if is_tf_op_name(name) && is_tf_op_type(kind) {
        op(true, name, kind)
    } else {
        jax_op_type(name, kind).map_or_else(|| op(false, full, ""), |kind| op(true, name, kind))
    }
}

fn shared(op: TfOp) -> Op {
    (op.name.into(), op.kind.into(), op.id)
}

fn host_line(plane: &Plane, map: &[u8], line: &Line, ops: &FxHashMap<i64, Op>, ids: &[Option<usize>; 6]) -> (Db, (u64, u64)) {
    let mut activities: Vec<Activity> = Vec::new();
    if !is_derived(line.id) {
        let mut parsed: FxHashMap<Cow<str>, Op> = FxHashMap::default();
        for event in &line.events {
            let [stage, other_stage, eager, tf_op] = event_stats(event.stats_raw(map), [ids[2], ids[4], ids[0], ids[1]]);
            let key = stage.or(other_stage).map_or(event.meta as i64, |value| value.int().unwrap_or(0));
            let (begin, end) = (event.ts, event.ts.wrapping_add(event.dur));
            if let Some(op) = ops.get(&key) {
                let (next, eager) = (activities.len() as u32 / 2 + 1, eager.is_some_and(|value| value.int().unwrap_or(0) != 0));
                activities.extend([(begin, next, None), (end, next, Some((op.clone(), eager)))]);
            }
            let Some(full) = tf_op.map(|value| plane.text_cow(&value)).filter(|full| !full.is_empty()) else { continue };
            let op = match parsed.get(&*full) {
                Some(op) => op.clone(),
                None => parsed.entry(full).or_insert_with_key(|full| shared(parse_tf_op(full))).clone(),
            };
            let next = activities.len() as u32 / 2 + 1;
            activities.extend([(begin, next, None), (end, next, Some((op, false)))]);
        }
    }
    let (mut db, mut index) = (Db::default(), FxHashMap::default());
    activities.sort_by_key(|activity| activity.0);
    let mut stack: Vec<(u32, u64, u64)> = Vec::new();
    let (mut enqueue, mut last_enqueue) = ((0u64, 0u64), None::<(u64, u64)>);
    for (ts, id, end) in &activities {
        let Some(((name, kind, module), eager)) = end else {
            stack.push((*id, *ts, 0));
            continue;
        };
        let Some(position) = stack.iter().rposition(|entry| entry.0 == *id) else {
            stack.clear();
            continue;
        };
        let (_, start, children) = stack[position];
        stack.truncate(position);
        let duration = ts.saturating_sub(start);
        let self_time = duration.wrapping_sub(children);
        let metrics = db.entry(&mut index, *module as u64, name);
        if metrics.category.is_empty() {
            metrics.category = kind.clone();
        }
        metrics.num_cores = 1;
        metrics.is_eager |= eager;
        metrics.occurrences += 1;
        metrics.time_ps += duration;
        metrics.self_time_ps += self_time;
        db.total_op_time_ps += self_time;
        if let Some(top) = stack.last_mut() {
            top.2 += duration;
        }
        if kind.starts_with("InfeedEnqueue") {
            if let Some((previous_start, previous_duration)) = last_enqueue {
                enqueue = (enqueue.0.wrapping_add(previous_duration), enqueue.1.wrapping_add(start.wrapping_sub(previous_start)));
            }
            last_enqueue = Some((start, duration));
        }
    }
    if let (Some(first), Some(last)) = (activities.first(), activities.last()) {
        db.total_time_ps = db.total_op_time_ps.max(last.0 - first.0);
    }
    let idle = db.total_time_ps.wrapping_sub(db.total_op_time_ps);
    db.metrics.push(Metrics { name: IDLE.into(), category: IDLE.into(), time_ps: idle, self_time_ps: idle, ..Default::default() });
    (db, enqueue)
}

pub fn host_db(planes: &[Plane], map: &[u8]) -> (Db, (u64, u64)) {
    let Some(plane) = planes.iter().find(|plane| plane.name == HOST_PLANE) else { return Default::default() };
    let ids = ["is_eager", "tf_op", "_ipl_stage_id", "_ipl_stage_cat", "input_pipeline_stage_id", "input_pipeline_stage_category"].map(|wanted| plane.id(wanted));
    let (mut ops, mut parsed) = (FxHashMap::default(), vec![false; plane.meta.len()]);
    for event in plane.lines.iter().flat_map(|line| &line.events) {
        let meta = &plane.meta[event.meta as usize];
        if meta.name.is_empty() {
            continue;
        }
        let [stage, other_stage, category, other_category] = event_stats(event.stats_raw(map), [ids[2], ids[4], ids[3], ids[5]]);
        if let Some(stage) = stage.or(other_stage) {
            if let Some(category) = category.or(other_category) {
                let (id, kind) = (stage.int().unwrap_or(0), plane.text(&category));
                let op = if kind.is_empty() { TfOp { id, ..parse_tf_op(&meta.name) } } else { TfOp { known: true, name: meta.name.to_string(), kind, id } };
                ops.entry(id).or_insert_with(|| shared(op));
            }
            continue;
        }
        if std::mem::replace(&mut parsed[event.meta as usize], true) {
            continue;
        }
        let op = parse_tf_op(&meta.name);
        if op.known {
            ops.entry(event.meta as i64).or_insert_with(|| shared(op));
        }
    }
    let dbs: Vec<(Db, (u64, u64))> = plane.lines.par_iter().map(|line| host_line(plane, map, line, &ops, &ids)).collect();
    let enqueue = dbs.iter().fold((0, 0), |sum, (_, enqueue)| (sum.0 + enqueue.0, sum.1 + enqueue.1));
    (Db::combined(dbs.iter().map(|(db, _)| db), false), enqueue)
}

pub fn device_tf_db(device: &Db) -> Db {
    let (mut db, mut index) = (Db::default(), FxHashMap::default());
    for metrics in &device.metrics {
        let op = if metrics.category == IDLE {
            TfOp { known: true, name: IDLE.into(), kind: IDLE.into(), id: 0 }
        } else if metrics.provenance.is_empty() {
            TfOp { known: false, name: metrics.name.to_string(), kind: String::new(), id: 0 }
        } else {
            parse_tf_op(&metrics.provenance)
        };
        let tf = db.entry(&mut index, 0, &op.name.as_str().into());
        if tf.category.is_empty() {
            tf.category = if op.kind.is_empty() { UNKNOWN } else { &op.kind }.into();
        }
        tf.is_eager = metrics.is_eager;
        tf.occurrences = tf.occurrences.max(metrics.occurrences);
        add!(tf, metrics, time_ps, self_time_ps, flops_v2, model_flops_v2, bytes_accessed);
        tf.core_type = metrics.core_type;
    }
    db.total_op_time_ps = device.total_op_time_ps;
    db.total_time_ps = device.total_time_ps;
    db
}

fn training(name: &str, kind: &str) -> bool {
    let jax = |name: &str| name.split('/').any(|scope| scope.starts_with("transpose("));
    if is_tf_op_type(kind) && is_tf_op_name(name) {
        let mut scopes: Vec<&str> = name.split('/').collect();
        scopes.pop();
        scopes.iter().any(|scope| scope.strip_prefix("gradient").is_some_and(|rest| rest == "_tape" || rest.starts_with('s')))
    } else if !name.is_empty() && is_jax_op_type(kind) && name.rsplit('/').next().unwrap().contains(kind) {
        jax(name)
    } else {
        kind.is_empty() && jax(name)
    }
}

fn table(stats: &OpStats, host: &Db, device: &Db, exclude_idle: bool) -> String {
    let gpu = stats.extra.device_type.contains("GPU");
    let mut table = Table::new(&COLUMNS[..COLUMNS.len() - 2]);
    if gpu {
        table.column("gpu_tensorcore_utilization", "number", "GPU TensorCore utilization");
    }
    COLUMNS[COLUMNS.len() - 2..].iter().for_each(|(id, kind, label)| table.column(id, kind, label));
    let kernels = crate::xplane::gpu::by_op_name(&stats.kernels);
    let (mut rank, mut device_cumulative, mut host_cumulative, mut train) = (0u64, 0.0, 0.0, false);
    for (on_device, db) in [(true, device), (false, host)] {
        let total_us = pico_to_micro(if exclude_idle { db.total_op_time_ps } else { db.total_time_ps });
        for metrics in db.sorted().into_iter().take(MAX_OPS) {
            if exclude_idle && metrics.category == IDLE {
                continue;
            }
            let roof = roofline(metrics, stats, false);
            let total_time = pico_to_micro(metrics.time_ps);
            let self_time = pico_to_micro(metrics.self_time_ps);
            let fraction = safe_divide(self_time, total_us);
            let (device_fraction, host_fraction) = if on_device { (fraction, 0.0) } else { (0.0, fraction) };
            (device_cumulative, host_cumulative) = if on_device { (device_cumulative + fraction, 0.0) } else { (0.0, host_cumulative + fraction) };
            rank += 1;
            let text = |text: &str| Cell::Text(text.to_string());
            let mut row = vec![Cell::Number(rank as f64), text(if on_device { "Device" } else { "Host" }), text(&metrics.category), text(&metrics.name)];
            row.extend(
                [
                    metrics.occurrences as f64,
                    total_time,
                    safe_divide(total_time, metrics.occurrences as f64),
                    self_time,
                    safe_divide(self_time, metrics.occurrences as f64),
                    device_fraction,
                    device_cumulative,
                    host_fraction,
                    host_cumulative,
                    roof.measured_flop_rate,
                    roof.model_flop_rate,
                    roof.measured_memory_bw,
                    roof.operational_intensity,
                ]
                .map(Cell::Number),
            );
            if gpu {
                let utilization = kernels.get(metrics.name.as_str()).filter(|_| on_device).map_or(0.0, |&(_, total, tensor_core)| safe_divide(tensor_core as f64, total as f64));
                row.push(Cell::Number(utilization));
            }
            row.extend([text(roof.bound_by), text(if metrics.is_eager { "Eager" } else { "Function" })]);
            table.rows.push(row);
            train |= training(&metrics.name, &metrics.category);
        }
    }
    table.prop("architecture_type", "Unknown");
    table.prop("task_type", if train { "Training" } else { "Inference" });
    table.json()
}

pub fn json(stats: &OpStats) -> String {
    let device = device_tf_db(&stats.db);
    format!("[{},{}]", table(stats, &stats.host, &device, false), table(stats, &stats.host, &device, true))
}

#[cfg(test)]
#[path = "../tests/inline/tools/framework_op_stats.rs"]
mod tests;
