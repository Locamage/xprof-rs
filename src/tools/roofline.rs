use crate::hlo::{OPCODES, general};
use crate::tools::hlo_stats::{roofline, source_text};
use crate::tools::input_pipeline_analyzer::{NO_STEP_MARKER, diagnostics_table};
use crate::tools::opstats::{CMEM, Db, GIBI_IN_GIGA, Metrics, OpStats, READ, SPARSE_CORE, VMEM, WRITE, combine_memory, giga_to_gibi, pico_to_micro, safe_divide};
use crate::tools::table::{Cell, Table};
use crate::xplane::steps::GPU;
use rustc_hash::FxHashSet;
use std::sync::LazyLock;

const MAX_RECORDS: usize = 1000;
const PROGRAM: &str = "Program";
const MEMORIES: [&str; 5] = ["hbm", "cmem_read", "cmem_write", "vmem_read", "vmem_write"];
static OPCODE_SET: LazyLock<FxHashSet<&str>> = LazyLock::new(|| OPCODES.split(' ').collect());
const COLUMNS: [(&str, &str, &str); 37] = [
    ("step", "string", "Step"),
    ("rank", "number", "Rank"),
    ("category", "string", "Category"),
    ("operation", "string", "Operation"),
    ("occurrences", "number", "# Occurrences"),
    ("total_time", "number", "Total Time (us)"),
    ("avg_time", "number", "Avg. time (us)"),
    ("total_self_time", "number", "Total self time (us)"),
    ("avg_self_time", "number", "Avg. self time (us)"),
    ("total_self_time_percent", "number", "Total self time (%)"),
    ("cumulative_total_self_time_percent", "number", "Cumulative total self time (%)"),
    ("dma_stall_percent", "number", "%time stalled by DMA"),
    ("measured_flop_rate", "number", "Normalized FLOP Rate (GFLOP/s)"),
    ("model_flop_rate", "number", "Model FLOP Rate (GFLOP/s)"),
    ("measured_memory_bw", "number", "Memory BW (GiB/s)"),
    ("hbm_bw", "number", "HBM BW (GiB/s)"),
    ("cmem_read_bw", "number", "CMEM Read BW (GiB/s)"),
    ("cmem_write_bw", "number", "CMEM Write BW (GiB/s)"),
    ("vmem_read_bw", "number", "VMEM Read BW (GiB/s)"),
    ("vmem_write_bw", "number", "VMEM Write BW (GiB/s)"),
    ("operational_intensity", "number", "Operational Intensity (FLOP/Byte)"),
    ("hbm_operational_intensity", "number", "HBM Operational Intensity (FLOP/Byte)"),
    ("cmem_read_operational_intensity", "number", "CMEM Read Operational Intensity (FLOP/Byte)"),
    ("cmem_write_operational_intensity", "number", "CMEM Write Operational Intensity (FLOP/Byte)"),
    ("vmem_read_operational_intensity", "number", "VMEM Read Operational Intensity (FLOP/Byte)"),
    ("vmem_write_operational_intensity", "number", "VMEM Write Operational Intensity (FLOP/Byte)"),
    ("bottleneck_operational_intensity", "number", "Bottleneck Operational Intensity (FLOP/Byte)"),
    ("bound_by", "string", "Bound by"),
    ("total_time_per_core", "number", "Total Time per core (us)"),
    ("total_time_in_percentage", "number", "Total Time (%)"),
    ("optimal_flop_rate", "number", "Optimal FLOP Rate (GFLOP/s)"),
    ("roofline_efficiency", "number", "Roofline efficiency (%)"),
    ("compute_efficiency", "number", "FLOP Rate / Peak (%)"),
    ("max_mem_bw_utilization", "number", "Max memory BW utilization (among supported memories) (%)"),
    ("include_infeed_outfeed", "boolean", "Include Infeed/Outfeed"),
    ("hlo_module_id", "string", "Program ID"),
    ("source_info", "string", "Source Info"),
];

const GPU_COLUMNS: [usize; 30] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36];
const GPU_LABELS: [(usize, &str); 2] = [(19, "SHM/L1 BW (GiB/s)"), (25, "SHM/L1 Operational Intensity (FLOP/Byte)")];

struct Placement<'a> {
    step: String,
    steps: Option<usize>,
    total_time_ps: u64,
    rank: u64,
    fraction: f64,
    cumulative: f64,
    include: bool,
    category: &'a str,
    gpu: bool,
}

pub fn category(metrics: &Metrics) -> &str {
    if !metrics.category.is_empty() && metrics.category != "unknown" && metrics.category != "Unknown" {
        &metrics.category
    } else if OPCODE_SET.contains(&*metrics.name) {
        &metrics.name
    } else if metrics.category.is_empty() {
        "unknown"
    } else {
        &metrics.category
    }
}

fn is_infeed_or_outfeed(category: &str) -> bool {
    category.contains("infeed") || category.contains("outfeed")
}

fn row(table: &mut Table, stats: &OpStats, peaks: &[f64; 6], metrics: &Metrics, place: &Placement) {
    let roof = roofline(metrics, stats, false);
    let occurrences = metrics.occurrences as f64;
    let (mut total_time, mut self_time) = (pico_to_micro(metrics.time_ps), pico_to_micro(metrics.self_time_ps));
    let (average, self_average) = (safe_divide(total_time, occurrences), safe_divide(self_time, occurrences));
    if let Some(steps) = place.steps {
        total_time = safe_divide(total_time, steps as f64);
        self_time = safe_divide(self_time, steps as f64);
    }
    let flops_utilization = safe_divide(roof.measured_flop_rate, peaks[0]);
    let bandwidths = [roof.hbm_bw, roof.cmem_read_bw, roof.cmem_write_bw, roof.vmem_read_bw, roof.vmem_write_bw];
    let memory_utilization = bandwidths.iter().zip(&peaks[1..]).map(|(bandwidth, peak)| safe_divide(*bandwidth, *peak)).fold(f64::NEG_INFINITY, f64::max);
    let efficiency = memory_utilization.max(flops_utilization);
    let intensity = |key: (u64, u8)| safe_divide(metrics.flops_v2, metrics.memory.iter().filter(|entry| (entry.1, entry.0) == key).map(|entry| entry.2).sum::<u64>() as f64);
    let text = |value: &str| Cell::Text(value.to_string());
    let mut cells = vec![text(&place.step), Cell::Number(place.rank as f64), text(place.category), text(&metrics.name)];
    cells.extend(
        [
            occurrences,
            total_time,
            average,
            self_time,
            self_average,
            place.fraction,
            place.cumulative,
            safe_divide(metrics.dma_stall_ps as f64, metrics.time_ps as f64),
            roof.measured_flop_rate,
            roof.model_flop_rate,
            roof.measured_memory_bw,
            roof.hbm_bw,
            roof.cmem_read_bw,
            roof.cmem_write_bw,
            roof.vmem_read_bw,
            roof.vmem_write_bw,
            roof.operational_intensity,
            roof.hbm_operational_intensity,
            intensity((CMEM, READ)),
            intensity((CMEM, WRITE)),
            intensity((VMEM, READ)),
            intensity((VMEM, WRITE)),
            roof.bottleneck_operational_intensity,
        ]
        .map(Cell::Number),
    );
    cells.push(text(roof.bound_by));
    cells.extend(
        [
            safe_divide(total_time, stats.extra.core_count as f64),
            safe_divide(metrics.time_ps as f64, place.total_time_ps as f64),
            safe_divide(roof.measured_flop_rate, efficiency),
            efficiency,
            flops_utilization,
            memory_utilization,
        ]
        .map(Cell::Number),
    );
    cells.extend([Cell::Boolean(place.include), text(&metrics.module.to_string()), text(&source_text(metrics))]);
    if place.gpu {
        let mut cells: Vec<Option<Cell>> = cells.into_iter().map(Some).collect();
        table.rows.push(GPU_COLUMNS.iter().map(|&index| cells[index].take().unwrap()).collect());
    } else {
        table.rows.push(cells);
    }
}

pub fn accumulate(sum: &mut Metrics, part: &Metrics) {
    sum.flops_v2 += part.flops_v2;
    sum.model_flops_v2 += part.model_flops_v2;
    sum.bytes_accessed += part.bytes_accessed;
    combine_memory(&part.memory, &mut sum.memory);
}

pub fn program(db: &Db) -> ([Metrics; 2], u64) {
    let mut program = Default::default();
    db.metrics
        .iter()
        .for_each(|metrics| add_scaled(&mut program, metrics, (metrics.core_type, metrics.time_ps), (metrics.flops_v2, metrics.model_flops_v2, metrics.bytes_accessed), &metrics.memory, 1));
    program
}

/// Adds `part`, an operation of `kind`, with the given flops, model flops and bytes, and with `memory` times `scale`.
pub fn add_scaled((sums, infeed_outfeed): &mut ([Metrics; 2], u64), kind: &Metrics, (core_type, time_ps): (u8, u64), (flops, model, bytes): (f64, f64, u64), memory: &[(u8, u64, u64)], scale: u64) {
    let category = category(kind);
    if matches!(category, "call" | "conditional" | "while" | "megacore fusion") || core_type == SPARSE_CORE {
        return;
    }
    let infeed = is_infeed_or_outfeed(category);
    for sum in &mut sums[..if infeed { 1 } else { 2 }] {
        sum.flops_v2 += flops;
        sum.model_flops_v2 += model;
        sum.bytes_accessed += bytes;
        combine_memory(memory.iter().map(|&(operation, space, amount)| (operation, space, amount.saturating_mul(scale))), &mut sum.memory);
    }
    if infeed {
        *infeed_outfeed += time_ps;
    }
}

fn records(table: &mut Table, stats: &OpStats, peaks: &[f64; 6], include: bool, total_only: bool) {
    let side = usize::from(!include);
    let gpu = stats.extra.hardware == GPU;
    let program_row = |table: &mut Table, sums: &Metrics, step: String, steps: Option<usize>| {
        let metrics = Metrics { name: PROGRAM.into(), category: PROGRAM.into(), occurrences: 1, ..sums.clone() };
        row(table, stats, peaks, &metrics, &Placement { step, steps, total_time_ps: metrics.time_ps, rank: 0, fraction: 0.0, cumulative: 0.0, include, category: PROGRAM, gpu });
    };
    let db = &stats.db;
    let (mut sums, program_infeed_outfeed) = program(db);
    sums[0].time_ps = db.total_time_ps;
    sums[1].time_ps = db.total_time_ps.wrapping_sub(program_infeed_outfeed);
    program_row(table, &sums[side], "Total".into(), None);
    if total_only {
        return;
    }
    let infeed_outfeed: u64 = if include { 0 } else { db.metrics.iter().filter(|metrics| is_infeed_or_outfeed(&metrics.category)).map(|metrics| metrics.time_ps).sum() };
    let total_time_ps = db.total_time_ps.wrapping_sub(infeed_outfeed);
    let total_us = pico_to_micro(total_time_ps);
    let (mut rank, mut cumulative) = (0u64, 0.0);
    for metrics in db.sorted().into_iter().take(MAX_RECORDS).filter(|metrics| metrics.occurrences != 0) {
        let category = category(metrics);
        if !include && is_infeed_or_outfeed(category) {
            continue;
        }
        let fraction = safe_divide(pico_to_micro(metrics.self_time_ps), total_us);
        rank += 1;
        cumulative += fraction;
        row(table, stats, peaks, metrics, &Placement { step: "Total".into(), steps: None, total_time_ps, rank, fraction, cumulative, include, category, gpu });
    }
    let extra = &stats.extra;
    if gpu {
        for record in &extra.steps {
            program_row(table, &Metrics::default(), record.num.to_string(), None);
        }
        return;
    }
    program_row(table, &extra.program_total[side], "Average".into(), Some(extra.program_steps.len()));
    for (number, sums) in &extra.program_steps {
        program_row(table, &sums[side], number.to_string(), None);
    }
}

pub fn json(stats: &OpStats) -> String {
    json_rows(stats, false)
}

/// With `total_only`, the table has the first row only. The overview command needs no other row.
pub fn json_rows(stats: &OpStats, total_only: bool) -> String {
    let extra = &*stats.extra;
    let bandwidth = |index: usize| giga_to_gibi(stats.perf.bandwidths.get(index).copied().unwrap_or(0.0));
    if extra.hardware == GPU {
        let mut table = Table::default();
        for &index in &GPU_COLUMNS {
            let (id, kind, label) = COLUMNS[index];
            table.column(id, kind, GPU_LABELS.iter().find(|(position, _)| *position == index).map_or(label, |(_, label)| label));
        }
        let peaks = [stats.perf.peak_tera_flops * 1e3, bandwidth(0), 0.0, 0.0, bandwidth(1), bandwidth(2)];
        table.prop("device_type", extra.device_type.as_str());
        table.prop("peak_flop_rate", general(peaks[0], 6));
        table.prop("peak_hbm_bw", general(peaks[1], 6));
        table.prop("peak_vmem_write_bw", general(peaks[5], 6));
        table.prop("hbm_ridge_point", general(safe_divide(peaks[0], peaks[1] * GIBI_IN_GIGA), 6));
        table.prop("vmem_write_ridge_point", general(safe_divide(peaks[0], peaks[5] * GIBI_IN_GIGA), 6));
        records(&mut table, stats, &peaks, true, total_only);
        records(&mut table, stats, &peaks, false, total_only);
        let warnings = if extra.steps.is_empty() { vec![NO_STEP_MARKER.to_string()] } else { Vec::new() };
        return format!("[{},{}]", table.json(), diagnostics_table(&warnings, &[]).json());
    }
    let mut table = Table::new(&COLUMNS);
    let cmem = stats.perf.cmem;
    let vmem = !cmem && extra.merged_vmem;
    let pick = |enabled: bool, index: usize| if enabled { bandwidth(index) } else { 0.0 };
    let peaks = [stats.perf.peak_tera_flops * 1e3, bandwidth(0), pick(cmem, 3), pick(cmem, 4), pick(vmem, 5), pick(vmem, 6)];
    let tpu_only = |value: String| if stats.tpu { value } else { "0".into() };
    table.prop("megacore", tpu_only((extra.megacore as u8).to_string()));
    table.prop("has_cmem", tpu_only((cmem as u8).to_string()));
    table.prop("has_merged_vmem", tpu_only((extra.merged_vmem as u8).to_string()));
    table.prop("peak_flop_rate", tpu_only(general(peaks[0], 6)));
    table.prop("time_scale_multiplier", tpu_only(general(safe_divide(stats.db.normalized_total_op_time_ps as f64, stats.db.total_op_time_ps as f64), 6)));
    for (name, peak) in MEMORIES.iter().zip(&peaks[1..]) {
        table.prop(&format!("peak_{name}_bw"), tpu_only(general(*peak, 6)));
        table.prop(&format!("{name}_ridge_point"), tpu_only(general(safe_divide(peaks[0], peak * GIBI_IN_GIGA), 6)));
    }
    table.prop("device_type", if stats.tpu { extra.device_type.as_str() } else { "" });
    let mut warnings = Vec::new();
    if stats.tpu {
        records(&mut table, stats, &peaks, true, total_only);
        if !total_only {
            records(&mut table, stats, &peaks, false, false);
        }
        if extra.program_steps.is_empty() {
            warnings.push(NO_STEP_MARKER.to_string());
        }
    }
    format!("[{},{}]", table.json(), diagnostics_table(&warnings, &[]).json())
}
