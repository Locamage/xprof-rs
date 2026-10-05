use crate::tools::opstats::{CMEM, HBM, Metrics, OpStats, READ, SPARSE_CORE, TENSOR_CORE, VMEM, WRITE, giga_to_gibi, pico_to_micro, pico_to_nano, safe_divide};
use crate::tools::table::{Cell, Table};
use crate::xplane::steps::GPU;
use rayon::prelude::*;

const COLUMNS: [(&str, &str, &str); 29] = [
    ("rank", "number", "Rank"),
    ("program_id", "string", "Program id"),
    ("category", "string", "HLO op category"),
    ("hlo_op_name", "string", "HLO op name"),
    ("hlo_op_expression", "string", "HLO op text"),
    ("tf_op_name", "string", "Framework op name"),
    ("occurrences", "number", "#Occurrences"),
    ("total_time", "number", "Total time (us)"),
    ("avg_time", "number", "Avg. time (us)"),
    ("total_self_time", "number", "Total self time (us)"),
    ("avg_self_time", "number", "Avg. self time (us)"),
    ("total_self_time_percent", "number", "Total self time (%)"),
    ("cumulative_total_self_time_percent", "number", "Cumulative total self time (%)"),
    ("dma_stall_percent", "number", "%time stalled by DMA"),
    ("model_flop_rate", "number", "Model GFLOP/s"),
    ("normalized_flop_rate", "number", "Normalized GFLOP/s"),
    ("measured_memory_bw", "number", "Measured memory BW (GiB/s)"),
    ("hbm_bw", "number", "HBM BW (GiB/s)"),
    ("cmem_read_bw", "number", "CMEM Read BW (GiB/s)"),
    ("cmem_write_bw", "number", "CMEM Write BW (GiB/s)"),
    ("operational_intensity", "number", "Operational intensity (FLOPS/Byte)"),
    ("bound_by", "string", "Bound by"),
    ("hlo_rematerialization", "string", "Rematerialization"),
    ("outside_compilation", "string", "Outside Compilation"),
    ("autotuned", "string", "Autotuned"),
    ("source_info", "string", "Source Info"),
    ("core_type", "string", "TPU core type"),
    ("parent_op_name", "string", "Parent op name"),
    ("vdd_energy", "number", "VDD Energy (J)"),
];

const MEMORIES: [(&str, usize); 5] = [("HBM", 0), ("CMEM Read", 3), ("CMEM Write", 4), ("VMEM Read", 5), ("VMEM Write", 6)];

#[derive(Default)]
pub struct Roofline {
    pub measured_flop_rate: f64,
    pub model_flop_rate: f64,
    pub measured_memory_bw: f64,
    pub operational_intensity: f64,
    pub hbm_bw: f64,
    pub cmem_read_bw: f64,
    pub cmem_write_bw: f64,
    pub vmem_read_bw: f64,
    pub vmem_write_bw: f64,
    pub hbm_operational_intensity: f64,
    pub bound_by: &'static str,
    pub bottleneck_operational_intensity: f64,
}

pub fn roofline(metrics: &Metrics, stats: &OpStats, scale_time: bool) -> Roofline {
    let (perf, time_ns) = (&stats.perf, pico_to_nano(metrics.time_ps));
    let device_ns = pico_to_nano(if scale_time { metrics.normalized_time_ps } else { metrics.time_ps });
    let mut bytes = [0u64; MEMORIES.len()];
    for &(operation, space, amount) in &metrics.memory {
        let slot = match (space, operation) {
            (HBM, _) => 0,
            (CMEM, READ) => 1,
            (CMEM, WRITE) => 2,
            (VMEM, READ) => 3,
            (VMEM, WRITE) => 4,
            _ => continue,
        };
        bytes[slot] += amount;
    }
    if metrics.memory.is_empty() {
        bytes[0] = metrics.bytes_accessed;
    }
    let bandwidths: [f64; MEMORIES.len()] = std::array::from_fn(|slot| giga_to_gibi(safe_divide(bytes[slot] as f64, if slot == 0 { time_ns } else { device_ns })));
    let peak = |index: usize| perf.bandwidths.get(index).copied().unwrap_or(0.0);
    let out = Roofline {
        measured_flop_rate: safe_divide(metrics.flops_v2, time_ns),
        model_flop_rate: safe_divide(metrics.model_flops_v2, time_ns),
        measured_memory_bw: giga_to_gibi(safe_divide(metrics.bytes_accessed as f64, time_ns)),
        operational_intensity: safe_divide(metrics.flops_v2, metrics.bytes_accessed as f64),
        hbm_bw: bandwidths[0],
        cmem_read_bw: bandwidths[1],
        cmem_write_bw: bandwidths[2],
        vmem_read_bw: bandwidths[3],
        vmem_write_bw: bandwidths[4],
        hbm_operational_intensity: safe_divide(metrics.flops_v2, bytes[0] as f64),
        ..Default::default()
    };
    let compute = safe_divide(out.measured_flop_rate, perf.peak_tera_flops * 1e3);
    let mut bound = if 0.0 < compute { ("Compute", compute, out.operational_intensity) } else { ("Unknown", 0.0, 0.0) };
    for (slot, &(name, index)) in MEMORIES.iter().enumerate() {
        let candidate = safe_divide(bandwidths[slot], giga_to_gibi(peak(index)));
        if (slot == 0 || (stats.tpu && bytes[slot] != 0 && peak(index) != 0.0)) && bound.1 < candidate {
            bound = (name, candidate, safe_divide(metrics.flops_v2, bytes[slot] as f64));
        }
    }
    let shared = safe_divide(out.hbm_bw, giga_to_gibi(peak(2)));
    if stats.extra.hardware == GPU && peak(2) != 0.0 && bound.1 < shared {
        bound = ("Shm/L1", shared, out.hbm_operational_intensity);
    }
    Roofline { bound_by: bound.0, bottleneck_operational_intensity: bound.2, ..out }
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#39;")
}

pub fn source_text(metrics: &Metrics) -> String {
    match &metrics.source {
        Some(source) if !source.file.is_empty() && source.line != -1 => {
            format!("<div class='source-info-cell' title='{}'>{}:{}</div>", escape_html(&source.stack), escape_html(&source.file), source.line)
        }
        _ => String::new(),
    }
}

struct Record<'a> {
    metrics: &'a Metrics,
    parent: &'a str,
    rank: u64,
    self_fraction: f64,
    cumulative: f64,
}

fn collect<'a>(out: &mut Vec<Record<'a>>, metrics: &'a Metrics, previous: &mut Option<(u64, f64)>, total_us: f64, parent: &'a str) {
    if metrics.occurrences == 0 {
        return;
    }
    if parent.is_empty() || metrics.core_type == SPARSE_CORE {
        let mut record = Record { metrics, parent, rank: 0, self_fraction: 0.0, cumulative: 0.0 };
        if let Some((rank, cumulative)) = *previous {
            record.rank = rank + 1;
            record.self_fraction = safe_divide(pico_to_micro(metrics.self_time_ps), total_us);
            record.cumulative = cumulative + record.self_fraction;
            *previous = Some((record.rank, record.cumulative));
        }
        out.push(record);
    }
    if !metrics.children.metrics.is_empty() {
        let mut child_previous = None;
        let child_total = pico_to_micro(metrics.children.total_time_ps);
        for child in metrics.children.sorted() {
            collect(out, child, &mut child_previous, child_total, &metrics.name);
        }
    }
}

fn hlo_op_name(expression: &str) -> &str {
    let name = expression.split(" = ").next().unwrap();
    name.strip_prefix('%').unwrap_or(name)
}

pub fn json(stats: &OpStats) -> String {
    let total_us = pico_to_micro(stats.db.total_time_ps);
    let mut trees: Vec<Vec<Record>> = stats
        .db
        .sorted()
        .into_par_iter()
        .map(|metrics| {
            let mut out = Vec::new();
            collect(&mut out, metrics, &mut Some((0, 0.0)), total_us, "");
            out
        })
        .collect();
    let mut previous = (0, 0.0);
    for top in trees.iter_mut().filter_map(|tree| tree.first_mut()) {
        previous = (previous.0 + 1, previous.1 + top.self_fraction);
        (top.rank, top.cumulative) = previous;
    }
    let records: Vec<Record> = trees.into_iter().flatten().collect();
    let mut table = Table::new(&COLUMNS);
    table.rows = records
        .par_iter()
        .map(|record| {
            let metrics = record.metrics;
            let roof = roofline(metrics, stats, false);
            let long = if metrics.long_name.is_empty() { &metrics.name } else { &metrics.long_name };
            let text = |text: &str| Cell::Text(text.to_string());
            let yes_no = |flag: bool| text(if flag { "Yes" } else { "No" });
            let (total_time, self_time, occurrences) = (pico_to_micro(metrics.time_ps), pico_to_micro(metrics.self_time_ps), metrics.occurrences as f64);
            let mut row = vec![Cell::Number(record.rank as f64), Cell::Text(metrics.module.to_string()), text(&metrics.category), text(hlo_op_name(long)), text(long), text(&metrics.provenance)];
            row.extend(
                [
                    occurrences,
                    total_time,
                    safe_divide(total_time, occurrences),
                    self_time,
                    safe_divide(self_time, occurrences),
                    record.self_fraction * 100.0,
                    record.cumulative * 100.0,
                    safe_divide(metrics.dma_stall_ps as f64, metrics.time_ps as f64),
                    roof.model_flop_rate,
                    roof.measured_flop_rate,
                    roof.measured_memory_bw,
                    roof.hbm_bw,
                    roof.cmem_read_bw,
                    roof.cmem_write_bw,
                    roof.operational_intensity,
                ]
                .map(Cell::Number),
            );
            row.extend([
                text(roof.bound_by),
                yes_no(metrics.long_name.split('=').next().unwrap().contains(".remat") || metrics.provenance.contains("rematted_computation")),
                yes_no(
                    metrics.provenance.ends_with(":XlaSendToHost")
                        || metrics.provenance.ends_with(":XlaRecvFromHost")
                        || (metrics.long_name.contains("send-done") && metrics.long_name.contains("is_host_transfer=true")),
                ),
                yes_no(metrics.autotuned),
                Cell::Text(source_text(metrics)),
                text(match metrics.core_type {
                    TENSOR_CORE => "TensorCore",
                    SPARSE_CORE => "SparseCore",
                    _ => "",
                }),
                text(record.parent),
                Cell::Number(metrics.vdd_energy.unwrap_or(0.0)),
            ]);
            row
        })
        .collect();
    let json = table.json();
    crate::release(table);
    json
}
