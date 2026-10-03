use crate::framework_op_stats::device_tf_db;
use crate::gpu::by_op_name;
use crate::hlo::general;
use crate::inference_profile::InferenceStats;
use crate::input_pipeline_analyzer::{Analysis, BOTTLENECK_PREFIXES, HARDWARE, analyze, diagnostics_table, fixed};
use crate::memory_viewer::std_sort;
use crate::opstats::{Db, IDLE, OpStats, safe_divide};
use crate::steps::{CPU_ONLY, Extra, GPU, TPU};
use crate::table::{Cell, Table};
use std::path::PathBuf;

const TOP_OPS: usize = 10;
const NO_DEVICE_TRACE: &str =
    "No TensorCore device trace was collected. This might happen if your job hadn't been run on the device when sampling was turned on. You could try the sampling again later.";
const TOP_COLUMNS: [(&str, &str, &str); 5] = [
    ("selfTimePercent", "number", "Time (%)"),
    ("cumulativeTimePercent", "number", "Cumulative time (%)"),
    ("category", "string", "Category"),
    ("operation", "string", "Operation"),
    ("flopRate", "number", "Bf16 Normalized Flop Rate(GFLOPs/Sec)"),
];
const GENERIC_TOP_COLUMNS: [(&str, &str, &str); 7] = [
    ("selfTimePercent", "number", "Time (%)"),
    ("cumulativeTimePercent", "number", "Cumulative time (%)"),
    ("category", "string", "Category"),
    ("operation", "string", "Operation"),
    ("flopRate", "number", "GFLOPs/Sec"),
    ("tcEligibility", "boolean", "TensorCore eligibility"),
    ("tcUtilization", "boolean", "Op is using TensorCore"),
];
const LOW_PRECISION_PERCENT: f64 = 10.0;
const LATENCY_PERCENTILES: [f64; 5] = [50.0, 75.0, 90.0, 99.0, 99.9];
const LATENCY_EPSILON: f64 = 1e-20;
const LATENCY_COLUMNS: [(&str, &str, &str); 5] = [
    ("percentile", "string", "percentile"),
    ("hostTimeMs", "number", "Host time (in ms)"),
    ("deviceTimeMs", "number", "Device time (in ms)"),
    ("communicationTimeMs", "number", "Host-device communication time (in ms)"),
    ("totalTimeMs", "number", "Total latency (in ms)"),
];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LatencyBreakdown {
    pub total_latency_us: f64,
    pub host_latency_us: f64,
    pub device_latency_us: f64,
    pub communication_latency_us: f64,
}

pub fn inference_latency(stats: &InferenceStats) -> Vec<LatencyBreakdown> {
    let sessions: Vec<LatencyBreakdown> = stats
        .inference_stats_per_model
        .values()
        .flat_map(|model| &model.request_details)
        .map(|request| {
            let total_latency_us = request.end_time_ps.wrapping_sub(request.start_time_ps) as f64 / 1e6;
            let device_latency_us = request.device_time_ps as f64 / 1e6;
            let communication_latency_us = request.read_from_device_time_ps.wrapping_add(request.write_to_device_time_ps) as f64 / 1e6;
            LatencyBreakdown { total_latency_us, host_latency_us: total_latency_us - device_latency_us - communication_latency_us, device_latency_us, communication_latency_us }
        })
        .collect();
    if sessions.is_empty() {
        return Vec::new();
    }
    let mean = |value: fn(&LatencyBreakdown) -> f64| {
        let sum = sessions.iter().fold(0.0, |sum, session| sum + value(session));
        if sum.abs() < LATENCY_EPSILON { 0.0 } else { sum / sessions.len() as f64 }
    };
    let average = LatencyBreakdown {
        total_latency_us: mean(|session| session.total_latency_us),
        host_latency_us: mean(|session| session.host_latency_us),
        device_latency_us: mean(|session| session.device_latency_us),
        communication_latency_us: mean(|session| session.communication_latency_us),
    };
    let mut order: Vec<usize> = (0..sessions.len()).collect();
    std_sort(&mut order, &|a, b| sessions[a].total_latency_us < sessions[b].total_latency_us);
    let percentiles = LATENCY_PERCENTILES.iter().map(|percentile| sessions[order[(percentile / 100.0 * sessions.len() as f64) as usize]]);
    std::iter::once(average).chain(percentiles).collect()
}

fn percentage(value: f64) -> String {
    if value >= 0.0 { format!("{}%", fixed(value, 1)) } else { "unknown".into() }
}

fn top_ops_table(stats: &OpStats, device_hardware: u8) -> Table {
    let extra = &*stats.extra;
    let ops = device_tf_db(&stats.db);
    let generic = device_hardware == GPU;
    let mut analysis_table = Table::new(if generic { &GENERIC_TOP_COLUMNS } else { &TOP_COLUMNS });
    let kernels = by_op_name(&stats.kernels);
    let mut cumulative = 0.0;
    for op in ops.sorted().into_iter().filter(|op| op.category != IDLE).take(TOP_OPS) {
        let fraction = safe_divide(op.self_time_ps as f64, stats.db.total_op_time_ps as f64);
        cumulative += fraction;
        let mut row = vec![
            Cell::Number(fraction),
            Cell::Number(cumulative),
            Cell::Text(op.category.to_string()),
            Cell::Text(op.name.to_string()),
            Cell::Number(safe_divide(op.flops_v2, op.time_ps as f64 / 1e3)),
        ];
        if generic {
            let (eligible, _, tensor_core_ns) = kernels.get(op.name.as_str()).copied().unwrap_or_default();
            row.extend([Cell::Boolean(eligible), Cell::Boolean(tensor_core_ns != 0)]);
        }
        analysis_table.rows.push(row);
    }
    let [compute_32, compute_16] = extra.precision.map(|ps| ps as f64);
    let precision_percent = |part: f64| percentage(100.0 * safe_divide(part, compute_16 + compute_32));
    let host = &stats.host;
    let host_ops: u64 = host.metrics.iter().map(|metrics| metrics.occurrences).sum();
    let host_busy: u64 = host.metrics.iter().filter(|metrics| metrics.category != IDLE).map(|metrics| metrics.self_time_ps).sum();
    let host_eager: u64 = host.metrics.iter().filter(|metrics| metrics.category != IDLE && metrics.is_eager).map(|metrics| metrics.self_time_ps).sum();
    let device_ops: u64 = ops.metrics.iter().map(|op| op.occurrences).sum();
    let device_busy: u64 = ops.metrics.iter().filter(|op| op.category != IDLE).map(|op| op.self_time_ps).sum();
    let device_eager: u64 = ops.metrics.iter().filter(|op| op.category != IDLE && op.is_eager).map(|op| op.self_time_ps).sum();
    let all_ops = host_ops + device_ops;
    let tpu_only = |value: f64| percentage(if device_hardware == TPU { value } else { 0.0 });
    let duty = |index: usize| tpu_only(safe_divide(extra.busy_ps[index] as f64, extra.busy_ps[index].wrapping_add(extra.idle_ps[index]) as f64) * 100.0);
    let idle = |db: &Db| tpu_only((1.0 - safe_divide(db.total_op_time_ps as f64, db.total_time_ps as f64)) * 100.0);
    let common = [
        ("host_trace_level", "0".to_string()),
        ("host_tf_op_percent", percentage(100.0 * safe_divide(host_ops as f64, all_ops as f64))),
        ("device_tf_op_percent", percentage(100.0 * safe_divide(device_ops as f64, all_ops as f64))),
        ("host_op_time_eager_percent", percentage(100.0 * safe_divide(host_eager as f64, host_busy as f64))),
        ("device_op_time_eager_percent", percentage(100.0 * safe_divide(device_eager as f64, device_busy as f64))),
        ("remark_text", String::new()),
        ("remark_color", String::new()),
        ("mxu_utilization_percent", percentage(extra.mxu)),
        ("hbm_utilization_percent", percentage(extra.hbm)),
    ];
    let specific = if generic {
        vec![("device_compute_16bit_percent", precision_percent(compute_16)), ("device_compute_32bit_percent", precision_percent(compute_32))]
    } else {
        vec![
            ("flop_rate_utilization_relative_to_roofline", percentage(0.0)),
            ("memory_bw_utilization_relative_to_hw_limit", percentage(0.0)),
            ("device_idle_time_percent", idle(&stats.db)),
            ("device_duty_cycle_percent", duty(0)),
            ("device_duty_cycle_high_confidence_percent", duty(1)),
            ("host_idle_time_percent", idle(host)),
        ]
    };
    for (key, value) in common.into_iter().chain(specific) {
        analysis_table.prop(key, value);
    }
    if !generic && device_hardware == TPU && extra.idle_ps[1] > 0 {
        analysis_table.prop("idle_time_high_confidence_ps", general(extra.idle_ps[1] as f64, 6));
    }
    analysis_table
}

fn environment_table(extra: &Extra) -> Table {
    let mut environment = Table::new(&["host_id", "command_line", "start_time", "bns_address"].map(|id| (id, "string", id)));
    let hosts = extra.hostnames.len() as i32;
    environment.prop("host_count", hosts.to_string());
    environment.prop("task_count", if extra.tasks > 0 && hosts > 0 { format!("{} (num tasks per host = {})", extra.tasks, extra.tasks / hosts) } else { extra.tasks.to_string() });
    environment.prop("device_type", extra.device_type.clone());
    environment.prop("device_core_count", extra.core_count.to_string());
    environment.prop("is_training", extra.training.to_string());
    environment
}

fn recommendation_table(extra: &Extra, analysis: &Analysis, device_hardware: u8) -> Table {
    let [compute_32, compute_16] = extra.precision.map(|ps| ps as f64);
    let bottleneck = &analysis.bottleneck;
    let mut recommendation = Table::new(&["tip_type", "link", "description"].map(|id| (id, "string", id)));
    let device_name = HARDWARE[device_hardware as usize];
    let (timeline, tool) = if device_hardware == TPU { ("TPU core", "op_profile") } else { (device_name, "framework_op_stats") };
    let tips = [
        ("faq", "Refer to the TF2 Profiler FAQ".to_string(), "Tool troubleshooting / FAQ"),
        ("host", "input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)".into(), "Next steps for reducing the Host time"),
        ("host", "trace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)".into(), "Next steps for reducing the Host time"),
        ("device", format!("{tool} (identify the time-consuming operations executed on the {device_name})"), "Next steps for reducing the Device time"),
        ("device", format!("trace_viewer (look at the activities on the timeline of each {timeline} in the trace view)"), "Next steps for reducing the Device time"),
        ("doc", r#"<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>"#.into(), "Other useful resources"),
        ("doc", r#"<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>"#.into(), "Other useful resources"),
    ];
    recommendation.rows = tips.into_iter().map(|(kind, link, description)| vec![Cell::Text(kind.into()), Cell::Text(link), Cell::Text(description.into())]).collect();
    for ((kind, statement), prefix) in bottleneck.iter().zip(BOTTLENECK_PREFIXES) {
        recommendation.prop(&format!("{prefix}bottleneck"), kind.clone());
        recommendation.prop(&format!("{prefix}statement"), statement.clone());
    }
    let percent_16 = 100.0 * compute_16 / (compute_16 + compute_32);
    let low_precision = extra.precision.iter().sum::<u64>() > 0 && percent_16 < LOW_PRECISION_PERCENT;
    recommendation.prop(
        "precision_statement",
        if low_precision {
            format!(
                "Only {}% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).",
                fixed(percent_16, 1)
            )
        } else {
            String::new()
        },
    );
    let non_bottleneck = match bottleneck[0].0.as_str() {
        "host" => "device",
        "device" => "host",
        _ => "",
    };
    recommendation.prop("non_bottleneck_tip_types", non_bottleneck);
    recommendation
}

fn latency_json(extra: &Extra, paths: &[PathBuf]) -> String {
    let mut latency = Table::new(&LATENCY_COLUMNS);
    let breakdowns = if extra.training { Vec::new() } else { inference_latency(&crate::inference_profile::load(paths).unwrap_or_default()) };
    let labels = std::iter::once("Avg".to_string()).chain(LATENCY_PERCENTILES.iter().map(|percentile| format!("{}%", fixed(*percentile, 1))));
    for (label, breakdown) in labels.zip(breakdowns) {
        let times = [breakdown.host_latency_us, breakdown.device_latency_us, breakdown.communication_latency_us, breakdown.total_latency_us];
        latency.row().extend(std::iter::once(Cell::Text(label)).chain(times.map(|us| Cell::Number(us / 1e3))));
    }
    latency.json()
}

pub fn json(stats: &OpStats, paths: &[PathBuf]) -> String {
    let extra = &*stats.extra;
    let device_hardware = match extra.device_type.as_str() {
        kind if kind.contains("GPU") => GPU,
        "CPU" => CPU_ONLY,
        kind if kind.contains("TPU") => TPU,
        _ => 0,
    };
    let analysis = analyze(stats);
    let mut errors = extra.errors.clone();
    errors.sort();
    if errors.is_empty() && extra.device_type != "CPU" && extra.core_count <= 0 {
        errors.push(NO_DEVICE_TRACE.into());
    }
    let warnings: Vec<String> = extra.warnings.iter().chain(&analysis.warnings).cloned().collect();
    format!(
        "[{},{},{},{},{}, {{}},{}]",
        top_ops_table(stats, device_hardware).json(),
        analysis.steps.json(),
        environment_table(extra).json(),
        recommendation_table(extra, &analysis, device_hardware).json(),
        latency_json(extra, paths),
        diagnostics_table(&warnings, &errors).json()
    )
}
