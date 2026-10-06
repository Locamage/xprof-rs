use crate::tools::opstats::{Db, IDLE, OpStats, safe_divide};
use crate::tools::table::{Cell, Table};
use crate::xplane::steps::{
    Breakdown, CPU_ONLY, Core, DEVICE_COLLECTIVES, DEVICE_COMPUTE_16, DEVICE_COMPUTE_32, DEVICE_TO_DEVICE, DEVICE_TO_HOST, DEVICE_WAIT_DEVICE, DEVICE_WAIT_HOST, Extra, GPU, HOST_COMPILE,
    HOST_COMPUTE, HOST_PREPARE, HOST_TO_DEVICE, HOST_WAIT_INPUT, SPARSE_CORE_START, StepRecord, TPU, UNKNOWN_TIME,
};
use std::collections::BTreeMap;

const NO_STEP: &str = "No step time measured. Therefore we cannot tell where the performance bottleneck is.";
pub const NO_STEP_MARKER: &str = "No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.";
pub const EMPTY_INTERSECT: &str = "Although there are steps observed on some host(s), the intersection of the steps over all hosts is empty (because the differences among individual host's step sequences are too big). Consequently, the overall step time is unknown.";
const INPUT_BOUND: [(&str, &str); 4] = [
    ("host", "Your program is HIGHLY input-bound because {}% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time."),
    ("both", "Your program is MODERATELY input-bound because {}% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time."),
    ("both", "Your program is POTENTIALLY input-bound because {}% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both)."),
    ("device", "Your program is NOT input-bound because only {}% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time."),
];
const OUTPUT_BOUND: [(f64, &str); 2] = [
    (20.0, "Your program is HIGHLY output-bound because {}% of the total step time sampled is spent on output. Therefore, you should first focus on reducing the output time."),
    (5.0, "Your program is MODERATELY output-bound because {}% of the total step time sampled is spent on output. Therefore, you would need to reduce both the output time and other time."),
];
const KERNEL_LAUNCH_TF_DATA: &str = ". It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.";
pub const HARDWARE: [&str; 4] = ["UNKNOWN_HARDWARE", "CPU_ONLY", "GPU", "TPU"];
pub const BOTTLENECK_PREFIXES: [&str; 4] = ["", "kernel_launch_", "all_other_", "device_collectives_"];
const RECOMMENDATIONS: [&str; 5] = [
    r"Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.",
    r#"Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE."#,
    r#"Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)"#,
    r#"Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)"#,
    r#"Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)"#,
];
const CONSIDER_TF_DATA: &str = r#"Consider using <a href="https://www.tensorflow.org/guide/data" target="_blank">the tf.data API</a> to enable profiler's host-side analysis for input pipeline. Profiler currently does not support custom input pipeline (please ignore Section 3 below)."#;
pub(crate) const TC_COMPUTE: usize = 0;
pub(crate) const SCV0_COMPUTE: usize = 1;
pub(crate) const TC_INFEED: usize = 2;
pub(crate) const TC_OUTFEED: usize = 3;
pub(crate) const SCV0_INFEED: usize = 4;
pub const TC_IDLE: usize = 5;
pub(crate) const HOST_TRANSFER: usize = 6;
pub(crate) const SC_COMPUTE: usize = 7;
const OTHER: usize = 0;
const TO_DEVICE: usize = 2;
const INPUT: usize = 3;
const COLLECTIVES: usize = 7;
const KERNEL_LAUNCH: usize = 9;
const GENERIC_COLUMNS: [(&str, &str, &str, &str, usize); 9] = [
    ("deviceComputeTimeMs", "Device compute", "device_compute_time_ms", "Device compute", 5),
    ("deviceToDeviceTimeMs", "Device to device", "device_to_device_time_ms", "Device to device", 6),
    ("deviceCollectivesTimeMs", "Device collective communication", "device_collectives_time_ms", "Device collectives", COLLECTIVES),
    ("hostComputeTimeMs", "Host compute", "host_compute_time_ms", "Host compute", 8),
    ("kernelLaunchTimeMs", "Kernel launch", "kernel_launch_time_ms", "Kernel launch", KERNEL_LAUNCH),
    ("infeedTimeMs", "Input", "infeed_time_ms", "Input", INPUT),
    ("outfeedTimeMs", "Output", "outfeed_time_ms", "Output", 4),
    ("compileTimeMs", "Compilation", "compile_time_ms", "Compilation", 10),
    ("otherTimeMs", "All others", "other_time_ms", "All others", OTHER),
];
const TPU_COLUMNS: [(&str, &str, &str, usize); 7] = [
    ("tcComputeTimeMs", "TensorCore compute (in ms)", "tc_compute_ms_average", TC_COMPUTE),
    ("scv0ComputeTimeMs", "SparseCoreV0 compute (in ms)", "scv0_compute_ms_average", SCV0_COMPUTE),
    ("scv0InfeedTimeMs", "SparseCoreV0 input (in ms)", "scv0_infeed_ms_average", SCV0_INFEED),
    ("tcInfeedTimeMs", "TensorCore input (in ms)", "tc_infeed_ms_average", TC_INFEED),
    ("tcOutfeedTimeMs", "TensorCore output (in ms)", "tc_outfeed_ms_average", TC_OUTFEED),
    ("tcIdleTimeMs", "TensorCore idle (in ms)", "tc_idle_ms_average", TC_IDLE),
    ("hostTransferTimeMs", "Host transfer (in ms)", "host_transfer_ms_average", HOST_TRANSFER),
];
const HOST_COLUMNS: [(&str, &str, &str); 7] = [
    ("opName", "string", "Input Op"),
    ("count", "number", "Count"),
    ("timeInMs", "number", "Total Time (in ms)"),
    ("timeInPercent", "number", "Total Time (as % of total input-processing time)"),
    ("selfTimeInMs", "number", "Total Self Time (in ms)"),
    ("selfTimeInPercent", "number", "Total Self Time (as % of total input-processing time)"),
    ("category", "string", "Category"),
];
const INPUT_TIMES: [&str; 5] = ["enqueue_us", "demanded_file_read_us", "advanced_file_read_us", "preprocessing_us", "unclassified_nonequeue_us"];
const SUMMARY: [&str; 4] = ["average", "standard_deviation", "minimum", "maximum"];
const INPUT_CATEGORIES: [&str; 5] = ["Enqueue", "Demanded file read", "Advanced file read", "Preprocessing", "Unknown"];
const FILE_READERS: [&str; 6] = ["::TFRecord", "::TextLine", "::FixedLengthRecord", "::SSTable", "::RecordIO", "::ArrayRecord"];
const ADVANCED_READERS: [&str; 5] = ["::MemoryReader", "::MemoryWriter", "::Interleave", "::Prefetch", "::ParallelMap"];

pub(crate) fn hardware(device_type: &str) -> u8 {
    match device_type {
        kind if kind.contains("GPU") => GPU,
        "CPU" => CPU_ONLY,
        kind if kind.contains("TPU") => TPU,
        _ => 0,
    }
}

pub fn fixed(value: f64, digits: usize) -> String {
    match value {
        value if value.is_nan() => "-nan".into(),
        value => format!("{value:.digits$}"),
    }
}

pub(crate) fn ms(ps: u64) -> f64 {
    ps as f64 / 1e9
}

fn summary(values: impl Iterator<Item = f64>) -> [f64; 4] {
    let (mut count, mut sum, mut squared, mut min, mut max) = (0u64, 0.0, 0.0, f64::MAX, f64::MIN_POSITIVE);
    for value in values {
        max = value.max(max);
        min = value.min(min);
        count += 1;
        sum += value;
        squared += value * value;
    }
    if count == 0 {
        return [0.0; 4];
    }
    let variance = if min == max { 0.0 } else { (squared - sum * sum / count as f64) / (count - 1) as f64 };
    [sum / count as f64, variance.sqrt(), min, max]
}

#[derive(Default)]
pub struct TpuStep {
    pub(crate) step_number: i32,
    pub fields: [f64; 10],
    infeed_percent: [f64; 3],
    pub(crate) all_reduce_ms: [f64; 2],
    core_name: String,
}

impl TpuStep {
    fn infeed_ms(&self) -> f64 {
        self.fields[TC_INFEED] + self.fields[SCV0_INFEED]
    }

    fn all_reduce(&self) -> f64 {
        self.all_reduce_ms[0] + self.all_reduce_ms[1]
    }

    fn step_ms(&self) -> f64 {
        self.fields[TC_COMPUTE] + self.fields[SCV0_COMPUTE] + self.infeed_ms() + self.all_reduce() + self.fields[TC_OUTFEED] + self.fields[TC_IDLE]
    }
}

pub fn tpu_step_details(record: &StepRecord, cores: &BTreeMap<u32, Core>) -> TpuStep {
    let mut minimum: BTreeMap<u64, u64> = BTreeMap::new();
    for reduce in record.collectives.values().flatten() {
        let duration = reduce.2.wrapping_sub(reduce.1);
        minimum.entry(reduce.0).and_modify(|value| *value = (*value).min(duration)).or_insert(duration);
    }
    let all_reduce = |core: u32| {
        record.collectives.get(&core).into_iter().flatten().fold((0u64, 0u64), |(compute, sync), reduce| {
            let (duration, min) = (reduce.2.wrapping_sub(reduce.1), minimum.get(&reduce.0).copied().unwrap_or(0));
            (compute.wrapping_add(min), sync.wrapping_add(duration - min))
        })
    };
    let mut step = TpuStep { step_number: if record.cores.is_empty() { -1 } else { record.num as i32 }, ..Default::default() };
    let mut infeed_percent = Vec::new();
    let (mut tc_count, mut step_min, mut optimal_min, mut outfeed_sum, mut send_recv_max) = (0u64, u64::MAX, u64::MAX, 0u64, 0u64);
    let (mut sc_count, mut sc_compute_min, mut sc_idle_sum, mut sc_step_sum) = (0u64, u64::MAX, 0u64, 0u64);
    let (mut max_scv0, mut max_all_reduce, mut max_infeed) = (0u64, (0u64, 0u64), (None, 0u64));
    for (&core, info) in &record.cores {
        let Breakdown::Categories(categories) = &info.breakdown else { continue };
        if core >= SPARSE_CORE_START {
            let idle = categories.get(IDLE).copied().unwrap_or(0);
            sc_count += 1;
            sc_step_sum = sc_step_sum.wrapping_add(info.duration);
            sc_compute_min = sc_compute_min.min(info.duration.wrapping_sub(idle));
            sc_idle_sum = sc_idle_sum.wrapping_add(idle);
            continue;
        }
        let mut categories: BTreeMap<&str, u64> = categories.iter().filter(|(category, _)| *category != IDLE).map(|(category, time)| (category.as_str(), *time)).collect();
        let total: u64 = categories.values().fold(0u64, |sum, value| sum.wrapping_add(*value));
        if let Some(bc_infeed) = categories.remove("sparsecorev0 infeed").filter(|value| *value != 0) {
            let transform = bc_infeed.min(categories.get("sparsecorev0 outfeed").copied().unwrap_or(0));
            categories.insert("sparsecorev0 infeed wait", bc_infeed - transform);
            categories.insert("sparsecorev0 infeed transform", transform);
        }
        let get = |name: &str| categories.get(name).copied().unwrap_or(0);
        let (infeed, outfeed, wait_scv0) = (get("infeed"), get("outfeed"), get("sparsecorev0 infeed wait"));
        let send_recv = get("host send") + get("host send-done") + get("host recv") + get("host recv-done");
        let tc_idle = info.duration.wrapping_sub(total);
        tc_count += 1;
        step_min = step_min.min(info.duration);
        max_scv0 = max_scv0.max(wait_scv0);
        outfeed_sum = outfeed_sum.wrapping_add(outfeed);
        let breakdown = all_reduce(core);
        if breakdown.0.wrapping_add(breakdown.1) > max_all_reduce.0.wrapping_add(max_all_reduce.1) {
            max_all_reduce = breakdown;
        }
        infeed_percent.push(100.0 * infeed as f64 / info.duration as f64);
        optimal_min = optimal_min.min(info.duration.wrapping_sub(infeed.wrapping_add(outfeed).wrapping_add(send_recv).wrapping_add(tc_idle).wrapping_add(wait_scv0)));
        send_recv_max = send_recv_max.max(send_recv);
        if infeed > max_infeed.1 {
            max_infeed = (Some(core), infeed);
        }
    }
    step.fields[TC_INFEED] = ms(max_infeed.1);
    step.core_name = max_infeed.0.and_then(|core| cores.get(&core)).map_or_else(|| "unknown".into(), |core| format!("{}:{}", core.hostname, core.ordinal));
    step.fields[SCV0_COMPUTE] = ms(max_scv0);
    if tc_count > 0 {
        step.fields[TC_OUTFEED] = ms((outfeed_sum as f64 / tc_count as f64) as u64);
        step.fields[TC_COMPUTE] = ms(optimal_min);
        step.fields[HOST_TRANSFER] = ms(send_recv_max);
        step.fields[TC_IDLE] = (ms(step_min) - step.step_ms()).max(0.0);
        let [average, _, minimum, maximum] = summary(infeed_percent.into_iter());
        step.infeed_percent = [average, minimum, maximum];
    }
    step.all_reduce_ms = [ms(max_all_reduce.0), ms(max_all_reduce.1)];
    if sc_count > 0 {
        step.fields[SC_COMPUTE..].copy_from_slice(&[ms(sc_compute_min), ms((sc_idle_sum as f64 / sc_count as f64) as u64), ms((sc_step_sum as f64 / sc_count as f64) as u64)]);
    }
    step
}

pub struct Analysis {
    cores: Table,
    host: Table,
    pub steps: Table,
    pub bottleneck: [(String, String); 4],
    pub warnings: Vec<String>,
}

fn summarize<const N: usize>(rows: &(impl Iterator<Item = [f64; N]> + Clone)) -> Vec<[f64; 4]> {
    (0..N).map(|index| summary(rows.clone().map(|row| row[index]))).collect()
}

fn input_analysis(input_percent: f64, all_other_percent: f64) -> (String, String, bool) {
    let (index, percent) = match (input_percent, all_other_percent) {
        (input, _) if input >= 20.0 => (0, input),
        (input, _) if input >= 5.0 => (1, input),
        (_, other) if other >= 3.0 => (2, other),
        (input, _) => (3, input),
    };
    (INPUT_BOUND[index].0.into(), INPUT_BOUND[index].1.replace("{}", &fixed(percent, 1)), index == 2)
}

fn host_result(host: &Db, enqueue: (u64, u64), fallback_ratio: f64) -> (Table, [f64; 5]) {
    let mut table = Table::new(&HOST_COLUMNS);
    let input_op = |category: &str| {
        let lower = category.to_ascii_lowercase();
        category.starts_with("InfeedEnqueue") || category == "Dataset" || category == "MemcpyHToD" || matches!(lower.as_str(), "enqueue" | "read" | "preprocessing" | "unknown")
    };
    let selected: Vec<_> = host.sorted().into_iter().filter(|metrics| input_op(&metrics.category)).collect();
    if selected.is_empty() {
        return (table, [0.0; 5]);
    }
    let total_ps: u64 = selected.iter().map(|metrics| metrics.self_time_ps).sum();
    let mut aggregated = [0.0f64; 5];
    for metrics in &selected {
        let category = match metrics.category.to_ascii_lowercase().as_str() {
            "enqueue" => 0,
            "read" => 1,
            "preprocessing" => 3,
            "unknown" => 4,
            _ if metrics.category.starts_with("InfeedEnqueue") || metrics.category == "MemcpyHToD" => 0,
            _ if FILE_READERS.iter().any(|suffix| metrics.name.ends_with(suffix)) => 1 + usize::from(ADVANCED_READERS.iter().any(|part| metrics.name.contains(part))),
            _ => 3,
        };
        let percent = |time: u64| Cell::Number(100.0 * safe_divide(time as f64, total_ps as f64) / 100.0);
        table.rows.push(vec![
            Cell::Text(metrics.name.to_string()),
            Cell::Number(metrics.occurrences as f64),
            Cell::Number(ms(metrics.time_ps)),
            percent(metrics.time_ps),
            Cell::Number(ms(metrics.self_time_ps)),
            percent(metrics.self_time_ps),
            Cell::Text(INPUT_CATEGORIES[category].into()),
        ]);
        aggregated[category] += metrics.self_time_ps as f64 / 1e6;
    }
    let (enqueue_us, total_input_us) = (aggregated[0], aggregated[1] + aggregated[2] + aggregated[3]);
    let ratio = if enqueue.1 > 0 { safe_divide(enqueue.0 as f64, enqueue.1 as f64) } else { fallback_ratio }.min(1.0);
    let non_enqueue = if ratio == 0.0 { total_input_us } else { enqueue_us * (1.0 - ratio) / ratio };
    let scaled = |value: f64| safe_divide(non_enqueue * value, total_input_us);
    let (demanded, advanced, preprocessing) = (scaled(aggregated[1]), scaled(aggregated[2]), scaled(aggregated[3]));
    (table, [enqueue_us, demanded, advanced, preprocessing, 0.0f64.max(non_enqueue - demanded - advanced - preprocessing)])
}

fn no() -> (String, String) {
    ("no".to_string(), String::new())
}

fn unknown() -> [(String, String); 4] {
    [("unknown".into(), NO_STEP.into()), no(), no(), no()]
}

fn tpu_steps(extra: &Extra, steps: &mut Table, cores: &mut Table, step_summary: [f64; 4]) -> ([(String, String); 4], String) {
    let (two, one) = (|value: f64| fixed(value, 2), |value: f64| fixed(value, 1));
    let has_sparse_core = extra.cores.values().any(|core| core.sparse);
    let tpu_steps: Vec<TpuStep> = extra.steps.iter().map(|record| tpu_step_details(record, &extra.cores)).collect();
    let breakdown = summarize(&tpu_steps.iter().map(|step| step.fields));
    let input_percent = summary(tpu_steps.iter().map(|step| step.infeed_percent[2]));
    let columns: Vec<_> = TPU_COLUMNS.iter().filter(|column| !has_sparse_core || !column.0.starts_with("scv0")).collect();
    steps.column("stepnum", "string", "stepnum");
    columns.iter().for_each(|(id, label, _, _)| steps.column(id, "number", label));
    let ids: Vec<&str> = steps.columns.iter().map(|column| column.0.as_str()).chain(["tooltip"]).collect();
    steps.prop("step_time_graph_column_ids", ids.join(","));
    steps.columns.push(("tooltip".into(), "string", "tooltip".into(), Some("tooltip")));
    steps.column("infeedPercentAverage", "number", "% step time waiting for input data");
    steps.columns.push(("infeedPercentMin".into(), "number", "Infeed percent min".into(), Some("interval")));
    steps.columns.push(("infeedPercentMax".into(), "number", "Infeed percent max".into(), Some("interval")));
    for (index, name) in SUMMARY.iter().enumerate() {
        steps.prop(&format!("steptime_ms_{name}"), two(step_summary[index]));
        steps.prop(&format!("infeed_percent_{name}"), one(input_percent[index]));
    }
    columns.iter().for_each(|(_, _, key, index)| steps.prop(key, two(breakdown[*index][0])));
    if has_sparse_core && breakdown[SC_COMPUTE + 2][2] > 0.0 {
        for (key, value) in [("sc_compute", breakdown[SC_COMPUTE][0]), ("sc_infeed", 0.0), ("sc_outfeed", 0.0), ("sc_idle", breakdown[SC_COMPUTE + 1][0])] {
            steps.prop(&format!("{key}_ms_average"), two(value));
        }
        steps.prop("sc_step_time_ms_average", one(breakdown[SC_COMPUTE + 2][0]));
    }
    for step in &tpu_steps {
        let tooltip = format!("step {}:\nTime waiting for input data = {} ms, Step time = {} ms", step.step_number, fixed(step.infeed_ms(), 3), fixed(step.step_ms() - step.all_reduce(), 3));
        let row = steps.row();
        row.push(Cell::Text(step.step_number.to_string()));
        row.extend(columns.iter().map(|column| Cell::Number(step.fields[column.3])));
        row.push(Cell::Text(tooltip));
        row.extend(step.infeed_percent.map(Cell::Number));
    }
    cores.column("index", "string", "Index");
    (0..tpu_steps.len()).for_each(|index| cores.column(&index.to_string(), "string", &index.to_string()));
    let numbers = tpu_steps.iter().map(|step| Cell::Text(step.step_number.to_string()));
    let names = tpu_steps.iter().map(|step| Cell::Text(step.core_name.clone()));
    cores.rows = vec![std::iter::once(Cell::Text("Step number".into())).chain(numbers).collect(), std::iter::once(Cell::Text("Core name".into())).chain(names).collect()];
    let sum = |field: fn(&TpuStep) -> f64| tpu_steps.iter().map(field).fold(0.0, |sum, value| sum + value);
    let total = sum(TpuStep::step_ms);
    let (classification, input_statement, _) = if total == 0.0 { ("unknown".into(), NO_STEP.into(), false) } else { input_analysis(100.0 * sum(TpuStep::infeed_ms) / total, 0.0) };
    let outfeed_percent = 100.0 * sum(|step| step.fields[TC_OUTFEED]) / total;
    let output = OUTPUT_BOUND.iter().find(|(threshold, _)| total != 0.0 && outfeed_percent >= *threshold).map_or("", |(_, text)| text);
    steps.prop("output_conclusion", output.replace("{}", &fixed(outfeed_percent, 1)));
    steps.prop("input_conclusion", input_statement);
    (if tpu_steps.is_empty() { unknown() } else { Default::default() }, classification)
}

fn generic_steps_table(steps: &mut Table, generic_steps: &[(String, f64, [f64; 11])], generic_breakdown: &[[f64; 4]], step_summary: [f64; 4], tf_data: bool) -> ([(String, String); 4], String) {
    let (two, one) = (|value: f64| fixed(value, 2), |value: f64| fixed(value, 1));
    let total: f64 = generic_steps.iter().map(|step| step.1).fold(0.0, |sum, value| sum + value);
    let percent = |index: usize| 100.0 * generic_steps.iter().map(|step| step.2[index]).fold(0.0, |sum, value| sum + value) / total;
    let assess = |percent: f64, reported: bool, name: &str, suffix: &str| match percent {
        _ if reported || percent < 3.0 => no(),
        _ => ((if percent >= 15.0 { "high" } else { "moderate" }).to_string(), format!("{} % of the total step time sampled is spent on '{name}'{suffix}", fixed(percent, 1))),
    };
    let bottleneck = if total == 0.0 {
        unknown()
    } else {
        let (input_classification, input_statement, all_other_reported) = input_analysis(percent(INPUT), percent(OTHER));
        [
            (input_classification, input_statement),
            assess(percent(KERNEL_LAUNCH), false, "Kernel Launch", if tf_data { KERNEL_LAUNCH_TF_DATA } else { "." }),
            assess(percent(OTHER), all_other_reported, "All Others", " time. This could be due to Python execution overhead."),
            assess(percent(COLLECTIVES), false, "Device Collective Communication", "."),
        ]
    };
    steps.column("stepnum", "string", "Step number");
    GENERIC_COLUMNS.iter().for_each(|(id, label, _, _, _)| steps.column(id, "number", label));
    let ids: Vec<&str> = steps.columns.iter().map(|column| column.0.as_str()).collect();
    steps.prop("step_time_graph_column_ids", ids.join(",") + ", tooltip");
    steps.columns.push(("tooltip".into(), "string", "tooltip".into(), Some("tooltip")));
    steps.column("stepTimeMs", "number", "Step time");
    for (name, value) in SUMMARY.iter().zip(step_summary) {
        steps.prop(&format!("steptime_ms_{name}"), one(value));
    }
    for (_, _, key, _, index) in GENERIC_COLUMNS {
        steps.prop(&format!("{key}_avg"), one(generic_breakdown[index][0]));
        steps.prop(&format!("{key}_sdv"), one(generic_breakdown[index][1]));
    }
    steps.prop("input_conclusion", bottleneck[0].1.clone());
    for ((_, statement), prefix) in bottleneck.iter().zip(BOTTLENECK_PREFIXES).skip(1) {
        steps.prop(&format!("{prefix}bottleneck"), statement.clone());
        steps.prop(&format!("{prefix}statement"), statement.clone());
    }
    for (name, step_time, values) in generic_steps {
        let mut tooltip = format!("Step {name}, duration: {} ms", two(*step_time));
        for (_, _, _, label, index) in GENERIC_COLUMNS.iter().rev() {
            tooltip += &format!("\n-{label}: {} ms", two(values[*index]));
        }
        let row = steps.row();
        row.push(Cell::Text(name.clone()));
        row.extend(GENERIC_COLUMNS.iter().map(|column| Cell::Number(values[column.4])));
        row.extend([Cell::Text(tooltip), Cell::Number(*step_time)]);
    }
    let classification = bottleneck[0].0.clone();
    (bottleneck, classification)
}

pub fn analyze(stats: &OpStats) -> Analysis {
    let extra = &*stats.extra;
    let (mut steps, mut cores) = (Table::default(), Table::default());
    steps.prop("hardware_type", HARDWARE[extra.hardware as usize]);
    let longest = |record: &StepRecord| record.cores.iter().filter(|(core, _)| **core < SPARSE_CORE_START).fold(0.0f64, |longest, (_, info)| ms(info.duration).max(longest));
    let step_summary = summary(extra.steps.iter().map(longest));
    let warnings = match (extra.steps.is_empty(), extra.empty_intersect) {
        (false, _) => Vec::new(),
        (true, true) => vec![EMPTY_INTERSECT.to_string()],
        (true, false) => vec![NO_STEP_MARKER.to_string()],
    };
    let mut generic_steps: Vec<(String, f64, [f64; 11])> = Vec::new();
    for (record, info) in extra.steps.iter().filter_map(|record| Some((record, record.cores.get(&1)?))) {
        let Breakdown::Types(types) = &info.breakdown else { continue };
        let time = |kind: u32| ms(types.get(&kind).copied().unwrap_or(0));
        let (wait, to_device) = (time(HOST_WAIT_INPUT), time(HOST_TO_DEVICE) + time(DEVICE_WAIT_HOST));
        let values = [
            time(UNKNOWN_TIME),
            wait,
            to_device,
            wait + to_device,
            time(DEVICE_TO_HOST),
            time(DEVICE_COMPUTE_16) + time(DEVICE_COMPUTE_32),
            time(DEVICE_TO_DEVICE) + time(DEVICE_WAIT_DEVICE),
            time(DEVICE_COLLECTIVES),
            time(HOST_COMPUTE),
            time(HOST_PREPARE),
            time(HOST_COMPILE),
        ];
        generic_steps.push((if info.name.is_empty() { record.num.to_string() } else { info.name.clone() }, ms(info.duration), values));
    }
    let generic_breakdown = summarize(&generic_steps.iter().map(|step| step.2));
    let tpu = extra.hardware == TPU;
    let fallback_ratio = if step_summary[0] > 0.0 && !tpu { safe_divide(generic_breakdown[TO_DEVICE][0], step_summary[0]) } else { 0.0 };
    let (mut host, input_time) = host_result(&stats.host, stats.infeed_enqueue, fallback_ratio);
    INPUT_TIMES.iter().zip(input_time).for_each(|(key, value)| host.prop(key, fixed(value, 3)));
    let tf_data = input_time[1..4].iter().any(|&time| time > 0.0);
    let (bottleneck, classification) =
        if tpu { tpu_steps(extra, &mut steps, &mut cores, step_summary) } else { generic_steps_table(&mut steps, &generic_steps, &generic_breakdown, step_summary, tf_data) };
    let next_step = match classification.as_str() {
        "host" | "both" if tf_data => "Look at Section 3 for the breakdown of input time on the host.",
        "host" | "both" => CONSIDER_TF_DATA,
        _ => "You may skip the rest of this page.",
    };
    steps.prop("summary_nextstep", next_step);
    Analysis { cores, host, steps, bottleneck, warnings }
}

pub fn diagnostics_table(warnings: &[String], errors: &[String]) -> Table {
    let mut table = Table::new(&[("severity", "string", "Severity"), ("message", "string", "Message")]);
    for (severity, messages) in [("WARNING", warnings), ("ERROR", errors)] {
        table.rows.extend(messages.iter().map(|message| vec![Cell::Text(severity.into()), Cell::Text(message.clone())]));
    }
    table
}

pub fn json(stats: &OpStats) -> String {
    let analysis = analyze(stats);
    let mut recommendation = Table::new(&[("link", "string", "link")]);
    recommendation.rows = RECOMMENDATIONS.iter().map(|detail| vec![Cell::Text(detail.to_string())]).collect();
    format!("[{},{},{},{},{}]", analysis.cores.json(), analysis.steps.json(), analysis.host.json(), recommendation.json(), diagnostics_table(&analysis.warnings, &[]).json())
}

#[cfg(test)]
#[path = "../tests/inline/tools/input_pipeline_analyzer.rs"]
mod tests;
