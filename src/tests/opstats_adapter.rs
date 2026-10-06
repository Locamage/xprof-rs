use super::xspace::descriptor;
use crate::tools::opstats::{Db, Metrics, OpStats, Perf, Source};
use crate::xplane::gpu::{KernelKey, KernelReport};
use crate::xplane::steps::{Breakdown, Core, Extra, StepInfo, StepRecord};
use prost_reflect::{DynamicMessage, MapKey, ReflectMessage, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

const ANY_PREFIX: &str = "type.googleapis.com/";
const GENERIC_STEP_BREAKDOWN: &str = "tensorflow.profiler.GenericStepBreakdown";

fn field(message: &DynamicMessage, name: &str) -> Value {
    message.get_field_by_name(name).unwrap_or_else(|| panic!("{} has no field {name}", message.descriptor().full_name())).into_owned()
}

fn number(message: &DynamicMessage, name: &str) -> u64 {
    match field(message, name) {
        Value::U64(value) => value,
        Value::U32(value) => value.into(),
        Value::I64(value) => value as u64,
        Value::I32(value) => value as u64,
        Value::EnumNumber(value) => value as u64,
        Value::Bool(value) => value.into(),
        other => panic!("{name} is not an integer: {other:?}"),
    }
}

fn double(message: &DynamicMessage, name: &str) -> f64 {
    match field(message, name) {
        Value::F64(value) => value,
        Value::F32(value) => value.into(),
        other => panic!("{name} is not a double: {other:?}"),
    }
}

fn text(message: &DynamicMessage, name: &str) -> String {
    field(message, name).as_str().unwrap().to_string()
}

fn flag(message: &DynamicMessage, name: &str) -> bool {
    field(message, name).as_bool().unwrap()
}

fn child(message: &DynamicMessage, name: &str) -> DynamicMessage {
    field(message, name).as_message().unwrap().clone()
}

fn list(message: &DynamicMessage, name: &str) -> Vec<Value> {
    field(message, name).as_list().unwrap().to_vec()
}

fn messages(message: &DynamicMessage, name: &str) -> Vec<DynamicMessage> {
    list(message, name).iter().map(|value| value.as_message().unwrap().clone()).collect()
}

fn key_number(key: &MapKey) -> u64 {
    match key {
        MapKey::U32(value) => (*value).into(),
        MapKey::U64(value) => *value,
        MapKey::I32(value) => *value as u64,
        MapKey::I64(value) => *value as u64,
        other => panic!("{other:?} is not an integer key"),
    }
}

fn map(message: &DynamicMessage, name: &str) -> Vec<(MapKey, Value)> {
    let mut entries: Vec<(MapKey, Value)> = field(message, name).as_map().unwrap().iter().map(|(key, value)| (key.clone(), value.clone())).collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

fn value_number(value: &Value) -> u64 {
    value.as_u64().or_else(|| value.as_u32().map(u64::from)).unwrap()
}

pub fn parse(name: &str, text: &str) -> DynamicMessage {
    DynamicMessage::parse_text_format(descriptor(name), text).unwrap()
}

pub fn metrics(message: &DynamicMessage) -> Metrics {
    let source = message.has_field_by_name("source_info").then(|| {
        let info = child(message, "source_info");
        Source { file: text(&info, "file_name").into(), line: number(&info, "line_number") as i32, stack: text(&info, "stack_frame").into() }
    });
    let memory = messages(message, "memory_accessed_breakdown").iter().map(|entry| (number(entry, "operation_type") as u8, number(entry, "memory_space"), number(entry, "bytes_accessed"))).collect();
    Metrics {
        module: number(message, "hlo_module_id"),
        name: text(message, "name").into(),
        long_name: text(message, "long_name").into(),
        category: text(message, "category").into(),
        provenance: text(message, "provenance").into(),
        deduplicated_name: text(message, "deduplicated_name").into(),
        occurrences: number(message, "occurrences"),
        time_ps: number(message, "time_ps"),
        normalized_time_ps: number(message, "normalized_time_ps"),
        min_time_ps: number(message, "min_time_ps"),
        self_time_ps: number(message, "self_time_ps"),
        dma_stall_ps: number(message, "dma_stall_ps"),
        flops_v2: double(message, "flops_v2"),
        model_flops_v2: double(message, "model_flops_v2"),
        bytes_accessed: number(message, "bytes_accessed"),
        memory,
        vdd_energy: message.has_field_by_name("vdd_energy_j").then(|| double(message, "vdd_energy_j")),
        num_cores: number(message, "num_cores") as u32,
        core_type: number(message, "core_type") as u8,
        autotuned: flag(message, "autotuned"),
        is_eager: flag(message, "is_eager"),
        source,
        children: if message.has_field_by_name("children") { db(&child(message, "children")) } else { Db::default() },
    }
}

pub fn db(message: &DynamicMessage) -> Db {
    Db {
        metrics: messages(message, "metrics_db").iter().map(metrics).collect(),
        total_time_ps: number(message, "total_time_ps"),
        total_op_time_ps: number(message, "total_op_time_ps"),
        normalized_total_op_time_ps: number(message, "normalized_total_op_time_ps"),
    }
}

fn breakdown(info: &DynamicMessage) -> Breakdown {
    if !info.has_field_by_name("step_breakdown") {
        return Breakdown::Types(BTreeMap::new());
    }
    let any = child(info, "step_breakdown");
    let type_url = text(&any, "type_url");
    assert_eq!(type_url.strip_prefix(ANY_PREFIX), Some(GENERIC_STEP_BREAKDOWN));
    let bytes = field(&any, "value").as_bytes().unwrap().clone();
    let generic = DynamicMessage::decode(descriptor(GENERIC_STEP_BREAKDOWN), bytes).unwrap();
    let categories: BTreeMap<String, u64> = map(&generic, "category_ps").into_iter().map(|(key, value)| (key.as_str().unwrap().to_string(), value_number(&value))).collect();
    if categories.is_empty() {
        Breakdown::Types(map(&generic, "type_ps").into_iter().map(|(key, value)| (key_number(&key) as u32, value_number(&value))).collect())
    } else {
        Breakdown::Categories(categories)
    }
}

pub fn steps(step_db: &DynamicMessage) -> Vec<StepRecord> {
    messages(step_db, "step_sequence")
        .iter()
        .map(|step| StepRecord {
            num: number(step, "step_num") as u32,
            cores: map(step, "step_info_per_core")
                .into_iter()
                .map(|(core, info)| {
                    let info = info.as_message().unwrap();
                    let record = StepInfo { name: text(info, "step_name"), begin: number(info, "begin_ps"), duration: number(info, "duration_ps"), breakdown: breakdown(info) };
                    (key_number(&core) as u32, record)
                })
                .collect(),
            collectives: map(step, "all_reduce_db_per_core")
                .into_iter()
                .map(|(core, reduces)| {
                    let reduces = messages(reduces.as_message().unwrap(), "all_reduce_info");
                    (key_number(&core) as u32, reduces.iter().map(|reduce| (number(reduce, "id"), number(reduce, "start_time_ps"), number(reduce, "end_time_ps"))).collect())
                })
                .collect(),
        })
        .collect()
}

fn kernel(report: &DynamicMessage) -> KernelReport {
    let dims = |name: &str| {
        let values: Vec<u32> = list(report, name).iter().map(|value| value.as_u32().unwrap()).collect();
        std::array::from_fn(|index| values.get(index).copied().unwrap_or(0))
    };
    KernelReport {
        key: KernelKey {
            name: text(report, "name"),
            grid: dims("grid_dim"),
            block: dims("block_dim"),
            registers: number(report, "registers_per_thread") as u32,
            static_shmem: number(report, "static_shmem_bytes") as u32,
            dynamic_shmem: number(report, "dynamic_shmem_bytes") as u32,
            tensor_core: flag(report, "is_kernel_using_tensor_core"),
            eligible: flag(report, "is_op_tensor_core_eligible"),
            op_name: text(report, "op_name"),
        },
        occupancy: match field(report, "occupancy_pct") {
            Value::F32(value) => value.into(),
            other => panic!("occupancy_pct is not a float: {other:?}"),
        },
        total_ns: number(report, "total_duration_ns"),
        min_ns: number(report, "min_duration_ns"),
        max_ns: number(report, "max_duration_ns"),
        occurrences: number(report, "occurrences"),
    }
}

pub fn extra(stats: &DynamicMessage) -> Extra {
    let environment = child(stats, "run_environment");
    let device = child(stats, "device_op_metrics_db");
    let step_db = child(stats, "step_db");
    let diagnostics = child(stats, "diagnostics");
    let counters = child(stats, "performance_counter_result");
    let precision = child(&device, "precision_stats");
    let perf_env = child(stats, "perf_env");
    let strings = |name: &str| list(&diagnostics, name).iter().map(|value| value.as_str().unwrap().to_string()).collect();
    Extra {
        hostnames: map(&environment, "hostnames").into_iter().map(|(key, _)| key.as_str().unwrap().to_string()).collect(),
        tasks: number(&environment, "task_count") as i32,
        empty_intersect: flag(&step_db, "empty_intersect"),
        device_type: text(&environment, "device_type"),
        core_count: number(&environment, "device_core_count") as i32,
        hardware: number(&environment, "hardware_type") as u8,
        training: flag(&environment, "is_training"),
        busy_ps: [number(&device, "busy_time_ps"), number(&device, "busy_time_high_confidence_ps")],
        idle_ps: [number(&device, "idle_time_ps"), number(&device, "idle_time_high_confidence_ps")],
        steps: steps(&step_db),
        cores: map(stats, "core_id_to_details")
            .into_iter()
            .map(|(core, details)| {
                let details = details.as_message().unwrap();
                let core_details = Core {
                    hostname: text(details, "hostname"),
                    ordinal: number(details, "device_ordinal") as u32,
                    chip: number(details, "local_chip_id") as u32,
                    sparse: flag(details, "is_sparse_core"),
                };
                (key_number(&core) as u32, core_details)
            })
            .collect(),
        mxu: double(&counters, "matrix_unit_utilization_percent"),
        hbm: double(&counters, "hbm_utilization_percent"),
        errors: strings("errors"),
        warnings: strings("warnings"),
        megacore: flag(&perf_env, "has_megacore"),
        merged_vmem: flag(&perf_env, "has_merged_vmem"),
        precision: [number(&precision, "compute_32bit_ps"), number(&precision, "compute_16bit_ps")],
        ..Default::default()
    }
}

pub fn op_stats_message(stats: &DynamicMessage) -> OpStats {
    let perf_env = child(stats, "perf_env");
    let extra = extra(stats);
    OpStats {
        db: db(&child(stats, "device_op_metrics_db")),
        perf: Perf {
            peak_tera_flops: double(&perf_env, "peak_tera_flops_per_second"),
            bandwidths: list(&perf_env, "peak_bws_giga_bytes_per_second").iter().map(|value| value.as_f64().unwrap()).collect(),
            ridge_point: double(&perf_env, "ridge_point"),
            cmem: false,
        },
        tpu: extra.hardware == crate::xplane::steps::TPU,
        host: db(&child(stats, "host_op_metrics_db")),
        extra: Arc::new(extra),
        programs: map(stats, "program_id_to_name_map").into_iter().map(|(key, value)| (key_number(&key), value.as_str().unwrap().to_string())).collect::<HashMap<_, _>>(),
        kernels: messages(&child(stats, "kernel_stats_db"), "reports").iter().map(kernel).collect(),
    }
}

pub fn op_stats(text: &str) -> OpStats {
    op_stats_message(&parse("tensorflow.profiler.OpStats", text))
}
