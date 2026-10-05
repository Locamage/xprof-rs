use super::xspace::{V, XPlane, XSpace, op_stats, proto_bytes};
use crate::tools::opstats::IDLE;
use crate::xplane::steps::{Breakdown, SPARSE_CORE_START};

const MAX_ERROR: f64 = 0.01;
const DEFAULT_GPU_LOCAL_CORE_ID: u32 = 1;

type CustomCall<'a> = (&'a str, &'a str, i64, &'a [(&'a str, V)]);

fn near(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= MAX_ERROR, "{actual} != {expected}");
}

fn amd_mi300x(space: &mut XSpace) -> &mut XPlane {
    let plane = space.gpu(0);
    plane.add_stat("device_vendor", "AMD");
    plane.add_stat("gpu_device_name", "gfx942");
    plane.add_stat("clock_rate", 2100000);
    plane.add_stat("core_count", 304);
    plane.add_stat("memory_bandwidth", 5300u64 * 1000 * 1000 * 1000);
    plane
}

fn perf_env_of(space: &XSpace) -> crate::tools::opstats::Perf {
    let (_, planes) = space.parsed();
    crate::xplane::gpu::perf_env(&planes[0])
}

#[test]
fn gpu_perf_env() {
    let mut space = XSpace::default();
    let plane = space.gpu(0);
    plane.add_stat("device_vendor", "Nvidia");
    plane.add_stat("clock_rate", 1530000);
    plane.add_stat("core_count", 80);
    plane.add_stat("memory_bandwidth", 900u64 * 1000 * 1000 * 1000);
    plane.add_stat("compute_cap_major", 7);
    plane.add_stat("compute_cap_minor", 0);
    let stats = op_stats(&[space]).unwrap();
    near(stats.perf.peak_tera_flops, 125.34);
    near(stats.perf.bandwidths[0], 900.0);
    near(stats.perf.ridge_point, 139.26);
}

#[test]
fn gpu_perf_env_prefers_backend_peaks() {
    let mut space = XSpace::default();
    let plane = amd_mi300x(&mut space);
    plane.add_stat("peak_teraflops_per_second", 1250.0);
    plane.add_stat("peak_sram_rd_bw_gigabytes_per_second", 65000.0);
    plane.add_stat("peak_sram_wr_bw_gigabytes_per_second", 48000.0);
    let perf = perf_env_of(&space);
    near(perf.peak_tera_flops, 1250.0);
    near(perf.bandwidths[1], 65000.0);
    near(perf.bandwidths[2], 48000.0);
    near(perf.bandwidths[0], 5300.0);
}

#[test]
fn gpu_perf_env_derives_peaks_without_backend_stats() {
    let mut space = XSpace::default();
    amd_mi300x(&mut space);
    let perf = perf_env_of(&space);
    near(perf.peak_tera_flops, 1307.44);
    near(perf.bandwidths[1], 81715.2);
    near(perf.bandwidths[2], 81715.2);
}

#[test]
fn gpu_perf_env_derives_peaks_when_backend_reports_zero() {
    let mut space = XSpace::default();
    let plane = amd_mi300x(&mut space);
    plane.add_stat("peak_teraflops_per_second", 0.0);
    plane.add_stat("peak_sram_rd_bw_gigabytes_per_second", 0.0);
    plane.add_stat("peak_sram_wr_bw_gigabytes_per_second", 0.0);
    let perf = perf_env_of(&space);
    near(perf.peak_tera_flops, 1307.44);
    near(perf.bandwidths[1], 81715.2);
    near(perf.bandwidths[2], 81715.2);
}

#[test]
fn gpu_run_environment() {
    let mut space = XSpace::default();
    space.gpu(0).add_stat("device_vendor", "Nvidia");
    space.gpu(1).add_stat("device_vendor", "Nvidia");
    let stats = op_stats(&[space]).unwrap();
    let extra = &stats.extra;
    assert_eq!((extra.device_type.as_str(), extra.hostnames.len(), extra.tasks, extra.core_count), ("Nvidia GPU", 1, 1, 2));
}

fn tf_step(space: &mut XSpace, executor_duration: i64, op: (&str, i64, &[(&str, V)])) {
    let host = space.host();
    host.event(0, "TraceContext", 0, 100, &[("step_num", 123.into())]);
    host.event(0, "FunctionRun", 10, 90, &[("id", 0.into()), ("_pt", 1.into()), ("_p", 0.into())]);
    host.event(1, "ExecutorState::Process", 20, executor_duration, &[("id", 0.into()), ("_ct", 1.into()), ("_c", 0.into())]);
    host.event(1, op.0, 30, op.1, op.2);
}

#[test]
fn cpu_only_step_db_test() {
    let mut space = XSpace::default();
    tf_step(&mut space, 80, ("matmul", 70, &[]));
    assert_eq!(op_stats(&[space]).unwrap().extra.steps.len(), 1);
}

#[test]
fn gpu_step_db_test() {
    let mut space = XSpace::default();
    tf_step(&mut space, 20, ("matmul", 10, &[("correlation_id", 100.into())]));
    space.gpu(0).event(0, "matmul", 50, 40, &[("correlation_id", 100.into())]);
    let stats = op_stats(&[space]).unwrap();
    assert_eq!(stats.extra.steps.len(), 1);
    assert_eq!((stats.extra.precision[1], stats.extra.precision[0]), (0, 40));
}

#[test]
fn propagate_and_dedup_errors() {
    let space = XSpace { errors: vec!["host: error".into(); 2], ..Default::default() };
    assert_eq!(op_stats(&[space]).unwrap().extra.errors, ["host: error"]);
}

#[test]
fn hostnames() {
    let space = XSpace { hostnames: vec!["host1".into()], ..Default::default() };
    assert_eq!(op_stats(&[space]).unwrap().extra.cores[&DEFAULT_GPU_LOCAL_CORE_ID].hostname, "host1");
}

#[test]
fn test_convert_multi_x_spaces_to_combined_op_stats() {
    let spaces: Vec<XSpace> = ["host1", "host2"]
        .map(|host| {
            let mut space = XSpace { hostnames: vec![host.into()], ..Default::default() };
            tf_step(&mut space, 80, ("aaa:bbb", 70, &[]));
            space
        })
        .into();
    let stats = op_stats(&spaces).unwrap();
    assert_eq!(stats.host.metrics.len(), 2);
    let metric = &stats.host.metrics[1];
    assert_eq!((&*metric.name, &*metric.category, metric.self_time_ps), ("aaa", "bbb", 140));
    assert_eq!(stats.extra.steps.len(), 1);
    let cores: Vec<u32> = stats.extra.steps[0].cores.keys().copied().collect();
    assert_eq!(cores, [DEFAULT_GPU_LOCAL_CORE_ID, 1000 + DEFAULT_GPU_LOCAL_CORE_ID]);
    assert_eq!(stats.extra.cores[&DEFAULT_GPU_LOCAL_CORE_ID].hostname, "host1");
    assert_eq!(stats.extra.cores[&(1000 + DEFAULT_GPU_LOCAL_CORE_ID)].hostname, "host2");
}

#[test]
fn run_environment_extracted_from_tpu_plane() {
    let mut space = XSpace::default();
    for ordinal in 0..4 {
        space.tpu(ordinal, "TPU V4", 0.0, 0.0, None);
    }
    let stats = op_stats(&[space]).unwrap();
    assert_eq!((stats.extra.device_type.as_str(), stats.extra.core_count), ("TPU V4", 4));
}

#[test]
fn tpu_perf_env() {
    let mut space = XSpace::default();
    let plane = space.tpu(0, "TPU V4", 141.0, 900.0, None);
    plane.add_stat("clock_rate", 1530000);
    plane.add_stat("core_count", 80);
    plane.add_stat("memory_bandwidth", 900u64 * 1000 * 1000 * 1000);
    plane.add_stat("compute_cap_major", 7);
    plane.add_stat("compute_cap_minor", 0);
    let peaks = [("sram_rd", 101.0), ("sram_wr", 102.0), ("cmem_rd", 101.0), ("cmem_wr", 102.0), ("vmem_rd", 201.0), ("vmem_wr", 202.0)];
    for (memory, peak) in peaks {
        plane.add_stat(&format!("peak_{memory}_bw_gigabytes_per_second"), peak);
    }
    let stats = op_stats(&[space]).unwrap();
    near(stats.perf.peak_tera_flops, 141.0);
    near(stats.perf.bandwidths[0], 900.0);
    for (index, (_, peak)) in peaks.into_iter().enumerate() {
        near(stats.perf.bandwidths[index + 1], peak);
    }
    near(stats.perf.ridge_point, 156.67);
}

#[test]
fn tpu_run_environment() {
    let mut space = XSpace::default();
    space.tpu(0, "TPU V4", 0.0, 0.0, None);
    space.tpu(1, "TPU V4", 0.0, 0.0, None);
    let stats = op_stats(&[space]).unwrap();
    let extra = &stats.extra;
    assert_eq!((extra.device_type.as_str(), extra.hostnames.len(), extra.tasks, extra.core_count), ("TPU V4", 1, 1, 2));
}

#[test]
fn tpu_device_trace_to_step_db() {
    let mut space = XSpace::default();
    let plane = space.tpu(0, "TPU V4", 141.0, 1000.0, None);
    plane.metadata_stats("op_name", &[("program_id", 1.into()), ("symbol_id", 1.into()), ("self_duration_ps", 10.into()), ("tf_op", "tf_op_name".into()), ("hlo_category", "category".into())]);
    plane.named_line(1, "Framework Ops");
    plane.event(1, "op_name", 0, 10 * 1000, &[]);
    let stats = op_stats(&[space]).unwrap();
    let mut names: Vec<&str> = stats.db.metrics.iter().map(|metrics| &*metrics.name).collect();
    names.sort_unstable();
    assert_eq!(names, [IDLE, "op_name"]);
}

#[test]
fn tpu_multi_device_step_db_test() {
    let mut space = XSpace::default();
    let first = space.tpu(0, "TPU V4", 0.0, 0.0, None);
    first.named_line(0, "Steps");
    first.event(0, "Step 1", 100, 100000, &[("group_id", 1.into())]);
    first.named_line(1, "XLA Ops");
    first.event(1, "op.1", 110, 10, &[("hlo_category", "infeed".into()), ("group_id", 1.into())]);
    let second = space.tpu(1, "TPU V4", 0.0, 0.0, None);
    second.named_line(0, "Steps");
    second.event(0, "Step 1", 300, 100, &[("group_id", 1.into())]);
    second.event(0, "Step 2", 300, 100, &[("group_id", 2.into())]);
    second.named_line(1, "XLA Ops");
    second.event(1, "op.1", 310, 10, &[("hlo_category", "infeed".into()), ("group_id", 1.into())]);
    second.event(1, "op.2", 310, 10, &[("hlo_category", "infeed".into()), ("group_id", 2.into())]);
    assert_eq!(op_stats(&[space]).unwrap().extra.steps.len(), 1);
}

#[test]
fn tpu_tc_and_sc_step_db_test() {
    let mut space = XSpace::default();
    let core = space.tpu(0, "TPU V4", 0.0, 0.0, None);
    core.id = 1;
    core.named_line(0, "Steps");
    core.event(0, "Step 1", 100, 100000, &[("group_id", 1.into())]);
    core.named_line(1, "XLA Ops");
    core.event(1, "op.1", 110, 10, &[("hlo_category", "infeed".into()), ("group_id", 1.into())]);
    let sparse = space.tpu(1, "TPU V4", 0.0, 0.0, None);
    sparse.id = 2;
    sparse.name = "/device:TPU:0 SparseCore 0".into();
    sparse.named_line(0, "Sparse Core Steps");
    sparse.event(0, "Step 1", 1000, 10000, &[("group_id", 1.into()), ("step_idle_time_ps", 9000.into())]);
    sparse.named_line(1, "Sparse Core Ops");
    sparse.event(1, "op.2", 1010, 1000, &[("hlo_category", "sparse_core_op".into()), ("group_id", 1.into())]);
    let stats = op_stats(&[space]).unwrap();
    assert_eq!(stats.extra.steps.len(), 1);
    let cores = &stats.extra.steps[0].cores;
    assert_eq!(cores.len(), 2);
    assert_eq!((cores[&1].duration, cores[&1].begin), (100000, 100));
    assert_eq!((cores[&(SPARSE_CORE_START + 2)].duration, cores[&(SPARSE_CORE_START + 2)].begin), (10000, 1000));
}

fn duty_cycle(space: XSpace) -> ([u64; 2], [u64; 2]) {
    let stats = op_stats(&[space]).unwrap();
    (stats.extra.busy_ps, stats.extra.idle_ps)
}

fn device_with_module(module_duration: i64) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.tpu(0, "TPU v4", 0.0, 0.0, None);
    plane.named_line(0, "XLA Ops");
    plane.named_line(1, "XLA Modules");
    plane.event(1, "module.1", 5, module_duration, &[]);
    space
}

#[test]
fn construct_duty_cycle_tracker_from_xla_ops() {
    let mut space = device_with_module(50);
    let plane = &mut space.planes[0];
    for (index, category) in ["infeed", "call", "call", "outfeed"].into_iter().enumerate() {
        let name = format!("op.{}", index + 1);
        plane.metadata_stats(&name, &[("hlo_category", category.into())]);
        plane.event(0, &name, 10 * (index as i64 + 1), 10, &[]);
    }
    let (busy, idle) = duty_cycle(space);
    assert_eq!((busy[0], idle[0]), (20, 30));
}

#[test]
fn construct_duty_cycle_tracker_from_sparse_core() {
    let mut space = XSpace::default();
    let plane = space.tpu(0, "TPU v4", 0.0, 0.0, None);
    plane.named_line(0, "Sparse Core Ops");
    for index in 1..=4 {
        plane.event(0, &format!("op.{index}"), 10 * index, 10, &[]);
    }
    plane.named_line(1, "Sparse Core Modules");
    plane.event(1, "module.1", 5, 50, &[]);
    let (busy, idle) = duty_cycle(space);
    assert_eq!((busy[0], idle[0]), (40, 10));
}

#[test]
fn multi_core_chip_busy_and_idle_time_test() {
    let core_details = proto_bytes("tensorflow.profiler.CoreDetails", "local_chip_id: 0");
    let mut space = XSpace::default();
    let core = space.tpu(0, "TPU v4", 0.0, 0.0, None);
    core.add_stat("core_details", V::Bytes(core_details.clone()));
    core.named_line(0, "XLA Ops");
    core.event(0, "op.1", 10, 10, &[("hlo_category", "infeed".into())]);
    core.event(0, "op.2", 20, 10, &[("hlo_category", "call".into())]);
    core.event(0, "op.3", 30, 10, &[]);
    core.event(0, "op.4", 40, 10, &[("hlo_category", "outfeed".into())]);
    core.named_line(1, "XLA Modules");
    core.event(1, "module.1", 5, 50, &[]);
    let sparse = space.tpu(1, "TPU v4", 0.0, 0.0, None);
    sparse.add_stat("core_details", V::Bytes(core_details));
    sparse.named_line(0, "Sparse Core Ops");
    for index in 1..=4 {
        sparse.event(0, &format!("op.{index}"), 10 * index, 10, &[]);
    }
    sparse.named_line(1, "Sparse Core Modules");
    sparse.event(1, "module.1", 5, 50, &[]);
    let (busy, idle) = duty_cycle(space);
    assert_eq!((idle[0], busy[0]), (10, 40));
}

#[test]
fn handle_sparse_core_busy_op_metrics() {
    let mut space = XSpace::default();
    let core = space.tpu(0, "TPU v4", 0.0, 0.0, None);
    core.id = 0;
    core.named_line(0, "Steps");
    core.named_line(1, "XLA Modules");
    core.named_line(2, "XLA Ops");
    for index in 1..=4 {
        let (offset, group) = (10 * index, [("group_id", V::from(index))]);
        core.metadata_stats(&format!("op.{index}"), &[("program_id", 1.into()), ("symbol_id", index.into())]);
        core.event(0, &format!("step.{index}"), offset, 10, &group);
        core.event(1, &format!("module.{index}"), offset, 10, &group);
        core.event(2, &format!("op.{index}"), offset + 5, 5, &group);
    }
    let sparse = space.tpu(1, "TPU v4", 0.0, 0.0, None);
    sparse.id = 1;
    sparse.name = "/device:TPU:1 SparseCore 1".into();
    sparse.named_line(0, "Sparse Core Steps");
    sparse.named_line(1, "Sparse Core Modules");
    sparse.named_line(2, "Sparse Core Ops");
    for index in 1..=4 {
        let (offset, group) = (10 * index, [("group_id", V::from(index))]);
        sparse.event(0, &format!("step.{index}"), offset, 10, &[("step_idle_time_ps", 5.into()), ("group_id", index.into())]);
        sparse.event(1, &format!("module.{index}"), offset, 10, &group);
        sparse.event(2, &format!("scs op.{index}"), offset + 5, 5, &group);
    }
    let stats = op_stats(&[space]).unwrap();
    assert_eq!((stats.db.total_time_ps, stats.db.total_op_time_ps), (40, 20));
    assert_eq!(stats.extra.steps.len(), 4);
}

#[test]
fn handle_input_pipeline_slowness_causing_device_idleness() {
    let mut space = XSpace::default();
    let device = space.tpu(0, "TPU V4", 0.0, 0.0, None);
    device.id = 0;
    device.named_line(0, "Steps");
    device.event(0, "Step 1", 1000, 10000, &[("group_id", 1.into())]);
    device.named_line(1, "XLA Ops");
    device.metadata_stats("op.1", &[("hlo_category", "arithmetic".into()), ("program_id", 1.into()), ("symbol_id", 1.into()), ("flops", 1000.into())]);
    device.event(1, "op.1", 2000, 8000, &[("group_id", 1.into())]);
    let host = space.host();
    host.named_line(0, "main");
    host.event(0, "Iterator::Batch::Map::TFRecord", 500, 2300, &[("group_id", 1.into()), ("_ipl_stage_id", 1.into()), ("_ipl_stage_name", "TFRecord".into())]);
    let stats = op_stats(&[space]).unwrap();
    assert_eq!(stats.extra.steps.len(), 1);
    let cores = &stats.extra.steps[0].cores;
    assert_eq!(cores.len(), 1);
    let Breakdown::Categories(categories) = &cores[&0].breakdown else { panic!("expected a generic step breakdown") };
    assert_eq!((categories.get(IDLE), categories.get("infeed"), categories.get("arithmetic")), (Some(&0), Some(&2000), Some(&8000)));
}

fn custom_call_duty_cycle(ops: &[CustomCall], module_duration: i64) -> ([u64; 2], [u64; 2]) {
    let mut space = device_with_module(module_duration);
    let plane = &mut space.planes[0];
    for &(name, category, offset, stats) in ops {
        plane.metadata_stats(name, &[("hlo_category", category.into())]);
        plane.event(0, name, offset, 10, stats);
    }
    duty_cycle(space)
}

#[test]
fn construct_hc_duty_cycle_tracker_from_custom_call() {
    let (busy, idle) = custom_call_duty_cycle(
        &[("custom_call_0_flops", "custom-call", 10, &[("flops", 0.into())]), ("custom_call_with_flops", "custom-call", 20, &[("flops", 5.into())]), ("custom_call_no_flops", "custom-call", 30, &[])],
        50,
    );
    assert_eq!((busy[1], idle[1]), (10, 40));
}

#[test]
fn construct_hc_duty_cycle_tracker_from_custom_call_with_ici() {
    let (busy, idle) = custom_call_duty_cycle(
        &[
            ("custom_call_with_ici", "custom-call", 10, &[("uses_ici", 1.into())]),
            ("custom_call_no_ici_no_flops", "custom-call", 20, &[("uses_ici", 0.into())]),
            ("custom_call_with_ici_and_flops", "custom-call", 30, &[("uses_ici", 1.into()), ("flops", 5.into())]),
            ("custom_call_no_ici", "custom-call", 40, &[]),
            ("custom_call_with_model_flops", "custom-call", 50, &[("uses_ici", 0.into()), ("model_flops", 5.into())]),
            ("megacore_op", "megacore", 15, &[]),
            ("dot", "dot", 60, &[("uses_ici", 1.into()), ("flops", 10.into())]),
            ("add", "add", 70, &[("uses_ici", 1.into())]),
        ],
        90,
    );
    assert_eq!((busy[1], idle[1]), (55, 35));
}
