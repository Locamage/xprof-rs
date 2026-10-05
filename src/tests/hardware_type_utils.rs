use super::xspace::{V, XSpace, op_stats};
use crate::xplane::steps::GPU;

fn device(stats: &[(&str, V)]) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.gpu(0);
    for (name, value) in stats {
        plane.add_stat(name, value.clone());
    }
    space
}

fn perf(space: &XSpace) -> crate::tools::opstats::Perf {
    let (_, planes) = space.parsed();
    crate::xplane::gpu::perf_env(&planes[0])
}

fn peak_tflops(space: &XSpace) -> f64 {
    perf(space).peak_tera_flops
}

fn aggregate_shared_memory_giga_bytes_per_second(space: &XSpace) -> f64 {
    perf(space).bandwidths[1]
}

fn gpu_model_name(space: &XSpace) -> String {
    let (_, planes) = space.parsed();
    crate::xplane::gpu::model_name(&planes[0])
}

fn near(actual: f64, expected: f64, abs_error: f64) {
    assert!((actual - expected).abs() <= abs_error, "{actual} != {expected}");
}

fn giga_to_uni(giga: f64) -> u64 {
    (giga * 1e9) as u64
}

fn nvidia(clock: i64, cores: i64, bandwidth_tib: f64, major: i64, minor: i64) -> XSpace {
    device(&[
        ("clock_rate", clock.into()),
        ("core_count", cores.into()),
        ("memory_bandwidth", giga_to_uni(bandwidth_tib * 1024.0).into()),
        ("device_vendor", "Nvidia".into()),
        ("compute_cap_major", major.into()),
        ("compute_cap_minor", minor.into()),
    ])
}

fn amd(clock: i64, cores: i64, arch: &str) -> XSpace {
    device(&[("clock_rate", clock.into()), ("core_count", cores.into()), ("device_vendor", "AMD".into()), ("gpu_device_name", arch.into())])
}

macro_rules! peaks { ($($name:ident: $space:expr => $peak:expr;)*) => { $(#[test] fn $name() { near(peak_tflops(&$space), $peak, 1.0); })* } }

peaks! {
    b200_peak_comput_t_flops: nvidia(1830000, 148, 7.68, 10, 0) => 2218.0;
    future_blackwell_peak_comput_t_flops: nvidia(1830000, 148, 7.68, 10, 9) => 2218.0;
    h100_peak_comput_t_flops: nvidia(1620000, 114, 2.04, 9, 0) => 756.0;
    a100_peak_comput_t_flops: nvidia(1410000, 108, 2.04, 8, 0) => 312.0;
    mi300_x_peak_compute_t_flops: amd(2100000, 304, "gfx942") => 1307.4;
    mi100_peak_compute_t_flops: amd(1502000, 120, "gfx908") => 184.6;
    mi250_peak_compute_t_flops_per_gcd: amd(1700000, 104, "gfx90a") => 181.0;
    mi355_x_peak_compute_t_flops: amd(2400000, 256, "gfx950") => 2516.6;
}

#[test]
fn ambiguous_amd_compute_capability_reports_no_peak() {
    let space = device(&[("clock_rate", 1700000.into()), ("core_count", 104.into()), ("device_vendor", "AMD".into()), ("compute_cap_major", 9.into()), ("compute_cap_minor", 0.into())]);
    assert_eq!(peak_tflops(&space), 0.0);
}

#[test]
fn mi300_x_shared_memory_bandwidth() {
    let space = amd(2100000, 304, "gfx942");
    near(aggregate_shared_memory_giga_bytes_per_second(&space), 81715.2, 1.0);
}

#[test]
fn mi355_x_shared_memory_bandwidth_doubles() {
    let space = amd(2400000, 256, "gfx950");
    near(aggregate_shared_memory_giga_bytes_per_second(&space), 157286.4, 1.0);
}

#[test]
fn shared_memory_bandwidth_from_compute_capability() {
    let space = device(&[("clock_rate", 2100000.into()), ("core_count", 304.into()), ("device_vendor", "AMD".into()), ("compute_cap_major", 9.into()), ("compute_cap_minor", 4.into())]);
    near(aggregate_shared_memory_giga_bytes_per_second(&space), 81715.2, 1.0);
}

#[test]
fn shared_memory_bandwidth_unknown_amd_arch_reports_nothing() {
    let space = device(&[("clock_rate", 1700000.into()), ("core_count", 104.into()), ("device_vendor", "AMD".into()), ("compute_cap_major", 9.into()), ("compute_cap_minor", 0.into())]);
    assert_eq!(aggregate_shared_memory_giga_bytes_per_second(&space), 0.0);
}

#[test]
fn gpu_model_name_ignores_nvidia_product_name() {
    let space = device(&[("device_vendor", "Nvidia".into()), ("compute_cap_major", 9.into()), ("gpu_device_name", "NVIDIA H100 80GB HBM3".into())]);
    assert_eq!(gpu_model_name(&space), "Nvidia GPU (Hopper)");
    let stats = op_stats(&[space]).unwrap();
    assert_eq!((stats.extra.device_type.as_str(), stats.extra.hardware), ("Nvidia GPU (Hopper)", GPU));
}

#[test]
fn gpu_model_name_falls_back_to_family_without_name() {
    let space = device(&[("device_vendor", "Nvidia".into()), ("compute_cap_major", 9.into())]);
    assert_eq!(gpu_model_name(&space), "Nvidia GPU (Hopper)");
}

#[test]
fn gpu_model_name_reports_amd_arch_from_device_name() {
    let space = device(&[("device_vendor", "AMD".into()), ("gpu_device_name", "gfx950".into())]);
    assert_eq!(gpu_model_name(&space), "AMD GPU - gfx950");
    let stats = op_stats(&[space]).unwrap();
    assert_eq!((stats.extra.device_type.as_str(), stats.extra.hardware), ("AMD GPU - gfx950", GPU));
}

#[test]
fn gpu_model_name_rejects_malformed_amd_device_name() {
    let space = device(&[("device_vendor", "AMD".into()), ("gpu_device_name", "gfx942:sramecc+:xnack-".into())]);
    assert_eq!(gpu_model_name(&space), "AMD GPU");
}

#[test]
fn gpu_model_name_resolves_amd_arch_from_compute_capability() {
    let space = device(&[("device_vendor", "AMD".into()), ("compute_cap_major", 9.into()), ("compute_cap_minor", 4.into())]);
    assert_eq!(gpu_model_name(&space), "AMD GPU - gfx942");
}

#[test]
fn gpu_model_name_falls_back_to_amd_family_without_name() {
    let space = device(&[("device_vendor", "AMD".into()), ("compute_cap_major", 9.into()), ("compute_cap_minor", 0.into())]);
    assert_eq!(gpu_model_name(&space), "AMD GPU - gfx-9XX series");
}
