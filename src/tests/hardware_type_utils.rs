use super::xspace::{V, XSpace, op_stats};
use crate::steps::GPU;

fn device(stats: &[(&str, V)]) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.gpu(0);
    for (name, value) in stats {
        plane.add_stat(name, value.clone());
    }
    space
}

fn perf(space: &XSpace) -> crate::opstats::Perf {
    let (_, planes) = space.parsed();
    crate::gpu::perf_env(&planes[0])
}

fn peak_tflops(space: &XSpace) -> f64 {
    perf(space).peak_tera_flops
}

fn aggregate_shared_memory_giga_bytes_per_second(space: &XSpace) -> f64 {
    perf(space).bandwidths[1]
}

fn gpu_model_name(space: &XSpace) -> String {
    let (_, planes) = space.parsed();
    crate::gpu::model_name(&planes[0])
}

fn near(actual: f64, expected: f64, abs_error: f64) {
    assert!((actual - expected).abs() <= abs_error, "{actual} != {expected}");
}

fn giga_to_uni(giga: f64) -> u64 {
    (giga * 1e9) as u64
}

#[test]
fn b200_peak_comput_t_flops() {
    let space = device(&[
        ("clock_rate", 1830000.into()),
        ("core_count", 148.into()),
        ("memory_bandwidth", giga_to_uni(7.68 * 1024.0).into()),
        ("device_vendor", "Nvidia".into()),
        ("compute_cap_major", 10.into()),
        ("compute_cap_minor", 0.into()),
    ]);
    near(peak_tflops(&space), 2218.0, 1.0);
}

#[test]
fn future_blackwell_peak_comput_t_flops() {
    let space = device(&[
        ("clock_rate", 1830000.into()),
        ("core_count", 148.into()),
        ("memory_bandwidth", giga_to_uni(7.68 * 1024.0).into()),
        ("device_vendor", "Nvidia".into()),
        ("compute_cap_major", 10.into()),
        ("compute_cap_minor", 9.into()),
    ]);
    near(peak_tflops(&space), 2218.0, 1.0);
}

#[test]
fn h100_peak_comput_t_flops() {
    let space = device(&[
        ("clock_rate", 1620000.into()),
        ("core_count", 114.into()),
        ("memory_bandwidth", giga_to_uni(2.04 * 1024.0).into()),
        ("device_vendor", "Nvidia".into()),
        ("compute_cap_major", 9.into()),
        ("compute_cap_minor", 0.into()),
    ]);
    near(peak_tflops(&space), 756.0, 1.0);
}

#[test]
fn a100_peak_comput_t_flops() {
    let space = device(&[
        ("clock_rate", 1410000.into()),
        ("core_count", 108.into()),
        ("memory_bandwidth", giga_to_uni(2.04 * 1024.0).into()),
        ("device_vendor", "Nvidia".into()),
        ("compute_cap_major", 8.into()),
        ("compute_cap_minor", 0.into()),
    ]);
    near(peak_tflops(&space), 312.0, 1.0);
}

#[test]
fn mi300_x_peak_compute_t_flops() {
    let space = device(&[("clock_rate", 2100000.into()), ("core_count", 304.into()), ("device_vendor", "AMD".into()), ("gpu_device_name", "gfx942".into())]);
    near(peak_tflops(&space), 1307.4, 1.0);
}

#[test]
fn mi100_peak_compute_t_flops() {
    let space = device(&[("clock_rate", 1502000.into()), ("core_count", 120.into()), ("device_vendor", "AMD".into()), ("gpu_device_name", "gfx908".into())]);
    near(peak_tflops(&space), 184.6, 1.0);
}

#[test]
fn mi250_peak_compute_t_flops_per_gcd() {
    let space = device(&[("clock_rate", 1700000.into()), ("core_count", 104.into()), ("device_vendor", "AMD".into()), ("gpu_device_name", "gfx90a".into())]);
    near(peak_tflops(&space), 181.0, 1.0);
}

#[test]
fn mi355_x_peak_compute_t_flops() {
    let space = device(&[("clock_rate", 2400000.into()), ("core_count", 256.into()), ("device_vendor", "AMD".into()), ("gpu_device_name", "gfx950".into())]);
    near(peak_tflops(&space), 2516.6, 1.0);
}

#[test]
fn ambiguous_amd_compute_capability_reports_no_peak() {
    let space = device(&[("clock_rate", 1700000.into()), ("core_count", 104.into()), ("device_vendor", "AMD".into()), ("compute_cap_major", 9.into()), ("compute_cap_minor", 0.into())]);
    assert_eq!(peak_tflops(&space), 0.0);
}

#[test]
fn mi300_x_shared_memory_bandwidth() {
    let space = device(&[("clock_rate", 2100000.into()), ("core_count", 304.into()), ("device_vendor", "AMD".into()), ("gpu_device_name", "gfx942".into())]);
    near(aggregate_shared_memory_giga_bytes_per_second(&space), 81715.2, 1.0);
}

#[test]
fn mi355_x_shared_memory_bandwidth_doubles() {
    let space = device(&[("clock_rate", 2400000.into()), ("core_count", 256.into()), ("device_vendor", "AMD".into()), ("gpu_device_name", "gfx950".into())]);
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
