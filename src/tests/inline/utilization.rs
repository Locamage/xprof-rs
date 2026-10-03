use super::*;

fn counters(entries: &[(u64, u64)]) -> Counters {
    entries.iter().copied().collect()
}

fn find<'a>(metrics: &'a [Metric], name: &str) -> &'a Metric {
    metrics.iter().find(|metric| metric.name == name).unwrap()
}

#[test]
fn v6e_tensor_core_metrics() {
    let id = |name: &str| Device::V6e.id(&format!("{TC_V6E}{name}"));
    let values = counters(&[
        (id("CYCLES"), 1000),
        (id("SCALAR_ALU_INSTRUCTION_0"), 300),
        (id("SCALAR_ALU_INSTRUCTION_1"), 200),
        (id("MXU_BUSY_1"), 100),
        (id("MXU_BUSY_2"), 50),
        (id("MATMUL_LMR_BF16_MXU_0"), 10),
        (id("MATMUL_VREG_I8_MXU_1"), 3),
        (id("XLU_BUSY_1"), 8),
        (id("TRANSPOSE_XLU_0"), 7),
    ]);
    let mut metrics = Vec::new();
    tensor_core(&values, Device::V6e, 0, &mut metrics);
    assert_eq!(metrics.len(), 21);
    assert_eq!(metrics[0], Metric { node: 0, name: "Scalar Unit".into(), achieved: 500.0, peak: 2000.0, unit: INSTRUCTIONS });
    assert_eq!(find(&metrics, "Avg MXU Busy").achieved, 100.0);
    assert_eq!(find(&metrics, "MXU0").achieved, 40.0);
    assert_eq!(find(&metrics, "MXU1").achieved, 24.0);
    assert_eq!(find(&metrics, "MXU BF16").peak, 2000.0);
    assert_eq!(find(&metrics, "MXU I8").achieved, 24.0);
    assert_eq!(find(&metrics, "XLU0").achieved, 7.0);
    assert_eq!(find(&metrics, "XLU0").peak, 250.0);
    assert_eq!(find(&metrics, "Avg XLU Busy").achieved, 4.0);
    let mut empty = Vec::new();
    tensor_core(&counters(&[]), Device::V6e, 0, &mut empty);
    assert!(empty.is_empty());
}

#[test]
fn v7x_tensor_core_uses_power_manager_cycles_per_die() {
    let power = Device::V7x.id("VF_CHIP_DIE1_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT");
    let skipped = Device::V7x.id("VF_CHIP_DIE1_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CLOCKS_SKIPPED");
    let f8 = Device::V7x.id("VF_CHIP_DIE1_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_MATMUL_VREG_F8_MXU_0");
    let values = counters(&[(power, 400), (skipped, 4), (f8, 5)]);
    let mut metrics = Vec::new();
    tensor_core(&values, Device::V7x, 0, &mut metrics);
    assert!(metrics.is_empty());
    tensor_core(&values, Device::V7x, 1, &mut metrics);
    assert_eq!(metrics[0], Metric { node: 1, name: "Clocks Skipped".into(), achieved: 4.0, peak: 400.0, unit: CYCLES });
    assert_eq!(find(&metrics, "MXU E4M3 + E5M2").achieved, 40.0);
    assert_eq!(find(&metrics, "XLU1").peak, 400.0);
}

#[test]
fn bandwidth_and_ici() {
    let cycles = Device::V6e.id(&format!("{TC_V6E}CYCLES"));
    let read = Device::V6e.id("VF_CHIP_HBM_1_HBMC_15_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_RD_RESP_PS1");
    let write = Device::V6e.id("VF_CHIP_HBM_0_HBMC_3_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_PARTIAL_WRITE_REQ_PS0");
    let ici_read = Device::V6e.id("VF_CHIP_OCI_ICR_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_RD_REQ_1_INSERTION_COUNT");
    let values = counters(&[(cycles, 1_750_000), (read, 3), (write, 1), (ici_read, 2)]);
    let mut metrics = Vec::new();
    bandwidth(&values, Device::V6e, 0, &mut metrics);
    ici(&values, Device::V6e, &mut metrics);
    assert_eq!(metrics[0], Metric { node: 0, name: "HBM Rd+Wr - core 0".into(), achieved: 128.0, peak: 1525.5 * GIB * (1_750_000.0 / 1.75e9), unit: BYTES });
    assert_eq!(metrics[1].achieved, 96.0);
    assert_eq!(metrics[2].achieved, 32.0);
    assert_eq!(metrics[3], Metric { node: 0, name: "ICI (Read)".into(), achieved: 256.0, peak: PEAK_ICI_BYTES_PER_SECOND * (1_750_000.0 / 1e9), unit: BYTES });
    assert_eq!(metrics[4].achieved, 0.0);
    assert_eq!(hbm_counters(Device::V6e, 0, true).len(), 64);
    assert_eq!(hbm_counters(Device::V6e, 0, false).len(), 128);
    assert_eq!(hbm_counters(Device::V7x, 1, true).len(), 128);
}

#[test]
fn sparse_core_metrics() {
    let tile = |tile: usize, name: &str| Device::V6e.id(&format!("VF_CHIP_SC_1_SCT_{tile}{SC_SUFFIX}{name}"));
    let values = counters(&[(tile(0, "CYCLES"), 10), (tile(3, "VEX_INSTRUCTION"), 4), (tile(15, "VEX_INSTRUCTION"), 5), (Device::V6e.id(&format!("VF_CHIP_SC_1_SCS{SC_SUFFIX}CYCLES")), 9)]);
    let mut metrics = Vec::new();
    sparse_core(&values, Device::V6e, 0, 1, &mut metrics);
    assert_eq!(metrics.len(), 16);
    assert_eq!(*find(&metrics, "SC Vector Issue:VEX Slot"), Metric { node: 1, name: "SC Vector Issue:VEX Slot".into(), achieved: 9.0, peak: 160.0, unit: INSTRUCTIONS });
    assert_eq!(find(&metrics, "SCS: Smisc Slot").peak, 9.0);
    let sct = Device::V7x.id(&format!("VF_CHIP_DIE1_SC_0_SCTC_2{SC_SUFFIX}V1_INSTRUCTION"));
    let mut v7 = Vec::new();
    sparse_core(&counters(&[(sct, 6)]), Device::V7x, 1, 0, &mut v7);
    assert_eq!(*find(&v7, "SC Vector Issue: V1 Slot"), Metric { node: 2, name: "SC Vector Issue: V1 Slot".into(), achieved: 6.0, peak: 0.0, unit: INSTRUCTIONS });
}

#[test]
fn compute_orders_units() {
    let metrics = compute(&counters(&[]), Device::V7x);
    assert_eq!(metrics.len(), 64);
    assert_eq!(metrics[0].name, "TEC Scalar Issue");
    assert_eq!(metrics[63].node, 3);
    assert_eq!(compute(&counters(&[]), Device::V6e).len(), 32);
}
