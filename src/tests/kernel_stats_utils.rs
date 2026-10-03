use crate::gpu::{KernelKey, KernelReport, by_op_name, launch_params, top_kernels};

fn report(name: &str, op_name: &str, total_ns: u64, tensor_core: bool, eligible: bool) -> KernelReport {
    KernelReport { key: KernelKey { name: name.into(), op_name: op_name.into(), tensor_core, eligible, ..Default::default() }, total_ns, ..Default::default() }
}

#[test]
fn test_group_kernel_reports_by_op_name() {
    let reports = [report("op1_kernel1", "op1", 1000, true, true), report("op1_kernel2", "op1", 1000, false, true), report("op2_kernel1", "op2", 100, false, false)];
    let by_op = by_op_name(&reports);
    assert_eq!(by_op.len(), 2);
    assert_eq!(by_op["op1"], (true, 2000, 1000));
    assert_eq!(by_op["op2"], (false, 100, 0));
}

#[test]
fn kernel_details_x_stat_parser() {
    let (registers, static_shared, dynamic_shared, block, grid, occupancy_pct) = (10, 128, 256, [32, 8, 4], [3, 2, 1], 50.0);
    let details = format!(
        "regs:{registers} static_shared:{static_shared} dynamic_shared:{dynamic_shared} grid:{},{},{} block:{},{},{} occ_pct:{occupancy_pct}",
        grid[0], grid[1], grid[2], block[0], block[1], block[2]
    );
    let (mut kernel, mut occupancy) = (KernelKey::default(), 0.0);
    launch_params(&details, &mut kernel, &mut occupancy);
    assert_eq!((kernel.registers, kernel.static_shmem, kernel.dynamic_shmem), (10, 128, 256));
    assert_eq!(kernel.block, [32, 8, 4]);
    assert_eq!(kernel.grid, [3, 2, 1]);
}

#[test]
fn kernel_details_tokenizer() {
    let (mut kernel, mut occupancy) = (KernelKey::default(), 0.0);
    launch_params("odd grid:3,2,1", &mut kernel, &mut occupancy);
    assert_eq!(kernel.grid, [3, 2, 1]);
    launch_params("block:6,5,4 odd ", &mut kernel, &mut occupancy);
    assert_eq!(kernel.block, [6, 5, 4]);
    launch_params("block:1,2,3 odd grid:4,5,6", &mut kernel, &mut occupancy);
    assert_eq!(kernel.block, [1, 2, 3]);
    assert_eq!(kernel.grid, [4, 5, 6]);
    launch_params("static_shared:7 dynamic_shared:8", &mut kernel, &mut occupancy);
    assert_eq!((kernel.static_shmem, kernel.dynamic_shmem), (7, 8));
}

#[test]
fn test_insert_or_update_kernel_report() {
    let key = KernelKey { name: "op1_kernel1".into(), op_name: "op1".into(), block: [32, 8, 4], grid: [3, 2, 1], ..Default::default() };
    let value = |total_ns, min_ns, max_ns, occurrences| KernelReport { key: key.clone(), total_ns, min_ns, max_ns, occurrences, ..Default::default() };
    let fields = |reports: Vec<KernelReport>| (reports[0].total_ns, reports[0].min_ns, reports[0].max_ns, reports[0].occurrences);
    assert_eq!(fields(top_kernels([value(1700, 500, 1200, 2), value(900, 900, 900, 1)])), (2600, 500, 1200, 3));
    assert_eq!(fields(top_kernels([value(900, 900, 900, 1), value(1700, 500, 1200, 2)])), (2600, 500, 1200, 3));
}
