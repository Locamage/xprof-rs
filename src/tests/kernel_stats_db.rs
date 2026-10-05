use super::xspace::XSpace;
use crate::xplane::gpu::{Infos, device_plane, top_kernels};

#[test]
fn multi_kernels() {
    let mut space = XSpace::default();
    let device_trace = space.add_plane();
    device_trace.line(0);
    device_trace.event(
        0,
        "kernel_name_shortest",
        10000,
        1000,
        &[("tf_op", "mul_786".into()), ("kernel_details", "regs:16\nstatic_shared:0\ndynamic_shared:0\ngrid:1,1,1\nblock:1,1,1\nocc_pct:50.0".into()), ("equation", "".into())],
    );
    device_trace.event(
        0,
        "kernel_name_middle",
        20000,
        2000,
        &[("tf_op", "Conv2D".into()), ("kernel_details", "regs:32\nstatic_shared:0\ndynamic_shared:16384\ngrid:2,1,1\nblock:32,1,1\nocc_pct=13.0".into()), ("equation", "".into())],
    );
    device_trace.event(
        0,
        "volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_tn",
        30000,
        3000,
        &[("tf_op", "Einsum_80".into()), ("kernel_details", "regs:32\nstatic_shared:0\ndynamic_shared:16384\ngrid:3,1,1\nblock:64,1,1\nocc_pct:25.0".into()), ("equation", "".into())],
    );
    let (map, planes) = space.parsed();
    let kernel_stats = top_kernels(device_plane(&planes[0], &map, &Infos::default(), 0).1);
    assert_eq!(kernel_stats.len(), 3);
    let expected = [
        ("volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_tn", 32, 0, 16384, [3, 1, 1], [64, 1, 1], 3, true, true, "Einsum_80"),
        ("kernel_name_middle", 32, 0, 16384, [2, 1, 1], [32, 1, 1], 2, false, true, "Conv2D"),
        ("kernel_name_shortest", 16, 0, 0, [1, 1, 1], [1, 1, 1], 1, false, false, "mul_786"),
    ];
    for (kernel, expected) in kernel_stats.iter().zip(expected) {
        let key = &kernel.key;
        assert_eq!((key.name.as_str(), key.registers, key.static_shmem, key.dynamic_shmem, key.grid, key.block, kernel.total_ns, key.tensor_core, key.eligible, key.op_name.as_str()), expected);
    }
}
