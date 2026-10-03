use super::xspace::XSpace;
use crate::event_fractions::{BARRIER_CORES, Fractions, accumulate, analyze};

type Op = (i64, i64, u64);

fn tpu_space(plane_name: &str, op_name: &str, step_duration_ps: i64, ops: &[Op]) -> XSpace {
    let mut xspace = XSpace::default();
    let plane = xspace.plane(plane_name);
    plane.id = 0;
    plane.named_line(0, "Steps");
    plane.event(0, "step", 0, step_duration_ps, &[("group_id", 1.into())]);
    plane.named_line(1, "XLA Ops");
    for &(offset_ps, duration_ps, device_duration_ps) in ops {
        plane.event(1, op_name, offset_ps, duration_ps, &[("group_id", 1.into()), ("device_duration_ps", device_duration_ps.into())]);
    }
    xspace
}

fn convert(xspace: &XSpace, hostname: &str, target: &str) -> Fractions {
    let (map, planes) = xspace.parsed();
    analyze(&planes, &map, hostname, target)
}

fn near(actual: f32, expected: f64) {
    assert!((f64::from(actual) - expected).abs() <= 1e-6, "{actual} != {expected}");
}

#[test]
fn basic_test() {
    let result = convert(&tpu_space("/device:TPU:0", "matmul", 1000, &[(100, 200, 200), (400, 100, 100)]), "", "matmul");
    assert_eq!(result.chips.len(), 1);
    let fractions = &result.chips["/device:TPU:0"];
    assert_eq!(fractions.len(), 1);
    near(fractions[0], 0.3);
}

#[test]
fn barrier_cores_filtering() {
    let result = convert(&tpu_space("/device:TPU:0", BARRIER_CORES, 10000, &[(100, 0, 0), (200, 1250, 1250), (300, 5000, 5000)]), "", BARRIER_CORES);
    assert_eq!(result.chips.len(), 1);
    let fractions = &result.chips["/device:TPU:0"];
    assert_eq!(fractions.len(), 1);
    near(fractions[0], 0.5);
}

#[test]
fn multi_x_space_test() {
    let mut result = Fractions::default();
    for (hostname, plane_name, step_duration_ps, op) in [("host1", "/device:TPU:0", 1000, (100, 200, 200)), ("host2", "/device:TPU:1", 2000, (100, 1000, 1000))] {
        let xspace = XSpace { hostnames: vec![hostname.into()], ..tpu_space(plane_name, "matmul", step_duration_ps, &[op]) };
        accumulate(convert(&xspace, hostname, "matmul"), &mut result);
    }
    assert_eq!(result.chips.len(), 2);
    near(result.chips["/device:TPU:0"][0], 0.2);
    near(result.chips["/device:TPU:1"][0], 0.5);
    assert_eq!(result.hosts.len(), 2);
    near(result.hosts["host1"][0], 0.2);
    near(result.hosts["host2"][0], 0.5);
}
