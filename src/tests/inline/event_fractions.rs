use super::*;
use crate::tests::legacy::{bytes, entry as named, number};

const GROUP: u64 = 1;
const DURATION: u64 = 2;

fn event(meta: u64, offset: u64, duration: u64, group: u64, device_ps: Option<u64>) -> Vec<u8> {
    let mut stats = bytes(4, &[number(1, GROUP), number(4, group)].concat());
    if let Some(ps) = device_ps {
        stats.extend(bytes(4, &[number(1, DURATION), number(3, ps)].concat()));
    }
    bytes(4, &[number(1, meta), number(2, offset), number(3, duration), stats].concat())
}

fn tpu_plane(name: &str, op: &str, steps: &[(u64, u64)], ops: &[(u64, u64, u64)]) -> Vec<u8> {
    let step_line = [number(1, 0), bytes(2, b"Steps"), steps.iter().flat_map(|&(group, duration)| event(1, 0, duration, group, None)).collect()].concat();
    let op_line = [number(1, 1), bytes(2, b"XLA Ops"), ops.iter().flat_map(|&(group, offset, duration)| event(2, offset, duration, group, Some(duration))).collect()].concat();
    let body = [bytes(2, name.as_bytes()), bytes(3, &step_line), bytes(3, &op_line), named(4, 1, "step"), named(4, 2, op), named(5, GROUP, "group_id"), named(5, DURATION, "device_duration_ps")];
    bytes(1, &body.concat())
}

fn run(space: &[u8], host: &str, target: &str) -> Fractions {
    analyze(&crate::xplane::parse(space).unwrap(), space, host, target)
}

#[test]
fn edge_steps_are_trimmed() {
    let ops: Vec<(u64, u64, u64)> = (1..=4).map(|step| (step, step * 1000, step * 10)).collect();
    let steps: Vec<(u64, u64)> = (1..=4).map(|step| (step, 1000)).collect();
    assert_eq!(run(&tpu_plane("/device:TPU:0", "fusion", &steps, &ops), "h", "fusion").chips["/device:TPU:0"], [0.02f32, 0.03]);
    let short = run(&tpu_plane("/device:TPU:0", "fusion", &[(1, 100_000), (2, 900)], &[(1, 0, 1000), (2, 5000, 90)]), "h", "fusion");
    assert_eq!(short.chips["/device:TPU:0"], [0.01f32]);
    let unmarked = run(&tpu_plane("/device:TPU:0", "fusion", &[], &[(1, 0, 1000)]), "h", "fusion");
    assert!(unmarked.chips.is_empty() && unmarked.hosts["h"].is_empty());
}
