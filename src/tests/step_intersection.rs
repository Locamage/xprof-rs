use crate::steps::{Breakdown, Extra, StepInfo, StepRecord, combine};
use std::collections::BTreeMap;

const STEP_DURATION_PS: u64 = 2000000000;
const NUM_STEPS_PER_HOST: u32 = 10;
const STEP_GAP_PS: u64 = 0;
const NUM_CORES_PER_HOST: u32 = 8;

fn create_one_test_step(host_id: u32, num_steps: u32, step_idx: u32, step_begin_ps: u64) -> StepRecord {
    let step_num = step_idx * host_id;
    let duration = if host_id == 0 && step_idx == num_steps - 1 { STEP_DURATION_PS - 1 } else { STEP_DURATION_PS };
    let info = StepInfo { name: String::new(), begin: step_begin_ps, duration, breakdown: Breakdown::Types(BTreeMap::new()) };
    StepRecord { num: step_num, cores: (0..NUM_CORES_PER_HOST).map(|core| (core, info.clone())).collect(), ..Default::default() }
}

fn host(steps: Vec<StepRecord>) -> Extra {
    Extra { device_type: "TPU".into(), steps, ..Default::default() }
}

fn create_step_database_result_for_host(host_id: u32, num_steps: u32, first_step_begin_ps: u64, step_gap_ps: u64) -> Extra {
    host((0..num_steps).map(|step_idx| create_one_test_step(host_id, num_steps, step_idx, first_step_begin_ps + step_idx as u64 * (STEP_DURATION_PS + step_gap_ps))).collect())
}

fn create_test_steps(num_hosts: u32, num_steps_per_host: u32, shift_ps: u64) -> Vec<Extra> {
    (0..num_hosts).map(|host_id| create_step_database_result_for_host(host_id, num_steps_per_host, host_id as u64 * shift_ps, STEP_GAP_PS)).collect()
}

fn intersection(perhost_stepdb: &[Extra]) -> Extra {
    combine(&perhost_stepdb.iter().collect::<Vec<_>>())
}

fn first_step_index(intersection: &Extra, host_id: u32) -> u64 {
    intersection.steps[0].cores[&(host_id * 1000)].begin / STEP_DURATION_PS
}

fn dst_step_numbers(intersection: &Extra) -> Vec<u32> {
    intersection.steps.iter().map(|step| step.num).collect()
}

fn single_core_step(duration: u64) -> Extra {
    let info = StepInfo { name: String::new(), begin: 0, duration, breakdown: Breakdown::Types(BTreeMap::new()) };
    host(vec![StepRecord { num: 0, cores: BTreeMap::from([(0, info)]), ..Default::default() }])
}

#[test]
fn each_host_shifted_by_1_step_duration() {
    let num_hosts = 4;
    let intersection = intersection(&create_test_steps(num_hosts, NUM_STEPS_PER_HOST, STEP_DURATION_PS));
    let dst_num_steps = NUM_STEPS_PER_HOST - num_hosts + 1;
    assert_eq!(intersection.steps.len() as u32, dst_num_steps);
    assert_eq!(first_step_index(&intersection, 0), (num_hosts - 1) as u64);
    assert_eq!(dst_step_numbers(&intersection), (0..dst_num_steps).collect::<Vec<_>>());
}

#[test]
fn exactly_no_shift() {
    let num_hosts = 4;
    let intersection = intersection(&create_test_steps(num_hosts, NUM_STEPS_PER_HOST, 0));
    assert_eq!(intersection.steps.len() as u32, NUM_STEPS_PER_HOST);
    assert_eq!(dst_step_numbers(&intersection), (0..NUM_STEPS_PER_HOST).collect::<Vec<_>>());
    for host_id in 0..num_hosts {
        assert_eq!(first_step_index(&intersection, host_id), 0);
    }
}

#[test]
fn each_host_shifted_by_just_a_bit() {
    let num_hosts = 4;
    let intersection = intersection(&create_test_steps(num_hosts, NUM_STEPS_PER_HOST, 100));
    assert_eq!(intersection.steps.len() as u32, NUM_STEPS_PER_HOST);
    assert_eq!(dst_step_numbers(&intersection), (0..NUM_STEPS_PER_HOST).collect::<Vec<_>>());
    for host_id in 0..num_hosts {
        assert_eq!(intersection.steps[0].cores[&(host_id * 1000)].begin, host_id as u64 * 100);
    }
}

#[test]
fn single_host() {
    let intersection = intersection(&create_test_steps(1, NUM_STEPS_PER_HOST, 0));
    assert_eq!(intersection.steps.len() as u32, NUM_STEPS_PER_HOST);
    assert_eq!(dst_step_numbers(&intersection), (0..NUM_STEPS_PER_HOST).collect::<Vec<_>>());
    assert_eq!(first_step_index(&intersection, 0), 0);
}

#[test]
fn no_step() {
    let intersection = intersection(&(0..4).map(|_| host(Vec::new())).collect::<Vec<_>>());
    assert_eq!(intersection.steps.len(), 0);
    assert!(!intersection.empty_intersect);
}

#[test]
fn empty_intersection() {
    let host_0_num_steps = 10;
    let host_1_num_steps = 5;
    let perhost_stepdb = [
        create_step_database_result_for_host(0, host_0_num_steps, 0, STEP_GAP_PS),
        create_step_database_result_for_host(1, host_1_num_steps, (host_0_num_steps - 2) as u64 * (STEP_DURATION_PS + STEP_GAP_PS), STEP_GAP_PS),
        create_step_database_result_for_host(2, 10, (host_0_num_steps + host_1_num_steps - 4) as u64 * (STEP_DURATION_PS + STEP_GAP_PS), STEP_GAP_PS),
    ];
    let intersection = intersection(&perhost_stepdb);
    assert_eq!(intersection.steps.len(), 0);
    assert!(intersection.empty_intersect);
}

#[test]
fn unequal_step_counts() {
    let intersection = intersection(&[create_step_database_result_for_host(0, 5, 0, STEP_GAP_PS), create_step_database_result_for_host(1, 10, 0, STEP_GAP_PS)]);
    assert_eq!(intersection.steps.len(), 5);
}

#[test]
fn one_host_empty() {
    let intersection = intersection(&[create_step_database_result_for_host(0, 5, 0, STEP_GAP_PS), host(Vec::new())]);
    assert_eq!(intersection.steps.len(), 0);
    assert!(!intersection.empty_intersect);
}

#[test]
fn varying_step_durations() {
    let steps = (0..5)
        .map(|index| {
            let mut step = create_one_test_step(0, 5, index, index as u64 * 2 * STEP_DURATION_PS);
            step.cores.values_mut().for_each(|info| info.duration = 2 * STEP_DURATION_PS);
            step
        })
        .collect();
    let intersection = intersection(&[host(steps), create_step_database_result_for_host(1, 10, 0, STEP_GAP_PS)]);
    assert!(!intersection.steps.is_empty());
}

#[test]
fn gaps_between_steps() {
    let intersection = intersection(&[create_step_database_result_for_host(0, 5, 0, STEP_DURATION_PS), create_step_database_result_for_host(1, 5, 0, STEP_DURATION_PS)]);
    assert_eq!(intersection.steps.len(), 5);
}

#[test]
fn empty_per_host_step_db() {
    let intersection = intersection(&[]);
    assert_eq!(intersection.steps.len(), 0);
    assert!(!intersection.empty_intersect);
}

#[test]
fn step_with_steps_but_no_cores() {
    let intersection = intersection(&[host(vec![StepRecord::default()])]);
    assert_eq!(intersection.steps.len(), 1);
}

#[test]
fn zero_duration_steps() {
    let perhost_stepdb: Vec<Extra> = (0..2)
        .map(|host_id| {
            let mut zero_duration_step = create_one_test_step(host_id, 3, 1, STEP_DURATION_PS);
            zero_duration_step.cores.values_mut().for_each(|info| info.duration = 0);
            host(vec![create_one_test_step(host_id, 3, 0, 0), zero_duration_step, create_one_test_step(host_id, 3, 2, STEP_DURATION_PS)])
        })
        .collect();
    assert_eq!(intersection(&perhost_stepdb).steps.len(), 3);
}

#[test]
fn one_host_with_no_steps() {
    let intersection = intersection(&[create_step_database_result_for_host(0, 5, 0, STEP_GAP_PS), host(Vec::new())]);
    assert_eq!(intersection.steps.len(), 0);
    assert!(!intersection.empty_intersect);
}

#[test]
fn single_core_steps() {
    assert_eq!(intersection(&[single_core_step(STEP_DURATION_PS)]).steps.len(), 1);
}

#[test]
fn no_positive_duration_steps() {
    assert_eq!(intersection(&[single_core_step(0)]).steps.len(), 1);
}
