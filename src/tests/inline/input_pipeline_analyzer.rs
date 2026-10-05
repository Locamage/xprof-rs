use super::*;
use crate::steps::StepInfo;

#[test]
fn collective_that_ends_before_it_starts() {
    let info = StepInfo { name: String::new(), begin: 0, duration: 1000, breakdown: Breakdown::Categories(BTreeMap::new()) };
    let record = StepRecord { num: 1, cores: BTreeMap::from([(0, info)]), collectives: BTreeMap::from([(0, vec![(1, 100, 50), (2, 10, 20)])]) };
    let step = tpu_step_details(&record, &BTreeMap::new());
    assert_eq!(step.all_reduce_ms, [ms(50u64.wrapping_sub(100).wrapping_add(10)), 0.0]);
}
