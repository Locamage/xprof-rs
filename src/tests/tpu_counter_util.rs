use crate::utilization::{Counters, count};

#[test]
fn basic_functionality() {
    let counters: Counters = [(1, 100), (2, 200)].into_iter().collect();
    assert_eq!(count(&counters, 1), 100);
    assert_eq!(count(&counters, 2), 200);
    assert_eq!(count(&counters, 3), 0);
}
