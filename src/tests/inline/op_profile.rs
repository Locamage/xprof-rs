use super::*;

#[test]
fn partial_sort_that_keeps_no_items() {
    let keys = [3, 1, 2];
    let mut items = vec![0, 1, 2];
    partial_sort(&mut items, 0, |a, b| keys[a] < keys[b]);
    assert_eq!(items, [1, 0, 2]);
    partial_sort(&mut [], 0, |a, b| keys[a] < keys[b]);
}
