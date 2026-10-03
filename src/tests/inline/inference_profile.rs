use super::*;
use crate::tests::legacy::{bytes, entry as named, number};

fn host(name: &str, root: Option<i64>) -> Vec<u8> {
    let stats = root.map_or_else(Vec::new, |level| bytes(4, &[number(1, 9), number(4, level as u64)].concat()));
    let line = bytes(3, &bytes(4, &[number(1, 1), number(2, 0), number(3, 10), stats].concat()));
    bytes(1, &[bytes(2, HOST_PLANE.as_bytes()), line, named(4, 1, name), named(5, 9, "_r")].concat())
}

#[test]
fn serving_roots_mark_inference_traces() {
    assert!(serves_requests(&host("TfrtModelRun", None)));
    assert!(serves_requests(&host("custom", Some(-1))));
    assert!(serves_requests(&host("ModelServerRequest_predict", Some(1))));
    assert!(!serves_requests(&host("train_step", Some(1))));
    assert!(!serves_requests(&host("SessionRunner", None)));
}
