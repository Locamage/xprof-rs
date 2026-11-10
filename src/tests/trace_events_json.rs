use super::legacy::same;
use super::xspace::XSpace;
use crate::trace::legacy::render;

#[test]
fn test_json_conversion() {
    let mut space = XSpace::default();
    let first = space.gpu(0);
    first.named_line(2, "R1.2");
    first.event(2, "E1.2.1", 100_000, 10_000, &[("label", "E1.2.1".into()), ("extra", "extra info".into())]);
    let second = space.gpu(1);
    second.named_line(2, "R2.2");
    second.event(2, "E2.2.1", 105_000, 0, &[]);
    let (map, planes) = space.parsed();
    let converted: serde_json::Value = serde_json::from_str(&render(&planes, &map)).unwrap();
    let expected = serde_json::json!({
        "displayTimeUnit": "ns",
        "metadata": {"highres-ticks": true},
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}},
            {"ph": "M", "pid": 1, "name": "process_sort_index", "args": {"sort_index": 1}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "R1.2"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_sort_index", "args": {"sort_index": 2}},
            {"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "/device:GPU:1"}},
            {"ph": "M", "pid": 2, "name": "process_sort_index", "args": {"sort_index": 2}},
            {"ph": "M", "pid": 2, "tid": 2, "name": "thread_name", "args": {"name": "R2.2"}},
            {"ph": "M", "pid": 2, "tid": 2, "name": "thread_sort_index", "args": {"sort_index": 2}},
            {"ph": "X", "pid": 1, "tid": 2, "name": "E1.2.1", "ts": 0.1, "dur": 0.01, "args": {"label": "E1.2.1", "extra": "extra info"}},
            {"ph": "i", "pid": 2, "tid": 2, "name": "E2.2.1", "ts": 0.105, "s": "t"},
            {},
        ]
    });
    assert!(same(&converted, &expected), "{converted}");
}
