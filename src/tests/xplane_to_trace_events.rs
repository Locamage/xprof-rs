use super::xspace::XSpace;
use crate::trace::legacy::render;

fn converted(space: &XSpace) -> Vec<serde_json::Value> {
    let (map, planes) = space.parsed();
    serde_json::from_str::<serde_json::Value>(&render(&planes, &map)).unwrap()["traceEvents"].as_array().unwrap().clone()
}

#[test]
fn convert() {
    let mut space = XSpace::default();
    let host = space.host();
    host.named_line(10, "thread1");
    host.event(10, "event1", 150_000_000, 10_000_000, &[("tf_op", super::xspace::V::Ref("Relu".into()))]);
    host.named_line(20, "thread2");
    host.event(20, "event2", 160_000_000, 10_000_000, &[("tf_op", super::xspace::V::Ref("Conv2D".into()))]);
    let device = space.gpu(0);
    device.id = 0;
    device.named_line(30, "gpu stream 1");
    device.event(30, "kernel1", 180_000_000, 10_000_000, &[("correlation id", 55.into())]);
    let events = converted(&space);
    let threads = |pid: u64| events.iter().filter(|event| event["name"] == "thread_name" && event["pid"] == pid).count();
    assert_eq!(events.iter().filter(|event| event["name"] == "process_name").count(), 2);
    assert_eq!((threads(701), threads(1)), (2, 1));
    assert_eq!(events.iter().filter(|event| event["ph"] == "X").count(), 3);
}

#[test]
fn skip_async_ops() {
    let mut space = XSpace::default();
    let device = space.gpu(0);
    device.named_line(10, "Async XLA Ops");
    device.event(10, "event1", 100_000, 1_000, &[]);
    assert!(converted(&space).iter().all(|event| event["ph"] != "X"));
}
