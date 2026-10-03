use super::xspace::{V, XSpace, grouped};
use serde_json::Value;

#[test]
fn one_allocator_multi_activities_test() {
    let mut space = XSpace::default();
    let host = space.host();
    let counters = |reserved: i64, allocated: i64, available: i64, peak: i64, requested: i64, allocation: i64, address: i64, step: i64, data_type: i64| {
        [
            ("bytes_reserved", V::from(reserved)),
            ("bytes_allocated", allocated.into()),
            ("bytes_available", available.into()),
            ("peak_bytes_in_use", peak.into()),
            ("requested_bytes", requested.into()),
            ("allocation_bytes", allocation.into()),
            ("addr", address.into()),
            ("id", step.into()),
            ("data_type", data_type.into()),
        ]
    };
    let named =
        |counters: [(&'static str, V); 9], names: &[(&'static str, &str)]| -> Vec<(&'static str, V)> { counters.into_iter().chain(names.iter().map(|&(key, text)| (key, V::from(text)))).collect() };
    let first =
        named(counters(2000, 3000, 5000, 8500, 200, 256, 222333, -93746, 1), &[("allocator_name", "GPU_0_bfc"), ("tf_op", "foo/bar"), ("region_type", "output"), ("shape", "[3, 3, 512, 512]")]);
    host.event(0, "MemoryAllocation", 40000, 1000, &first);
    let second = named(counters(2000, 2744, 5256, 8500, 200, 256, 222333, 0, 0), &[("allocator_name", "GPU_0_bfc"), ("region_type", ""), ("shape", "")]);
    host.event(0, "MemoryDeallocation", 50000, 1000, &second);
    let third = named(counters(2000, 5000, 3000, 9500, 300, 300, 345678, -93746, 9), &[("allocator_name", "GPU_0_bfc"), ("tf_op", "mul_grad/Sum"), ("region_type", "temp"), ("shape", "[1, 2]")]);
    host.event(0, "MemoryAllocation", 70000, 1000, &third);
    let (map, planes, _) = grouped(&space);
    let profile: Value = serde_json::from_str(&crate::memory_profile::json(&planes, &map)).unwrap();
    let allocators = profile["memoryProfilePerAllocator"].as_object().unwrap();
    assert_eq!(allocators.len(), 1);
    assert_eq!((&profile["numHosts"], profile["memoryIds"].as_array().unwrap().len(), &profile["version"]), (&Value::from(1), 1, &Value::from(1)));
    let (name, allocator) = allocators.iter().next().unwrap();
    assert_eq!(name, "GPU_0_bfc");
    let summary = &allocator["profileSummary"];
    assert_eq!((&summary["peakBytesUsageLifetime"], &summary["peakStats"]["peakBytesInUse"], &summary["peakStatsTimePs"]), (&Value::from("9500"), &Value::from("7000"), &Value::from("70000")));
    let count = |key: &str| allocator[key].as_array().unwrap().len();
    assert_eq!((count("sampledTimelineSnapshots"), count("memoryProfileSnapshots"), count("activeAllocations"), count("specialAllocations")), (3, 1, 3, 2));
    assert_eq!(allocator["memoryProfileSnapshots"][0]["activityMetadata"]["tfOpName"], "mul_grad/Sum");
    assert_eq!(allocator["activeAllocations"][2]["snapshotIndex"], "0");
    assert_eq!((&allocator["specialAllocations"][1]["tfOpName"], &allocator["specialAllocations"][1]["allocationBytes"]), (&Value::from("stack"), &Value::from("2000")));
}
