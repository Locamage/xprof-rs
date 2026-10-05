use crate::tools::inference_profile::{InferenceStats, PerModelInferenceStats, RequestDetail};
use crate::tools::overview_page::inference_latency;

const MAX_ERROR: f64 = 0.0001;

fn assert_near(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= MAX_ERROR, "{actual} is not within {MAX_ERROR} of {expected}");
}

#[test]
fn inference_latency_test() {
    let request_details = (0..100).map(|i| RequestDetail { start_time_ps: 0, end_time_ps: i * 10000, device_time_ps: i * 1000, write_to_device_time_ps: i * 1000, ..Default::default() }).collect();
    let stats = InferenceStats { inference_stats_per_model: [(0, PerModelInferenceStats { request_details, ..Default::default() })].into(), ..Default::default() };

    let result = inference_latency(&stats);

    assert_eq!(result.len(), 6);
    for (index, [total, host, device, communication]) in [(0, [0.495, 0.396, 0.0495, 0.0495]), (1, [0.5, 0.4, 0.05, 0.05]), (5, [0.99, 0.792, 0.099, 0.099])] {
        assert_near(result[index].total_latency_us, total);
        assert_near(result[index].host_latency_us, host);
        assert_near(result[index].device_latency_us, device);
        assert_near(result[index].communication_latency_us, communication);
    }
}
