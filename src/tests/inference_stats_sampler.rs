use super::xspace::inference_stats;
use crate::tools::inference_profile::sample;

#[test]
fn test_sort() {
    let stats = inference_stats(
        r"
        inference_stats_per_model {
          key: 1
          value {
            request_details { request_id: 0 start_time_ps: 0 end_time_ps: 10000 batching_request_delay_ps: 2000 batching_request_size: 200 }
            request_details { request_id: 1 start_time_ps: 0 end_time_ps: 20000 batching_request_delay_ps: 1000 batching_request_size: 100 }
            request_details { request_id: 2 start_time_ps: 0 end_time_ps: 30000 batching_request_delay_ps: 3000 batching_request_size: 300 }
            batch_details { batch_id: 3 start_time_ps: 0 end_time_ps: 10000 batch_delay_ps: 2000 padding_amount: 20 batch_size_after_padding: 200 }
            batch_details { batch_id: 4 start_time_ps: 0 end_time_ps: 20000 batch_delay_ps: 1000 padding_amount: 10 batch_size_after_padding: 100 }
            batch_details { batch_id: 5 start_time_ps: 0 end_time_ps: 30000 batch_delay_ps: 3000 padding_amount: 30 batch_size_after_padding: 300 }
          }
        }",
    );

    let by_latency = &sample("Latency", "Latency", &stats)[&1];
    assert_eq!(by_latency.sampled_requests.iter().map(|request| request.request_id).collect::<Vec<_>>(), [0, 1, 2]);
    assert_eq!(by_latency.sampled_batches.iter().map(|batch| batch.batch_id).collect::<Vec<_>>(), [3, 4, 5]);

    let by_size = &sample("Request size", "Padding amount", &stats)[&1];
    assert_eq!(by_size.sampled_requests.iter().map(|request| request.batching_request_size).collect::<Vec<_>>(), [100, 200, 300]);
    assert_eq!(by_size.sampled_batches.iter().map(|batch| batch.padding_amount).collect::<Vec<_>>(), [10, 20, 30]);

    let by_delay = &sample("Request delay for batching", "Batching delay", &stats)[&1];
    assert_eq!(by_delay.sampled_requests.iter().map(|request| request.batching_request_delay_ps).collect::<Vec<_>>(), [1000, 2000, 3000]);
    assert_eq!(by_delay.sampled_batches.iter().map(|batch| batch.batch_delay_ps).collect::<Vec<_>>(), [1000, 2000, 3000]);
}
