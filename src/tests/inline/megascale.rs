use super::*;

fn send(kind: Opcode, rendezvous: Option<&str>) -> Instruction {
    Instruction { opcode: kind, channel_id: 7, rendezvous: rendezvous.map(String::from), transfer_type: Some("ALL_REDUCE".into()), size: 1000 }
}

fn visit(timestamp_ns: i64, duration_ps: i64) -> Visit<'static> {
    Visit { timestamp_ns, duration_ps, display_name: "op", collective_info: &[0x12, 0x08, 0x0a, 0x00, 0x0a, 0x00, 0x0a, 0x00, 0x0a, 0x00] }
}

#[test]
fn send_to_recv_done_measures_slack_minus_overlapped_ops() {
    let mut tracker = Tracker::default();
    tracker.visit(&send(Opcode::Send, Some("r")), &visit(1000, 2_000_000));
    tracker.visit(&send(Opcode::Recv, None), &visit(4000, 1_000_000));
    tracker.visit(&send(Opcode::SendDone, None), &visit(10_000, 500_000));
    tracker.visit(&send(Opcode::RecvDone, None), &visit(20_000, 3_000_000));
    let summary = &averaged(tracker.summaries)[0];
    assert_eq!((summary.occurrences, summary.bytes_transmitted_over_network), (1, 1500));
    assert_eq!([summary.times_us[SLACK], summary.times_us[STALL], summary.times_us[OBSERVED], summary.times_us[SEND], summary.times_us[RECV]], [15, 6, 22, 2, 0]);
    let json = table(std::slice::from_ref(summary));
    assert!(json.contains(r#"{"v":"r"},{"v":"op"},{"v":"op"},{"v":"ALL_REDUCE"},{"v":0.015},{"v":"-"},{"v":0.022}"#), "{json}");
    assert!(json.contains(r#"{"f":"1.5K","v":1500.0},{"v":0.8}"#), "{json}");
}

#[test]
fn ops_without_a_known_channel_are_ignored() {
    let mut tracker = Tracker::default();
    tracker.visit(&send(Opcode::RecvDone, None), &visit(20_000, 3_000_000));
    assert!(tracker.summaries.is_empty());
}

#[test]
fn transfer_types_scale_the_receive_buffer() {
    assert_eq!(["ONE_TO_ONE", "ALL_GATHER", "ALL_REDUCE", "ALL_TO_ALL", "REDUCE_SCATTER", "OTHER"].map(|kind| transmitted_bytes(100, 4, kind)), [400, 75, 150, 75, 300, 0]);
    assert_eq!(transmitted_bytes(100, 0, "ONE_TO_ONE"), 0);
    assert_eq!(replica_group_size(&[0x1a, 0x00]), 1);
}

#[test]
fn hosts_combine_weighted_by_occurrences() {
    let host = |slack, occurrences| Summary { rendezvous: "r".into(), times_us: [slack, 0, 0, 0, 0, 0, 0], occurrences, ..Default::default() };
    assert_eq!(combine(&[vec![host(10, 1)], vec![host(40, 3)]])[0].times_us[SLACK], 32);
    assert_eq!(human_bytes(512), "512B");
    assert_eq!(human_bytes(3 << 20), "3.00M");
    assert_eq!(human_bytes(1280), "1.2K");
    assert_eq!(table(&[]), format!("[{COLUMNS}]}}]"));
}
