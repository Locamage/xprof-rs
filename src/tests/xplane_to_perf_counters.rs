use super::xspace::XSpace;
use crate::counters::perf_counters;
use prost::Message;

fn create_session_snapshot() -> Vec<u8> {
    let mut xspace = XSpace::default();
    xspace.plane("host:0");
    let device_plane = xspace.gpu(0);
    device_plane.add_stat("global_chip_id", 0);
    device_plane.named_line(0, "Stream 1");
    device_plane.event(0, "KernelA", 0, 0, &[("counter_value", 123u64.into()), ("performance_counter_description", "Description A".into()), ("performance_counter_sets", "Set A".into())]);
    xspace.encode_to_vec()
}

#[test]
fn convert_multi_x_spaces_to_perf_counters() {
    let map = create_session_snapshot();
    let json = perf_counters(&[("hostname".to_string(), &map[..])]);
    assert!(json.contains("kernela"), "{json}");
    assert!(json.contains("Description A"));
    assert!(json.contains("0x7b"));
}
