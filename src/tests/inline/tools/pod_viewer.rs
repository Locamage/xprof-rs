use super::*;
use crate::xplane::steps::{Core, StepRecord};
use std::collections::BTreeMap;

fn step(num: u32, cores: &[(u32, Breakdown)]) -> StepRecord {
    StepRecord {
        num,
        cores: cores.iter().map(|(core, breakdown)| (*core, StepInfo { name: String::new(), begin: 0, duration: 2_500_000, breakdown: breakdown.clone() })).collect(),
        collectives: BTreeMap::new(),
    }
}

#[test]
fn generic_breakdowns_sum_event_types_and_pick_the_largest_bottleneck() {
    let types = Breakdown::Types(BTreeMap::from([(HOST_WAIT_INPUT, 1_000_000), (HOST_TO_DEVICE, 500_000), (HOST_COMPUTE, 1_000_000)]));
    let extra = Extra { steps: vec![step(4, &[(1, types)])], cores: BTreeMap::from([(1, Core { hostname: "worker".into(), ..Default::default() })]), device_type: "CPU".into(), ..Default::default() };
    let json = render(&extra);
    assert!(json.starts_with(r#"{"podStatsSequence":{"podStatsMap":[{"stepNum":4,"podStatsPerCore":{"1":{"hostName":"worker","chipId":0,"nodeId":0,"stepNum":4,"totalDurationUs":2.5,"bottleneck":"Input","stepBreakdownUs":{"1":0,"2":0,"3":0,"4":1,"5":0,"6":1.5,"7":0,"8":0,"9":0}}},"channelDb":[],"coreIdToReplicaIdMap":{},"allReduceOpDb":[]}]},"diagnostics":{"info":[],"warnings":[],"errors":[]},"stepBreakdownEvents":[{"id":1,"name":"Device compute"}"#), "{json}");
    assert!(json.ends_with(r#"{"id":9,"name":"All others"}],"deviceType":"CPU"}"#));
}

#[test]
fn tpu_breakdowns_do_not_unpack_so_every_type_is_zero_and_output_wins_ties() {
    let extra = Extra {
        steps: vec![step(0, &[(2, Breakdown::Categories(BTreeMap::from([("IDLE".to_string(), 9)]))), (5, Breakdown::Categories(BTreeMap::new()))])],
        cores: BTreeMap::from([(5, Core { hostname: "tpu".into(), ..Default::default() })]),
        device_type: "TPU v4".into(),
        ..Default::default()
    };
    let json = render(&extra);
    assert!(json.contains(r#""podStatsPerCore":{"2":{"hostName":"tpu","chipId":0,"nodeId":0,"stepNum":0,"totalDurationUs":2.5,"bottleneck":"Output""#), "{json}");
    assert!(json.contains(r#""5":{"hostName":"","chipId":0,"nodeId":0,"stepNum":0,"totalDurationUs":0,"bottleneck":"","stepBreakdownUs":{}}"#), "{json}");
}

#[test]
fn missing_steps_warn_like_xprof() {
    let json = render(&Extra { device_type: "CPU".into(), ..Default::default() });
    assert!(json.starts_with(&format!(r#"{{"podStatsSequence":{{"podStatsMap":[]}},"diagnostics":{{"info":[],"warnings":["{NO_STEP_MARKER}"],"errors":[]}}"#)));
    assert!(render(&Extra { empty_intersect: true, ..Default::default() }).contains(EMPTY_INTERSECT));
}
