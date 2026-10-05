use crate::tools::opstats::pico_to_micro;
use crate::tools::pod_viewer::render;
use crate::xplane::steps::{
    Breakdown, Core, DEVICE_COLLECTIVES, DEVICE_COMPUTE_16, DEVICE_COMPUTE_32, DEVICE_TO_DEVICE, DEVICE_TO_HOST, DEVICE_WAIT_DEVICE, DEVICE_WAIT_HOST, Extra, HOST_COMPILE, HOST_COMPUTE, HOST_PREPARE,
    HOST_TO_DEVICE, HOST_TO_HOST, HOST_WAIT_INPUT, StepInfo, StepRecord, UNKNOWN_TIME,
};
use serde_json::Value;
use std::collections::BTreeMap;

const MAX_ERROR: f64 = 1e-6;
const STEP_NUM: u32 = 2;
const CORE_ID: u32 = 1001;
const STEP_TIME_PS: u64 = 1000;
const HOST_COMPUTE_PS: u64 = 50;
const HOST_COMPILE_PS: u64 = 50;
const HOST_TO_HOST_PS: u64 = 50;
const HOST_TO_DEVICE_PS: u64 = 50;
const HOST_PREPARE_PS: u64 = 50;
const DEVICE_COLLECTIVE_PS: u64 = 350;
const HOST_WAIT_INPUT_PS: u64 = 50;
const DEVICE_TO_DEVICE_PS: u64 = 50;
const DEVICE_TO_HOST_PS: u64 = 50;
const DEVICE_COMPUTE_32_PS: u64 = 50;
const DEVICE_COMPUTE_16_PS: u64 = 50;
const DEVICE_WAIT_DEVICE_PS: u64 = 50;
const DEVICE_WAIT_HOST_PS: u64 = 50;
const UNKNOWN_TIME_PS: u64 = 50;
const HOSTNAME: &str = "host:123";
const DEVICE_COMPUTE: &str = "1";
const DEVICE_TO_DEVICE_EVENT: &str = "2";
const DEVICE_COLLECTIVES_EVENT: &str = "3";
const HOST_COMPUTE_EVENT: &str = "4";
const HOST_PREPARE_EVENT: &str = "5";
const INPUT: &str = "6";
const OUTPUT: &str = "7";
const COMPILE: &str = "8";
const ALL_OTHERS: &str = "9";

pub fn create_op_stats() -> Extra {
    let type_ps = BTreeMap::from([
        (HOST_COMPUTE, HOST_COMPUTE_PS),
        (HOST_COMPILE, HOST_COMPILE_PS),
        (HOST_TO_HOST, HOST_TO_HOST_PS),
        (HOST_TO_DEVICE, HOST_TO_DEVICE_PS),
        (HOST_PREPARE, HOST_PREPARE_PS),
        (DEVICE_COLLECTIVES, DEVICE_COLLECTIVE_PS),
        (HOST_WAIT_INPUT, HOST_WAIT_INPUT_PS),
        (DEVICE_TO_DEVICE, DEVICE_TO_DEVICE_PS),
        (DEVICE_TO_HOST, DEVICE_TO_HOST_PS),
        (DEVICE_COMPUTE_32, DEVICE_COMPUTE_32_PS),
        (DEVICE_COMPUTE_16, DEVICE_COMPUTE_16_PS),
        (DEVICE_WAIT_DEVICE, DEVICE_WAIT_DEVICE_PS),
        (DEVICE_WAIT_HOST, DEVICE_WAIT_HOST_PS),
        (UNKNOWN_TIME, UNKNOWN_TIME_PS),
    ]);
    let step_info = StepInfo { name: String::new(), begin: 0, duration: STEP_TIME_PS, breakdown: Breakdown::Types(type_ps) };
    Extra {
        steps: vec![StepRecord { num: STEP_NUM, cores: BTreeMap::from([(CORE_ID, step_info)]), ..Default::default() }],
        cores: BTreeMap::from([(CORE_ID, Core { hostname: HOSTNAME.into(), ..Default::default() })]),
        ..Default::default()
    }
}

pub fn check_record(record: &Value) {
    let near = |actual: &Value, expected_ps: u64| assert!((actual.as_f64().unwrap() - pico_to_micro(expected_ps)).abs() <= MAX_ERROR, "{actual} != {expected_ps} ps");
    assert_eq!(record["stepNum"], STEP_NUM);
    assert_eq!(record["hostName"], HOSTNAME);
    near(&record["totalDurationUs"], STEP_TIME_PS);
    let breakdown = &record["stepBreakdownUs"];
    near(&breakdown[DEVICE_COMPUTE], DEVICE_COMPUTE_32_PS + DEVICE_COMPUTE_16_PS);
    near(&breakdown[DEVICE_TO_DEVICE_EVENT], DEVICE_TO_DEVICE_PS + DEVICE_WAIT_DEVICE_PS);
    near(&breakdown[DEVICE_COLLECTIVES_EVENT], DEVICE_COLLECTIVE_PS);
    near(&breakdown[HOST_COMPUTE_EVENT], HOST_COMPUTE_PS);
    near(&breakdown[HOST_PREPARE_EVENT], HOST_PREPARE_PS);
    near(&breakdown[INPUT], HOST_WAIT_INPUT_PS + HOST_TO_DEVICE_PS + DEVICE_WAIT_HOST_PS);
    near(&breakdown[OUTPUT], DEVICE_TO_HOST_PS);
    near(&breakdown[COMPILE], HOST_COMPILE_PS);
    near(&breakdown[ALL_OTHERS], UNKNOWN_TIME_PS);
    assert_eq!(record["bottleneck"], "Device collective communication");
}

#[test]
fn gpu_pod_viewer() {
    let pod_viewer_db: Value = serde_json::from_str(&render(&create_op_stats())).unwrap();
    let pod_stats_map = pod_viewer_db["podStatsSequence"]["podStatsMap"].as_array().unwrap();
    assert_eq!(pod_stats_map.len(), 1);
    assert_eq!(pod_stats_map[0]["stepNum"], STEP_NUM);
    check_record(&pod_stats_map[0]["podStatsPerCore"][CORE_ID.to_string()]);
}

#[test]
fn device_type() {
    let pod_viewer_db: Value = serde_json::from_str(&render(&Extra { device_type: "GPU".into(), ..Default::default() })).unwrap();
    assert_eq!(pod_viewer_db["deviceType"], "GPU");
}
