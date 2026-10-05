use crate::tools::input_pipeline_analyzer::{HOST_TRANSFER, SC_COMPUTE, SCV0_COMPUTE, SCV0_INFEED, TC_COMPUTE, TC_IDLE, TC_INFEED, TC_OUTFEED, tpu_step_details};
use crate::tools::opstats::{Db, IDLE, Metrics};
use crate::xplane::steps::{Breakdown, Core, Extra, SPARSE_CORE_START, StepInfo, StepRecord, fix};
use std::collections::{BTreeMap, HashMap};

const STEP_NUM: u32 = 1;
const HOST_INPUT_EVENTS: [(u64, u64); 3] = [(50, 100), (110, 200), (430, 500)];

fn step_info(begin_ps: u64, duration_ps: u64, category_ps: &[(&str, u64)]) -> StepInfo {
    StepInfo { name: String::new(), begin: begin_ps, duration: duration_ps, breakdown: Breakdown::Categories(category_ps.iter().map(|(name, ps)| (name.to_string(), *ps)).collect()) }
}

fn may_fix_tpu_step_analysis(device_op_metrics_db: &Db, step_info_per_core: BTreeMap<u32, StepInfo>) -> BTreeMap<u32, BTreeMap<String, u64>> {
    let total_input: u64 = HOST_INPUT_EVENTS.iter().map(|(begin, end)| end - begin).sum();
    let mut extra =
        Extra { steps: vec![StepRecord { num: STEP_NUM, cores: step_info_per_core, ..Default::default() }], host_input: Some(HashMap::from([(STEP_NUM as i64, total_input)])), ..Default::default() };
    fix(&mut extra, device_op_metrics_db);
    extra.steps.remove(0).cores.into_iter().map(|(core, info)| (core, if let Breakdown::Categories(categories) = info.breakdown { categories } else { BTreeMap::new() })).collect()
}

fn sparse_core_step(duration_ps: u64, idle_ps: u64) -> (StepRecord, BTreeMap<u32, Core>) {
    let core = SPARSE_CORE_START + 1;
    let record = StepRecord { num: STEP_NUM, cores: BTreeMap::from([(core, step_info(100, duration_ps, &[(IDLE, idle_ps)]))]), ..Default::default() };
    (record, BTreeMap::from([(core, Core { sparse: true, ..Default::default() })]))
}

#[test]
fn attribute_host_input_time_to_tc_when_infeed_missing() {
    let sir = step_info(40, 1000, &[(IDLE, 300), ("multiply", 300), ("all-gather", 300), ("async-start", 50), ("async-done", 50)]);
    let updated = may_fix_tpu_step_analysis(&Db::default(), BTreeMap::from([(2, sir)]));
    assert_eq!(updated[&2][IDLE], 90);
    assert_eq!(updated[&2]["infeed"], 210);
}

#[test]
fn attribute_host_input_time_to_tc_when_infeed_missing_multi_core() {
    let sir = step_info(40, 1000, &[(IDLE, 300), ("multiply", 300), ("all-gather", 300), ("async-start", 50), ("async-done", 50)]);
    let sir2 = step_info(45, 900, &[(IDLE, 250), ("multiply", 300), ("all-gather", 250), ("async-start", 50), ("async-done", 50)]);
    let updated = may_fix_tpu_step_analysis(&Db::default(), BTreeMap::from([(2, sir), (1, sir2)]));
    assert_eq!(updated[&2][IDLE], 48);
    assert_eq!(updated[&2]["infeed"], 252);
    assert_eq!(updated[&1][IDLE], 40);
    assert_eq!(updated[&1]["infeed"], 210);
}

#[test]
fn skip_may_fix_tpu_step_analysis_when_infeed_exists() {
    let sir = step_info(40, 1000, &[(IDLE, 300), ("multiply", 300), ("all-gather", 300), ("async-start", 50), ("infeed", 50)]);
    let device_op_metrics_db = Db { metrics: vec![Metrics { category: "infeed".into(), ..Default::default() }], ..Default::default() };
    let updated = may_fix_tpu_step_analysis(&device_op_metrics_db, BTreeMap::from([(2, sir)]));
    assert_eq!(updated[&2][IDLE], 300);
    assert_eq!(updated[&2]["infeed"], 50);
}

#[test]
fn ensure_sparse_core_steps_set_step_number() {
    let (per_core_step_info, core_details_map) = sparse_core_step(1000, 500);
    assert_eq!(tpu_step_details(&per_core_step_info, &core_details_map).step_number, 1);
}

#[test]
fn compute_tpu_per_step_data_across_cores_sparse_core_only() {
    let (per_core_step_info, core_details_map) = sparse_core_step(1_000_000_000_000, 200_000_000_000);
    let per_step_data = tpu_step_details(&per_core_step_info, &core_details_map);
    for field in [TC_COMPUTE, TC_INFEED, TC_OUTFEED, TC_IDLE, SCV0_COMPUTE, SCV0_INFEED, HOST_TRANSFER] {
        assert_eq!(per_step_data.fields[field], 0.0);
    }
    assert_eq!(per_step_data.all_reduce_ms, [0.0, 0.0]);
    assert_eq!(per_step_data.fields[SC_COMPUTE], 800.0);
    assert_eq!(per_step_data.fields[SC_COMPUTE + 1], 200.0);
    assert_eq!(per_step_data.fields[SC_COMPUTE + 2], 1000.0);
}
