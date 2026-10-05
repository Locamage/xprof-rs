use super::xspace::{V, XSpace};
use crate::tools::inference_profile::{InferenceStats, PerModelInferenceStats, SampledPerModelInferenceStats, generate, load, tables};
use crate::xplane::group::{Metadata, Relatives};
use prost::Message;
use serde_json::Value;

const GROUP_ID: i64 = 42;

fn batch_with_programs(programs: &[(&str, u64, i64, i64)]) -> InferenceStats {
    let mut space = XSpace::default();
    let device = space.plane("/device:TPU:0");
    device.named_line(0, "XLA Modules");
    for &(module, program, offset_ps, duration_ps) in programs {
        device.metadata_stats(module, &[("program_id", V::Uint(program))]);
        device.event(0, module, offset_ps, duration_ps, &[("group_id", GROUP_ID.into())]);
    }
    space.host().named_line(0, "BatchThread");
    space.host().event(0, "ProcessBatch", 500_000, 3_000_000, &[("group_id", GROUP_ID.into())]);
    let (map, planes) = space.parsed();
    generate(&planes, &map, &Metadata::from([(GROUP_ID, Relatives::default())]), 0)
}

#[test]
fn test_with_multiple_xspaces() {
    let paths: Vec<_> = (1..=2)
        .map(|host| {
            let path = crate::tests::temp_dir().join(format!("xprof-rs-inference-{}-hostname{host}.xplane.pb", std::process::id()));
            std::fs::write(&path, XSpace::default().encode_to_vec()).unwrap();
            path
        })
        .collect();
    let stats = load(&paths);
    paths.iter().for_each(|path| std::fs::remove_file(path).unwrap());
    assert!(stats.is_some());
}

#[test]
fn populates_program_id_from_tpu_module_metadata() {
    const EXPECTED_PROGRAM_ID: u64 = 9_876_543_210;
    let mut stats = batch_with_programs(&[("my_hlo_module", EXPECTED_PROGRAM_ID, 1_000_000, 2_000_000)]);
    let batch = stats.inference_stats_per_host[&0].batch_details.iter().find(|batch| batch.batch_id == GROUP_ID).unwrap().clone();
    assert!(batch.program_ids.contains(&EXPECTED_PROGRAM_ID));

    stats.inference_stats_per_model.insert(0, PerModelInferenceStats { batch_details: vec![batch.clone()], ..Default::default() });
    stats.sampled_inference_stats.insert(0, SampledPerModelInferenceStats { sampled_batches: vec![batch], ..Default::default() });
    stats.model_id_db.ids = vec!["my_model".into()];
    stats.model_id_db.id_to_index.insert("my_model".into(), 0);
    let tables: Vec<Value> = serde_json::from_str(&tables(&stats)).unwrap();
    assert_eq!(tables.len(), 3);
    assert!(tables[2]["cols"].as_array().unwrap().iter().any(|column| column["type"] == "string" && column["label"] == "Program ID(s)"));
    assert!(!tables[2]["rows"].as_array().unwrap().is_empty());
}

#[test]
fn appends_program_id_on_batch_collision() {
    const PROGRAM_ID_1: u64 = 11111;
    const PROGRAM_ID_2: u64 = 22222;
    let stats = batch_with_programs(&[("hlo_module_1", PROGRAM_ID_1, 1_000_000, 1_000_000), ("hlo_module_2", PROGRAM_ID_2, 2_000_000, 1_000_000)]);
    assert!(stats.inference_stats_per_host[&0].batch_details.iter().any(|batch| batch.program_ids == [PROGRAM_ID_1, PROGRAM_ID_2]));
}
