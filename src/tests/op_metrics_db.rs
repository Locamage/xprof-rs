use super::xspace::{V, XPlane, XSpace, assert_db};
use crate::tools::framework_op_stats::host_db;
use crate::tools::opstats::{Db, IDLE, convert_tensor_core, templates};

const NS: i64 = 1000;

fn tensor_flow_op_event(plane: &mut XPlane, line: i64, tf_op_fullname: &str, start_ns: i64, duration_ns: i64, kernel: Option<&str>) {
    let name = kernel.unwrap_or(tf_op_fullname);
    let stats = kernel.map(|_| vec![("tf_op", V::Ref(tf_op_fullname.into()))]).unwrap_or_default();
    plane.event(line, name, start_ns * NS, duration_ns * NS, &stats);
}

fn summary(db: &Db) -> Vec<(String, String, u64, u64)> {
    db.metrics.iter().map(|metrics| (metrics.name.to_string(), metrics.category.to_string(), metrics.occurrences, metrics.time_ps)).collect()
}

#[test]
fn host_op_metrics_db() {
    let mut space = XSpace::default();
    let host = space.host();
    tensor_flow_op_event(host, 10, "TfOp1:TfOp1", 100000, 8000, None);
    tensor_flow_op_event(host, 20, "TfOp1:TfOp1", 100000, 8000, None);
    tensor_flow_op_event(host, 20, "TfOp2:TfOp2", 110000, 10000, None);
    let (map, planes) = space.parsed();
    let db = host_db(&planes, &map).0;
    assert_eq!(db.metrics.len(), 3);
    assert_eq!(db.total_op_time_ps, ((8000 * 2 + 10000) * NS) as u64);
    assert_eq!(db.total_time_ps, ((110000 - 100000 + 10000 + 8000) * NS) as u64);
    let expected = [("TfOp1", "TfOp1", 2, 8000 * 2), (IDLE, IDLE, 0, 2000), ("TfOp2", "TfOp2", 1, 10000)];
    assert_eq!(summary(&db), expected.map(|(name, category, occurrences, ns)| (name.to_string(), category.to_string(), occurrences, (ns * NS) as u64)));
}

#[test]
fn device_op_metrics_db() {
    let mut space = XSpace::default();
    let gpu = space.gpu(0);
    tensor_flow_op_event(gpu, 10, "TfOp1:TfOp1", 100000, 8000, Some("kernel1"));
    tensor_flow_op_event(gpu, 10, "TfOp1:TfOp1", 110000, 10000, Some("kernel2"));
    tensor_flow_op_event(gpu, 20, "TfOp1:TfOp1", 100000, 8000, Some("kernel1"));
    tensor_flow_op_event(gpu, 20, "TfOp1:TfOp1", 110000, 10000, Some("kernel2"));
    tensor_flow_op_event(gpu, 20, "TfOp2:TfOp2", 120000, 10000, Some("kernel3"));
    let (map, planes) = space.parsed();
    let gpus = crate::xplane::gpu::devices(&planes);
    let infos = crate::xplane::gpu::infos(&planes, &map, &gpus);
    let (db, _) = crate::xplane::gpu::device_plane(gpus[0], &map, &infos, crate::xplane::origin_ns(&planes));
    assert_eq!(db.metrics.len(), 4);
    let total_op = ((8000 * 2 + 10000 * 2 + 10000) * NS) as u64;
    assert_eq!(db.total_op_time_ps, total_op);
    assert_eq!(db.total_time_ps, total_op.max(((120000 + 10000 - 100000) * NS) as u64));
    let expected = [("TfOp1/kernel1", "TfOp1", 2, 8000 * 2), ("TfOp1/kernel2", "TfOp1", 2, 10000 * 2), ("TfOp2/kernel3", "TfOp2", 1, 10000), (IDLE, IDLE, 0, 0)];
    assert_eq!(summary(&db), expected.map(|(name, category, occurrences, ns)| (name.to_string(), category.to_string(), occurrences, (ns * NS) as u64)));
}

fn tpu_plane<'a>(space: &'a mut XSpace, line: i64, line_name: &str) -> &'a mut XPlane {
    let plane = space.tpu(0, "TPU V4", 0.0, 0.0, None);
    plane.named_line(line, line_name);
    plane
}

#[test]
fn tensor_core_device_op_metrics_db() {
    let mut space = XSpace::default();
    let plane = tpu_plane(&mut space, 10, "Framework Ops");
    plane.metadata_stats("MatMul", &[("tf_op", "while:MatMul".into()), ("hlo_category", "MatMul".into()), ("flops", 34u64.into()), ("symbol_id", 1.into()), ("program_id", 1.into())]);
    plane.event(10, "MatMul", 0, 10 * NS, &[("Time Scale Multiplier", 2.0.into())]).set_num_occurrences(2);
    let (map, planes) = space.parsed();
    assert_db(
        &convert_tensor_core(&planes[0], &map, &templates(&planes[0], &map)),
        r#"metrics_db {
             hlo_module_id: 1
             self_time_ps: 10000
             flops: 68
             flops_v2: 68
             model_flops: 68
             model_flops_v2: 68
             num_cores: 1
             occurrences: 2
             name: "MatMul"
             time_ps: 10000
             category: "MatMul"
             normalized_time_ps: 20000
             provenance: "while:MatMul"
             min_time_ps: 10000
             core_type: TENSOR_CORE
           }
           metrics_db { name: "IDLE" category: "IDLE" }
           total_time_ps: 10000
           total_op_time_ps: 10000
           normalized_total_op_time_ps: 20000"#,
    );
}

#[test]
fn tensor_core_device_op_metrics_db_extracts_vdd_energy() {
    let mut space = XSpace::default();
    let plane = tpu_plane(&mut space, 10, "XLA Ops");
    plane.metadata_stats("MatMul", &[("hlo_op", "while:MatMul".into()), ("symbol_id", 1.into()), ("program_id", 1.into()), ("hlo_category", "MatMul".into()), ("flops", 0u64.into())]);
    plane.event(10, "MatMul", 0, 10 * NS, &[("vdd_energy_j", 42.0.into())]).set_num_occurrences(1);
    let (map, planes) = space.parsed();
    let db = convert_tensor_core(&planes[0], &map, &templates(&planes[0], &map));
    assert!(!db.metrics.is_empty());
    assert_eq!(db.metrics[0].vdd_energy, Some(42.0));
}

#[test]
fn host_x_plane_with_xla_ops() {
    let mut space = XSpace::default();
    let host = space.host();
    host.event(10, "xla_op", 100000 * NS, 8000 * NS, &[("tf_op", "tf_op".into())]);
    host.event(10, "xla_op2", 110000 * NS, 10000 * NS, &[("tf_op", "tf_op2".into())]);
    let (map, planes) = space.parsed();
    assert_db(
        &host_db(&planes, &map).0,
        r#"metrics_db { self_time_ps: 8000000 occurrences: 1 name: "tf_op" time_ps: 8000000 }
           metrics_db { self_time_ps: 10000000 occurrences: 1 name: "tf_op2" time_ps: 10000000 }
           metrics_db { self_time_ps: 2000000 name: "IDLE" time_ps: 2000000 category: "IDLE" }
           total_time_ps: 20000000
           total_op_time_ps: 18000000
           precision_stats {}"#,
    );
}

#[test]
fn host_x_plane_with_input_pipeline_traceme_ops() {
    let mut space = XSpace::default();
    let host = space.host();
    for (name, offset, duration, stage) in
        [("ShuffleMapDataset", 100000000, 10000000, 1), ("MapMapDataset", 100000000, 8000000, 2), ("ShuffleMapDataset", 120000000, 10000000, 3), ("MapMapDataset", 120000000, 8000000, 4)]
    {
        host.event(10, name, offset, duration, &[("_ipl_stage_id", stage.into()), ("_ipl_stage_cat", "preprocessing".into())]);
    }
    let (map, planes) = space.parsed();
    let mut db = host_db(&planes, &map).0;
    db.metrics.sort_by_key(|metrics| (metrics.name == IDLE, metrics.module));
    assert_db(
        &db,
        r#"metrics_db { self_time_ps: 2000000 occurrences: 1 name: "ShuffleMapDataset" category: "preprocessing" hlo_module_id: 1 time_ps: 10000000 }
           metrics_db { self_time_ps: 8000000 occurrences: 1 name: "MapMapDataset" category: "preprocessing" hlo_module_id: 2 time_ps: 8000000 }
           metrics_db { self_time_ps: 2000000 occurrences: 1 name: "ShuffleMapDataset" category: "preprocessing" hlo_module_id: 3 time_ps: 10000000 }
           metrics_db { self_time_ps: 8000000 occurrences: 1 name: "MapMapDataset" category: "preprocessing" hlo_module_id: 4 time_ps: 8000000 }
           metrics_db { self_time_ps: 10000000 name: "IDLE" time_ps: 10000000 category: "IDLE" }
           total_time_ps: 30000000
           total_op_time_ps: 20000000
           precision_stats {}"#,
    );
}
