pub mod fixture {
    use prost::encoding::encode_varint;

    pub enum Stat<'a> {
        Int(u64),
        Text(&'a str),
    }

    fn key(tag: u64, wire: u64) -> Vec<u8> {
        let mut out = Vec::new();
        encode_varint(tag << 3 | wire, &mut out);
        out
    }

    pub fn number(tag: u64, value: u64) -> Vec<u8> {
        let mut out = key(tag, 0);
        encode_varint(value, &mut out);
        out
    }

    pub fn bytes(tag: u64, body: &[u8]) -> Vec<u8> {
        let mut out = key(tag, 2);
        encode_varint(body.len() as u64, &mut out);
        out.extend_from_slice(body);
        out
    }

    pub fn event(meta: u64, offset: u64, duration: u64, stats: &[(u64, Stat)]) -> Vec<u8> {
        let stats: Vec<u8> = stats
            .iter()
            .flat_map(|(id, value)| {
                let value = match value {
                    Stat::Int(number) => self::number(4, *number),
                    Stat::Text(text) => bytes(5, text.as_bytes()),
                };
                bytes(4, &[number(1, *id), value].concat())
            })
            .collect();
        bytes(4, &[number(1, meta), number(2, offset), number(3, duration), stats].concat())
    }

    pub fn line(id: u64, name: &str, events: &[Vec<u8>]) -> Vec<u8> {
        bytes(3, &[number(1, id), bytes(2, name.as_bytes()), number(3, 1000), events.concat()].concat())
    }

    pub fn plane(name: &str, lines: &[Vec<u8>], events: &[&str], stats: &[&str]) -> Vec<u8> {
        let entry = |tag: u64, id: usize, name: &str| bytes(tag, &[number(1, id as u64 + 1), bytes(2, &[number(1, id as u64 + 1), bytes(2, name.as_bytes())].concat())].concat());
        let metadata: Vec<u8> = events.iter().enumerate().map(|(id, name)| entry(4, id, name)).chain(stats.iter().enumerate().map(|(id, name)| entry(5, id, name))).flatten().collect();
        bytes(1, &[bytes(2, name.as_bytes()), lines.concat(), metadata].concat())
    }
}

use super::*;
use fixture::{Stat, event, line, plane as encode};

fn plane(stats: Vec<(&str, Own)>) -> Plane {
    let mut plane = Plane::default();
    plane.name = format!("{PREFIX}0");
    plane.own = stats.into_iter().map(|(name, value)| (name.into(), value)).collect();
    plane
}

#[test]
fn perf_env_from_device_caps() {
    let blackwell = plane(vec![
        ("device_vendor", Own::Text("Nvidia".into())),
        ("clock_rate", Own::Int(1965000)),
        ("core_count", Own::Int(148)),
        ("memory_bandwidth", Own::Int(7672320000000)),
        ("compute_cap_major", Own::Int(10)),
        ("compute_cap_minor", Own::Int(0)),
    ]);
    let perf = perf_env(&blackwell);
    assert_eq!(perf.peak_tera_flops, 2382.39744);
    assert_eq!(perf.bandwidths, vec![7672.32, 74449.92, 74449.92]);
    assert_eq!(perf.ridge_point, 310.5185185185186);
    assert_eq!(model_name(&blackwell), "Nvidia GPU (Blackwell)");
    let unknown = plane(vec![("device_vendor", Own::Text("Nvidia".into())), ("clock_rate", Own::Int(1000000)), ("compute_cap_major", Own::Int(8)), ("compute_cap_minor", Own::Int(7))]);
    assert_eq!(flops_per_core(&caps(&unknown)), 1024.0);
    let amd = plane(vec![("device_vendor", Own::Text("AMD".into())), ("gpu_device_name", Own::Text("gfx942".into())), ("clock_rate", Own::Int(2000000)), ("core_count", Own::Int(2))]);
    assert_eq!(perf_env(&amd).peak_tera_flops, 8.192);
    assert_eq!(model_name(&amd), "AMD GPU - gfx942");
    assert_eq!(model_name(&plane(vec![("device_vendor", Own::Text("AMD".into())), ("compute_cap_major", Own::Int(11))])), "AMD GPU - gfx-11XX series");
    assert_eq!(model_name(&plane(vec![("device_vendor", Own::Text("Intel".into()))])), "");
}

fn info(category: &str, cost: Option<Cost>) -> Info {
    Info { category: category.into(), provenance: "jit(f)/dot:".into(), deduplicated: "".into(), expression: "%x = f32[] add()".into(), source: Source { line: -1, ..Default::default() }, cost }
}

#[test]
fn device_db_merges_consecutive_kernels_of_one_hlo_op_per_group() {
    let stats = ["hlo_op", "program_id", "flops", "group_id"];
    let kernel = |offset: u64, op: &str, flops: u64, group: u64| event(1, offset, 10, &[(1, Stat::Text(op)), (2, Stat::Int(7)), (3, Stat::Int(flops)), (4, Stat::Int(group))]);
    let lines = [line(1, "Stream #1", &[kernel(0, "fusion", 4, 0), kernel(20, "a::fusion", 4, 0), kernel(40, "dot", 6, 0), kernel(60, "missing", 1, 0), kernel(80, "dot", 6, 1)])];
    let map = encode("/device:GPU:0", &lines, &["kernel"], &stats);
    let planes = crate::xplane::parse(&map).unwrap();
    let cost = Cost { model_flops: 100, device_flops: 50, bytes_accessed: 8, memory: vec![(true, 1, 6), (false, 1, 2)] };
    let infos: Infos = [(7, [("fusion".to_string(), info("loop fusion", Some(cost))), ("dot".to_string(), info("", None))].into_iter().collect())].into_iter().collect();
    let (db, _) = device_plane(&planes[0], &map, &infos, 0);
    let summary: Vec<(&str, &str, u64, u64, f64, f64, u64)> =
        db.metrics.iter().map(|metrics| (&*metrics.name, &*metrics.category, metrics.occurrences, metrics.time_ps, metrics.flops_v2, metrics.model_flops_v2, metrics.bytes_accessed)).collect();
    assert_eq!(summary, vec![("fusion", "loop fusion", 2, 20, 100.0, 200.0, 16), ("dot", "unknown", 2, 20, 12.0, 12.0, 0), ("IDLE", "IDLE", 0, 50, 0.0, 0.0, 0)]);
    assert_eq!(db.metrics[0].memory, vec![(READ, 1, 12), (WRITE, 1, 4)]);
    assert_eq!((db.total_time_ps, db.total_op_time_ps), (90, 40));
}

#[test]
fn kernels_take_the_hlo_op_name_and_skip_empty_or_non_kernel_events() {
    let stats = ["kernel_details", "tf_op", "hlo_op", "program_id", "equation"];
    let details = Stat::Text("regs:32 grid:2,1,1 block:64,1,1 occ_pct:50");
    let lines = [line(
        1,
        "Stream #1",
        &[
            event(1, 0, 4000, &[(1, Stat::Text("regs:32 grid:2,1,1 block:64,1,1 occ_pct:50")), (2, Stat::Text("XlaModule:")), (3, Stat::Text("fusion")), (4, Stat::Int(7))]),
            event(1, 10_000, 2000, &[(1, details), (2, Stat::Text("model/MatMul:MatMul")), (5, Stat::Text("ab,bc->ac"))]),
            event(2, 20_000, 999, &[(1, Stat::Text("regs:1"))]),
            event(2, 25_000, 0, &[(1, Stat::Text("regs:1"))]),
            event(2, 30_000, 5000, &[]),
        ],
    )];
    let map = encode("/device:GPU:0", &lines, &["volta_h884gemm", "copy"], &stats);
    let planes = crate::xplane::parse(&map).unwrap();
    let infos: Infos = [(7, [("fusion".to_string(), info("loop fusion", None))].into_iter().collect())].into_iter().collect();
    let top = top_kernels(device_plane(&planes[0], &map, &infos, 0).1);
    let summary: Vec<(&str, &str, bool, bool, u64, f64)> =
        top.iter().map(|report| (&*report.key.name, &*report.key.op_name, report.key.eligible, report.key.tensor_core, report.total_ns, report.occupancy)).collect();
    assert_eq!(summary, vec![("volta_h884gemm", "jit(f)/dot:", true, true, 4, 50.0), ("volta_h884gemm", "model/MatMul", true, true, 2, 50.0), ("copy", "", false, false, 0, 0.0)]);
}

#[test]
fn ungrouped_events_take_the_group_of_the_previous_event_in_time() {
    let lines = [line(1, "Stream #1", &[event(1, 0, 10, &[]), event(1, 30, 10, &[])]), line(2, "Stream #2", &[event(1, 20, 5, &[(1, Stat::Int(4))]), event(1, 50, 5, &[])])];
    let map = encode("/device:GPU:0", &lines, &["kernel"], &["group_id"]);
    let planes = crate::xplane::parse(&map).unwrap();
    assert_eq!(groups(&planes[0], &map), vec![vec![None, Some(4)], vec![Some(4), Some(4)]]);
}

#[test]
fn source_info_prefers_the_file_and_falls_back_to_the_first_stack_frame() {
    use crate::hlo::xla::{
        HloModuleProto, HloProto, OpMetadata, StackFrameIndexProto,
        stack_frame_index_proto::{FileLocation, StackFrame},
    };
    use prost::Message;
    let index = StackFrameIndexProto {
        file_names: vec!["a.py".into(), "b.py".into()],
        function_names: vec!["f".into()],
        file_locations: vec![
            FileLocation { file_name_id: 1, function_name_id: 1, line: 3, column: 4, ..Default::default() },
            FileLocation { file_name_id: 2, function_name_id: 1, line: 9, column: 1, ..Default::default() },
        ],
        stack_frames: vec![StackFrame { file_location_id: 2, parent_frame_id: 0 }, StackFrame { file_location_id: 1, parent_frame_id: 1 }],
    };
    let proto = HloProto { hlo_module: Some(HloModuleProto { stack_frame_index: Some(index), ..Default::default() }), ..Default::default() }.encode_to_vec();
    let module = Module::parse(Cow::Owned(proto));
    let from_stack = source(&module, &OpMetadata { stack_frame_id: 2, ..Default::default() });
    assert_eq!((&*from_stack.file, from_stack.line, &*from_stack.stack), ("a.py", 3, "a.py:3:4\nb.py:9:1\n"));
    let from_file = source(&module, &OpMetadata { stack_frame_id: 1, source_file: "c.py".into(), source_line: 5, ..Default::default() });
    assert_eq!((&*from_file.file, from_file.line, &*from_file.stack), ("c.py", 5, "b.py:9:1\n"));
    let missing = source(&module, &OpMetadata::default());
    assert_eq!((&*missing.file, missing.line, &*missing.stack), ("", -1, ""));
}

#[test]
fn launch_params_and_tensor_core_rules() {
    let (mut key, mut occupancy) = (KernelKey::default(), 0.0);
    launch_params("regs:96 static_shared:21520 dynamic_shared:82240 grid:32,1,1 block:640,1,1 occ_pct:12.5 bad block:1,2", &mut key, &mut occupancy);
    assert_eq!((key.registers, key.static_shmem, key.dynamic_shmem, key.grid, key.block, occupancy), (96, 21520, 82240, [32, 1, 1], [640, 1, 1], 12.5));
    assert!(tensor_core_eligible("model/dense/MatMul") && tensor_core_eligible("a/BatchMatMulV2") && !tensor_core_eligible("MatMul"));
    assert!(einsum_eligible("ab,bc->ac") && !einsum_eligible("ab->ba") && !einsum_eligible(""));
    assert_eq!(tf_op_fullname("", ""), "");
    assert_eq!(tf_op_fullname("", "XLA_Args"), "XLA_Args:XLA_Args");
    assert_eq!(tf_op_fullname("", "jit(f)/add"), "jit(f)/add:");
}

#[test]
fn kernel_reports_merge_and_sort() {
    let report =
        |name: &str, total: u64| KernelReport { key: KernelKey { name: name.into(), ..Default::default() }, total_ns: total, min_ns: total, max_ns: total, occurrences: 1, ..Default::default() };
    let top = top_kernels([("b", 5), ("a", 5), ("b", 7), ("c", 1)].map(|(name, total)| report(name, total)));
    let summary: Vec<(&str, u64, u64, u64, u64)> = top.iter().map(|report| (report.key.name.as_str(), report.total_ns, report.min_ns, report.max_ns, report.occurrences)).collect();
    assert_eq!(summary, vec![("b", 12, 5, 7, 2), ("a", 5, 5, 5, 1), ("c", 1, 1, 1, 1)]);
}

fn module_with_source(stack_frames: bool) -> (Module<'static>, crate::hlo::xla::OpMetadata) {
    use crate::hlo::xla::{
        HloModuleProto, HloProto, OpMetadata, StackFrameIndexProto,
        stack_frame_index_proto::{FileLocation, StackFrame},
    };
    use prost::Message;
    let index = StackFrameIndexProto {
        file_names: vec!["main.py".into()],
        function_names: vec!["func1".into(), "func2".into()],
        file_locations: vec![
            FileLocation { file_name_id: 1, function_name_id: 1, line: 10, column: 5, ..Default::default() },
            FileLocation { file_name_id: 1, function_name_id: 2, line: 20, column: 1, ..Default::default() },
        ],
        stack_frames: vec![StackFrame { file_location_id: 1, parent_frame_id: 0 }, StackFrame { file_location_id: 2, parent_frame_id: 1 }],
    };
    let metadata = OpMetadata { stack_frame_id: if stack_frames { 2 } else { 0 }, source_file: "main.py".into(), source_line: 20, ..Default::default() };
    let module = HloModuleProto { stack_frame_index: stack_frames.then_some(index), ..Default::default() };
    (Module::parse(Cow::Owned(HloProto { hlo_module: Some(module), ..Default::default() }.encode_to_vec())), metadata)
}

#[test]
fn test_get_location_stack() {
    let (module_with_stack_frames, _) = module_with_source(true);
    assert_eq!(stack(&module_with_stack_frames, 2), "main.py:20:1\nmain.py:10:5\n");
}

#[test]
fn test_get_source_info() {
    let (module_with_stack_frames, root_metadata) = module_with_source(true);
    let source_info = source(&module_with_stack_frames, &root_metadata);
    assert_eq!((&*source_info.file, source_info.line, &*source_info.stack), ("main.py", 20, "main.py:20:1\nmain.py:10:5\n"));
}

#[test]
fn test_get_source_info_fallback() {
    let (module_with_source_location, root_metadata) = module_with_source(false);
    let source_info = source(&module_with_source_location, &root_metadata);
    assert_eq!((&*source_info.file, source_info.line, &*source_info.stack), ("main.py", 20, ""));
}

fn op_info(category: &str, provenance: &str, deduplicated: &str, expression: &str, cost: Option<Cost>) -> Info {
    Info { category: category.into(), provenance: provenance.into(), deduplicated: deduplicated.into(), expression: expression.into(), source: Source::default(), cost }
}

fn op<'a>(info: &'a Info, name: &str, occurrences: u64, duration: u64, flops: u64, bytes: u64) -> Tracker<'a> {
    Tracker { info: Some(info), name: name.into(), program: 123, occurrences, duration, flops, bytes, ..Default::default() }
}

#[test]
fn enter_op_accumulates_basic_metrics() {
    let mut builder = Builder::default();
    let info = op_info("test_cat", "test_prov", "test_dedup", "long_test_op", None);
    builder.enter(&op(&info, "test_op", 2, 1000, 10, 20));
    assert_eq!(builder.db.metrics.len(), 1);
    let metrics = &builder.db.metrics[0];
    assert_eq!((&*metrics.name, &*metrics.category, &*metrics.provenance, &*metrics.deduplicated_name, &*metrics.long_name), ("test_op", "test_cat", "test_prov", "test_dedup", "long_test_op"));
    assert!(!metrics.is_eager);
    assert_eq!((metrics.occurrences, metrics.time_ps), (2, 1000));
    assert_eq!((metrics.flops_v2, metrics.model_flops_v2, metrics.bytes_accessed), (20.0, 20.0, 40));
}

#[test]
fn enter_op_accumulates_multiple_occurrences() {
    let mut builder = Builder::default();
    let info = op_info("", "", "", "", None);
    builder.enter(&op(&info, "test_op", 2, 1000, 10, 20));
    builder.enter(&Tracker { eager: true, ..op(&info, "test_op", 1, 500, 5, 10) });
    assert_eq!(builder.db.metrics.len(), 1);
    let metrics = &builder.db.metrics[0];
    assert!(!metrics.is_eager);
    assert_eq!((metrics.occurrences, metrics.time_ps), (3, 1500));
    assert_eq!((metrics.flops_v2, metrics.bytes_accessed), (25.0, 50));
}

#[test]
fn enter_op_preserves_eager_from_first_occurrence() {
    let mut builder = Builder::default();
    let info = op_info("", "", "", "", None);
    builder.enter(&Tracker { eager: true, ..op(&info, "test_op", 1, 0, 0, 0) });
    builder.enter(&op(&info, "test_op", 1, 0, 0, 0));
    assert_eq!(builder.db.metrics.len(), 1);
    assert!(builder.db.metrics[0].is_eager);
}

#[test]
fn enter_op_handles_model_flops() {
    let mut builder = Builder::default();
    let info = op_info("", "", "", "", Some(Cost { model_flops: 15, device_flops: 10, ..Default::default() }));
    builder.enter(&op(&info, "test_op", 1, 0, 10, 0));
    assert_eq!(builder.db.metrics.len(), 1);
    assert_eq!((builder.db.metrics[0].flops_v2, builder.db.metrics[0].model_flops_v2), (10.0, 15.0));
}

#[test]
fn enter_op_handles_source_info() {
    let mut builder = Builder::default();
    let info = Info { source: Source { file: "file.py".into(), line: 42, stack: "stack_frame_info".into() }, ..op_info("", "", "", "", None) };
    builder.enter(&op(&info, "test_op", 0, 0, 0, 0));
    assert_eq!(builder.db.metrics.len(), 1);
    let source = builder.db.metrics[0].source.clone().unwrap();
    assert_eq!((&*source.file, source.line, &*source.stack), ("file.py", 42, "stack_frame_info"));
}

#[test]
fn enter_op_accumulates_memory_accessed_breakdown() {
    let mut builder = Builder::default();
    let first = op_info("", "", "", "", Some(Cost { memory: vec![(true, 1, 100)], ..Default::default() }));
    builder.enter(&op(&first, "test_op", 1, 0, 0, 0));
    assert_eq!(builder.db.metrics.len(), 1);
    assert_eq!(builder.db.metrics[0].memory, [(READ, 1, 100)]);
    let second = op_info("", "", "", "", Some(Cost { memory: vec![(true, 1, 50), (false, 2, 200)], ..Default::default() }));
    builder.enter(&op(&second, "test_op", 1, 0, 0, 0));
    assert_eq!(builder.db.metrics[0].memory, [(READ, 1, 150), (WRITE, 2, 200)]);
}

#[test]
fn enter_op_handles_gpu_fields() {
    let mut builder = Builder::default();
    let info = op_info("", "", "", "", Some(Cost { model_flops: 100, device_flops: 100, bytes_accessed: 200, memory: vec![(true, 1, 50)] }));
    builder.enter(&op(&info, "gpu_op", 2, 0, 0, 0));
    assert_eq!(builder.db.metrics.len(), 1);
    let metrics = &builder.db.metrics[0];
    assert_eq!((metrics.flops_v2, metrics.bytes_accessed, metrics.model_flops_v2), (200.0, 400, 200.0));
    assert_eq!(metrics.memory, [(READ, 1, 100)]);
}

#[test]
fn enter_op_handles_gpu_negative_bytes_fallback() {
    let mut builder = Builder::default();
    let info = op_info("", "", "", "", Some(Cost { memory: vec![(true, 1, -50)], ..Default::default() }));
    builder.enter(&op(&info, "gpu_op", 2, 0, 0, 0));
    assert_eq!(builder.db.metrics.len(), 1);
    assert_eq!(builder.db.metrics[0].memory, [(READ, 1, 0)]);
}

#[test]
fn enter_op_metadata_populates_metadata() {
    let mut builder = Builder::default();
    let info = Info { source: Source { file: "file.py".into(), line: 42, stack: "stack_frame_info".into() }, ..op_info("test_cat", "test_prov", "test_dedup", "long_test_op", None) };
    builder.enter(&Tracker { eager: true, ..op(&info, "test_op", 0, 0, 0, 0) });
    assert_eq!(builder.db.metrics.len(), 1);
    let metrics = &builder.db.metrics[0];
    assert_eq!((&*metrics.name, &*metrics.category, &*metrics.provenance, &*metrics.deduplicated_name, &*metrics.long_name), ("test_op", "test_cat", "test_prov", "test_dedup", "long_test_op"));
    assert!(metrics.is_eager);
    let source = metrics.source.clone().unwrap();
    assert_eq!((&*source.file, source.line, &*source.stack), ("file.py", 42, "stack_frame_info"));
}

#[test]
fn enter_op_metadata_does_not_overwrite_if_occurrences_exist() {
    let mut builder = Builder::default();
    let initial = op_info("initial_cat", "initial_prov", "", "", None);
    builder.enter(&op(&initial, "test_op", 1, 0, 0, 0));
    let new = op_info("new_cat", "new_prov", "", "", None);
    builder.enter(&Tracker { eager: true, ..op(&new, "test_op", 0, 0, 0, 0) });
    assert_eq!(builder.db.metrics.len(), 1);
    assert_eq!((&*builder.db.metrics[0].category, &*builder.db.metrics[0].provenance), ("initial_cat", "initial_prov"));
}

#[test]
fn enter_op_accumulates_vdd_energy() {
    let mut builder = Builder::default();
    let info = op_info("test_cat", "test_prov", "test_dedup", "", None);
    builder.enter(&Tracker { vdd: 1.5, ..op(&info, "test_op", 2, 1000, 10, 20) });
    assert_eq!(builder.db.metrics.len(), 1);
    assert_eq!((&*builder.db.metrics[0].name, builder.db.metrics[0].vdd_energy), ("test_op", Some(1.5)));
    builder.enter(&Tracker { vdd: 2.2, ..op(&info, "test_op", 1, 500, 5, 10) });
    assert_eq!(builder.db.metrics[0].vdd_energy, Some(3.7));
    builder.enter(&op(&info, "other_op", 1, 500, 5, 10));
    assert_eq!(builder.db.metrics.len(), 2);
    assert_eq!((&*builder.db.metrics[1].name, builder.db.metrics[1].vdd_energy), ("other_op", None));
}
