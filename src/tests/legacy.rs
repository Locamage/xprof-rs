use crate::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

fn varint(value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    prost::encoding::encode_varint(value, &mut out);
    out
}

pub fn number(tag: u64, value: u64) -> Vec<u8> {
    [varint(tag << 3), varint(value)].concat()
}

pub fn bytes(tag: u64, body: &[u8]) -> Vec<u8> {
    [varint(tag << 3 | 2), varint(body.len() as u64), body.to_vec()].concat()
}

pub fn entry(tag: u64, id: u64, name: &str) -> Vec<u8> {
    bytes(tag, &[number(1, id), bytes(2, &[number(1, id), bytes(2, name.as_bytes())].concat())].concat())
}

fn entries(tag: u64, names: &[&str]) -> Vec<u8> {
    names.iter().zip(1..).flat_map(|(name, id)| entry(tag, id, name)).collect()
}

const EVERYTHING: Options = Options { start_ms: 0.0, end_ms: 0.0, resolution: 0.0, full_dma: false };

fn event(meta: u64, offset: u64, duration: u64, stats: &[(u64, u64)]) -> Vec<u8> {
    let stats: Vec<u8> = stats.iter().flat_map(|&(id, value)| bytes(4, &[number(1, id), number(4, value)].concat())).collect();
    bytes(4, &[number(1, meta), number(2, offset), number(3, duration), stats].concat())
}

fn annotated_space() -> Vec<u8> {
    let line = [number(1, 1), bytes(2, b"python"), event(1, 1_000_000, 10_000_000, &[(1, 1), (2, 7)]), event(2, 2_000_000, 1_000_000, &[])].concat();
    let plane = [bytes(2, b"/host:CPU"), bytes(3, &line), entries(4, &["train", "inner"]), entries(5, &["_r", "step_num"])].concat();
    bytes(1, &plane)
}

fn host_of(space: &[u8]) -> Host {
    crate::tests::with_file(space, |path| load_host(path).unwrap())
}

fn render_space(space: &[u8], options: &Options) -> String {
    let host = host_of(space);
    let view = View { trace: &host.trace, map: &host.map, planes: &host.planes, events: host.trace.load(options) };
    String::from_utf8(render(&[view], false, false)).unwrap()
}

#[test]
fn root_events_name_and_group_their_descendants() {
    let json = render_space(&annotated_space(), &EVERYTHING);
    assert!(json.contains("\"name\":\"train 7\""), "{json}");
    assert!(json.contains("\"name\":\"inner\""), "{json}");
    assert_eq!(json.matches("\"group_id\":0").count(), 2, "{json}");
    assert!(json.contains("\"returnedEventsSize\":2"), "{json}");
}

#[test]
fn details_always_list_the_mpmd_toggle() {
    let json = render_space(&annotated_space(), &EVERYTHING);
    assert!(json.contains("\"details\":[{\"name\":\"mpmd_pipeline_view\",\"value\":false}],"), "{json}");
}

#[test]
fn events_sharing_a_timestamp_beyond_the_serial_limit_are_dropped() {
    let events: Vec<u8> = (0..300).flat_map(|_| event(1, 5_000_000, 1_000_000, &[])).collect();
    let plane = [bytes(2, b"/host:CPU"), bytes(3, &[number(1, 1), bytes(2, b"python"), events].concat()), entry(4, 1, "tick")].concat();
    let json = render_space(&bytes(1, &plane), &EVERYTHING);
    assert!(json.contains("\"returnedEventsSize\":256"), "{json}");
}

#[test]
fn framework_op_names_parse_like_xprof() {
    let parsed = |full: &str| {
        let op = tools::framework_op_stats::parse_tf_op(full);
        (op.known, op.name, op.kind)
    };
    let expect = |known: bool, name: &str, kind: &str| (known, name.to_string(), kind.to_string());
    assert_eq!(parsed("jit(step)/shard_map/layer_1/dot_general:"), expect(true, "jit(step)/shard_map/layer_1/dot_general", "dot_general"));
    assert_eq!(parsed("scope/add_3:"), expect(true, "scope/add_3", "add"));
    assert_eq!(parsed("a/transpose[permutation=(0, 1)]:"), expect(true, "a/transpose[permutation=(0, 1)]", "transpose"));
    assert_eq!(parsed("model/MatMul_1:MatMul"), expect(true, "model/MatMul_1", "MatMul"));
    assert_eq!(parsed("Iterator::Batch"), expect(true, "Iterator::Batch", "Dataset"));
    assert_eq!(parsed("memcpyHToD"), expect(true, "memcpyHToD", "MemcpyHToD"));
    assert_eq!(parsed("tpu::System::Execute"), expect(false, "tpu::System::Execute", ""));
    assert_eq!(parsed("dummy"), expect(false, "dummy", ""));
}

#[test]
fn host_ops_split_self_time_and_idle_per_thread() {
    let text = |id: u64, value: &str| bytes(4, &[number(1, id), bytes(5, value.as_bytes())].concat());
    let host_event = |meta: u64, offset: u64, duration: u64, op: &str| bytes(4, &[number(1, meta), number(2, offset), number(3, duration), text(1, op)].concat());
    let line = [number(1, 7), bytes(2, b"worker"), host_event(1, 0, 10, "a/foo:foo"), host_event(2, 2, 3, "a/bar:bar"), host_event(3, 20, 0, "dummy")].concat();
    let plane = [bytes(2, b"/host:CPU"), bytes(3, &line), entries(4, &["outer", "inner", "MemoryAllocation"]), entry(5, 1, "tf_op")].concat();
    let space = bytes(1, &plane);
    let (db, _) = tools::framework_op_stats::host_db(&xplane::parse(&space).unwrap(), &space);
    let rows: Vec<(&str, &str, u64, u64, u64)> =
        db.metrics.iter().map(|metrics| (metrics.name.as_str(), metrics.category.as_str(), metrics.occurrences, metrics.time_ps, metrics.self_time_ps)).collect();
    assert_eq!(rows, [("a/bar", "bar", 1, 3, 3), ("a/foo", "foo", 1, 10, 7), ("dummy", "", 1, 0, 0), ("IDLE", "IDLE", 0, 10, 10)]);
    assert_eq!((db.total_time_ps, db.total_op_time_ps), (20, 10));
}

#[test]
fn memory_profile_doubles_print_like_protobuf() {
    let printed: Vec<String> = [0.000401934, 2.33222e-05, 0.1 + 0.2, 0.0, 1e20, 123.0, 0.5 + 8.0 / 1048576.0, 1.1444091796875e-05]
        .map(|value| {
            let mut out = String::new();
            tools::op_profile::proto_double(&mut out, value);
            out
        })
        .to_vec();
    assert_eq!(printed, ["0.000401934", "2.33222e-05", "0.30000000000000004", "0", "1e+20", "123", "0.50000762939453125", "1.1444091796875e-05"]);
}

#[test]
fn numbers_print_like_nlohmann_grisu2() {
    for (value, expected) in [
        (1.0, "1.0"),
        (0.5, "0.5"),
        (100.0, "100.0"),
        (0.001234, "0.001234"),
        (1.5e20, "1.5e+20"),
        (1e-7, "1e-07"),
        (-2.25, "-2.25"),
        (123456.789, "123456.789"),
        (0.0, "0.0"),
        (2.1194219481587213e-6, "2.1194219481587212e-06"),
    ] {
        let mut out = String::new();
        crate::tools::table::number(&mut out, value);
        assert_eq!(out, expected);
    }
}

fn op_stats_of(space: &[u8]) -> std::sync::Arc<OpStats> {
    crate::tests::with_file(space, tools::opstats::load).unwrap().unwrap()
}

#[test]
fn cpu_steps_come_from_host_step_markers() {
    let stats = op_stats_of(&annotated_space());
    let pipeline = tools::input_pipeline_analyzer::json(&stats);
    assert!(pipeline.starts_with("[{\"cols\":[],\"rows\":[]},"), "{pipeline}");
    assert!(pipeline.contains("{\"c\":[{\"v\":\"train 7\"},{\"v\":0.0},{\"v\":0.0},{\"v\":0.0},{\"v\":0.001},"), "{pipeline}");
    assert!(pipeline.contains("\"hardware_type\":\"CPU_ONLY\""), "{pipeline}");
    let overview = tools::overview_page::json(&stats, &[]);
    assert!(overview.contains("\"device_type\":\"CPU\""), "{overview}");
    assert!(overview.contains("\"rows\":[]}, {},{\"cols\":[{\"id\":\"severity\""), "{overview}");
}

#[test]
fn absl_six_digit_doubles() {
    for (value, expected) in [(1.04411e13, "1.04411e+13"), (669084641.0, "6.69085e+08"), (123456.0, "123456"), (0.5, "0.5"), (1e-5, "1e-05"), (999999.5, "1e+06")] {
        assert_eq!(hlo::general(value, 6), expected);
    }
}

#[test]
fn op_profile_partial_sort_breaks_ties_like_libstdcxx() {
    let keys = [3, 1, 3, 2, 3, 1, 2, 3, 0];
    for (k, expected) in [(9, vec![7, 4, 0, 2, 3, 6, 1, 5, 8]), (4, vec![4, 0, 7, 2]), (1, vec![0])] {
        let mut items: Vec<usize> = (0..keys.len()).collect();
        tools::op_profile::partial_sort(&mut items, k, |a, b| keys[a] > keys[b]);
        assert_eq!(items[..k], expected[..]);
    }
}

#[test]
fn printf_general_formatting_matches_c() {
    for (value, precision, expected) in
        [(0.5, 15, "0.5"), (1e21, 15, "1e+21"), (123456.0, 6, "123456"), (1234567.0, 6, "1.23457e+06"), (0.0001, 6, "0.0001"), (0.00001, 6, "1e-05"), (572.2, 6, "572.2"), (-0.0, 6, "-0")]
    {
        assert_eq!(hlo::general(value, precision), expected);
    }
}

#[test]
fn round_trip_matches_printf() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let special = [0.1, 0.3, 0.1 + 0.2, -0.0, 1e-5, 1e-4, 9.99999999999999e14, 1e15, 123456789012345.6, 5e-324, f64::MAX, 2.0 / 3.0];
    let random = (0..200_000).map(|index| match index % 4 {
        0 => f64::from_bits(next() >> 1),
        1 => (next() % 1_000_000_000) as f64 / (next() % 100_000 + 1) as f64,
        2 => (next() % 100_000) as f64 * 10f64.powi((next() % 40) as i32 - 20),
        _ => -((next() >> 11) as f64 / (1u64 << 53) as f64),
    });
    for value in special.into_iter().chain(random).filter(|value| value.is_finite()) {
        let short = hlo::general(value, 15);
        let expected = if short.parse::<f64>().ok() == Some(value) { short } else { hlo::general(value, 17) };
        let mut out = String::new();
        hlo::round_trip(&mut out, value);
        assert_eq!(out, expected, "{value:e}");
    }
}

#[test]
fn fused_instructions_print_like_xla() {
    let shape = bytes(3, &[number(2, 11), bytes(3, &varint(4)), bytes(5, &bytes(1, &varint(0)))].concat());
    let instruction = |id: u64, name: &str, opcode: &str, operands: &[u64], extra: Vec<u8>| {
        bytes(2, &[number(35, id), bytes(1, name.as_bytes()), bytes(2, opcode.as_bytes()), shape.clone(), operands.iter().flat_map(|&operand| number(36, operand)).collect(), extra].concat())
    };
    let fused = bytes(3, &[bytes(1, b"fused"), instruction(1, "p0", "parameter", &[], Vec::new()), instruction(2, "add.1", "add", &[1, 1], bytes(7, &bytes(2, b"jit(f)/add"))), number(5, 7)].concat());
    let main = bytes(
        3,
        &[bytes(1, b"main"), instruction(1, "p1", "parameter", &[], Vec::new()), instruction(2, "fusion.1", "fusion", &[1], [number(38, 7), bytes(11, b"kLoop")].concat()), number(5, 8)].concat(),
    );
    let proto = bytes(1, &[bytes(1, b"jit_f"), fused, main].concat());
    let module = hlo::Module::parse(std::borrow::Cow::Owned(proto));
    let printer = hlo::text::Printer::new(&module, hlo::text::Style::Expression, false);
    let (add, fusion) = (module.find("add.1").unwrap(), module.find("fusion.1").unwrap());
    let expression = |node: usize| {
        let mut out = String::new();
        printer.instruction(node, &mut out);
        out
    };
    assert_eq!(expression(add), "%add.1 = f32[4]{0} add(f32[4]{0} %p0, f32[4]{0} %p0)");
    assert_eq!(expression(fusion), "%fusion.1 = f32[4]{0} fusion(f32[4]{0} %p1), kind=kLoop, calls=%fused");
    assert_eq!((module.category(add).as_str(), module.category(fusion).as_str()), ("non-fusion elementwise", "loop fusion"));
    assert_eq!(module.graphs[module.nodes[fusion].called[0]].nodes.iter().map(|&index| module.nodes[index].name.as_str()).collect::<Vec<_>>(), ["p0", "add.1"]);
    assert_eq!(module.inst(add).metadata.unwrap().op_name, "jit(f)/add");
}

#[test]
fn roofline_category_falls_back_to_opcode_names() {
    let metrics = |name: &str, category: &str| tools::opstats::Metrics { name: name.into(), category: category.into(), ..Default::default() };
    assert_eq!(tools::roofline::category(&metrics("copy", "")), "copy");
    assert_eq!(tools::roofline::category(&metrics("copy.1", "unknown")), "unknown");
    assert_eq!(tools::roofline::category(&metrics("x", "")), "unknown");
    assert_eq!(tools::roofline::category(&metrics("copy", "data formatting")), "data formatting");
}

#[test]
fn std_sort_matches_libstdcxx_tie_order() {
    let mut seed = 12345u32;
    let keys: Vec<u32> = (0..60)
        .map(|_| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) % 7
        })
        .collect();
    let mut order: Vec<usize> = (0..60).collect();
    hlo::memory::std_sort(&mut order, &|a, b| keys[a] > keys[b]);
    let expected = [
        34, 50, 37, 28, 7, 31, 35, 32, 9, 3, 2, 53, 38, 41, 16, 44, 36, 49, 42, 45, 11, 10, 56, 14, 40, 13, 27, 29, 8, 54, 58, 47, 48, 30, 59, 6, 25, 17, 24, 4, 20, 33, 51, 5, 19, 46, 43, 23, 39, 26,
        57, 55, 1, 12, 52, 15, 18, 21, 22, 0,
    ];
    assert_eq!(order, expected);
}

#[test]
fn memory_viewer_simulates_shared_buffers_like_xprof() {
    let proto = b"\x0a\xcc\x01\x0a\x03\x73\x79\x6e\x12\x04\x6d\x61\x69\x6e\x30\x01\x1a\xbc\x01\x0a\x04\x6d\x61\x69\x6e\x12\x24\x0a\x01\x70\x12\x09\x70\x61\x72\x61\x6d\x65\x74\x65\x72\x1a\x0c\x10\x0b\x1a\x02\x40\x40\x2a\x04\x0a\x02\x01\x00\x3a\x03\x12\x01\x78\x98\x02\x01\x12\x2b\x0a\x01\x61\x12\x03\x65\x78\x70\x1a\x0c\x10\x0b\x1a\x02\x40\x40\x2a\x04\x0a\x02\x01\x00\x3a\x0c\x12\x0a\x6a\x69\x74\x28\x66\x29\x2f\x65\x78\x70\x98\x02\x02\xa2\x02\x01\x01\x12\x2c\x0a\x01\x62\x12\x03\x6c\x6f\x67\x1a\x0d\x10\x0b\x1a\x03\x40\x80\x01\x2a\x04\x0a\x02\x01\x00\x3a\x0c\x12\x0a\x6a\x69\x74\x28\x66\x29\x2f\x6c\x6f\x67\x98\x02\x03\xa2\x02\x01\x02\x12\x2f\x0a\x01\x63\x12\x06\x6e\x65\x67\x61\x74\x65\x1a\x0d\x10\x0b\x1a\x03\x40\x80\x01\x2a\x04\x0a\x02\x01\x00\x3a\x0c\x12\x0a\x6a\x69\x74\x28\x66\x29\x2f\x6e\x65\x67\x98\x02\x04\xa2\x02\x01\x03\x28\x01\x30\x04\x1a\xa8\x01\x0a\x0a\x08\x0a\x10\x80\x80\x01\x1a\x02\x20\x01\x0a\x0a\x08\x0b\x10\x80\x80\x01\x1a\x02\x20\x02\x0a\x0a\x08\x0c\x10\x80\x80\x02\x1a\x02\x20\x03\x0a\x0a\x08\x0d\x10\x80\x80\x02\x1a\x02\x20\x04\x1a\x12\x08\x00\x10\x80\x80\x01\x28\x01\x4a\x08\x08\x0a\x10\x00\x18\x80\x80\x01\x1a\x28\x08\x01\x10\x80\x80\x04\x4a\x08\x08\x0b\x10\x00\x18\x80\x80\x01\x4a\x0a\x08\x0c\x10\x80\x80\x01\x18\x80\x80\x02\x4a\x0a\x08\x0d\x10\x80\x80\x01\x18\x80\x80\x02\x22\x38\x0a\x07\x08\x00\x10\x0b\x22\x01\x61\x0a\x07\x08\x00\x10\x0c\x22\x01\x62\x0a\x07\x08\x01\x10\x0b\x22\x01\x62\x0a\x09\x08\x02\x10\x0d\x22\x01\x63\x28\x0c\x0a\x07\x08\x01\x10\x0c\x22\x01\x63\x0a\x07\x08\x01\x10\x0d\x22\x01\x63";
    let (body, kind) = hlo::memory::render(&hlo::Module::parse(std::borrow::Cow::Borrowed(proto)), 0, 16 * 1024, false).unwrap();
    assert_eq!(kind, "application/json");
    assert_eq!(
        body,
        r##"{"heapSizes":[0.015625,0.03125,0.0625,0.046875,0.046875,0.046875,0.015625],"unpaddedHeapSizes":[0.015625,0.03125,0.0625,0.046875,0.046875,0.046875,0.015625],"maxHeap":[{"numbered":0,"label":"p: f32[64,64]{1,0} # x","logicalBufferId":10,"logicalBufferSizeMib":0.015625,"unpaddedShapeMib":0.015625,"instructionName":"p","shapeString":"f32[64,64]{1,0}","tfOpName":"x","groupName":"Parameter","opCode":"parameter"},{"numbered":1,"label":"a: f32[64,64]{1,0} # jit(f)/exp","logicalBufferId":11,"logicalBufferSizeMib":0.015625,"unpaddedShapeMib":0.015625,"instructionName":"a","shapeString":"f32[64,64]{1,0}","tfOpName":"jit(f)/exp","groupName":"Temporary","opCode":"exp"},{"numbered":2,"label":"b: f32[64,128]{1,0} # jit(f)/log","logicalBufferId":12,"logicalBufferSizeMib":0.03125,"unpaddedShapeMib":0.03125,"instructionName":"b","shapeString":"f32[64,128]{1,0}","tfOpName":"jit(f)/log","groupName":"Temporary","opCode":"log"}],"maxHeapBySize":[{"numbered":2,"label":"b: f32[64,128]{1,0} # jit(f)/log","logicalBufferId":12,"logicalBufferSizeMib":0.03125,"unpaddedShapeMib":0.03125,"instructionName":"b","shapeString":"f32[64,128]{1,0}","tfOpName":"jit(f)/log","groupName":"Temporary","opCode":"log"},{"numbered":0,"label":"p: f32[64,64]{1,0} # x","logicalBufferId":10,"logicalBufferSizeMib":0.015625,"unpaddedShapeMib":0.015625,"instructionName":"p","shapeString":"f32[64,64]{1,0}","tfOpName":"x","groupName":"Parameter","opCode":"parameter"},{"numbered":1,"label":"a: f32[64,64]{1,0} # jit(f)/exp","logicalBufferId":11,"logicalBufferSizeMib":0.015625,"unpaddedShapeMib":0.015625,"instructionName":"a","shapeString":"f32[64,64]{1,0}","tfOpName":"jit(f)/exp","groupName":"Temporary","opCode":"exp"}],"logicalBufferSpans":{"11":{"limit":2},"12":{"start":1,"limit":5}},"maxHeapToBySize":[1,2,0],"bySizeToMaxHeap":[2,0,1],"moduleName":"syn","entryComputationName":"main","peakHeapMib":0.0625,"peakUnpaddedHeapMib":0.0625,"peakHeapSizePosition":1,"entryComputationParametersMib":0.015625,"indefiniteLifetimes":[{"sizeMib":0.015625,"attributes":["entry computation parameter","reusable"],"logicalBuffers":[{"id":"10","shape":"f32[64,64]{1,0}","sizeMib":0.015625,"hloName":"p"}],"commonShape":"f32[64,64]{1,0}"}],"totalBufferAllocationMib":0.078125,"indefiniteBufferAllocationMib":0.015625,"hloInstructionNames":["a","b","b","c","c","c",""],"bufferBlocks":[{"logicalBufferId":-1,"name":"Temporary","size":65536,"endStep":6,"category":"Temporary","color":"#ffffff"},{"logicalBufferId":11,"name":"a","size":16384,"endStep":2,"tfOpName":"jit(f)/exp","category":"Temporary","shapeString":"f32[64,64]{1,0}","unpaddedSize":16384,"color":"#2196f3"},{"logicalBufferId":12,"name":"b","offset":16384,"size":32768,"startStep":1,"endStep":5,"tfOpName":"jit(f)/log","category":"Temporary","shapeString":"f32[64,128]{1,0}","unpaddedSize":32768,"color":"#81c784"}]}"##
    );
}

#[test]
fn backend_configs_print_bare_only_when_they_lex_as_one_json_dict() {
    assert!(hlo::text::lexes_as_json_dict("{\"a\":{\"b\":\"}\"}}"));
    assert!(hlo::text::lexes_as_json_dict(" {\"x\":1}\n"));
    assert!(!hlo::text::lexes_as_json_dict("{\"a\":1}}"));
    assert!(!hlo::text::lexes_as_json_dict("x{}"));
    assert!(!hlo::text::lexes_as_json_dict("{\"a"));
    assert_eq!(hlo::text::frontend_attributes(&[("z", "1"), ("a", "{\"k\":2}")].map(|(key, value)| (key.to_string(), value.to_string())).into()), "{a={\"k\":2},z=\"1\"}");
}

#[test]
fn iota_tile_assignments_print_canonicalized_like_xla() {
    assert_eq!(hlo::text::iota_text(&[4, 1], &[4], &[0]), "[4,1]<=[4]");
    assert_eq!(hlo::text::iota_text(&[60], &[3, 4, 5], &[1, 2, 0]), "[60]<=[3,20]T(1,0)");
    assert_eq!(hlo::text::iota_text(&[60], &[3, 4, 5], &[0, 1, 2]), "[60]<=[60]");
    assert_eq!(hlo::text::iota_text(&[60], &[1, 3, 1, 4, 1, 5], &[4, 3, 2, 5, 1, 0]), "[60]<=[3,20]T(1,0)");
}

pub fn same(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value::{Array, Number, Object};
    match (a, b) {
        (Number(a), Number(b)) => a.as_f64() == b.as_f64(),
        (Array(a), Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b)),
        (Object(a), Object(b)) => a.len() == b.len() && a.iter().all(|(key, value)| b.get(key).is_some_and(|other| same(value, other))),
        (a, b) => a == b,
    }
}

fn normalized(mut trace: serde_json::Value) -> serde_json::Value {
    let frames = trace["stackFrames"].take();
    for event in trace["traceEvents"].as_array_mut().unwrap() {
        event.as_object_mut().unwrap().remove("bind_id");
        if let Some(frame) = event.as_object_mut().unwrap().remove("sf") {
            event["stack"] = frames[frame.to_string()]["name"].clone();
        }
    }
    trace.as_object_mut().unwrap().remove("stackFrames");
    trace
}

#[test]
fn demo_trace_matches_xprof_outputs() {
    let demo = include_bytes!("../../tests/data/demo.xplane.pb");
    let stats = op_stats_of(demo);
    let tools: [(&str, &str, String); 7] = [
        ("hlo_stats", include_str!("../../tests/data/hlo_stats.json"), tools::hlo_stats::json(&stats)),
        ("framework_op_stats", include_str!("../../tests/data/framework_op_stats.json"), tools::framework_op_stats::json(&stats)),
        ("overview_page", include_str!("../../tests/data/overview_page.json"), tools::overview_page::json(&stats, &[])),
        ("input_pipeline_analyzer", include_str!("../../tests/data/input_pipeline_analyzer.json"), tools::input_pipeline_analyzer::json(&stats)),
        ("roofline_model", include_str!("../../tests/data/roofline_model.json"), tools::roofline::json(&stats)),
        ("op_profile", include_str!("../../tests/data/op_profile.json"), tools::op_profile::json(&stats, None)),
        ("memory_profile", include_str!("../../tests/data/memory_profile.json"), tools::memory_profile::from_map(demo).unwrap().unwrap()),
    ];
    for (name, golden, produced) in tools {
        assert!(golden == produced, "{name} differs from XProf");
    }
    let dir = crate::tests::temp_dir().join(format!("xprof-rs-golden-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tpu-vm-demo-host-0.xplane.pb");
    std::fs::write(&path, demo).unwrap();
    let host = load_host(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let view = View { trace: &host.trace, map: &host.map, planes: &host.planes, events: host.trace.load(&Options { start_ms: 0.0, end_ms: 0.0, resolution: 8000.0, full_dma: false }) };
    let produced: serde_json::Value = serde_json::from_slice(&render(&[view], false, false)).unwrap();
    let golden = serde_json::from_str(include_str!("../../tests/data/trace_viewer.json")).unwrap();
    assert!(same(&normalized(golden), &normalized(produced)), "trace viewer differs from XProf");
}

#[test]
fn csv_numbers_print_like_python_repr() {
    for (value, expected) in [
        (1e16, "1e+16"),
        (1e15, "1000000000000000.0"),
        (12345678901234567.0, "1.2345678901234568e+16"),
        (0.0001, "0.0001"),
        (1e-5, "1e-05"),
        (-0.5, "-0.5"),
        (123.456, "123.456"),
        (-0.0, "-0.0"),
        (f64::INFINITY, "inf"),
        (f64::NEG_INFINITY, "-inf"),
        (f64::NAN, "nan"),
    ] {
        assert_eq!(crate::tools::table::repr(value), expected);
    }
}

#[test]
fn layouts_print_every_section_like_xla() {
    let array = |layout: Option<hlo::xla::LayoutProto>| hlo::Shape { element_type: 11, dimensions: vec![2, 3], layout: layout.map(Box::new), ..Default::default() };
    let layout = hlo::xla::LayoutProto {
        minor_to_major: vec![1, 0],
        tiles: vec![hlo::xla::TileProto { dimensions: vec![8, i64::MIN, -5] }],
        tail_padding_alignment_in_elements: 2,
        index_primitive_type: 4,
        pointer_primitive_type: 11,
        element_size_in_bits: 4,
        memory_space: 1,
        split_configs: vec![hlo::xla::SplitConfigProto { dimension: 0, split_indices: vec![1, 2] }],
        physical_shape: Some(Box::new(array(None))),
        dynamic_shape_metadata_prefix_bytes: 8,
        ..Default::default()
    };
    assert_eq!(array(Some(layout)).text(true), "f32[2,3]{1,0:T(8,*,Invalid value -5)L(2)#(s32)*(invalid)E(4)S(1)SC(0:1,2)P(f32[2,3])M(8)}");
    let mut buffer = hlo::Shape { element_type: 34, dimensions: vec![7], tuple_shapes: vec![array(None), array(None)], ..Default::default() };
    buffer.normalize();
    assert_eq!((buffer.text(true), buffer.dimensions.len()), ("b(f32[2,3])".to_string(), 0));
}

#[test]
fn backend_config_lexer_skips_comments() {
    assert!(hlo::text::lexes_as_json_dict("/* c */ {\"a\":1} // tail"));
    assert!(!hlo::text::lexes_as_json_dict("/* open {\"a\":1}"));
}

fn text_stat(id: u64, value: &str) -> Vec<u8> {
    bytes(4, &[number(1, id), bytes(5, value.as_bytes())].concat())
}

#[test]
fn serving_roots_take_their_graph_type_and_tensorflow_step_names() {
    let root = bytes(4, &[number(1, 1), number(2, 1_000_000), number(3, 10_000_000), text_stat(1, "train"), bytes(4, &[number(1, 2), number(4, 3)].concat())].concat());
    let named = bytes(4, &[number(1, 2), number(2, 20_000_000), number(3, 1_000_000), text_stat(3, "my step")].concat());
    let line = [number(1, 1), number(10, 9), bytes(2, b"python"), root, event(3, 2_000_000, 1_000_000, &[]), named].concat();
    let plane = [bytes(2, b"/host:CPU"), bytes(3, &line), entries(4, &["TraceContext", "other", "inner"]), entries(5, &["graph_type", "step_num", "step_name"])].concat();
    let json = render_space(&bytes(1, &plane), &EVERYTHING);
    assert!(json.contains("\"name\":\"train 3\""), "{json}");
    assert!(json.contains("\"name\":\"my step\""), "{json}");
    assert!(json.contains("\"tid\":9"), "{json}");
    assert_eq!(json.matches("\"group_id\":0").count(), 2, "{json}");
}

fn stat_event(meta: u64, offset: u64, duration: u64, stats: &[Vec<u8>]) -> Vec<u8> {
    bytes(4, &[number(1, meta), number(2, offset), number(3, duration), stats.concat()].concat())
}

fn int_stat(id: u64, value: u64) -> Vec<u8> {
    bytes(4, &[number(1, id), number(4, value)].concat())
}

fn gpu_space() -> Vec<u8> {
    let host_line =
        [number(1, 1), bytes(2, b"python"), stat_event(1, 1_000_000, 10_000_000, &[int_stat(1, 1), int_stat(2, 7)]), stat_event(2, 2_000_000, 1_000_000, &[int_stat(3, 5), int_stat(4, 0)])].concat();
    let host = [bytes(2, b"/host:CPU"), bytes(3, &host_line), entries(4, &["train", "fusion_kernel"]), entries(5, &["_r", "step_num", "correlation_id", "device_id"])].concat();
    let kernel = stat_event(1, 3_000_000, 2_000_000, &[int_stat(3, 5), text_stat(1, "regs:8 grid:2,1,1 block:32,1,1"), text_stat(2, "jit_f"), text_stat(4, "fusion"), int_stat(5, 3)]);
    let gpu_line = [number(1, 7), bytes(2, b"Stream #7(Compute)"), kernel].concat();
    let gpu =
        [number(1, 0), bytes(2, b"/device:GPU:0"), bytes(3, &gpu_line), entry(4, 1, "fusion_kernel"), entries(5, &["kernel_details", "hlo_module", "correlation_id", "hlo_op", "program_id"])].concat();
    [bytes(1, &gpu), bytes(1, &host)].concat()
}

#[test]
fn gpu_kernels_join_their_launch_step_and_derive_stream_lines() {
    let host = host_of(&gpu_space());
    let view = View { trace: &host.trace, map: &host.map, planes: &host.planes, events: host.trace.load(&EVERYTHING) };
    let json = String::from_utf8(render(&[view], false, true)).unwrap();
    for expected in [
        "/device:GPU:0\"},\"name\":\"process_name\"",
        "{\"args\":{\"name\":\"XLA Modules - from #7\"},\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1001,\"tid\":4}",
        "{\"args\":{\"name\":\"XLA Ops - from #7\"},\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1001,\"tid\":5}",
        "\"tid\":3735928559,\"name\":\"train 7\"",
        "\"step_name\":\"train 7\"",
        "\"name\":\"Launch Stats for train 7\"",
        "\"num_launches\":1,\"max_launch_time_us\":1,\"avg_launch_time_us\":1",
        "\"name\":\"jit_f(3)\"",
        "\"name\":\"fusion\"",
        "\"long_name\":\"3/fusion\"",
        "\"cat\":\"gpu_launch\"",
    ] {
        assert!(json.contains(expected), "{expected} missing from {json}");
    }
    assert_eq!(json.matches("\"is_eager\":0").count(), 1, "{json}");
}

#[test]
fn tf_op_categories_follow_parse_tf_op_fullname() {
    let cases = [
        ("model/dense/MatMul_1:", xplane::derive::Category::TensorFlow, "model/dense/MatMul_1", "MatMul", "MatMul", vec!["model", "dense"]),
        ("jit(f)/dot_general[x=1]:", xplane::derive::Category::Jax, "jit(f)/dot_general[x=1]", "dot_general", "dot_general", vec!["jit(f)"]),
        ("Iterator::Batch::Map:", xplane::derive::Category::TfData, "Iterator::Batch::Map:", "Dataset", "Iterator::Map:", vec![]),
        ("MemcpyHToD", xplane::derive::Category::Memcpy, "MemcpyHToD", "MemcpyHToD", "MemcpyHToD", vec![]),
        (":", xplane::derive::Category::TensorFlow, "", "", "", vec![]),
        ("$threading.py:323 wait ", xplane::derive::Category::Unknown, "$threading.py:323 wait ", "", "$threading.py:323 wait", vec![]),
    ];
    for (full, category, name, kind, event_name, scopes) in cases {
        let op = xplane::derive::tf_op(full);
        assert_eq!((op.category, op.name, op.kind, op.event_name(), op.scopes()), (category, name, kind, event_name.to_string(), scopes), "{full}");
    }
}

#[test]
fn run_tools_list_kernel_stats_only_for_gpu_profiles_and_nothing_for_corrupt_files() {
    assert_eq!(server::run_tools::tools(&gpu_space()).unwrap()[8..], ["kernel_stats"]);
    assert!(server::run_tools::tools(&[bytes(1, &bytes(2, b"/host:CPU")), vec![0x0a, 0xff, 0x7f]].concat()).is_none());
}

#[test]
fn utilization_and_bandwidth_doubles_print_with_two_decimals() {
    let double = |id: u64, value: f64| bytes(4, &[number(1, id), varint(2 << 3 | 1), value.to_le_bytes().to_vec()].concat());
    let line = [number(1, 1), bytes(2, b"python"), bytes(4, &[number(1, 1), number(2, 0), number(3, 10), double(1, 12.3456), double(2, 0.125)].concat())].concat();
    let plane = [bytes(2, b"/host:CPU"), bytes(3, &line), entry(4, 1, "op"), entries(5, &["HBM (util %)", "ratio"])].concat();
    let host = host_of(&bytes(1, &plane));
    let view = View { trace: &host.trace, map: &host.map, planes: &host.planes, events: vec![0] };
    let json = String::from_utf8(render(&[view], false, true)).unwrap();
    assert!(json.contains("\"HBM (util %)\":12.35,\"ratio\":0.125"), "{json}");
}

#[test]
fn request_queue_producers_flow_to_their_consumers() {
    let link = |meta: u64, offset: u64| event(meta, offset, 10, &[(1, 5), (2, 6)]);
    let line = [number(1, 1), bytes(2, b"python"), link(1, 0), link(2, 100)].concat();
    let plane = [bytes(2, b"/host:CPU"), bytes(3, &line), entries(4, &["EnqueueRequestLocked", "PjrtAsyncWait"]), entries(5, &["request_id", "queue_addr"])].concat();
    let host = host_of(&bytes(1, &plane));
    let flows: Vec<(u64, u8)> = host.trace.events.iter().map(|event| (event.flow, event.flow_entry)).collect();
    assert_eq!(flows, [(0, trace::FLOW_START), (0, trace::FLOW_END)]);
}

#[test]
fn cpu_input_waits_come_from_iterator_ops_and_pipeline_stage_roots() {
    let staged = bytes(4, &[number(1, 3), number(2, 5_000_000), number(3, 2_000_000), bytes(4, &[number(1, 3), bytes(5, b"stage")].concat())].concat());
    let line = [number(1, 1), bytes(2, b"python"), event(1, 1_000_000, 10_000_000, &[(1, 1), (2, 7)]), event(2, 2_000_000, 1_000_000, &[]), staged, event(4, 8_000_000, 1_000_000, &[])].concat();
    let plane = [bytes(2, b"/host:CPU"), bytes(3, &line), entries(4, &["train", "IteratorGetNext", "stage", "inner"]), entries(5, &["_r", "step_num", "_ipl_stage_name"])].concat();
    let pipeline = tools::input_pipeline_analyzer::json(&op_stats_of(&bytes(1, &plane)));
    assert!(pipeline.contains("{\"c\":[{\"v\":\"train 7\"},{\"v\":0.0},{\"v\":0.0},{\"v\":0.0},{\"v\":0.001},{\"v\":0.0},{\"v\":0.003},"), "{pipeline}");
    assert!(pipeline.contains("Your program is HIGHLY input-bound because 30.0% of the total step time"), "{pipeline}");
}

pub fn logdir(name: &str) -> PathBuf {
    let dir = crate::tests::scratch(&format!("server-{name}"));
    std::fs::create_dir_all(dir.join("run/plugins/profile/s")).unwrap();
    dir.canonicalize().unwrap()
}

pub fn settle(path: &Path) {
    std::fs::File::options().write(true).open(path).unwrap().set_modified(SystemTime::now() - Duration::from_secs(60)).unwrap();
}

pub async fn fetch(state: &Shared, uri: &str) -> (StatusCode, HeaderMap, String) {
    use tower::ServiceExt;
    let reply = app(state.clone()).oneshot(axum::http::Request::builder().uri(uri).body(Body::empty()).unwrap()).await.unwrap();
    let (parts, body) = reply.into_parts();
    (parts.status, parts.headers, String::from_utf8_lossy(&axum::body::to_bytes(body, usize::MAX).await.unwrap()).into_owned())
}

pub async fn served(name: &str, hosts: &[(&str, Vec<u8>)], query: &str) -> (StatusCode, HeaderMap, String) {
    let dir = logdir(name);
    for (host, space) in hosts {
        std::fs::write(dir.join(format!("run/plugins/profile/s/{host}.xplane.pb")), space).unwrap();
    }
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let reply = fetch(&state, &format!("/data/plugin/profile/data?run=run/s&{query}")).await;
    std::fs::remove_dir_all(&dir).unwrap();
    reply
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn module_list_extracts_hlo_protos_from_a_profile_with_hostnames() {
    let dir = logdir("modules");
    std::fs::write(dir.join("run/plugins/profile/s/tpu-vm-demo-host-0.xplane.pb"), include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let (status, _, modules) = fetch(&state, "/data/plugin/profile/module_list?run=run/s").await;
    assert_eq!(status, StatusCode::OK, "{modules}");
    let module = modules.split(',').next().unwrap().to_string();
    assert!(module.starts_with("jit_"), "{modules}");
    let query = |tool: &str| format!("/data/plugin/profile/data?run=run/s&tag={tool}&module_name={}", module.replace('(', "%28").replace(')', "%29"));
    let (status, _, text) = fetch(&state, &format!("{}&type=short_txt", query("graph_viewer"))).await;
    assert!(status == StatusCode::OK && text.contains("ENTRY"), "{status} {text}");
    let (status, _, memory) = fetch(&state, &query("memory_viewer")).await;
    assert!(status == StatusCode::OK && memory.starts_with('{'), "{status} {memory}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_paths_stay_inside_the_logdir() {
    let (dir, outside_dir) = (logdir("confined"), logdir("confined-outside"));
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), annotated_space()).unwrap();
    std::fs::write(outside_dir.join("run/plugins/profile/s/h.xplane.pb"), annotated_space()).unwrap();
    std::os::unix::fs::symlink(outside_dir.join("run"), dir.join("escape")).unwrap();
    std::os::unix::fs::symlink(&dir, dir.join("loop")).unwrap();
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let base = "/data/plugin/profile";
    let inside = dir.join("run/plugins/profile/s");
    for (uri, expected) in [
        (format!("{base}/data?run=run/s&tag=trace_viewer@&host=h"), StatusCode::OK),
        (format!("{base}/data?run=s&session_path={}&tag=trace_viewer@&host=h", inside.display()), StatusCode::OK),
        (format!("{base}/data?run={}/run/s&tag=trace_viewer@&host=h", outside_dir.display()), StatusCode::BAD_REQUEST),
        (format!("{base}/data?run=../{}/run/s&tag=trace_viewer@&host=h", outside_dir.file_name().unwrap().to_string_lossy()), StatusCode::BAD_REQUEST),
        (format!("{base}/data?run=escape/s&tag=trace_viewer@&host=h"), StatusCode::BAD_REQUEST),
        (format!("{base}/data?session_path={}&tag=trace_viewer@&host=h", outside_dir.join("run/plugins/profile/s").display()), StatusCode::BAD_REQUEST),
        (format!("{base}/data?run=run/s&tag=graph_viewer&type=pb&module_name=../../x"), StatusCode::BAD_REQUEST),
        (format!("{base}/hosts?run=escape/s"), StatusCode::BAD_REQUEST),
        (format!("{base}/module_list?run=/{}/run/s", outside_dir.display()), StatusCode::BAD_REQUEST),
    ] {
        assert_eq!(fetch(&state, &uri).await.0, expected, "{uri}");
    }
    let runs = tokio::time::timeout(Duration::from_secs(5), fetch(&state, &format!("{base}/runs"))).await.unwrap();
    assert_eq!(runs.2, "[\"run/s\"]");
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&outside_dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rewritten_profiles_are_reloaded_and_truncation_is_harmless() {
    let dir = logdir("rewrite");
    let file = dir.join("run/plugins/profile/s/h.xplane.pb");
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let uri = "/data/plugin/profile/data?run=run/s&tag=trace_viewer@&host=h";
    std::fs::write(&file, include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    settle(&file);
    let (status, _, first) = fetch(&state, uri).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetch(&state, uri).await.2, first);
    std::fs::write(&file, annotated_space()).unwrap();
    settle(&file);
    let (_, _, second) = fetch(&state, uri).await;
    assert!(second != first && second.contains("train 7"), "{second}");
    std::fs::File::options().write(true).open(&file).unwrap().set_len(0).unwrap();
    let (status, _, third) = fetch(&state, uri).await;
    assert!(third != second, "{status} {third}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn trace_requests_default_to_resolution_8000_and_failures_answer_with_security_headers() {
    let dir = logdir("defaults");
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    std::fs::write(dir.join("run/plugins/profile/s/corrupt.xplane.pb"), [0x0c]).unwrap();
    std::fs::write(dir.join("run/plugins/profile/s/bad.xplane.pb"), bytes(1, &entry(4, 1 << 40, "op"))).unwrap();
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let base = "/data/plugin/profile/data?run=run/s";
    let (status, headers, implicit) = fetch(&state, &format!("{base}&tag=trace_viewer@&host=h")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
    assert_eq!(headers["content-security-policy"], SECURITY_POLICY);
    assert_eq!(fetch(&state, &format!("{base}&tag=trace_viewer@&host=h&resolution=8000")).await.2, implicit);
    for tool in ["trace_viewer@", "overview_page"] {
        let (status, headers, message) = fetch(&state, &format!("{base}&tag={tool}&host=bad")).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{tool} {message}");
        assert!(message.contains("An event metadata ID is out of range"), "{message}");
        assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        let (status, _, message) = fetch(&state, &format!("{base}&tag={tool}&host=corrupt")).await;
        assert_eq!((status, message.as_str()), (StatusCode::NOT_FOUND, "No Data"));
    }
    let detail = format!("{base}&tag=trace_viewer@&host=h&event_name=x&start_time_ms=1e10&duration_ms=1&unique_id=1e30");
    assert_eq!(fetch(&state, &detail).await.0, StatusCode::NOT_FOUND);
    assert_eq!(fetch(&state, "/data/plugin/profile/version").await.2, VERSION);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn hosts_match_exactly_like_xprof() {
    let dir = logdir("select");
    for name in ["a", "a.b"] {
        std::fs::write(dir.join(format!("run/plugins/profile/s/{name}.xplane.pb")), []).unwrap();
    }
    let session = dir.join("run/plugins/profile/s");
    let pick = |tool: &str, query: &[(&str, &str)]| {
        let params: Params = query.iter().map(|(key, value)| (key.to_string(), value.to_string())).chain([("run".to_string(), "run/s".to_string())]).collect();
        select(&session, tool, &params).map(|files| files.iter().map(|file| host_name(file)).collect::<Vec<_>>())
    };
    assert_eq!(pick("hlo_stats", &[("host", "a")]), Ok(vec!["a".to_string()]));
    assert_eq!(pick("hlo_stats", &[("host", "")]), Ok(vec!["a.b".to_string(), "a".to_string()]));
    assert_eq!(pick("hlo_stats", &[("host", "a.b"), ("hosts", "a")]), Ok(vec!["a.b".to_string()]));
    assert_eq!(pick("hlo_stats", &[("hosts", "a")]), Err((StatusCode::NOT_FOUND, "Host must be specified for tool hlo_stats in run run/s".to_string())));
    assert_eq!(pick("trace_viewer@", &[("hosts", "a,zz")]), Err((StatusCode::NOT_FOUND, "No xplane file found for host: zz in run: run/s".to_string())));
    assert_eq!(pick("trace_viewer@", &[("hosts", "a.b")]), Ok(vec!["a.b".to_string()]));
    assert_eq!(pick("overview_page", &[("host", "zz")]), Err((StatusCode::NOT_FOUND, "No xplane file found for host: zz in run: run/s".to_string())));
    std::fs::remove_dir_all(&dir).unwrap();
}

static SLOW_BUILDS: AtomicUsize = AtomicUsize::new(0);
static FRESH_BUILDS: AtomicUsize = AtomicUsize::new(0);

fn slow_build(path: &Path) -> anyhow::Result<usize> {
    std::thread::sleep(Duration::from_millis(300));
    SLOW_BUILDS.fetch_add(1, Ordering::SeqCst);
    Ok(path.as_os_str().len())
}

fn fresh_build(_: &Path) -> anyhow::Result<usize> {
    Ok(FRESH_BUILDS.fetch_add(1, Ordering::SeqCst))
}

static BROKEN_BUILDS: AtomicUsize = AtomicUsize::new(0);

fn broken_build(_: &Path) -> anyhow::Result<usize> {
    BROKEN_BUILDS.fetch_add(1, Ordering::SeqCst);
    panic!("corrupt profile")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_requests_leave_one_load_that_later_requests_share() {
    let dir = logdir("flights");
    let (settled, fresh) = (dir.join("settled"), dir.join("fresh"));
    std::fs::write(&settled, b"x").unwrap();
    std::fs::write(&fresh, b"x").unwrap();
    settle(&settled);
    let (memo, permits) = (Memo::new(|_, _: &usize| 1), Arc::new(Semaphore::new(1)));
    for _ in 0..3 {
        assert!(tokio::time::timeout(Duration::from_millis(20), memo.get(settled.clone(), &permits, slow_build)).await.is_err());
    }
    let (first, second) = tokio::join!(memo.get(settled.clone(), &permits, slow_build), memo.get(settled.clone(), &permits, slow_build));
    assert_eq!((first, second), (Ok(settled.as_os_str().len()), Ok(settled.as_os_str().len())));
    assert_eq!(memo.get(settled.clone(), &permits, slow_build).await, Ok(settled.as_os_str().len()));
    assert_eq!(SLOW_BUILDS.load(Ordering::SeqCst), 1);
    memo.get(fresh.clone(), &permits, fresh_build).await.unwrap();
    memo.get(fresh.clone(), &permits, fresh_build).await.unwrap();
    assert_eq!(FRESH_BUILDS.load(Ordering::SeqCst), 2);
    let broken = Memo::new(|_, _: &usize| 1);
    for _ in 0..3 {
        assert_eq!(broken.get(settled.clone(), &permits, broken_build).await, Err(Arc::from("corrupt profile")));
    }
    assert_eq!(BROKEN_BUILDS.load(Ordering::SeqCst), 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flights_run_shared_work_once_and_report_panics() {
    let (flights, runs) = (Flights::<u8, u8>::new(), Arc::new(AtomicUsize::new(0)));
    let work = || {
        let runs = runs.clone();
        move || async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            runs.fetch_add(1, Ordering::SeqCst);
            7
        }
    };
    let (first, second) = tokio::join!(flights.join(1, work()), flights.join(1, work()));
    assert_eq!((first, second, runs.load(Ordering::SeqCst)), (Ok(7), Ok(7), 1));
    assert_eq!(flights.join(2, || async { panic!("broken profile") }).await, Err(Arc::from("broken profile")));
    assert_eq!(flights.join(2, || async { 3 }).await, Ok(3));
}

#[tokio::test]
async fn caches_keep_entries_heavier_than_the_budget_and_count_tiny_ones() {
    let heavy = cache(1 << 20, |_: &u32, value: &Vec<u8>| value.len() as u64);
    heavy.insert(1, vec![0; 4 << 20]).await;
    heavy.run_pending_tasks().await;
    assert!(heavy.get(&1).await.is_some());
    let tiny = cache(1 << 20, |_: &u32, value: &Vec<u8>| value.len() as u64);
    for key in 0..4096 {
        tiny.insert(key, vec![0; 10]).await;
    }
    tiny.run_pending_tasks().await;
    assert!(tiny.entry_count() <= 1024, "{}", tiny.entry_count());
}

#[test]
fn graph_html_escapes_the_dot_inside_its_template_literal() {
    let plain = "digraph G {\nlabel = <<b>fusion.1</b>>;\n}";
    assert!(hlo::graph::wrap_dot_html(plain, "dot").contains(&format!("const data = `{plain}`;")));
    let hostile = hlo::graph::wrap_dot_html("a`b${c}\\d</SCRIPT><!--", "dot");
    assert!(hostile.contains("const data = `a\\`b\\${c}\\\\d<\\/SCRIPT><\\!--`;"), "{hostile}");
}

#[test]
fn command_line_errors_are_reported_not_panicked() {
    let dir = logdir("arguments");
    let run = |list: &[&str]| arguments(list.iter().map(std::string::ToString::to_string));
    let path = dir.to_string_lossy().into_owned();
    let base = Settings { logdir: dir.clone(), port: DEFAULT_PORT, ..Default::default() };
    assert_eq!(run(&["--logdir", &path]), Ok(Settings { logdir: dir.clone(), port: DEFAULT_PORT, ..Default::default() }));
    assert_eq!(run(&[&format!("--logdir={path}"), "--port", "9000", "--host", "127.0.0.1"]), Ok(Settings { port: 9000, host: Some("127.0.0.1".into()), ..base }));
    assert_eq!(
        run(&[
            "--logdir",
            &path,
            "--src_prefix",
            "/src",
            "--hide_capture_profile_button",
            "--enable_tab_name_label",
            "--grpc_port",
            "1",
            "--worker_service_address",
            "h:1",
            "--max_concurrent_worker_requests=2"
        ]),
        Ok(Settings { logdir: dir.clone(), port: DEFAULT_PORT, src_prefix: Some("/src".into()), hide_capture_profile_button: true, enable_tab_name_label: true, ..Default::default() })
    );
    assert_eq!(run(&["--port", "8000", "--logdir"]), Err("argument --logdir: expected one argument".to_string()));
    assert_eq!(run(&["--logdir", &path, "--port", "x"]), Err("argument --port: invalid port value: 'x'".to_string()));
    assert_eq!(run(&[]), Ok(Settings { port: DEFAULT_PORT, ..Default::default() }));
    assert_eq!(run(&["--bogus"]), Err("unrecognized argument: --bogus".to_string()));
    assert_eq!(run(&["--logdir", "/nonexistent/xprof-rs"]), Err("Log directory '/nonexistent/xprof-rs' does not exist or is not a directory.".to_string()));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn proto_json_strings_escape_like_protobuf() {
    let mut out = String::new();
    hlo::proto_text::json_string(&mut out, "<a>\u{1}\u{7f}\n\"\u{2028}");
    assert_eq!(out, "\"\\u003ca\\u003e\\u0001\\u007f\\n\\\"\\u2028\"");
}

#[test]
fn malformed_protobuf_fails_the_load_instead_of_panicking() {
    let good = annotated_space();
    let overlong = [vec![0x08], vec![0xff; 10], vec![0x01]].concat();
    let cases: [&[u8]; 7] = [&[0x0c], &[0x0a], &[0x0a, 0xff, 0xff, 0xff, 0xff, 0x0f], &overlong, &[0x02, 0x00], &good[..good.len() - 3], b"not a protobuf\n"];
    for case in cases {
        assert!(xplane::parse(case).is_err(), "{case:?}");
    }
    assert!(xplane::parse(&[]).unwrap().is_empty());
    assert_eq!(xplane::parse(&good).unwrap()[0].meta.len(), 3);
}

#[test]
fn metadata_ids_beyond_the_table_bound_are_rejected_without_allocating() {
    for tag in [4, 5] {
        for id in [1 << 28, 1 << 40, u64::MAX] {
            assert!(xplane::parse(&bytes(1, &[bytes(2, b"/host:CPU"), entry(tag, id, "x")].concat())).is_err(), "{tag} {id}");
        }
    }
    let sparse = xplane::parse(&bytes(1, &[bytes(2, b"/host:CPU"), entry(4, 1000, "x"), entry(5, 81, "y")].concat())).unwrap();
    assert_eq!((sparse[0].meta.len(), &*sparse[0].meta[1000].name, &*sparse[0].stat_names[81]), (1001, "x", "y"));
}

#[test]
fn events_with_undefined_metadata_resolve_to_an_empty_entry() {
    for plane in ["/host:CPU", "/device:TPU:0"] {
        let line = [number(1, 1), bytes(2, b"XLA Ops"), event(5, 1_000, 2_000, &[]), event(u64::from(u32::MAX) + 7, 4_000, 1_000, &[(9, 1)])].concat();
        let space = bytes(1, &[bytes(2, plane.as_bytes()), bytes(3, &line), entry(4, 1, "op")].concat());
        let planes = xplane::parse(&space).unwrap();
        assert_eq!(planes[0].meta.len(), 3);
        assert!(planes[0].lines[0].events.iter().all(|event| event.meta == 2 && planes[0].meta[2].name.is_empty()));
        render_space(&space, &EVERYTHING);
        op_stats_of(&space);
    }
    let empty = bytes(1, &[bytes(2, b"/host:CPU"), bytes(3, &[number(1, 1), event(0, 1, 1, &[])].concat())].concat());
    assert_eq!(xplane::parse(&empty).unwrap()[0].meta.len(), 1);
}

#[test]
fn extreme_timestamps_saturate_at_parse() {
    let line = |timestamp: u64, offset: u64, duration: u64| bytes(3, &[number(1, 1), bytes(2, b"XLA Ops"), number(3, timestamp), event(1, offset, duration, &[])].concat());
    let spans = |lines: &[Vec<u8>]| {
        let space = bytes(1, &[bytes(2, b"/device:TPU:0"), lines.concat(), entry(4, 1, "op")].concat());
        render_space(&space, &EVERYTHING);
        op_stats_of(&space);
        xplane::parse(&space).unwrap()[0].lines.iter().map(|line| (line.events[0].ts, line.events[0].dur)).collect::<Vec<(u64, u64)>>()
    };
    assert_eq!(spans(&[line(u64::MAX, 7, 3), line(5, u64::MAX, u64::MAX), line(1 << 60, 9, 4), line(0, 11, 2)]), [(7, 3), (u64::MAX, 0), (u64::MAX, 0), (11, 2)]);
    let epoch = 1_700_000_000_000_000_000;
    assert_eq!(spans(&[line(epoch, 7, 3), line(0, 11, 2), line(epoch + 1, 13, u64::MAX)]), [(7, 3), (11, 2), (1013, u64::MAX - 1013)]);
}

#[test]
fn deeply_nested_events_build_in_linear_time() {
    let events: Vec<u8> = (0..200_000u64).flat_map(|depth| event(1, depth, 1_000_000 - 2 * depth, &[])).collect();
    let space = bytes(1, &[bytes(2, b"/host:CPU"), bytes(3, &[number(1, 1), bytes(2, b"python"), events].concat()), entry(4, 1, "nested")].concat());
    let started = std::time::Instant::now();
    assert_eq!(host_of(&space).trace.events.len(), 200_000);
    assert!(started.elapsed() < std::time::Duration::from_secs(20), "{:?}", started.elapsed());
}

#[test]
fn hlo_split_copies_unmatched_fields_verbatim() {
    let fixed = [vec![3 << 3 | 1], 7u64.to_le_bytes().to_vec(), vec![4 << 3 | 5], 9u32.to_le_bytes().to_vec()].concat();
    let buf = [number(1, 300), bytes(2, b"a"), fixed.clone(), bytes(2, b"b"), bytes(5, b"c")].concat();
    let (rest, matched) = hlo::split(&buf, 2);
    assert_eq!((rest, matched), ([number(1, 300), fixed, bytes(5, b"c")].concat(), vec![&b"a"[..], b"b"]));
    assert_eq!(hlo::split(&[0x02, 0x00, 0x0c], 2), (Vec::new(), Vec::new()));
}

#[test]
fn lazily_decoded_stats_stop_at_malformed_bytes() {
    let stats = [bytes(4, &[number(1, 2), number(4, 5)].concat()), vec![4 << 3 | 2, 3, 0x08, 0xff]].concat();
    let found: Vec<(usize, Option<i64>)> = xplane::stats(&stats, 4, |_| true).map(|stat| (stat.id, stat.value.int())).collect();
    assert_eq!(found, [(2, Some(5))]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_trace_numbers_session_paths_csv_and_graph_errors_answer_like_xprof() {
    let dir = logdir("parity");
    std::fs::write(dir.join("run/plugins/profile/s/tpu-vm-demo-host-0.xplane.pb"), include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let base = "/data/plugin/profile";
    let trace = format!("{base}/data?run=run/s&tag=trace_viewer@&host=tpu-vm-demo-host-0");
    for invalid in ["start_time_ms=invalid", "end_time_ms=", "resolution=x"] {
        assert_eq!(fetch(&state, &format!("{trace}&{invalid}")).await.0, StatusCode::NOT_FOUND, "{invalid}");
    }
    for valid in ["start_time_ms=%205", "end_time_ms=1e3", "resolution=0"] {
        assert_eq!(fetch(&state, &format!("{trace}&{valid}")).await.0, StatusCode::OK, "{valid}");
    }
    let sessions = dir.join("run/plugins/profile");
    assert_eq!(fetch(&state, &format!("{base}/runs?run_path={}", sessions.display())).await.2, "[\"s\"]");
    assert_eq!(fetch(&state, &format!("{base}/runs?session_path={}", sessions.join("s").display())).await.2, "[\"s\"]");
    assert_eq!(fetch(&state, &format!("{base}/runs?session_path={}", sessions.display())).await.2, "[]");
    assert_eq!(fetch(&state, &format!("{base}/hosts?run=s&run_path={}&tag=trace_viewer@", sessions.display())).await.2, "[{\"hostname\": \"tpu-vm-demo-host-0\"}]");
    let (status, _, message) = fetch(&state, &format!("{base}/data_csv?run=run/s&tag=memory_viewer&host=ALL_HOSTS")).await;
    assert_eq!((status, message.as_str()), (StatusCode::NOT_FOUND, "No Data Found"));
    let (status, _, message) = fetch(&state, &format!("{base}/data_csv?run=run/s&tag=trace_viewer@&host=tpu-vm-demo-host-0")).await;
    assert_eq!((status, message.as_str()), (StatusCode::INTERNAL_SERVER_ERROR, "Data format not suitable for CSV (missing 'cols')"));
    let (status, _, modules) = fetch(&state, &format!("{base}/module_list?run=run/s")).await;
    assert_eq!(status, StatusCode::OK);
    let module = modules.split(',').next().unwrap().replace('(', "%28").replace(')', "%29");
    let graph = format!("{base}/data?run=run/s&host=ALL_HOSTS&tag=graph_viewer");
    for (query, message) in [
        (String::new(), "Can not load hlo proto from options."),
        (format!("&module_name={module}"), "Graph viewer must provide a type option."),
        (format!("&module_name={module}&type=bogus"), "Unknown graph viewer type option: bogus"),
        (format!("&module_name={module}&type=graph"), "node_name should not be empty"),
        (format!("&module_name={module}&type=adj_nodes&node_name=nope"), "Couldn't find HloInstruction or HloComputation named nope."),
        (format!("&module_name={module}&type=json"), "Not implemented"),
        (format!("&module_name={module}&type=custom_call_graph&node_name=x"), "Unknown graph viewer type option: custom_call_graph"),
        ("&module_name=nope&type=short_txt".into(), "nope.hlo_proto.pb; No such file or directory"),
        ("&program_id=zzz&type=short_txt".into(), "HLO proto file containing program ID zzz not found in"),
        ("&type=graph&node_name=nope".into(), "HLO proto file containing node name nope not found in"),
    ] {
        let (status, _, body) = fetch(&state, &format!("{graph}{query}")).await;
        assert!(status == StatusCode::INTERNAL_SERVER_ERROR && body.contains(message), "{query}: {status} {body}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routes_answer_at_the_root_and_under_the_plugin_prefix_and_generate_cache_validates_like_xprof() {
    use tower::ServiceExt;
    let dir = logdir("routes");
    std::fs::write(dir.join("run/plugins/profile/s/h.xplane.pb"), include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    let state = state(&Settings { logdir: dir.clone(), src_prefix: Some("/src".into()), hide_capture_profile_button: true, ..Default::default() });
    let config = "{\"enableTabNameLabel\": false, \"hideCaptureProfileButton\": true, \"srcPathPrefix\": \"/src\"}";
    for uri in ["/config", "/data/plugin/profile/config"] {
        assert_eq!(fetch(&state, uri).await.2, config, "{uri}");
    }
    assert_eq!(fetch(&state, "/runs").await.2, "[\"run/s\"]");
    assert_eq!(fetch(&state, "/local").await.0, StatusCode::OK);
    let post = |uri: String| app(state.clone()).oneshot(axum::http::Request::builder().method("POST").uri(uri).body(Body::empty()).unwrap());
    assert_eq!(fetch(&state, "/generate_cache?session_path=x").await.0, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(post("/generate_cache".into()).await.unwrap().status(), StatusCode::BAD_REQUEST);
    assert_eq!(post(format!("/generate_cache?session_path={}", dir.display())).await.unwrap().status(), StatusCode::NOT_FOUND);
    let session = dir.join("run/plugins/profile/s");
    assert_eq!(post(format!("/generate_cache?session_path={}&tools=bogus", session.display())).await.unwrap().status(), StatusCode::BAD_REQUEST);
    let accepted = post(format!("/generate_cache?session_path={}", session.display())).await.unwrap();
    assert_eq!(accepted.status(), StatusCode::ACCEPTED);
    assert_eq!(&axum::body::to_bytes(accepted.into_body(), usize::MAX).await.unwrap()[..], b"{\"message\": \"Cache generation started\", \"status\": \"ACCEPTED\"}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roofline_model_follows_the_host_selection() {
    let dir = logdir("roofline");
    for (host, space) in [("a", include_bytes!("../../tests/data/demo.xplane.pb").to_vec()), ("b", annotated_space())] {
        std::fs::write(dir.join(format!("run/plugins/profile/s/{host}.xplane.pb")), space).unwrap();
    }
    let state = state(&Settings { logdir: dir.clone(), ..Default::default() });
    let uri = |host: &str| format!("/data/plugin/profile/data?run=run/s&tag=roofline_model&host={host}");
    let (status, _, combined) = fetch(&state, &uri("ALL_HOSTS")).await;
    assert_eq!(status, StatusCode::OK, "{combined}");
    let alone = served("roofline-alone", &[("a", include_bytes!("../../tests/data/demo.xplane.pb").to_vec())], "tag=roofline_model&host=a").await.2;
    assert_eq!(fetch(&state, &uri("a")).await.2, alone);
    assert_ne!(alone, combined);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn non_streaming_trace_viewer_matches_xprofs_json_stream() {
    let dir = logdir("legacy-trace");
    let file = dir.join("run/plugins/profile/s/tpu-vm-demo-host-0.xplane.pb");
    std::fs::write(&file, include_bytes!("../../tests/data/demo.xplane.pb")).unwrap();
    let (map, planes) = prepare(&file, true).unwrap();
    let mut golden = String::new();
    flate2::read::GzDecoder::new(&include_bytes!("../../tests/data/trace_viewer_legacy.json.gz")[..]).read_to_string(&mut golden).unwrap();
    let flow = |text: &str| {
        text.lines()
            .map(|line| {
                line.split("\"flow\": \"")
                    .enumerate()
                    .map(|(index, part)| if index == 0 { part } else { part.trim_start_matches(|c: char| c.is_ascii_digit()) })
                    .collect::<Vec<_>>()
                    .join("\"flow\": \"")
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(flow(&trace::legacy::render(&planes, &map)), flow(&golden));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flows_of_hosts_with_different_hostnames_stay_apart_like_xprof() {
    let demo = include_bytes!("../../tests/data/demo.xplane.pb");
    let named = |hostname: &str| -> Vec<u8> {
        let planes = xplane::fields(demo).filter_map(|(tag, field)| if let (1..=3, xplane::Field::Bytes(_, body)) = (tag, field) { Some(bytes(tag as u64, body)) } else { None });
        planes.chain([bytes(4, hostname.as_bytes())]).flatten().collect()
    };
    let flows = |body: &str| {
        let mut hosts: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        for event in serde_json::from_str::<serde_json::Value>(body).unwrap()["traceEvents"].as_array().unwrap() {
            if let Some(id) = event.get("bind_id").and_then(serde_json::Value::as_i64) {
                hosts.entry(id).or_default().insert(event["pid"].as_i64().unwrap() / 1000);
            }
        }
        hosts.into_values().map(|hosts| hosts.len()).collect::<Vec<_>>()
    };
    let query = "tag=trace_viewer@&hosts=a,b&resolution=0";
    let (status, _, apart) = served("flows-apart", &[("a", named("host-a")), ("b", named("host-b"))], query).await;
    let (_, _, merged) = served("flows-merged", &[("a", named("host")), ("b", named("host"))], query).await;
    let (apart, merged) = (flows(&apart), flows(&merged));
    assert!(status == StatusCode::OK && !merged.is_empty() && apart.len() == 2 * merged.len(), "{} {}", apart.len(), merged.len());
    assert!(apart.iter().all(|&hosts| hosts == 1) && merged.iter().all(|&hosts| hosts == 2));
}

#[test]
fn accept_encoding_weights() {
    let accepts = |value: &str| accepts_gzip(&HeaderMap::from_iter([(header::ACCEPT_ENCODING, HeaderValue::from_str(value).unwrap())]));
    for (value, expected) in [("gzip, deflate, br", true), ("GZIP;q=0.5", true), ("*", true), ("gzip;q=0, identity", false), ("gzip;q=0, *", false), ("identity", false), ("br, *;q=0", false)] {
        assert_eq!(accepts(value), expected, "{value}");
    }
}
