use crate::tests::xspace::{XEventMetadata, XSpace, op_stats};

fn tpu_space() -> XSpace {
    let mut space = XSpace::default();
    let plane = space.plane("/device:TPU:0");
    plane.id = 1;
    plane.named_line(0, "XLA Ops");
    plane.event(0, "fusion.1", 0, 50, &[]);
    space
}

#[test]
fn metadata_that_is_its_own_child() {
    let mut space = tpu_space();
    let plane = &mut space.planes[0];
    let meta = plane.event_metadata("fusion.1");
    meta.child_id = vec![meta.id];
    assert!(op_stats(&[space]).is_some());
}

#[test]
fn metadata_with_a_very_deep_child_chain() {
    let mut space = tpu_space();
    let plane = &mut space.planes[0];
    let first = plane.event_metadata("fusion.1").id;
    for id in first..first + 200_000 {
        plane.event_metadata.entry(id).or_default().child_id = vec![id + 1];
        plane.event_metadata.insert(id + 1, XEventMetadata { id: id + 1, name: format!("child.{id}"), ..Default::default() });
    }
    assert!(op_stats(&[space]).is_some());
}

#[test]
fn metadata_with_a_display_name_and_no_name() {
    let mut space = tpu_space();
    let plane = &mut space.planes[0];
    plane.event_metadata.insert(9, XEventMetadata { id: 9, display_name: "shown".into(), ..Default::default() });
    plane.lines[0].events[0].metadata_id = 9;
    let (map, planes) = space.parsed();
    let meta = &planes[0].meta[9];
    assert_eq!(meta.long_name(&map).as_ref(), "");
    assert!(op_stats(&[space]).is_some());
}

#[test]
fn threadpool_region_at_the_origin() {
    use crate::tests::xspace::V;
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.event(0, "ThreadpoolListener::Record", 100, 100, &[("_pt", 15.into()), ("_p", 123.into())]);
    let consumer = [("_ct", V::from(15)), ("_c", 123.into())];
    plane.event(1, "ThreadpoolListener::StartRegion", 0, 0, &consumer);
    plane.event(1, "ThreadpoolListener::StopRegion", 300, 0, &consumer);
    plane.lines.iter_mut().for_each(|line| line.timestamp_ns = 5);
    let (map, mut planes) = space.parsed();
    planes[0].add_threadpool_regions(&map);
    let plane = &planes[0];
    assert!(plane.lines.iter().flat_map(|line| &line.events).any(|event| &*plane.meta[event.meta as usize].name == "ThreadpoolListener::Region"));
}

#[test]
fn metadata_id_from_the_map_key() {
    let mut space = tpu_space();
    space.planes[0].event_metadata.get_mut(&1).unwrap().id = 0;
    let (_, planes) = space.parsed();
    assert_eq!(&*planes[0].meta[1].name, "fusion.1");
}

#[test]
fn literals_that_are_not_valid() {
    use crate::hlo::xla::{LayoutProto, LiteralProto, ShapeProto};
    let module = crate::hlo::Module::parse(std::borrow::Cow::Owned(std::fs::read(format!("{}/tests/data/literal/literals.pb", env!("CARGO_MANIFEST_DIR"))).unwrap()));
    let printer = crate::hlo::text::Printer::new(&module, crate::hlo::text::Style::Long, false);
    let shape = |element_type, dimensions: Vec<i64>, layout: Option<Vec<i64>>| ShapeProto {
        element_type,
        dimensions,
        layout: layout.map(|minor_to_major| Box::new(LayoutProto { minor_to_major, ..Default::default() })),
        ..Default::default()
    };
    let cases = [
        LiteralProto { shape: Some(shape(11, vec![2], Some(vec![5]))), f32s: vec![1.0, 2.0], ..Default::default() },
        LiteralProto { shape: Some(shape(11, vec![3, 2], Some(vec![0, 0]))), f32s: vec![1.0; 6], ..Default::default() },
        LiteralProto { shape: Some(shape(3, vec![], None)), s16s: vec![1, 2, 3], ..Default::default() },
        LiteralProto { shape: Some(shape(15, vec![], None)), c64s: vec![1.0, 2.0, 3.0], ..Default::default() },
        LiteralProto { shape: Some(shape(11, vec![1; 200_000], None)), f32s: vec![1.0], ..Default::default() },
        LiteralProto { shape: Some(shape(11, vec![1 << 40, 0], None)), ..Default::default() },
        LiteralProto { shape: Some(shape(11, vec![1 << 32, 1 << 32], None)), ..Default::default() },
        LiteralProto { shape: Some(shape(11, vec![-1, -1], None)), f32s: vec![1.0], ..Default::default() },
    ];
    for literal in cases {
        printer.literal(&literal, true, &mut String::new());
    }
    let mut text = String::new();
    printer.literal(&LiteralProto { shape: Some(shape(11, vec![2], Some(vec![0]))), f32s: vec![1.0, 2.0], ..Default::default() }, true, &mut text);
    assert_eq!(text, "{1, 2}");
}

#[test]
fn cli_values_with_deep_brackets_or_many_signs_stay_text() {
    use crate::cli::{json::J, literal};
    let nested = |depth: usize| format!("{}1{}", "(".repeat(depth), ")".repeat(depth));
    assert_eq!(literal(&nested(200)), J::Int(1));
    for text in [nested(201), nested(100_000), format!("{}1", "-".repeat(100_000)), "-True".into(), "--1".into(), "-(-1)".into(), "+-1".into(), "[1, --2]".into()] {
        assert_eq!(literal(&text), J::Str(text.clone()));
    }
    assert_eq!(literal("-( 1)"), J::Int(-1));
}

#[test]
fn cli_trace_that_cannot_be_read() {
    use crate::cli::client::{Client, Local};
    let dir = crate::tests::scratch("cli-unreadable-trace");
    std::os::unix::fs::symlink(dir.join("missing"), dir.join("h.xplane.pb")).unwrap();
    assert_eq!(Local::default().fetch("overview_page", &dir.to_string_lossy(), &[]), Ok(None));
}

#[test]
fn recovery_module_that_is_not_valid() {
    use crate::hlo::cost::tests::{computation, parameter, shape};
    use crate::hlo::xla::original_value_recovery_table_proto::Entry;
    use crate::hlo::xla::{HloComputationProto, HloModuleProto, HloProto, OriginalValueRecoveryTableProto, ProgramShapeProto};
    let recovery = HloModuleProto {
        entry_computation_id: 7,
        computations: vec![HloComputationProto { id: 7, ..Default::default() }],
        host_program_shape: Some(ProgramShapeProto::default()),
        ..Default::default()
    };
    let table = OriginalValueRecoveryTableProto { entries: vec![Entry { recovery_module: Some(recovery), ..Default::default() }] };
    let hlo_module = HloModuleProto {
        name: "m".into(),
        entry_computation_id: 1,
        computations: vec![computation(1, vec![parameter(1, 0, shape(11, &[2]))])],
        host_program_shape: Some(ProgramShapeProto::default()),
        original_value_recovery_table: Some(table),
        ..Default::default()
    };
    let module = crate::tests::hlo_fixture::module(&HloProto { hlo_module: Some(hlo_module), ..Default::default() });
    assert!(module.valid);
    assert_eq!(crate::hlo::text::Printer::new(&module, crate::hlo::text::Style::Long, false).module_text(), None);
}

#[test]
fn provenance_with_many_parts() {
    let text = format!(
        "device_op_metrics_db {{ metrics_db {{ name: \"op\" category: \"fusion\" provenance: \"{}\" occurrences: 1 time_ps: 10 self_time_ps: 10 }} total_time_ps: 10 total_op_time_ps: 10 }}",
        "a/".repeat(100_000)
    );
    let stats = crate::tests::opstats_adapter::op_stats(&text);
    let json = std::thread::Builder::new().stack_size(1 << 20).spawn(move || crate::tools::op_profile::json(&stats, Some("provenance"))).unwrap().join().unwrap();
    assert_eq!(json.matches("\"name\":\"a\"").count(), 2 * 100_000);
}

#[test]
fn hlo_text_proto_with_deep_nesting() {
    use crate::hlo::xla::{HloComputationProto, HloInstructionProto, HloModuleProto, HloProto, ShapeProto};
    use prost::Message;
    use prost::encoding::{encode_varint, encoded_len_varint};
    let wrap = |tag: Vec<u8>, inner: Vec<u8>| {
        let mut out = vec![tag[0]];
        encode_varint(inner.len() as u64, &mut out);
        out.extend(inner);
        out
    };
    let hlo = |shapes: usize| {
        let tuple_shape = ShapeProto { tuple_shapes: vec![ShapeProto::default()], ..Default::default() }.encode_to_vec()[0];
        let mut lengths = vec![0u64];
        for _ in 1..shapes {
            let inner = *lengths.last().unwrap();
            lengths.push(inner + 1 + encoded_len_varint(inner) as u64);
        }
        let mut chain = Vec::new();
        for &length in lengths.iter().rev().skip(1) {
            chain.push(tuple_shape);
            encode_varint(length, &mut chain);
        }
        let shape = wrap(HloInstructionProto { shape: Some(Box::default()), ..Default::default() }.encode_to_vec(), chain);
        let instruction = wrap(HloComputationProto { instructions: vec![HloInstructionProto::default()], ..Default::default() }.encode_to_vec(), shape);
        let computation = wrap(HloModuleProto { computations: vec![HloComputationProto::default()], ..Default::default() }.encode_to_vec(), instruction);
        wrap(HloProto { hlo_module: Some(HloModuleProto::default()), ..Default::default() }.encode_to_vec(), computation)
    };
    assert!(HloProto::decode(hlo(97).as_slice()).is_ok() && HloProto::decode(hlo(98).as_slice()).is_err());
    assert!(crate::hlo::proto_text::print_hlo(&hlo(97)).is_some_and(|text| text.matches("tuple_shapes {").count() == 96));
    assert!(crate::hlo::proto_text::print_hlo(&hlo(98)).is_none());
    assert!(crate::hlo::proto_text::print_hlo(&hlo(100_000)).is_none());
}

#[test]
fn memory_viewer_with_many_live_buffers() {
    use crate::hlo::xla::buffer_allocation_proto::Assigned;
    use crate::hlo::xla::heap_simulator_trace::Event;
    use crate::hlo::xla::logical_buffer_proto::Location;
    use crate::hlo::xla::{BufferAllocationProto, BufferAssignmentProto, HeapSimulatorTrace, HloComputationProto, HloInstructionProto, HloModuleProto, HloProto, LogicalBufferProto};
    let count = 100_000;
    let instructions = (0..count).map(|id| HloInstructionProto { name: format!("f.{id}"), id, ..Default::default() }).collect();
    let hlo_module = HloModuleProto { name: "m".into(), computations: vec![HloComputationProto { name: "c".into(), instructions, ..Default::default() }], ..Default::default() };
    let size = |id: i64| if id == 1 { 3 << 20 } else { 1 << 20 };
    let logical_buffers =
        (0..count).map(|id| LogicalBufferProto { id: id + 1, size: size(id + 1), defined_at: Some(Location { instruction_id: id, ..Default::default() }), ..Default::default() }).collect();
    let assigned = (0..count).map(|id| Assigned { logical_buffer_id: id + 1, offset: id << 22, size: size(id + 1), ..Default::default() }).collect();
    let event = |kind: i32, id: i64| Event { kind, buffer_id: id + 1, ..Default::default() };
    let mut events = vec![event(0, 0), event(1, 0)];
    events.extend((1..count).map(|id| event(0, id)));
    events.extend((1..count).map(|id| event(1, id)));
    let assignment = BufferAssignmentProto {
        logical_buffers,
        buffer_allocations: vec![BufferAllocationProto { index: 0, size: count << 22, assigned, ..Default::default() }],
        heap_simulator_traces: vec![HeapSimulatorTrace { events, ..Default::default() }],
        ..Default::default()
    };
    let module = crate::tests::hlo_fixture::module(&HloProto { hlo_module: Some(hlo_module), buffer_assignment: Some(assignment) });
    let (body, _) = crate::hlo::memory::render(&module, 0, 16 * 1024, false).unwrap();
    let result: serde_json::Value = serde_json::from_str(&body).unwrap();
    let ids: Vec<i64> = result["maxHeap"].as_array().unwrap().iter().map(|object| object["logicalBufferId"].as_i64().unwrap()).collect();
    assert_eq!(ids, (2..=count).collect::<Vec<_>>());
    assert_eq!(result["peakHeapSizePosition"].as_i64(), Some(count));
}

#[test]
fn json_deeper_than_the_serde_limit() {
    use crate::cli::json::J;
    let wrap = |text: &str, depth: usize| format!("{}{text}{}", "[ ".repeat(depth), "\n]".repeat(depth));
    let nest = |value: J, depth: usize| (0..depth).fold(value, |inner, _| J::List(vec![inner]));
    for text in [
        "null",
        "true",
        "-0",
        "12345678901234567890123",
        "1.5e300",
        "-7",
        "1e400",
        "01",
        "1.",
        "-",
        "nul",
        "\"\\ud83d\\ude00\"",
        "\"\\ud800\"",
        "\"a\\\"b\\\\\"",
        "\"\\x\"",
        "\"\n\"",
        "{\"k\": 1, \"j\": [], \"k\": {\"x\": 2}}",
        "{\"k\" 1}",
        "{1: 2}",
        "[1, 2,]",
        "[1 2]",
        "{}",
        "[]",
        "1 2",
        "[1]]",
    ] {
        let expected = serde_json::from_str::<J>(text).ok();
        assert_eq!(J::parse(&wrap(text, 300)), expected.map(|value| nest(value, 300)), "{text}");
    }
    assert!(J::parse(&wrap("1", 1000)).is_some() && J::parse(&wrap("1", 1001)).is_none());
    let big = format!("{}{}", "[".repeat(1 << 20), "]".repeat(1 << 20));
    assert_eq!(std::thread::Builder::new().stack_size(2 << 20).spawn(move || J::parse(&big)).unwrap().join().unwrap(), None);
}

#[test]
fn op_profile_deeper_than_the_serde_limit() {
    use crate::cli::json::J;
    use crate::cli::ops::{get_hlo_op_profile, get_top_hlo_ops};
    use crate::tests::cli_support::{Fake, args, json};
    let chain = |nodes: usize| {
        let open = "{\"name\": \"n\", \"metrics\": {\"rawTime\": 1}, \"xla\": {\"category\": \"c\"}, \"children\": [".repeat(nodes);
        format!("{{\"byCategory\": {}}}{}}}", &open[..open.len() - ", \"children\": [".len()], "]}".repeat(nodes - 1))
    };
    let flat = json(get_hlo_op_profile(&Fake::fixed(&chain(98)), &args("s", &[("view", J::from("flat"))])));
    assert_eq!(flat.items()[0].at("name").text(), vec!["n"; 98].join("/"));
    let error = get_hlo_op_profile(&Fake::fixed(&chain(99)), &args("s", &[])).unwrap_err();
    assert_eq!(error.message, "Failed to parse op_profile proto: ParseError('Message too deep. Max recursion depth is 100')");
    let top = json(get_top_hlo_ops(&Fake::fixed(&chain(99)), &args("s", &[])));
    assert_eq!(top, J::Map(vec![("error".into(), J::from("Failed to parse JSON proto: Message too deep. Max recursion depth is 100"))]));
}

#[test]
fn request_paths_that_do_not_exist() {
    let (dir, outside) = (crate::tests::legacy::logdir("confine-missing"), crate::tests::legacy::logdir("confine-missing-outside"));
    std::os::unix::fs::symlink(outside.join("later"), dir.join("dangling")).unwrap();
    std::os::unix::fs::symlink(&outside, dir.join("out")).unwrap();
    let state = crate::state(&crate::Settings { logdir: dir.clone(), ..Default::default() });
    assert_eq!(crate::confine(&state, &dir.join("run/missing/s")), Some(dir.join("run/missing/s")));
    assert_eq!(crate::confine(&state, &dir.join("run/plugins/profile/s")), Some(dir.join("run/plugins/profile/s")));
    for path in ["dangling/s", "dangling", "out/missing", "missing/../run", "../missing"] {
        assert_eq!(crate::confine(&state, &dir.join(path)), None, "{path}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&outside).unwrap();
}
