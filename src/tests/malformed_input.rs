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
    let printer = crate::hlo_text::Printer::new(&module, crate::hlo_text::Style::Long, false);
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
    use crate::gpu_cost::tests::{computation, parameter, shape};
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
    assert_eq!(crate::hlo_text::Printer::new(&module, crate::hlo_text::Style::Long, false).module_text(), None);
}

#[test]
fn provenance_with_many_parts() {
    let text = format!(
        "device_op_metrics_db {{ metrics_db {{ name: \"op\" category: \"fusion\" provenance: \"{}\" occurrences: 1 time_ps: 10 self_time_ps: 10 }} total_time_ps: 10 total_op_time_ps: 10 }}",
        "a/".repeat(100_000)
    );
    let stats = crate::tests::opstats_adapter::op_stats(&text);
    let json = std::thread::Builder::new().stack_size(1 << 20).spawn(move || crate::op_profile::json(&stats, Some("provenance"))).unwrap().join().unwrap();
    assert_eq!(json.matches("\"name\":\"a\"").count(), 2 * 100_000);
}
