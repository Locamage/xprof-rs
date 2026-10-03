use super::xspace::{V, XSpace};
use crate::derive::{derive, derive_gpu, is_tensor_core};
use crate::xplane::{Ev, Line, NONE_GROUP, Plane};
use std::collections::{BTreeMap, HashMap};

const DERIVED_MIN: i64 = 0xdead_beef;
const STEP_INFO: i64 = DERIVED_MIN;
const TF_NAME_SCOPE: i64 = DERIVED_MIN + 2;
const TF_OP: i64 = DERIVED_MIN + 3;
const HLO_MODULE: i64 = DERIVED_MIN + 4;
const HLO_OP: i64 = DERIVED_MIN + 5;
const DEVICE_DERIVED_MIN: i64 = DERIVED_MIN + 290;

type Event<'a> = (&'a str, i64, i64, &'a [(&'a str, V)]);

fn generate(space: &XSpace, groups: &[(i64, &str)]) -> (Vec<u8>, Vec<Plane>) {
    let (map, mut planes) = space.parsed();
    let groups: HashMap<i64, String> = groups.iter().map(|&(id, name)| (id, name.to_string())).collect();
    derive_gpu(&mut planes, &map, Some(&groups), true);
    planes.iter_mut().filter(|plane| is_tensor_core(&plane.name)).for_each(|plane| derive(plane, &map));
    (map, planes)
}

fn event_name<'a>(plane: &'a Plane, line: &'a Line, event: &Ev) -> &'a str {
    if line.labels.is_empty() {
        return &plane.meta[event.meta as usize].name;
    }
    let long = &line.longs[event.meta as usize];
    if long.is_empty() { &line.labels[event.meta as usize] } else { long }
}

fn names_and_spans(plane: &Plane, line: &Line) -> Vec<(String, u64, u64)> {
    line.events.iter().map(|event| (event_name(plane, line, event).to_string(), event.ts, event.dur)).collect()
}

fn gpu_space(events: &[Event]) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.gpu(0);
    for &(name, offset, duration, stats) in events {
        plane.event(0, name, offset, duration, stats);
    }
    space
}

#[test]
fn empty_space_test() {
    let (_, planes) = generate(&XSpace::default(), &[]);
    assert!(planes.is_empty());
}

fn module_lines(scopes: [Option<i64>; 2]) -> Vec<(i64, Vec<String>)> {
    let stats = |scope: Option<i64>| {
        let mut stats = vec![("hlo_module", V::from("hlo_module")), ("kernel_details", "kernel_details".into())];
        stats.extend(scope.map(|scope| ("scope_range_id", scope.into())));
        stats
    };
    let (first, second) = (stats(scopes[0]), stats(scopes[1]));
    let (_, planes) = generate(&gpu_space(&[("op1", 0, 100, &first), ("op2", 200, 300, &second)]), &[]);
    let plane = &planes[0];
    plane.lines.iter().map(|line| (line.id, names_and_spans(plane, line).into_iter().map(|event| event.0).collect())).collect()
}

#[test]
fn hlo_module_name_test() {
    let lines = module_lines([None, None]);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1], (DEVICE_DERIVED_MIN + HLO_MODULE - TF_NAME_SCOPE, vec!["hlo_module".to_string()]));
}

#[test]
fn hlo_module_name_same_scope_range_id_test() {
    let lines = module_lines([Some(10), Some(10)]);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1], (DEVICE_DERIVED_MIN + HLO_MODULE - TF_NAME_SCOPE, vec!["hlo_module".to_string()]));
}

#[test]
fn hlo_module_name_different_scope_range_id_test() {
    let lines = module_lines([Some(10), Some(20)]);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1], (DEVICE_DERIVED_MIN + HLO_MODULE - TF_NAME_SCOPE, vec!["hlo_module".to_string(); 2]));
}

#[test]
fn no_hlo_module_name_test() {
    let details = [("kernel_details", V::from("kernel_details"))];
    let (_, planes) = generate(&gpu_space(&[("op1", 0, 100, &details), ("op2", 200, 300, &details), ("op3", 500, 100, &[("cuda_graph_exec_id", 1u64.into())])]), &[]);
    assert_eq!(planes[0].lines.len(), 1);
}

#[test]
fn tf_op_line_test() {
    let kernel = [("tf_op", V::from("mul:Mul")), ("kernel_details", "kernel_details".into())];
    let graph = [("tf_op", V::from("mul:Mul")), ("cuda_graph_exec_id", 1u64.into())];
    let (_, planes) = generate(&gpu_space(&[("op1", 0, 100, &kernel), ("op2", 200, 300, &kernel), ("op3", 500, 100, &graph)]), &[]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 2);
    assert_eq!(plane.lines[1].id - DEVICE_DERIVED_MIN, TF_OP - TF_NAME_SCOPE);
    assert_eq!(names_and_spans(plane, &plane.lines[1]), [("mul:Mul".to_string(), 0, 600)]);
}

#[test]
fn dependency_test() {
    let stats = |group: i64| [("group_id", V::from(group)), ("tf_op", "mul:Mul".into()), ("kernel_details", "kernel_details".into())];
    let (first, second) = (stats(0), stats(1));
    let (_, planes) = generate(&gpu_space(&[("op1", 0, 100, &first), ("op2", 200, 300, &second)]), &[(0, "train 0"), (1, "train 1")]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 3);
    for line in plane.lines.iter().filter(|line| line.id != 0) {
        assert!(line.id == STEP_INFO || line.id - DEVICE_DERIVED_MIN == TF_OP - TF_NAME_SCOPE, "{}", line.id);
        assert_eq!(line.events.len(), 2);
    }
}

#[test]
fn tf_op_name_scope_test() {
    let stats = [("tf_op", V::from("scope1/scope2/mul:Mul")), ("kernel_details", "kernel_details".into())];
    let (_, planes) = generate(&gpu_space(&[("op1", 0, 100, &stats), ("op2", 200, 300, &stats)]), &[]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 3);
    for line in &plane.lines {
        if line.id == TF_NAME_SCOPE {
            assert_eq!(line.events.iter().map(|event| (event.ts, event.dur)).collect::<Vec<_>>(), [(0, 500); 2]);
        } else if line.id == TF_OP {
            assert_eq!(names_and_spans(plane, line), [("scope1/scope2/mul:Mul".to_string(), 0, 500)]);
        }
    }
}

#[test]
fn tf_name_scope_maintains_order() {
    let mut space = XSpace::default();
    space.tpu(0, "TPU V4", 0.0, 0.0, None).event(0, "op1", 0, 10000, &[("tf_op", "scope1/scope2/mul:Mul".into()), ("kernel_details", "kernel_details".into())]);
    let (_, planes) = generate(&space, &[]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 3);
    let scopes = plane.lines.iter().find(|line| line.name == "Framework Name Scope").unwrap();
    assert_eq!(scopes.events.iter().map(|event| (event.ts, event.dur)).collect::<Vec<_>>(), [(0, 10000), (0, 9000)]);
}

#[test]
fn only_derived_events_from_all_lines() {
    let first = "stream1scope1/stream1scope2/mul:Mul1";
    let second = "stream2scope1/stream2scope2/mul:Mul2";
    let mut space = XSpace::default();
    let plane = space.gpu(0);
    plane.event(0, "op1", 0, 100, &[("tf_op", first.into()), ("kernel_details", "kernel_details".into())]);
    plane.event(0, "op2", 200, 300, &[("tf_op", first.into()), ("kernel_details", "kernel_details".into())]);
    plane.event(1, "op3", 50, 850, &[("tf_op", second.into()), ("kernel_details", "kernel_details".into())]);
    let (_, planes) = generate(&space, &[]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 6);
    let actual: BTreeMap<&str, BTreeMap<String, (u64, u64)>> = plane
        .lines
        .iter()
        .filter(|line| line.name.starts_with("Framework"))
        .map(|line| (line.name.as_str(), names_and_spans(plane, line).into_iter().map(|(name, ts, dur)| (name, (ts, dur))).collect()))
        .collect();
    let expected = BTreeMap::from([
        ("Framework Name Scope - from #0", BTreeMap::from([("stream1scope1".to_string(), (0, 500)), ("stream1scope2".to_string(), (0, 500))])),
        ("Framework Ops - from #0", BTreeMap::from([(first.to_string(), (0, 500))])),
        ("Framework Name Scope - from #1", BTreeMap::from([("stream2scope1".to_string(), (50, 850)), ("stream2scope2".to_string(), (50, 850))])),
        ("Framework Ops - from #1", BTreeMap::from([(second.to_string(), (50, 850))])),
    ]);
    assert_eq!(actual, expected);
}

#[test]
fn tf_op_name_scope_shrink_test() {
    let blah = |tf_op: &'static str| vec![("tf_op", V::from(tf_op)), ("kernel_details", "blah".into())];
    let (first, second) = (blah("a/b/c/Add:Add"), blah("a/d/Mul:Mul"));
    let (_, planes) = generate(&gpu_space(&[("op1", 0, 10000, &first), ("op2", 20000, 30000, &second)]), &[]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 3);
    for line in plane.lines.iter().filter(|line| line.id == TF_NAME_SCOPE) {
        let durations: BTreeMap<String, u64> = names_and_spans(plane, line).into_iter().map(|(name, _, dur)| (name, dur)).collect();
        assert_eq!(durations, BTreeMap::from([("a".into(), 50000), ("b".into(), 10000), ("c".into(), 9000), ("d".into(), 30000)]));
    }
    let (first, second, third) = (blah("a/b/c/d/e/Add:Add"), blah("a/b/c/d/f/Sub:Sub"), blah("a/g/Mul:Mul"));
    let (_, planes) = generate(&gpu_space(&[("op1", 0, 10000, &first), ("op2", 10000, 2000, &second), ("op3", 20000, 30000, &third)]), &[]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 3);
    let scopes = plane.lines.iter().find(|line| line.id == DEVICE_DERIVED_MIN).unwrap();
    assert_eq!(scopes.events.len(), 7);
    let durations: BTreeMap<String, u64> = names_and_spans(plane, scopes).into_iter().map(|(name, _, dur)| (name, dur)).collect();
    let expected = [("a", 50000), ("b", 12000), ("c", 11000), ("d", 11000), ("e", 10000), ("f", 1000), ("g", 30000)].map(|(name, dur)| (name.to_string(), dur));
    assert_eq!(durations, BTreeMap::from(expected));
}

#[test]
fn xlo_op_has_cuda_graph_stats() {
    let stats = [
        ("kernel_details", V::from("kernel_details")),
        ("group_id", 1.into()),
        ("hlo_module", "module".into()),
        ("hlo_op", "op_level_2".into()),
        ("correlation_id", 10000.into()),
        ("cuda_graph_id", 20u64.into()),
    ];
    let (_, planes) = generate(&gpu_space(&[("op1", 0, 100, &stats), ("op2", 200, 300, &stats)]), &[]);
    let lines: Vec<&Line> = planes[0].lines.iter().filter(|line| line.id - DEVICE_DERIVED_MIN == HLO_OP - TF_NAME_SCOPE).collect();
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].events.len(), 1);
    let args = &lines[0].args[&0];
    assert!(args.contains(&"\"correlation_id\":10000".to_string()) && args.contains(&"\"cuda_graph_id\":20".to_string()), "{args:?}");
}

#[test]
fn merge_and_no_merge() {
    let tf_op = "abc:model/layer/MatMul_1";
    let mut space = XSpace::default();
    let plane = space.tpu(0, "DummyTPU", 1.0, 1.0, None);
    for (name, offset, duration) in [("op1", 0, 100), ("op2", 200, 300), ("op3", 1501, 300)] {
        plane.event(0, name, offset, duration, &[("hlo_module", "Framework Ops".into()), ("tf_op", tf_op.into())]);
    }
    let (_, planes) = generate(&space, &[]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 2);
    assert_eq!(names_and_spans(plane, &plane.lines[1]).into_iter().map(|event| event.0).collect::<Vec<_>>(), [tf_op; 2]);
}

#[test]
fn ensure_all_gpu_events_are_grouped() {
    let stats = |group: Option<i64>| {
        let mut stats = vec![("tf_op", V::from("mul:Mul")), ("kernel_details", "kernel_details".into())];
        stats.extend(group.map(|group| ("group_id", group.into())));
        stats
    };
    let (first, second, eager) = (stats(Some(0)), stats(Some(1)), stats(None));
    let (map, planes) = generate(&gpu_space(&[("op1", 0, 100, &first), ("op2", 200, 300, &second), ("op3", 600, 100, &eager)]), &[(0, "train 0"), (1, "train 1")]);
    let plane = &planes[0];
    assert_eq!(plane.lines.len(), 3);
    for line in &plane.lines {
        for event in &line.events {
            let grouped = event.group != NONE_GROUP || (line.labels.is_empty() && plane.stat(&map, event.meta, event.raw, "group_id").is_some());
            assert!(grouped, "{} {}", line.name, event_name(plane, line, event));
        }
    }
}

#[test]
fn multi_threaded_tensor_core_plane_processing() {
    let mut space = XSpace::default();
    for ordinal in 0..4 {
        let plane = space.tpu(ordinal, "TPU V4", 0.0, 0.0, None);
        for index in 0..10 {
            plane.event(0, "kernel", (9 - index) * 200, 100, &[("tf_op", format!("MyOp:{ordinal}").into())]);
        }
    }
    let (_, planes) = generate(&space, &[]);
    for ordinal in 0..4 {
        let plane = planes.iter().find(|plane| plane.name == format!("/device:TPU:{ordinal}")).unwrap();
        let original = plane.lines.iter().find(|line| line.id == 0).unwrap();
        assert_eq!(original.events.len(), 10);
        assert!(original.events.windows(2).all(|pair| pair[0].ts <= pair[1].ts));
        let ops = plane.lines.iter().find(|line| line.name == "Framework Ops").unwrap();
        assert_eq!(names_and_spans(plane, ops), [(format!("MyOp:{ordinal}"), 0, 1900)]);
    }
}
