use super::xspace::{V, XPlane, XSpace, grouped};
use crate::tools::opstats::templates;
use crate::xplane::Plane;
use crate::xplane::derive::derive_gpu;
use crate::xplane::steps::{HOST_WAIT_INPUT, SPARSE_CORE_START, Span, StepEvents, device_plane, gpu_device, host_steps};

const TFRT_TPU_RUNTIME: i64 = 7;

fn device_steps(planes: &[Plane], map: &[u8], index: usize) -> StepEvents {
    device_plane(&planes[index], &[], map, &templates(&planes[index], map), 0, "localhost").events
}

fn host_step_events(planes: &[Plane], map: &[u8], device: &StepEvents) -> StepEvents {
    let mut host = host_steps(&planes[0], map, 0);
    host.retain(|step, _| device.contains_key(step));
    host
}

#[test]
fn cpu_only_step_db_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.event(0, "TraceContext", 0, 100, &[("step_num", 123.into())]);
    host.event(0, "FunctionRun", 10, 90, &[("id", 0.into()), ("_pt", 1.into()), ("_p", 0.into())]);
    host.event(0, "TraceContext", 300, 100, &[("step_num", 456.into())]);
    host.event(0, "FunctionRun", 310, 90, &[("id", 1.into()), ("_pt", 1.into()), ("_p", 1.into())]);
    host.event(1, "ExecutorState::Process", 20, 20, &[("id", 0.into()), ("_ct", 1.into()), ("_c", 0.into())]);
    host.event(1, "matmul", 30, 10, &[("correlation_id", 100.into())]);
    host.event(1, "ExecutorState::Process", 320, 20, &[("id", 1.into()), ("_ct", 1.into()), ("_c", 1.into())]);
    host.event(1, "matmul", 330, 10, &[("correlation_id", 200.into())]);
    space.gpu(0).event(0, "matmul", 50, 40, &[("correlation_id", 100.into())]);
    let (map, mut planes, names) = grouped(&space);
    derive_gpu(&mut planes, &map, names.as_ref(), true);
    let device = gpu_device(&planes[1], &map, 0);
    assert_eq!(device.len(), 1);
    assert_eq!(device[&0].events.len(), 1);
    let host = host_step_events(&planes, &map, &device);
    assert_eq!(host.len(), 1);
    assert_eq!(host[&0].markers.len(), 1);
    assert_eq!(host[&0].events.len(), 2);
}

fn op_metadata(plane: &mut XPlane, name: &str, display: &str, symbol: i64) {
    plane.event_metadata(name).display_name = display.into();
    plane.metadata_stats(name, &[("program_id", 1.into()), ("symbol_id", symbol.into())]);
}

fn tpu_op_plane(step_line: bool, events: &[(&str, i64, i64)]) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.plane("/device:TPU:0");
    plane.id = 1;
    plane.named_line(0, if step_line { "XLA Ops" } else { "Steps" });
    plane.named_line(1, if step_line { "Steps" } else { "XLA Ops" });
    let ops = i64::from(!step_line);
    op_metadata(plane, "op_long_name", "op_name", 1);
    op_metadata(plane, "op_long_name2", "op_name2", 2);
    for &(name, offset, group) in events {
        plane.event(ops, name, offset, 50, &[("group_id", group.into())]);
    }
    if step_line {
        plane.event(1, "", 0, 100, &[("group_id", 1.into())]);
        plane.event(1, "", 100, 100, &[("group_id", 2.into())]);
    }
    space
}

#[test]
fn tpu_device_plane_to_step_events() {
    let (map, planes) = tpu_op_plane(true, &[("op_long_name", 0, 1), ("op_long_name", 100, 2), ("op_long_name2", 50, 1)]).parsed();
    let steps = device_steps(&planes, &map, 0);
    assert_eq!(steps.len(), 2);
    for step in [1, 2] {
        assert!(steps[&step].cores.contains_key(&1), "{step}");
        assert_eq!(steps[&step].markers.len(), 1);
    }
}

#[test]
fn sparse_core_should_have_step_markers() {
    let mut space = XSpace::default();
    let plane = space.tpu(0, "TPUv3", 0.0, 0.0, Some(1));
    plane.id = 1;
    plane.named_line(0, "Sparse Core Steps");
    plane.event(0, "Step 1", 0, 1000, &[("group_id", 1.into())]);
    plane.named_line(1, "Sparse Core Ops");
    plane.metadata_stats("sparse_op", &[("program_id", 1.into()), ("symbol_id", 1.into())]);
    plane.event(1, "sparse_op", 100, 880, &[("group_id", 1.into())]);
    let (map, planes) = space.parsed();
    let steps = device_steps(&planes, &map, 0);
    assert_eq!(steps.len(), 1);
    let step = &steps[&1];
    assert_eq!(step.markers.iter().map(|marker| marker.span).collect::<Vec<_>>(), [Span { begin: 0, duration: 1000 }]);
    assert_eq!(step.cores[&(1 + SPARSE_CORE_START)].1, 880);
}

#[test]
fn tpu_device_plane_no_step_line() {
    let (map, planes) = tpu_op_plane(false, &[("op_long_name", 0, 1), ("op_long_name", 100, 2), ("op_long_name2", 50, 1), ("op_long_name2", 150, 2)]).parsed();
    assert!(device_steps(&planes, &map, 0).is_empty());
}

#[test]
fn cpu_only_py_grain_single_process_input_pipeline_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.event(0, "read_data 1", 0, 100, &[("step_num", 1.into()), ("_r", 1.into())]);
    host.event(0, "_BatchDatasetIterator.__next__", 1, 90, &[("_ipl_stage_name", "Batch".into())]);
    host.event(0, "FirstFitPackDatasetIterator.__next__", 2, 88, &[("_ipl_stage_name", "FirstFitPack".into())]);
    for (producer, (offset, duration)) in [(3, 9), (13, 9), (23, 50), (74, 10), (85, 5)].into_iter().enumerate() {
        host.event(0, "PrefetchDatasetIterator.__next__", offset, duration, &[("_ipl_stage_name", "Prefetch".into()), ("_pt", 9999.into()), ("_p", (producer as i64 + 1).into())]);
    }
    host.event(0, "tpu::System::Execute", 92, 1, &[("_pt", TFRT_TPU_RUNTIME.into()), ("_p", 1.into())]);
    host.event(1, "tpu::System::Execute=>IssueSequencedEvent", 94, 4, &[("_ct", TFRT_TPU_RUNTIME.into()), ("_c", 1.into())]);
    host.event(1, "DoEnqueueProgram", 95, 2, &[("run_id", 1.into()), ("queue_id", 0.into()), ("device_ordinal", 0.into())]);
    for worker in 1..=5 {
        host.named_line(worker + 1, &format!("WorkerThread-{worker}"));
        host.event(worker + 1, "MapDataset.__getitem__", 2, 2, &[("_ipl_stage_name", "MapDataset".into()), ("_ct", 9999.into()), ("_c", V::from(worker))]);
    }
    let device = space.tpu(0, "TPUv4", 0.0, 0.0, None);
    device.named_line(0, "Steps");
    device.named_line(1, "XLA Modules");
    device.named_line(2, "XLA Ops");
    device.event(0, "read_data 1", 0, 100, &[]);
    device.event(1, "jit_multiply(1)", 97, 2, &[("run_id", 1.into()), ("replica_id", 0.into()), ("queue_id", 0.into())]);
    device.event(2, "multiply.1", 98, 1, &[]);
    let (map, planes, names) = grouped(&space);
    assert_eq!(names.unwrap().len(), 1);
    let device = device_steps(&planes, &map, 1);
    assert_eq!(device.len(), 1);
    for plane in &planes {
        for event in plane.lines.iter().flat_map(|line| &line.events) {
            assert_eq!(event.group, 0, "{} {}", plane.name, plane.meta[event.meta as usize].name);
        }
    }
    let host = host_step_events(&planes, &map, &device);
    assert_eq!(host.len(), 1);
    let step = host.values().next().unwrap();
    assert_eq!(step.markers.len(), 1);
    assert_eq!(step.events.len(), 15);
    let waits: Vec<Span> = step.events.iter().filter(|(kind, _)| *kind == HOST_WAIT_INPUT).map(|(_, span)| *span).collect();
    assert_eq!(waits, [Span { begin: 1, duration: 90 }]);
}
