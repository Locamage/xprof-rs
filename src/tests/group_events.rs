use super::xspace::{V, XPlane, XSpace, group_id, grouped, name};
use crate::xplane::Plane;
use std::collections::{BTreeMap, HashMap};

const TF_EXECUTOR: i64 = 2;
const SHARED_BATCH_SCHEDULER: i64 = 4;
const TFRT_TPU_RUNTIME: i64 = 7;
const SC_OFFLOAD: i64 = 17;

fn groups(plane: &Plane) -> Vec<(String, Option<i64>)> {
    plane.lines.iter().flat_map(|line| line.events.iter().map(move |event| (format!("{} {}", line.name, name(plane, event)), group_id(event)))).collect()
}

fn gpu_trace(root: &str, root_stats: &[(&str, V)]) -> XSpace {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(1);
    host.event(0, root, 0, 100, root_stats);
    host.event(0, "FunctionRun", 10, 90, &[("id", 0.into()), ("_pt", TF_EXECUTOR.into()), ("_p", 0.into())]);
    host.event(1, "ExecutorState::Process", 20, 80, &[("id", 0.into()), ("_ct", TF_EXECUTOR.into()), ("_c", 0.into())]);
    host.event(1, "matmul", 30, 70, &[("correlation_id", 100.into())]);
    space.add_plane().event(0, "matmul", 200, 300, &[("correlation_id", 100.into())]);
    space
}

#[test]
fn group_gpu_trace_legacy_root_test() {
    let space = gpu_trace("TraceContext", &[("graph_type", "train".into()), ("step_num", 123.into())]);
    let (_, planes, names) = grouped(&space);
    let kernel = &planes[1].lines[0].events[0];
    assert_eq!((group_id(kernel), kernel.eager.is_some()), (Some(0), true));
    assert_eq!(names.unwrap(), HashMap::from([(0, "train 123".to_string())]));
}

#[test]
fn group_gpu_trace_test() {
    let space = gpu_trace("train", &[("step_num", 123.into()), ("_r", 1.into())]);
    let (_, planes, names) = grouped(&space);
    let kernel = &planes[1].lines[0].events[0];
    assert_eq!((group_id(kernel), kernel.eager.is_some()), (Some(0), true));
    assert_eq!(names.unwrap(), HashMap::from([(0, "train 123".to_string())]));
}

#[test]
fn group_tensor_flow_loop_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(1);
    let consumer = |iter: i64| [("id", V::from(0)), ("iter_num", iter.into()), ("_ct", TF_EXECUTOR.into()), ("_c", 0.into())];
    host.event(0, "ExecutorState::Process", 5, 10, &consumer(10));
    host.event(0, "ExecutorState::Process", 20, 80, &consumer(10));
    host.event(0, "matmul", 30, 70, &[("correlation_id", 100.into())]);
    let device = space.add_plane();
    device.event(0, "matmul", 200, 300, &[("correlation_id", 100.into())]);
    device.named_line(1, "Sync Flags");
    device.event(1, "SyncWait", 200, 300, &[("Raw Value", 10.into())]);
    let (_, planes, names) = grouped(&space);
    assert_eq!(group_id(&planes[1].lines[1].events[0]), None);
    assert_eq!(group_id(&planes[1].lines[0].events[0]), Some(0));
    assert_eq!(names.unwrap(), HashMap::from([(0, "10".to_string())]));
}

#[test]
fn group_multiple_tensor_flow_loops_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(1);
    let consumer = |step: i64, iter: i64| [("id", V::from(step)), ("iter_num", iter.into()), ("_ct", TF_EXECUTOR.into()), ("_c", step.into())];
    host.event(0, "ExecutorState::Process", 220, 80, &consumer(1, 0));
    host.event(0, "ExecutorState::Process", 320, 80, &consumer(1, 1));
    host.event(1, "ExecutorState::Process", 20, 80, &consumer(0, 10));
    host.event(1, "ExecutorState::Process", 120, 80, &consumer(0, 11));
    let (_, _, names) = grouped(&space);
    let expected = [(0, "10"), (1, "11"), (2, "0"), (3, "1")].map(|(id, name)| (id, name.to_string()));
    assert_eq!(names.unwrap(), expected.into_iter().collect());
}

#[test]
fn eager_op_test() {
    let mut space = XSpace::default();
    space.host().set_pid_if_not_set(1);
    space.add_plane();
    let (host, device) = space.planes.split_at_mut(1);
    let (host, device) = (&mut host[0], &mut device[0]);
    let launch = |host: &mut XPlane, device: &mut XPlane, launch: &str, kernel: &str, offset: i64, correlation: i64| {
        host.event(0, launch, offset + 10, offset + 90, &[("correlation_id", correlation.into())]);
        device.event(0, kernel, 200 + offset, 300 + offset, &[("correlation_id", correlation.into())]);
    };
    launch(host, device, "tf1 matmul", "tf1_kernel_matmul", 0, 100);
    host.event(0, "EagerExecute", 100, 200, &[]);
    launch(host, device, "legacy matmul", "legacy_kernel_matmul", 100, 101);
    host.event(0, "EagerExecute", 200, 300, &[("is_func", 0.into())]);
    launch(host, device, "eager op matmul", "eager_op_kernel_matmul", 200, 102);
    host.event(0, "EagerExecute", 300, 400, &[("is_func", 1.into())]);
    launch(host, device, "eager func matmul", "eager_func_kernel_matmul", 300, 103);
    host.event(0, "EagerExecute", 400, 500, &[("is_func", 0.into())]);
    host.event(0, "eager_op_cpu_kernel:Matmul", 410, 490, &[]);
    host.event(0, "EagerExecute", 500, 600, &[("is_func", 1.into())]);
    host.event(0, "eager_func_cpu_kernel:Matmul", 510, 590, &[]);
    let (_, planes, _) = grouped(&space);
    let eager = |plane: &Plane| -> BTreeMap<String, bool> { plane.lines.iter().flat_map(|line| line.events.iter()).map(|event| (name(plane, event).to_string(), event.eager == Some(true))).collect() };
    let (host, device) = (eager(&planes[0]), eager(&planes[1]));
    assert_eq!((host["eager_op_cpu_kernel:Matmul"], host["eager_func_cpu_kernel:Matmul"]), (true, false));
    let kernels = ["tf1_kernel_matmul", "legacy_kernel_matmul", "eager_op_kernel_matmul", "eager_func_kernel_matmul"].map(|kernel| device[kernel]);
    assert_eq!(kernels, [false, false, true, false]);
}

#[test]
fn function_op_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(1);
    host.event(0, "TraceContext", 0, 100, &[("step_num", 123.into())]);
    host.event(0, "EagerExecute", 10, 90, &[]);
    host.event(0, "FunctionRun", 10, 90, &[("id", 0.into()), ("_pt", TF_EXECUTOR.into()), ("_p", 0.into())]);
    host.event(1, "ExecutorState::Process", 20, 80, &[("id", 0.into()), ("_ct", TF_EXECUTOR.into()), ("_c", 0.into())]);
    host.event(1, "matmul", 30, 30, &[("correlation_id", 100.into())]);
    host.event(1, "add:Add", 70, 20, &[]);
    space.add_plane().event(0, "matmul", 200, 300, &[("correlation_id", 100.into())]);
    let (_, planes, _) = grouped(&space);
    assert_eq!(planes[0].lines[1].events[2].eager, Some(false));
    assert_eq!(planes[1].lines[0].events[0].eager, Some(false));
}

fn semantic_space(producer: V, consumer: V) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.set_pid_if_not_set(1);
    plane.event(0, "TraceContext", 0, 100, &[("_r", 1.into()), ("step_num", 100.into())]);
    plane.event(0, "FunctionRun", 10, 90, &[("_pt", 123.into()), ("_p", producer)]);
    plane.event(1, "ExecutorState::Process", 20, 80, &[("_ct", 123.into()), ("_c", consumer)]);
    space
}

#[test]
fn semantic_arg_test() {
    let (_, planes, _) = grouped(&semantic_space(456u64.into(), 456u64.into()));
    let found = groups(&planes[0]);
    assert_eq!(found.len(), 3);
    assert!(found.iter().all(|(_, group)| *group == Some(0)), "{found:?}");
}

#[test]
fn semantic_int_arg_no_match_test() {
    let (_, planes, _) = grouped(&semantic_space(456u64.into(), 789u64.into()));
    let found = groups(&planes[0]);
    assert_eq!(found.len(), 3);
    for (event, group) in found {
        assert_eq!(group, if event.ends_with("ExecutorState::Process") { None } else { Some(0) }, "{event}");
    }
}

#[test]
fn semantic_uint_arg_no_match_test() {
    let (_, planes, _) = grouped(&semantic_space(u64::MAX.into(), (u64::MAX - 1).into()));
    let found = groups(&planes[0]);
    assert_eq!(found.len(), 3);
    for (event, group) in found {
        assert_eq!(group, if event.ends_with("ExecutorState::Process") { None } else { Some(0) }, "{event}");
    }
}

#[test]
fn async_event_test() {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.set_pid_if_not_set(1);
    plane.event(0, "parent", 0, 100, &[("_r", 1.into())]);
    plane.event(0, "async", 10, 200, &[("_a", 1.into())]);
    plane.event(0, "child", 20, 80, &[]);
    let (_, planes, _) = grouped(&space);
    assert_eq!(groups(&planes[0]), [(" parent".to_string(), Some(0)), (" async".to_string(), None), (" child".to_string(), Some(0))]);
}

#[test]
fn batching_session_test() {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.set_pid_if_not_set(1);
    for offset in [0, 200] {
        plane.event(0, "BatchingSessionRun", offset, 100, &[("_r", 1.into())]);
        plane.event(0, "Schedule", offset, 100, &[("_pt", SHARED_BATCH_SCHEDULER.into()), ("_p", 123.into())]);
    }
    plane.event(1, "ProcessBatch", 200, 100, &[("_ct", SHARED_BATCH_SCHEDULER.into()), ("_c", 123.into()), ("_r", 2.into())]);
    let (_, planes, names) = grouped(&space);
    assert_eq!(names.unwrap().len(), 3);
    let found = groups(&planes[0]);
    assert!(found.iter().all(|(_, group)| group.is_some()), "{found:?}");
    assert_eq!(found.iter().filter(|(event, _)| event.ends_with("BatchingSessionRun") || event.ends_with("ProcessBatch")).count(), 3);
}

fn host_with_session_root(name: &str, stats: &[(&str, V)]) -> XSpace {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(1);
    host.event(0, "SessionRun", 0, 100, &[("_r", 1.into())]);
    host.event(0, name, 20, 50, stats);
    space
}

#[test]
fn tpu_execute_op_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(1);
    host.event(0, "ExecutorState::Process", 20, 50, &[("id", 123.into()), ("iter_num", 456.into())]);
    let (_, planes, names) = grouped(&space);
    assert_eq!(names.unwrap().len(), 1);
    assert!(groups(&planes[0]).iter().all(|(_, group)| group.is_some()));
}

#[test]
fn tpu_request_test() {
    let (_, planes, names) = grouped(&host_with_session_root("EnqueueRequestLocked", &[("queue_addr", 123.into()), ("request_id", 456.into())]));
    assert_eq!(names.unwrap().len(), 1);
    assert!(groups(&planes[0]).iter().all(|(_, group)| group.is_some()));
}

#[test]
fn tpu_program_callback_test() {
    let (_, planes, names) = grouped(&host_with_session_root("DoEnqueueProgram", &[("run_id", 123.into()), ("queue_id", 0.into()), ("device_ordinal", 1.into())]));
    assert_eq!(names.unwrap().len(), 1);
    assert!(groups(&planes[0]).iter().all(|(_, group)| group.is_some()));
}

#[test]
fn module_root_event_test() {
    let mut space = XSpace::default();
    let device = space.tpu(0, "TPUv4", 0.0, 0.0, None);
    device.named_line(0, "Steps");
    device.event(0, "1", 100, 200, &[("step_num", 1.into())]);
    device.named_line(1, "XLA Modules");
    device.event(1, "module", 105, 194, &[("run_id", 123.into()), ("queue_id", 0.into()), ("device_ordinal", 0.into())]);
    device.named_line(2, "XLA Ops");
    device.event(2, "matmul", 110, 190, &[]);
    let (_, planes, _) = grouped(&space);
    let found = groups(&planes[0]);
    assert!(found.iter().all(|(_, group)| group.is_some()), "{found:?}");
}

fn offload_start(index: i64, producer: i64, start_id: i64) -> [(&'static str, V); 5] {
    [("tc_offload_start_id", start_id.into()), ("offload_core_id", 0.into()), ("offload_execution_index", index.into()), ("_p", producer.into()), ("_pt", SC_OFFLOAD.into())]
}

fn offload_consumer(id: i64) -> [(&'static str, V); 2] {
    [("_ct", SC_OFFLOAD.into()), ("_c", id.into())]
}

fn device_step_span(planes: &[Plane], plane: usize, line: &str) -> Vec<(u64, u64, (u64, u64))> {
    let line = planes[plane].lines.iter().find(|candidate| candidate.name == line).unwrap();
    let stat = |index: usize, key: &str| line.steps.get(&index).and_then(|step| step.stats.iter().find(|(name, _)| *name == key).map(|(_, value)| *value));
    line.events
        .iter()
        .enumerate()
        .map(|(index, event)| {
            (
                event.ts,
                event.ts + event.dur,
                stat(index, "device_offset_ps").zip(stat(index, "device_duration_ps")).map_or((event.ts, event.ts + event.dur), |(offset, duration)| (offset as u64, (offset + duration) as u64)),
            )
        })
        .collect()
}

#[test]
fn merge_host_steps_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(123);
    host.named_line(0, "main");
    host.event(0, "train", 100, 10, &[("step_num", 1.into()), ("_r", 1.into())]);
    for (offset, duration, run) in [(100, 1, 2), (101, 2, 3), (103, 2, 4), (105, 4, 5)] {
        host.event(0, "DoEnqueueProgram", offset, duration, &[("run_id", run.into()), ("queue_id", 0.into()), ("device_ordinal", 0.into())]);
    }
    let spans = [(1000, 10), (1015, 100), (1125, 50), (1180, 25)];
    let device = space.tpu(0, "TPUv4", 0.0, 0.0, None);
    device.named_line(0, "XLA Modules");
    for ((offset, duration), run) in spans.into_iter().zip(2..) {
        device.event(0, "jit_something(1)", offset, duration, &[("run_id", run.into()), ("queue_id", 0.into())]);
    }
    device.named_line(1, "Steps");
    for (index, (offset, duration)) in spans.into_iter().enumerate() {
        device.event(1, &index.to_string(), offset, duration, &[("device_offset_ps", offset.into()), ("device_duration_ps", duration.into())]);
    }
    device.named_line(2, "XLA Ops");
    for (index, ((offset, _), done)) in spans.into_iter().zip([5, 95, 45, 20]).enumerate() {
        device.event(2, "offload.start.1", offset, 5, &offload_start(index as i64, index as i64 + 1, 1));
        device.event(2, "offload.done.1", offset + 5, done, &[]);
    }
    let sparse = space.tpu(0, "TPUv4", 0.0, 0.0, Some(0));
    sparse.named_line(0, "Sparse Core Modules");
    sparse.named_line(1, "Sparse Core Steps");
    sparse.named_line(2, "Sparse Core Ops");
    for (index, (offset, duration)) in spans.into_iter().enumerate() {
        sparse.event(0, "offloaded(1)", offset + 1, duration - 2, &[("tc_offload_start_id", 1.into())]);
        sparse.event(1, &format!("sc step {index}"), offset, duration, &[]);
        sparse.event(2, "sc_op_1", offset + 1, duration - 2, &offload_consumer(index as i64 + 1));
    }
    let (_, planes, names) = grouped(&space);
    assert_eq!(names.unwrap().len(), 1);
    assert_eq!(device_step_span(&planes, 1, "Steps"), [(1000, 1205, (1000, 1205))]);
    assert_eq!(device_step_span(&planes, 2, "Sparse Core Steps"), [(1000, 1205, (1000, 1205))]);
}

#[test]
fn merge_offloaded_sc_steps() {
    let mut space = XSpace::default();
    let core = space.tpu(0, "TPUv4", 0.0, 0.0, None);
    core.named_line(0, "XLA Modules");
    core.event(0, "jit_tc_module", 1010, 980, &[("run_id", 1.into())]);
    core.named_line(1, "Steps");
    core.event(1, "tc step 0", 1000, 1000, &[("device_offset_ps", 1000.into()), ("device_duration_ps", 1000.into())]);
    core.named_line(2, "XLA Ops");
    core.event(2, "offload.start.1", 1050, 50, &offload_start(0, 1, 1));
    core.event(2, "offload.done.1", 1100, 400, &[]);
    core.event(2, "offload.start.1", 1550, 50, &offload_start(1, 2, 1));
    core.event(2, "offload.done.1", 1600, 400, &[]);
    let sparse = space.tpu(0, "TPUv4", 0.0, 0.0, Some(1));
    sparse.named_line(0, "Sparse Core Modules");
    sparse.event(0, "offloaded(1)", 1101, 398, &[("tc_offload_start_id", 1.into())]);
    sparse.event(0, "offloaded(1)", 1601, 398, &[("tc_offload_start_id", 1.into())]);
    sparse.named_line(1, "Sparse Core Steps");
    sparse.event(1, "sc step 0", 1100, 400, &[]);
    sparse.event(1, "sc step 1", 1600, 400, &[]);
    sparse.named_line(2, "Sparse Core Ops");
    sparse.event(2, "sc_op_1a", 1110, 100, &offload_consumer(1));
    sparse.event(2, "sc_op_2a", 1610, 100, &offload_consumer(2));
    let (_, planes, names) = grouped(&space);
    let names = names.unwrap();
    assert_eq!(names.len(), 1);
    let expected = *names.keys().next().unwrap();
    for plane in &planes {
        let found = groups(plane);
        assert!(found.iter().all(|(_, group)| *group == Some(expected)), "{}: {found:?}", plane.name);
    }
    assert_eq!(device_step_span(&planes, 1, "Sparse Core Steps").iter().map(|span| (span.0, span.1)).collect::<Vec<_>>(), [(1100, 2000)]);
}

fn offloaded_sparse_core(space: &mut XSpace, module: (i64, i64), copy: (i64, i64, i64)) {
    let core = space.tpu(0, "TPUv4", 0.0, 0.0, None);
    core.named_line(0, "XLA Modules");
    core.event(0, "jit(123)", module.0, module.1, &[("run_id", 1.into()), ("queue_id", 0.into()), ("replica_id", 0.into()), ("core_type", 0.into())]);
    core.named_line(1, "Steps");
    core.event(1, "tc step 0", 1000, 1000, &[]);
    core.named_line(2, "XLA Ops");
    core.event(2, "offload_start", 1050, 100, &offload_start(0, 1, 123));
    core.event(2, "offload_done", 1200, 750, &[]);
    let sparse = space.tpu(0, "TPUv4", 0.0, 0.0, Some(0));
    sparse.named_line(0, "Sparse Core Modules");
    sparse.event(0, "offloaded(123)", 1100, 800, &[("tc_offload_start_id", 123.into())]);
    sparse.named_line(1, "Sparse Core Steps");
    sparse.event(1, "sc step 0", 1100, 800, &[]);
    sparse.named_line(2, "Sparse Core Ops");
    sparse.event(2, "offloaded_start.copy", 1100, copy.0, &offload_consumer(1));
    sparse.event(2, "offloaded_done.copy", copy.1, copy.2, &[]);
}

#[test]
fn group_offloaded_sparse_core_modules_host_loop_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(1);
    host.named_line(0, "main");
    host.event(0, "host step 0", 0, 200, &[("_r", 1.into())]);
    host.event(0, "DoEnqueueProgram", 100, 10, &[("run_id", 1.into()), ("queue_id", 0.into()), ("replica_id", 0.into()), ("device_ordinal", 0.into()), ("core_type", 0.into())]);
    offloaded_sparse_core(&mut space, (1000, 1000), (10, 1120, 180));
    let (_, planes, names) = grouped(&space);
    let names = names.unwrap();
    assert_eq!(names.len(), 1);
    let expected = *names.keys().next().unwrap();
    assert_eq!(groups(&planes[0]).len(), 2);
    for plane in &planes {
        let found = groups(plane);
        assert!(found.iter().all(|(_, group)| *group == Some(expected)), "{}: {found:?}", plane.name);
    }
}

#[test]
fn group_offloaded_sparse_core_modules_device_loop_test() {
    let mut space = XSpace::default();
    let host = space.host();
    host.set_pid_if_not_set(1);
    host.named_line(0, "main");
    host.event(0, "ExecutorState::Process", 100, 10, &[("id", 1.into()), ("iter_num", 99.into())]);
    host.event(0, "tpu::System::Execute", 100, 9, &[("_pt", TFRT_TPU_RUNTIME.into()), ("_p", 1.into())]);
    host.named_line(1, "tf_enqueue");
    host.event(1, "tpu::System::Execute=>IssueSequencedEvent", 102, 10, &[("_ct", TFRT_TPU_RUNTIME.into()), ("_c", 1.into())]);
    host.event(1, "DoEnqueueProgram", 103, 8, &[("run_id", 1.into()), ("queue_id", 0.into()), ("core_type", 0.into()), ("device_ordinal", 0.into())]);
    offloaded_sparse_core(&mut space, (900, 1200), (100, 1300, 100));
    let (_, planes, names) = grouped(&space);
    assert_eq!(names.unwrap().len(), 2);
    let host = groups(&planes[0]);
    assert_eq!(host.len(), 4);
    assert!(host.iter().all(|(_, group)| *group == Some(0)), "{host:?}");
    for plane in &planes[1..] {
        let found = groups(plane);
        assert!(found.iter().filter(|(event, _)| !event.starts_with("XLA Modules ")).all(|(_, group)| *group == Some(1)), "{}: {found:?}", plane.name);
    }
}

fn subprocess_space(consumer_pid: Option<i64>) -> XSpace {
    let mut space = XSpace::default();
    let producer = space.add_plane();
    producer.name = "Host CPUs".into();
    producer.add_stat("process_id", 1000);
    producer.event(0, "root1", 0, 10, &[("_r", 1.into())]);
    producer.event(0, "root2", 10, 100, &[("_r", 1.into())]);
    producer.event(0, "producer", 10, 90, &[("_pt", 123.into()), ("_p", 456u64.into())]);
    let consumer = space.add_plane();
    consumer.name = "Host CPUs [2000]".into();
    consumer.add_stat("process_id", 2000);
    let mut stats = vec![("_ct", V::from(123)), ("_c", 456u64.into())];
    stats.extend(consumer_pid.map(|pid| ("_pid", pid.into())));
    consumer.event(0, "consumer", 20, 80, &stats);
    space
}

fn named_groups(planes: &[Plane]) -> BTreeMap<String, i64> {
    planes.iter().flat_map(|plane| plane.lines.iter().flat_map(move |line| line.events.iter().filter_map(move |event| Some((name(plane, event).to_string(), group_id(event)?))))).collect()
}

#[test]
fn subprocess_grouping_test() {
    let (_, planes, _) = grouped(&subprocess_space(Some(1000)));
    let expected = [("root1", 0), ("root2", 1), ("producer", 1), ("consumer", 1)].map(|(name, group)| (name.to_string(), group));
    assert_eq!(named_groups(&planes), expected.into_iter().collect());
}

#[test]
fn subprocess_grouping_no_match_test() {
    let (_, planes, _) = grouped(&subprocess_space(None));
    let expected = [("root1", 0), ("root2", 1), ("producer", 1)].map(|(name, group)| (name.to_string(), group));
    assert_eq!(named_groups(&planes), expected.into_iter().collect());
}
