use crate::derive::{Category, is_tensor_core, tf_op};
use crate::xplane::{Ev, Line, NONE_GROUP, Plane, Step, Value, slice, stats};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::LazyLock;

const TPU: &str = "/device:TPU:";
const STEPS: [&str; 2] = ["Steps", "Sparse Core Steps"];
const MODULES: [&str; 2] = ["XLA Modules", "Sparse Core Modules"];
const COUNTERS: &str = "_counters_";
const IMPLICIT_ROOTS: [&str; 4] = ["Steps", "Sparse Core Steps", "XLA Modules", "Sparse Core Modules"];
const FANOUT_LIMIT: usize = 64;
const IDLE: &str = "step_idle_time_ps";
const OFFSET: &str = "device_offset_ps";
const DURATION: &str = "device_duration_ps";
const HOST: &str = "/host:CPU";
const LOOPS: [&str; 5] = ["WhileOp-EvalCond", "WhileOp-StartBody", "ForOp", "ParallelForOp", "ForeverOp"];
const HOST_EVENTS: &str = "UnknownHostEventType TraceContext SessionRun FunctionRun RunGraph RunGraphDone ExecutorState::Process ExecutorDoneCallback MemoryAllocation MemoryDeallocation RemotePerfCounter InstantiatedCapturedFunction::Run InstantiatedCapturedFunction::RunWithBorrowedArgs InstantiatedCapturedFunction::RunInstantiated InstantiatedCapturedFunction::RunAsync ParallelForOp ForeverOp WhileOp-EvalCond WhileOp-StartBody ForOp IteratorGetNextOp::DoCompute IteratorGetNextAsOptionalOp::DoCompute Iterator Iterator::Prefetch::Generator PrefetchProduce PrefetchConsume ParallelInterleaveProduce ParallelInterleaveConsume ParallelInterleaveInitializeInput ParallelMapProduce ParallelMapConsume MapAndBatchProduce MapAndBatchConsume ParseExampleProduce ParseExampleConsume ParallelBatchProduce ParallelBatchConsume BatchingSessionRun ProcessBatch BrainSessionRun ConcatInputTensors MergeInputTensors ScheduleWithoutSplit ScheduleWithSplit ScheduleWithEagerSplit ASBSQueue::Schedule OrbaxServing::ProcessBatch OrbaxServing::ConcatInputBuffers TfrtModelRun ServingModelRun EnqueueRequestLocked RunProgramRequest HostCallbackRequest TransferH2DRequest TransferPreprocessedH2DRequest TransferD2HRequest OnDeviceSendRequest OnDeviceRecvRequest OnDeviceSendRecvLocalRequest CustomWait OnDeviceSendRequestMulti OnDeviceRecvRequestMulti PjrtAsyncWait DoEnqueueProgram DoEnqueueContinuationProgram WriteHbm ReadHbm TpuExecuteOp CompleteCallbacks TPUPartitionedCallOp-InitializeVarOnTPU TPUPartitionedCallOp-ExecuteRemote TPUPartitionedCallOp-ExecuteLocal Linearize Delinearize TransferBufferFromDevice-FastPath tpu::System::TransferToDevice=>IssueEvent tpu::System::TransferToDevice=>IssueEvent=>Done tpu::System::TransferFromDevice=>IssueEvent tpu::System::TransferFromDevice=>IssueEvent=>Done tpu::System::Execute";
static HOST_EVENT_SET: LazyLock<FxHashSet<&str>> = LazyLock::new(|| HOST_EVENTS.split(' ').collect());
const OTHER: u8 = 0;
const UNTYPED: u8 = 1;
const TF_OP_RUN: u8 = 2;
const EAGER: u8 = 3;
const LAUNCH: u8 = 4;
const EXECUTE: u8 = 5;
const EXECUTOR: u8 = 6;
const TF_DATA: u8 = 7;
const TF_DATA_RUNS: [&str; 4] =
    ["InstantiatedCapturedFunction::Run", "InstantiatedCapturedFunction::RunAsync", "InstantiatedCapturedFunction::RunInstantiated", "InstantiatedCapturedFunction::RunWithBorrowedArgs"];
const IMPLICIT_ROOT_EVENTS: [&str; 4] = ["FunctionRun", "SessionRun", "RunGraph", "ExecutorState::Process"];

type Key = (u64, u64, Option<i32>);
type Members = (Vec<u32>, Vec<u32>);

#[derive(Default)]
struct Out {
    edges: Vec<(u32, u32)>,
    contexts: Vec<(Key, bool, u32)>,
    roots: Vec<(u32, i64)>,
    launches: Vec<(u64, u32)>,
    executes: Vec<(u64, u32)>,
    candidates: Vec<u32>,
    eager: Vec<u32>,
    executors: Vec<(u32, i64, i64)>,
    tf_data: Vec<i64>,
}

struct Typing {
    kinds: Vec<u8>,
    host: bool,
    correlation: Option<usize>,
    step: Option<usize>,
    iteration: Option<usize>,
}

impl Typing {
    fn new(plane: &Plane, map: &[u8]) -> Typing {
        let kinds = plane
            .meta
            .par_iter()
            .map(|meta| match &*meta.full_name(map) {
                "EagerExecute" => EAGER,
                "TfOpRun" => TF_OP_RUN,
                "KernelLaunch" => LAUNCH,
                "KernelExecute" => EXECUTE,
                "ExecutorState::Process" => EXECUTOR,
                name if TF_DATA_RUNS.contains(&name) => TF_DATA,
                name if HOST_EVENT_SET.contains(name) => OTHER,
                name => match tf_op(name).category {
                    Category::TensorFlow => TF_OP_RUN,
                    Category::TfData => OTHER,
                    _ => UNTYPED,
                },
            })
            .collect();
        Typing { kinds, host: plane.name == HOST, correlation: plane.id("correlation_id"), step: plane.id("id"), iteration: plane.id("iter_num") }
    }

    fn kind(&self, map: &[u8], event: &Ev) -> (u8, Option<u64>) {
        let kind = self.kinds[event.meta as usize];
        if !matches!(kind, UNTYPED | LAUNCH | EXECUTE) {
            return (kind, None);
        }
        let correlation = stats(slice(map, event.raw), 4, |id| Some(id) == self.correlation).find_map(|stat| stat.value.int()).map(|value| value as u64);
        match (kind, correlation) {
            (UNTYPED, Some(_)) => (if self.host { LAUNCH } else { EXECUTE }, correlation),
            _ => (kind, correlation),
        }
    }
}

enum Kind {
    Generic,
    Grouping,
    Child(u32, usize),
}

struct Job {
    plane: usize,
    line: usize,
    base: u32,
    kind: Kind,
}

#[derive(Default)]
pub struct Relatives {
    pub parents: BTreeSet<i64>,
    pub children: BTreeSet<i64>,
}

pub type Metadata = HashMap<i64, Relatives>;

pub struct Groups {
    pub names: HashMap<i64, String>,
    pub relatives: Metadata,
}

struct Graph {
    locs: Vec<(u32, u32, u32)>,
    kids: Members,
    parents: Members,
}

pub fn ordinal(plane: &Plane, line: &Line) -> Option<u64> {
    (line.name == "XLA Modules" && is_tensor_core(&plane.name)).then(|| plane.name.rsplit(':').next().and_then(|ordinal| ordinal.parse().ok())).flatten()
}

pub fn is_sparse_core(name: &str) -> bool {
    let Some(rest) = name.strip_prefix(TPU) else { return false };
    let parts: Vec<&str> = rest.split(' ').collect();
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    matches!(parts.len(), 3 | 4)
        && digits(parts[0])
        && parts[1] == "SparseCore"
        && digits(parts[parts.len() - 1])
        && (parts.len() == 3 || parts[2].bytes().all(|byte| byte.is_ascii_alphabetic()) && !parts[2].is_empty())
}

fn step_line(plane: &Plane) -> Option<usize> {
    plane.lines.iter().rposition(|line| STEPS.contains(&line.name.as_str()))
}

fn grouping_line(plane: &Plane) -> Option<usize> {
    let module = plane.lines.iter().rposition(|line| MODULES.contains(&line.name.as_str()));
    [step_line(plane), module].into_iter().flatten().find(|&index| !plane.lines[index].events.is_empty())
}

fn csr(nodes: usize, edges: &[(u32, u32)], by_child: bool) -> Members {
    let key = |edge: &(u32, u32)| if by_child { edge.1 } else { edge.0 } as usize;
    let mut offsets = vec![0u32; nodes + 1];
    edges.iter().for_each(|edge| offsets[key(edge) + 1] += 1);
    (0..nodes).for_each(|node| offsets[node + 1] += offsets[node]);
    let (mut cursor, mut targets) = (offsets.clone(), vec![0u32; edges.len()]);
    for edge in edges {
        targets[cursor[key(edge)] as usize] = if by_child { edge.0 } else { edge.1 };
        cursor[key(edge)] += 1;
    }
    (offsets, targets)
}

fn run(planes: &[Plane], map: &[u8], job: &Job, typing: &Typing) -> Out {
    let (plane, line) = (&planes[job.plane], &planes[job.plane].lines[job.line]);
    let (ordinal, mut out, mut stack) = (ordinal(plane, line), Out::default(), Vec::<(u32, u64, u64)>::new());
    let generic = matches!(job.kind, Kind::Generic);
    let mut nodes: Vec<(&Ev, u32)> = Vec::with_capacity(line.events.len());
    let mut parent = 0;
    for (index, event) in line.events.iter().enumerate() {
        nodes.push(match job.kind {
            Kind::Child(base, grouping) => {
                let parents = &plane.lines[grouping].events;
                while parent < parents.len() && parents[parent].ts + parents[parent].dur <= event.ts {
                    parent += 1;
                }
                if parent == parents.len() {
                    break;
                }
                if parents[parent].ts > event.ts || parents[parent].ts + parents[parent].dur < event.ts + event.dur {
                    continue;
                }
                (event, base + parent as u32)
            }
            _ => (event, job.base + index as u32),
        });
    }
    // Chunks decode their events in parallel. Only the nesting of the events needs them in order, and it needs one flag per event.
    let parts: Vec<(Out, Vec<bool>)> = nodes
        .par_chunks(1024)
        .map(|chunk| {
            let (mut part, mut nested) = (Out::default(), Vec::with_capacity(if generic { chunk.len() } else { 0 }));
            for &(event, node) in chunk {
                let links = plane.links(event.meta, slice(map, event.raw), ordinal);
                if let Some(level) = links.root.filter(|_| generic) {
                    part.roots.push((node, level));
                }
                let value = |wanted: Option<usize>| stats(slice(map, event.raw), 4, |id| Some(id) == wanted).find_map(|stat| stat.value.int());
                match generic.then(|| typing.kind(map, event)) {
                    Some((LAUNCH, Some(correlation))) => part.launches.push((correlation, node)),
                    Some((EXECUTE, correlation)) => {
                        part.executes.extend(correlation.map(|correlation| (correlation, node)));
                        part.candidates.push(node);
                    }
                    Some((TF_OP_RUN, _)) => part.candidates.push(node),
                    Some((EAGER, _)) => part.eager.push(node),
                    Some((EXECUTOR, _)) => {
                        if let (Some(step), Some(iteration)) = (value(typing.step), value(typing.iteration)) {
                            part.executors.push((node, step, iteration));
                        }
                    }
                    Some((TF_DATA, _)) => part.tf_data.extend(value(typing.step)),
                    _ => {}
                }
                for (link, producer) in [(links.producer, true), (links.consumer, false)] {
                    if let Some((id, kind)) = link {
                        part.contexts.push(((kind, id, links.pid), producer, node));
                    }
                }
                if generic {
                    nested.push(links.asynchronous.is_none_or(|value| value == 0));
                }
            }
            (part, nested)
        })
        .collect();
    let mut nested = Vec::with_capacity(if generic { nodes.len() } else { 0 });
    for (part, flags) in parts {
        out.roots.extend(part.roots);
        out.launches.extend(part.launches);
        out.executes.extend(part.executes);
        out.candidates.extend(part.candidates);
        out.eager.extend(part.eager);
        out.executors.extend(part.executors);
        out.tf_data.extend(part.tf_data);
        out.contexts.extend(part.contexts);
        nested.extend(flags);
    }
    for (&(event, node), nested) in nodes.iter().zip(nested) {
        if nested {
            let end = event.ts + event.dur;
            while let Some(&(top, begin, top_end)) = stack.last() {
                if begin <= event.ts && end <= top_end {
                    out.edges.push((top, node));
                    break;
                }
                stack.pop();
            }
            stack.push((node, event.ts, end));
        }
    }
    out
}

struct Walker<'a> {
    planes: &'a [Plane],
    map: &'a [u8],
    graph: &'a Graph,
    group: Vec<i64>,
    roots: FxHashMap<u32, i64>,
    seen: Vec<u32>,
    epoch: u32,
    names: HashMap<i64, String>,
    relatives: Metadata,
    renames: Vec<(u32, String)>,
}

fn neighbors(csr: &Members, node: u32) -> &[u32] {
    &csr.1[csr.0[node as usize] as usize..csr.0[node as usize + 1] as usize]
}

impl Walker<'_> {
    fn event(&self, node: u32) -> (&Plane, &Line, &Ev) {
        let (plane, line, event) = self.graph.locs[node as usize];
        let (plane, line) = (&self.planes[plane as usize], &self.planes[plane as usize].lines[line as usize]);
        (plane, line, &line.events[event as usize])
    }

    fn mark(&mut self, node: u32) -> bool {
        let fresh = self.seen[node as usize] != self.epoch;
        self.seen[node as usize] = self.epoch;
        fresh
    }

    fn needs_grouping(&mut self, node: u32) -> bool {
        if self.group[node as usize] != NONE_GROUP {
            return false;
        }
        let (graph, level) = (self.graph, self.roots.get(&node).copied().unwrap_or(0));
        self.epoch += 1;
        self.mark(node);
        let mut queue = VecDeque::new();
        for &parent in neighbors(&graph.parents, node) {
            self.mark(parent);
            queue.push_back(parent);
        }
        while let Some(current) = queue.pop_front() {
            if self.roots.get(&current) == Some(&level) {
                return false;
            }
            for &parent in neighbors(&graph.parents, current) {
                if self.mark(parent) {
                    queue.push_back(parent);
                }
            }
        }
        true
    }

    fn context(&self, root: u32, name: &str) -> Option<Value<'_>> {
        let (mut queue, mut seen) = (VecDeque::from([root]), FxHashSet::from_iter([root]));
        while let Some(node) = queue.pop_front() {
            let (plane, _, event) = self.event(node);
            if let Some(value) = plane.stat(self.map, event.meta, event.raw, name) {
                return Some(value);
            }
            queue.extend(neighbors(&self.graph.parents, node).iter().copied().filter(|&parent| seen.insert(parent)));
        }
        None
    }

    fn process(&mut self, group: i64, root: u32) {
        let graph = self.graph;
        self.epoch += 1;
        self.mark(root);
        let mut queue = VecDeque::from([root]);
        self.relatives.entry(group).or_default();
        while let Some(node) = queue.pop_front() {
            let current = self.group[node as usize];
            if current == NONE_GROUP {
                self.group[node as usize] = group;
                for &child in neighbors(&graph.kids, node) {
                    if self.mark(child) {
                        queue.push_back(child);
                    }
                }
            } else if current != group {
                self.relatives.entry(group).or_default().children.insert(current);
                self.relatives.entry(current).or_default().parents.insert(group);
            }
        }
        let (plane, line, event) = self.event(root);
        let implicit = IMPLICIT_ROOTS.contains(&line.name.as_str()) || IMPLICIT_ROOT_EVENTS.contains(&&*plane.meta[event.meta as usize].full_name(self.map));
        let step = self.context(root, "iter_num").and_then(|value| value.int()).or_else(|| self.context(root, "step_num").and_then(|value| value.int())).unwrap_or(group);
        let prefix = match self.context(root, "graph_type").map(|value| plane.text(&value)) {
            Some(text) => format!("{text} "),
            None if !implicit => format!("{} ", plane.meta[event.meta as usize].name),
            None => String::new(),
        };
        let name = format!("{prefix}{step}");
        if !implicit {
            self.renames.push((root, name.clone()));
        }
        self.names.insert(group, name);
    }
}

fn merge(plane: &mut Plane, map: &[u8], index: usize, names: &HashMap<i64, String>) {
    let events = std::mem::take(&mut plane.lines[index].events);
    if events.iter().all(|event| event.group == NONE_GROUP) {
        plane.lines[index].events = events;
        return;
    }
    let (idle_id, offset_id, duration_id) = (plane.id(IDLE), plane.id(OFFSET), plane.id(DURATION));
    let stat = |event: &Ev, id: Option<usize>| plane.find(map, event.meta, event.raw, id)?.int();
    let device_span = |event: &Ev| match (stat(event, offset_id), stat(event, duration_id)) {
        (Some(offset), Some(duration)) => (offset as u64, duration as u64),
        _ => (event.ts, event.dur),
    };
    let (mut kept, mut steps): (Vec<Ev>, HashMap<usize, Step>) = (Vec::new(), HashMap::new());
    let (mut group, mut idle, mut span) = (None, 0i64, None::<(u64, u64)>);
    for event in events {
        if event.group == NONE_GROUP {
            (group, idle) = (None, 0);
            continue;
        }
        if group != Some(event.group) {
            group = Some(event.group);
            idle = stat(&event, idle_id).unwrap_or(idle);
            span = (stat(&event, offset_id).is_some() && stat(&event, duration_id).is_some()).then(|| device_span(&event));
            steps.insert(kept.len(), Step { name: names.get(&event.group).cloned().unwrap_or_default(), stats: vec![(IDLE, idle)] });
            kept.push(event);
            continue;
        }
        let (last, end) = (kept.len() - 1, event.ts + event.dur);
        idle += stat(&event, idle_id).unwrap_or(0);
        let step = steps.get_mut(&last).unwrap();
        if let Some((begin, duration)) = span {
            let other = device_span(&event);
            if other.1 != 0 {
                span = Some(if duration == 0 { other } else { (begin.min(other.0), (begin + duration).max(other.0 + other.1) - begin.min(other.0)) });
            }
            let (begin, duration) = span.unwrap();
            step.stats.retain(|(key, _)| *key == IDLE);
            step.stats.extend([(OFFSET, begin as i64), (DURATION, duration as i64)]);
        }
        step.stats[0].1 = idle;
        kept[last].dur = end - kept[last].ts;
    }
    plane.lines[index].events = kept;
    plane.lines[index].steps = steps;
}

fn plan_jobs(planes: &[Plane], full: bool) -> (Vec<Job>, u32, Vec<(u32, usize)>) {
    let (mut jobs, mut nodes, mut cores) = (Vec::new(), 0u32, Vec::new());
    for (plane_index, plane) in planes.iter().enumerate() {
        let grouping = plane.name.starts_with(TPU).then(|| (is_tensor_core(&plane.name) || is_sparse_core(&plane.name)).then(|| grouping_line(plane)).flatten());
        match grouping {
            Some(None) => continue,
            Some(Some(line)) => {
                jobs.push(Job { plane: plane_index, line, base: nodes, kind: Kind::Grouping });
                if is_tensor_core(&plane.name) {
                    cores.push((nodes, plane.lines[line].events.len()));
                }
                if full {
                    jobs.extend((0..plane.lines.len()).filter(|&other| other != line).map(|other| Job { plane: plane_index, line: other, base: 0, kind: Kind::Child(nodes, line) }));
                }
                nodes += plane.lines[line].events.len() as u32;
            }
            None if full => {
                for (line_index, line) in plane.lines.iter().enumerate() {
                    jobs.push(Job { plane: plane_index, line: line_index, base: nodes, kind: Kind::Generic });
                    nodes += line.events.len() as u32;
                }
            }
            None => {}
        }
    }
    (jobs, nodes, cores)
}

fn build_graph(planes: &[Plane], jobs: &[Job], outs: &[Out], nodes: u32) -> (Graph, Vec<Vec<u32>>) {
    let locs: Vec<(u32, u32, u32)> = jobs
        .par_iter()
        .filter(|job| !matches!(job.kind, Kind::Child(..)))
        .flat_map_iter(|job| (0..planes[job.plane].lines[job.line].events.len() as u32).map(|event| (job.plane as u32, job.line as u32, event)))
        .collect();
    let mut edges: Vec<(u32, u32)> = outs.iter().flat_map(|out| out.edges.iter().copied()).collect();
    let tf_data: FxHashSet<i64> = outs.iter().flat_map(|out| out.tf_data.iter().copied()).collect();
    let mut loops: BTreeMap<i64, BTreeMap<i64, Vec<u32>>> = BTreeMap::new();
    for &(node, step, iteration) in outs.iter().flat_map(|out| &out.executors).filter(|(_, step, _)| !tf_data.contains(step)) {
        loops.entry(step).or_default().entry(iteration).or_default().push(node);
    }
    let order = |node: u32| {
        let (plane, line, event) = locs[node as usize];
        let event = &planes[plane as usize].lines[line as usize].events[event as usize];
        (event.ts, Reverse(event.ts + event.dur), node)
    };
    let mut iterations: Vec<Vec<u32>> = loops.values().filter(|iterations| !(iterations.len() == 1 && iterations.contains_key(&0))).flat_map(|iterations| iterations.values().cloned()).collect();
    iterations.iter_mut().for_each(|nodes| nodes.sort_by_key(|&node| order(node)));
    iterations.sort_by_key(|nodes| order(nodes[0]));
    edges.extend(iterations.iter().flat_map(|nodes| nodes[1..].iter().map(|&node| (nodes[0], node))));
    let launches: FxHashMap<u64, u32> = outs.iter().flat_map(|out| out.launches.iter().copied()).collect();
    edges.extend(outs.iter().flat_map(|out| out.executes.iter()).filter_map(|(correlation, node)| launches.get(correlation).map(|&launch| (launch, *node))));
    let mut contexts: FxHashMap<Key, Members> = FxHashMap::default();
    for (key, producer, node) in outs.iter().flat_map(|out| out.contexts.iter().copied()) {
        let entry = contexts.entry(key).or_default();
        if producer { &mut entry.0 } else { &mut entry.1 }.push(node);
    }
    let mut ordered: Vec<(&Key, &Members)> = contexts.iter().collect();
    ordered.sort_by_key(|(key, _)| **key);
    for (_, (producers, consumers)) in ordered.into_iter().filter(|(_, (producers, consumers))| producers.len() < FANOUT_LIMIT || consumers.len() < FANOUT_LIMIT) {
        edges.extend(producers.iter().flat_map(|&producer| consumers.iter().map(move |&consumer| (producer, consumer))));
    }
    let (kids, parents) = rayon::join(|| csr(nodes as usize, &edges, false), || csr(nodes as usize, &edges, true));
    (Graph { kids, parents, locs }, iterations)
}

fn classify_eager(walker: &Walker, outs: &[Out]) -> Vec<(u32, bool)> {
    let eager_nodes: FxHashSet<u32> = outs.iter().flat_map(|out| out.eager.iter().copied()).collect();
    outs.par_iter()
        .flat_map(|out| out.candidates.par_iter())
        .map(|&node| {
            let (mut queue, mut seen) = (VecDeque::from([node]), FxHashSet::from_iter([node]));
            while let Some(current) = queue.pop_front() {
                if eager_nodes.contains(&current) {
                    let (plane, _, event) = walker.event(current);
                    let is_func = stats(slice(walker.map, event.raw), 4, |id| Some(id) == plane.id("is_func")).find_map(|stat| stat.value.int());
                    return (node, is_func == Some(0));
                }
                queue.extend(neighbors(&walker.graph.parents, current).iter().copied().filter(|&parent| seen.insert(parent)));
            }
            (node, false)
        })
        .collect()
}

fn align_device_lines(plane: &mut Plane, map: &[u8], names: &HashMap<i64, String>) {
    let Some(grouping) = grouping_line(plane) else { return };
    let step = step_line(plane);
    if let Some(step) = step {
        merge(plane, map, step, names);
        let line = &mut plane.lines[step];
        for (index, event) in line.events.iter().enumerate() {
            if let Some(name) = names.get(&event.group) {
                line.steps.entry(index).or_default().name.clone_from(name);
            }
        }
    }
    let grouping: Vec<(u64, u64, i64)> = plane.lines[grouping].events.iter().map(|event| (event.ts, event.dur, event.group)).collect();
    for line in plane.lines.iter_mut().enumerate().filter(|(index, line)| Some(*index) != step && line.name != COUNTERS).map(|(_, line)| line) {
        let mut cursor: Option<usize> = None;
        for event in &mut line.events {
            let end = event.ts + event.dur;
            let overlaps = |index: usize| grouping[index].0 <= end && event.ts <= grouping[index].0 + grouping[index].1;
            let found = if cursor.is_some_and(overlaps) {
                cursor
            } else {
                let candidate =
                    (cursor.map_or(0, |current| current + 1)..grouping.len()).take_while(|&index| grouping[index].0 <= end).find(|&index| grouping[index].0 + grouping[index].1 >= event.ts);
                cursor = candidate.or(cursor);
                candidate
            };
            if let Some(index) = found.filter(|&index| grouping[index].2 != NONE_GROUP) {
                event.group = grouping[index].2;
            }
        }
    }
}

pub fn group(planes: &mut [Plane], map: &[u8]) -> Option<Groups> {
    if planes.par_iter().any(|plane| plane.meta.par_iter().any(|meta| LOOPS.contains(&&*meta.full_name(map)))) {
        return None;
    }
    let typings: Vec<Typing> = planes.par_iter().map(|plane| Typing::new(plane, map)).collect();
    let eager_exists = planes.iter().zip(&typings).any(|(plane, typing)| !plane.name.starts_with(TPU) && typing.kinds.contains(&EAGER));
    let full = eager_exists
        || planes.iter().any(Plane::has_roots)
        || typings.iter().any(|typing| typing.kinds.contains(&EXECUTOR))
        || planes.iter().any(|plane| plane.name.starts_with(TPU) && plane.has_contexts());
    let (jobs, nodes, cores) = plan_jobs(planes, full);
    let outs: Vec<Out> = jobs.par_iter().map(|job| run(planes, map, job, &typings[job.plane])).collect();
    let (graph, iterations) = build_graph(planes, &jobs, &outs, nodes);
    let mut walker = Walker {
        planes,
        map,
        graph: &graph,
        group: vec![NONE_GROUP; nodes as usize],
        roots: outs.iter().flat_map(|out| out.roots.iter().copied()).collect(),
        seen: vec![0; nodes as usize],
        epoch: 0,
        names: HashMap::new(),
        relatives: Metadata::new(),
        renames: Vec::new(),
    };
    let mut roots: Vec<u32> = walker.roots.keys().copied().collect();
    roots.sort_by_key(|&node| (std::cmp::Reverse(walker.roots[&node]), walker.event(node).2.ts, node));
    let mut next = 0;
    if iterations.is_empty() {
        for root in roots {
            if walker.needs_grouping(root) {
                walker.process(next, root);
                next += 1;
            }
        }
    }
    for nodes in &iterations {
        walker.process(next, nodes[0]);
        next += 1;
    }
    let steps: Vec<u32> = cores.iter().flat_map(|&(base, length)| base..base + length as u32).collect();
    if steps.iter().all(|&node| walker.needs_grouping(node)) {
        for &(base, length) in &cores {
            let mut id = next;
            for node in base..base + length as u32 {
                if walker.needs_grouping(node) {
                    walker.process(id, node);
                    id += 1;
                }
            }
        }
    }
    let eager = classify_eager(&walker, &outs);
    let (group, names, relatives, renames) = (walker.group, walker.names, walker.relatives, walker.renames);
    for (node, value) in eager {
        let (plane, line, event) = graph.locs[node as usize];
        planes[plane as usize].lines[line as usize].events[event as usize].eager = Some(value);
    }
    if !full {
        planes.par_iter_mut().zip(&typings).filter(|(plane, _)| !plane.name.starts_with(TPU)).for_each(|(plane, typing)| {
            for event in plane.lines.iter_mut().flat_map(|line| line.events.iter_mut()) {
                if matches!(typing.kind(map, event).0, EXECUTE | TF_OP_RUN) {
                    event.eager = Some(false);
                }
            }
        });
    }
    for (&(plane, line, event), &id) in graph.locs.iter().zip(&group) {
        if id != NONE_GROUP {
            planes[plane as usize].lines[line as usize].events[event as usize].group = id;
        }
    }
    for (node, name) in renames {
        let (plane, line, event) = graph.locs[node as usize];
        planes[plane as usize].lines[line as usize].steps.insert(event as usize, Step { name, stats: Vec::new() });
    }
    if planes.iter().any(|plane| plane.name.starts_with(TPU)) {
        planes.par_iter_mut().filter(|plane| plane.name.starts_with("/device")).for_each(|plane| align_device_lines(plane, map, &names));
    }
    Some(Groups { names, relatives })
}
