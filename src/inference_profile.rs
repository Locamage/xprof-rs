use crate::counters::{events, first, planes, valid_space};
use crate::derive::{is_grouped, is_tensor_core};
use crate::group::{Metadata, group};
use crate::hlo::general;
use crate::input_pipeline_analyzer::fixed;
use crate::memory_viewer::std_sort;
use crate::opstats::safe_divide;
use crate::steps::{DEVICE_COMPUTE_16, DEVICE_COMPUTE_32, DEVICE_TO_HOST, HOST_TO_DEVICE, Span, UNKNOWN_TIME, non_overlapped};
use crate::table::{Cell, Table};
use crate::xplane::{self, NONE_GROUP, Plane, Value, slice, stats};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::iter::once;
use std::path::PathBuf;

const HOST_PLANE: &str = "/host:CPU";
const TFSTREAMZ_PLANE: &str = "/host:tfstreamz";
const XLA_MODULES: &str = "XLA Modules";
const REQUEST_EVENTS: [&[u8]; 7] = [b"SessionRun", b"RunGraph", b"BatchingSessionRun", b"ProcessBatch", b"OrbaxServing::ProcessBatch", b"TfrtModelRun", b"ServingModelRun"];
const USER_REQUEST_PREFIX: &str = "ModelServerRequest_";
const HOST_PREPROCESS: u32 = 20;
const HOST_POSTPROCESS: u32 = 30;
const HOST_BATCH_FORMATION: u32 = 40;
const HOST_RUNTIME: u32 = 50;
const BATCHING_REQUEST: &str = "BatchingSessionRun";
const REQUEST_KINDS: [&[&str]; 3] = [&["TfrtModelRun"], &["ServingModelRun"], &["SessionRun", "RunGraph"]];
const MODEL_ID_EVENTS: [&str; 3] = ["SessionRun", "TfrtModelRun", "ServingModelRun"];
const SCHEDULE_EVENTS: [&str; 4] = ["ScheduleWithSplit", "ScheduleWithoutSplit", "ScheduleWithEagerSplit", "ASBSQueue::Schedule"];
const PROCESS_BATCH_EVENTS: [&str; 2] = ["ProcessBatch", "OrbaxServing::ProcessBatch"];
const PADDING_EVENTS: [&str; 4] = ["ConcatInputTensors", "OrbaxServing::ConcatInputBuffers", "MergeInputTensors", "BrainSessionRun"];
const TRANSFER_EVENTS: [(&str, u32); 5] =
    [("ReadHbm", DEVICE_TO_HOST), ("TransferD2HRequest", DEVICE_TO_HOST), ("WriteHbm", HOST_TO_DEVICE), ("TransferH2DRequest", HOST_TO_DEVICE), ("TransferPreprocessedH2DRequest", HOST_TO_DEVICE)];
const SYSTEM_TRANSFERS: [(&str, &str, u32); 2] = [
    ("tpu::System::TransferToDevice=>IssueEvent", "tpu::System::TransferToDevice=>IssueEvent=>Done", HOST_TO_DEVICE),
    ("tpu::System::TransferFromDevice=>IssueEvent", "tpu::System::TransferFromDevice=>IssueEvent=>Done", DEVICE_TO_HOST),
];
const EXECUTE_EVENTS: [&str; 4] = ["TPUPartitionedCallOp-ExecuteLocal", "TPUPartitionedCallOp-ExecuteRemote", "TPUPartitionedCallOp-InitializeVarOnTPU", "tpu::System::Execute"];
const LAUNCH_EVENTS: [&str; 2] = ["DoEnqueueProgram", "DoEnqueueContinuationProgram"];
const CALLBACK_EVENTS: [&str; 1] = ["CompleteCallbacks"];
const TENSOR_EVENTS: [&str; 3] = ["Linearize", "Delinearize", "TransferBufferFromDevice-FastPath"];
const BATCHING_PARAM_PREFIX: &str = "/tensorflow/serving/batching/";
const BATCHING_PARAMS: [&str; 5] = ["num_batch_threads", "batch_timeout_micros", "max_batch_size", "max_enqueued_batches", "allowed_batch_sizes"];
const REQUEST_OWNER: i32 = 1;
const BATCH_OWNER: i32 = 2;
const TENSOR_PERCENTILES: [f64; 6] = [50.0, 75.0, 90.0, 95.0, 99.0, 99.9];
const WANTED_PERCENTILES: [(f64, f64); 6] = [(50.0, 1.0), (75.0, 1.0), (90.0, 1.0), (99.0, 0.5), (99.9, 0.05), (99.99, 0.005)];
const PER_PERCENTILE: usize = 10;
const LATENCY_COLUMN: &str = "Latency";
const BATCH_COLUMNS: [(&str, &str, &str); 11] = [
    ("percentile", "string", "Percentile"),
    ("batch_id", "string", "Batch ID"),
    ("latency", "number", "Latency"),
    ("padding_amount", "number", "Padding amount"),
    ("batch_size_after_padding", "number", "Batch size after padding"),
    ("batching_efficiency", "number", "Batching efficiency"),
    ("batching_delay_us", "number", "Batching delay"),
    ("throughput", "string", "Throughput"),
    ("device_compute", "number", "Device compute"),
    ("program_id", "string", "Program ID(s)"),
    ("trace_viewer_url", "string", "Trace Viewer URL"),
];
const TPU_COLUMNS: [(&str, &str, &str); 7] = [
    ("host_preprocessing", "number", "Host preprocess"),
    ("host_runtime", "number", "Host runtime"),
    ("data_transfer_h2d", "number", "Data transfer H2D"),
    ("data_transfer_d2h", "number", "Data transfer D2H"),
    ("device_compute", "number", "Device compute"),
    ("host_postprocess", "number", "Host postprocess"),
    ("idle time", "number", "Idle time"),
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorEventDetail {
    pub tensor_pattern_index: i32,
    pub owner: i32,
    pub linearize_delinearize_time_ps: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestDetail {
    pub request_id: i64,
    pub model_id_index: i32,
    pub start_time_ps: u64,
    pub end_time_ps: u64,
    pub device_time_ps: u64,
    pub write_to_device_time_ps: u64,
    pub read_from_device_time_ps: u64,
    pub batching_request_delay_ps: u64,
    pub related_batch_ids: Vec<i64>,
    pub batching_request_size: i32,
    pub host_preprocessing_ps: u64,
    pub host_batch_formation_ps: u64,
    pub host_runtime_ps: u64,
    pub host_postprocessing_ps: u64,
    pub tensor_event_details: Vec<TensorEventDetail>,
    pub host_id: i32,
    pub percentile: f64,
    pub idle_time_ps: f64,
}

impl Default for RequestDetail {
    fn default() -> Self {
        Self {
            request_id: -1,
            model_id_index: -1,
            start_time_ps: 0,
            end_time_ps: 0,
            device_time_ps: 0,
            write_to_device_time_ps: 0,
            read_from_device_time_ps: 0,
            batching_request_delay_ps: 0,
            related_batch_ids: Vec::new(),
            batching_request_size: 0,
            host_preprocessing_ps: 0,
            host_batch_formation_ps: 0,
            host_runtime_ps: 0,
            host_postprocessing_ps: 0,
            tensor_event_details: Vec::new(),
            host_id: 0,
            percentile: 0.0,
            idle_time_ps: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchDetail {
    pub batch_id: i64,
    pub start_time_ps: u64,
    pub end_time_ps: u64,
    pub batch_delay_ps: u64,
    pub related_request_ids: Vec<i64>,
    pub padding_amount: i32,
    pub batch_size_after_padding: i32,
    pub model_id_index: i32,
    pub tensor_event_detail: Option<TensorEventDetail>,
    pub host_id: i32,
    pub percentile: f64,
    pub device_time_ps: u64,
    pub program_ids: Vec<u64>,
}

impl Default for BatchDetail {
    fn default() -> Self {
        Self {
            batch_id: -1,
            start_time_ps: 0,
            end_time_ps: 0,
            batch_delay_ps: 0,
            related_request_ids: Vec::new(),
            padding_amount: 0,
            batch_size_after_padding: 0,
            model_id_index: 0,
            tensor_event_detail: None,
            host_id: 0,
            percentile: 0.0,
            device_time_ps: 0,
            program_ids: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerHostInferenceStats {
    pub request_details: Vec<RequestDetail>,
    pub batch_details: Vec<BatchDetail>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TensorPatternResult {
    pub tensor_pattern_index: i32,
    pub count: u64,
    pub linearize_delinearize_percentile_time: Vec<(f64, u64)>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerBatchSizeAggregatedResult {
    pub batch_size: i32,
    pub aggregated_request_result: RequestDetail,
    pub aggregated_batch_result: BatchDetail,
    pub request_throughput: f64,
    pub batch_throughput: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerModelInferenceStats {
    pub request_details: Vec<RequestDetail>,
    pub aggregated_request_detail: RequestDetail,
    pub request_throughput: f64,
    pub request_average_latency_us: f64,
    pub batch_details: Vec<BatchDetail>,
    pub aggregated_batch_detail: BatchDetail,
    pub batch_throughput: f64,
    pub batch_average_latency_us: f64,
    pub tensor_pattern_results: Vec<TensorPatternResult>,
    pub per_batch_size_aggregated_result: Vec<PerBatchSizeAggregatedResult>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchingParameters {
    pub num_batch_threads: i64,
    pub batch_timeout_micros: i64,
    pub max_batch_size: i64,
    pub max_enqueued_batches: i64,
    pub allowed_batch_sizes: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelIdDatabase {
    pub ids: Vec<String>,
    pub id_to_index: HashMap<String, i32>,
    pub id_to_batching_params: HashMap<String, BatchingParameters>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SampledPerModelInferenceStats {
    pub sampled_requests: Vec<RequestDetail>,
    pub sampled_batches: Vec<BatchDetail>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InferenceStats {
    pub inference_stats_per_host: BTreeMap<i32, PerHostInferenceStats>,
    pub inference_stats_per_model: BTreeMap<i32, PerModelInferenceStats>,
    pub model_id_db: ModelIdDatabase,
    pub tensor_patterns: Vec<String>,
    pub sampled_inference_stats: BTreeMap<i32, SampledPerModelInferenceStats>,
}

struct HostEvent<'a> {
    name: &'a str,
    ts: u64,
    dur: u64,
    group: Option<i64>,
    raw: &'a [u8],
}

#[derive(Default)]
struct Stamps {
    schedule: Option<u64>,
    concat: Option<u64>,
    execute: Option<u64>,
    launch: Option<u64>,
    callback: Option<u64>,
}

#[derive(Default)]
struct Request<'a> {
    model: i32,
    span: Span,
    delay: u64,
    size: i32,
    stamps: BTreeMap<i64, Stamps>,
    tensors: Vec<&'a HostEvent<'a>>,
    details: Vec<TensorEventDetail>,
    batches: Vec<i64>,
    events: Vec<(u32, Span)>,
}

#[derive(Default)]
struct Batch<'a> {
    tensors: Vec<&'a HostEvent<'a>>,
    detail: BatchDetail,
    events: Vec<(u32, Span)>,
}

fn serves_requests(map: &[u8]) -> bool {
    let planes = planes(map, |name| name == HOST_PLANE.as_bytes());
    let Some(plane) = planes.first() else { return false };
    let root = [plane.stat_id("_r")];
    plane.lines.par_iter().any(|line| {
        events(line).any(|event| {
            let name = plane.metadata.get(&event.meta).map_or(&[][..], |(name, _)| name);
            let level = first(event.raw, 4, root)[0].as_ref().map(|value| if let Value::Int(level) = value { *level } else { 0 });
            REQUEST_EVENTS.contains(&name) || level == Some(-1) || (level == Some(1) && name.starts_with(USER_REQUEST_PREFIX.as_bytes()))
        })
    })
}

fn span(begin: u64, end: u64) -> Span {
    Span { begin, duration: end.saturating_sub(begin) }
}

fn ms(ps: u64) -> f64 {
    ps as f64 / 1e9
}

fn event_stat(raw: &[u8], id: Option<usize>) -> Option<Value<'_>> {
    stats(raw, 4, |stat| Some(stat) == id).next().map(|stat| stat.value)
}

fn int64(value: &Value) -> i64 {
    if let Value::Int(value) = value { *value } else { 0 }
}

fn group_of(raw: &[u8], group: i64, id: Option<usize>) -> Option<i64> {
    if group == NONE_GROUP { event_stat(raw, id).map(|value| int64(&value)) } else { Some(group) }
}

fn targets(relatives: &Metadata, group: i64) -> impl Iterator<Item = i64> + '_ {
    once(group).chain(relatives.get(&group).into_iter().flat_map(|relative| relative.parents.iter().copied()))
}

fn assign(models: &mut ModelIdDatabase, id: &str) -> i32 {
    if id.is_empty() {
        return -1;
    }
    let next = models.ids.len() as i32;
    *models.id_to_index.entry(id.to_string()).or_insert_with(|| {
        models.ids.push(id.to_string());
        next
    })
}

struct StatIds {
    group: Option<usize>,
    root: Option<usize>,
    model: Option<usize>,
    task_size: Option<usize>,
    padding: Option<usize>,
    padded_size: Option<usize>,
    consumer: Option<usize>,
    shape: Option<usize>,
    dims: Option<usize>,
    kind: Option<usize>,
    layout: Option<usize>,
}

impl StatIds {
    fn new(host: &Plane) -> Self {
        Self {
            group: host.id("group_id"),
            root: host.id("_r"),
            model: host.id("model_id"),
            task_size: host.id("batching_input_task_size"),
            padding: host.id("padding_amount"),
            padded_size: host.id("batch_size_after_padding"),
            consumer: host.id("_c"),
            shape: host.id("shape"),
            dims: host.id("dims"),
            kind: host.id("type"),
            layout: host.id("layout"),
        }
    }
}

struct Events<'a> {
    all: Vec<HostEvent<'a>>,
    by_name: HashMap<&'a str, Vec<usize>>,
    roots: Vec<usize>,
}

impl<'a> Events<'a> {
    fn collect(host: &'a Plane, map: &'a [u8], names: &'a [Cow<str>], ids: &StatIds) -> Self {
        let all: Vec<HostEvent> = host
            .lines
            .iter()
            .flat_map(|line| line.events.iter().map(move |event| (line, event)))
            .map(|(line, event)| {
                let raw = slice(map, event.raw);
                HostEvent { name: &names[event.meta as usize], ts: line.absolute_ps(event.ts, host.origin_ns), dur: event.dur, group: group_of(raw, event.group, ids.group), raw }
            })
            .collect();
        let mut by_name: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, event) in all.iter().enumerate() {
            by_name.entry(event.name).or_default().push(index);
        }
        let roots = (0..all.len())
            .filter(|&index| match event_stat(all[index].raw, ids.root) {
                Some(Value::Int(-1)) => true,
                Some(Value::Int(1)) => all[index].name.starts_with(USER_REQUEST_PREFIX),
                _ => false,
            })
            .collect();
        Events { all, by_name, roots }
    }

    fn named(&self, name: &str) -> impl Iterator<Item = &HostEvent<'a>> {
        self.by_name.get(name).map_or(&[][..], Vec::as_slice).iter().map(|&index| &self.all[index])
    }

    fn grouped(&self, names: &'static [&'static str]) -> impl Iterator<Item = (&HostEvent<'a>, i64)> {
        names.iter().flat_map(|name| self.named(name)).filter_map(|event| Some((event, event.group?)))
    }

    fn roots(&self) -> impl Iterator<Item = &HostEvent<'a>> {
        self.roots.iter().map(|&index| &self.all[index])
    }
}

fn model_ids(events: &Events, host: &Plane, ids: &StatIds) -> HashMap<i64, String> {
    let mut model_ids = HashMap::new();
    for (event, group) in events.grouped(&MODEL_ID_EVENTS).chain(events.roots().filter_map(|event| Some((event, event.group?)))) {
        let text = match event_stat(event.raw, ids.model) {
            Some(Value::Int(value)) => value.to_string(),
            Some(Value::Uint(value)) => value.to_string(),
            Some(Value::Double(value)) => general(value, 6),
            Some(Value::Bytes(_)) => "<opaque bytes>".into(),
            Some(value) => host.text(&value),
            None => continue,
        };
        model_ids.insert(group, text);
    }
    model_ids
}

fn build_requests<'a>(events: &Events, relatives: &Metadata, model_ids: &HashMap<i64, String>, models: &mut ModelIdDatabase) -> BTreeMap<i64, Request<'a>> {
    let model_of = |group: &i64| model_ids.get(group).map_or("", String::as_str);
    let batch_groups: HashSet<i64> = events.grouped(&PROCESS_BATCH_EVENTS).map(|(_, group)| group).collect();
    let batching = events.by_name.contains_key(BATCHING_REQUEST);
    let kinds: &[&str] = if batching { &[BATCHING_REQUEST] } else { REQUEST_KINDS.into_iter().find(|kinds| events.by_name.contains_key(kinds[0])).unwrap_or(REQUEST_KINDS[2]) };
    let mut requests: BTreeMap<i64, Request> = BTreeMap::new();
    let mut initialized = HashSet::new();
    for (event, user) in kinds.iter().flat_map(|kind| events.named(kind)).map(|event| (event, false)).chain(events.roots().map(|event| (event, true))) {
        let Some(group) = event.group else { continue };
        if batch_groups.contains(&group) || !initialized.insert(group) {
            continue;
        }
        let request = requests.entry(group).or_default();
        let Some(relative) = relatives.get(&group) else { continue };
        request.batches.extend(&relative.children);
        request.model = if batching && !user {
            let Some(child) = relative.children.first() else { continue };
            assign(models, model_of(child))
        } else if user && !model_ids.contains_key(&group) {
            if relative.children.is_empty() {
                continue;
            }
            let (mut common, mut same) = (-1, true);
            for child in &relative.children {
                let index = assign(models, model_of(child));
                if common == -1 {
                    common = index;
                } else if index != common {
                    same = false;
                }
            }
            if same { common } else { -1 }
        } else {
            assign(models, model_of(&group))
        };
    }
    for event in &events.all {
        if let Some(request) = event.group.and_then(|group| requests.get_mut(&group)) {
            let begin = if request.span.begin == 0 { event.ts } else { request.span.begin.min(event.ts) };
            request.span = span(begin, request.span.end().max(event.ts.wrapping_add(event.dur)));
        }
    }
    requests
}

fn stamp(requests: &mut BTreeMap<i64, Request>, relatives: &Metadata, group: i64, apply: &dyn Fn(&mut Stamps)) {
    for target in targets(relatives, group) {
        if let Some(request) = requests.get_mut(&target) {
            apply(request.stamps.entry(group).or_default());
        }
    }
}

fn stamp_schedules(events: &Events, relatives: &Metadata, ids: &StatIds, requests: &mut BTreeMap<i64, Request>) {
    for (event, group) in events.grouped(&SCHEDULE_EVENTS) {
        stamp(requests, relatives, group, &|stamps| stamps.schedule = Some(event.ts));
        if let (Some(request), Some(size)) = (requests.get_mut(&group), event_stat(event.raw, ids.task_size)) {
            request.size = int64(&size) as i32;
        }
    }
}

fn build_batches<'a>(events: &Events, relatives: &Metadata, ids: &StatIds, requests: &mut BTreeMap<i64, Request>) -> BTreeMap<i64, Batch<'a>> {
    let mut batches: BTreeMap<i64, Batch> = BTreeMap::new();
    for (event, group) in events.grouped(&PROCESS_BATCH_EVENTS) {
        let Some(relative) = relatives.get(&group) else { continue };
        let detail = &mut batches.entry(group).or_default().detail;
        (detail.batch_id, detail.start_time_ps, detail.end_time_ps) = (group, event.ts, event.ts.wrapping_add(event.dur));
        detail.related_request_ids.extend(&relative.parents);
        detail.related_request_ids.sort_unstable();
    }
    for (event, group) in events.grouped(&PADDING_EVENTS) {
        stamp(requests, relatives, group, &|stamps| stamps.concat = Some(event.ts));
        if let (Some(batch), Some(padding), Some(size)) = (batches.get_mut(&group), event_stat(event.raw, ids.padding), event_stat(event.raw, ids.padded_size)) {
            (batch.detail.batch_size_after_padding, batch.detail.padding_amount) = (int64(&size) as i32, int64(&padding) as i32);
        }
    }
    for batch in batches.values_mut() {
        if let Some(request) = batch.detail.related_request_ids.first().and_then(|&id| requests.get(&i64::from(id as i32))) {
            batch.detail.model_id_index = request.model;
        }
    }
    batches
}

type DeviceEvent = (i64, (u32, Span), Option<u64>);

fn device_events(planes: &[Plane], map: &[u8], events: &Events, ids: &StatIds) -> Vec<DeviceEvent> {
    let mut found: Vec<DeviceEvent> = Vec::new();
    for (name, kind) in TRANSFER_EVENTS {
        found.extend(events.named(name).filter_map(|event| Some((event.group?, (kind, Span { begin: event.ts, duration: event.dur }), None))));
    }
    for (start, end, kind) in SYSTEM_TRANSFERS {
        let mut transfers: HashMap<i64, (&HostEvent, Option<&HostEvent>)> = HashMap::new();
        let consumer = |event: &HostEvent| event.group.and(event_stat(event.raw, ids.consumer)).map(|value| int64(&value));
        for event in events.named(start) {
            if let Some(id) = consumer(event) {
                transfers.insert(id, (event, None));
            }
        }
        for event in events.named(end) {
            if let Some(transfer) = consumer(event).and_then(|id| transfers.get_mut(&id)).filter(|transfer| transfer.0.ts < event.ts) {
                transfer.1 = Some(event);
            }
        }
        for (start, end) in transfers.values().filter_map(|&(start, end)| Some((start, end?))) {
            found.push((start.group.unwrap_or_default(), (kind, Span { begin: start.ts, duration: end.ts.wrapping_add(end.dur).wrapping_sub(start.ts) }), None));
        }
    }
    for plane in planes.iter().filter(|plane| is_tensor_core(&plane.name)) {
        for (line, event) in plane.lines.iter().filter(|line| line.name == XLA_MODULES).flat_map(|line| line.events.iter().map(move |event| (line, event))) {
            let Some(group) = group_of(slice(map, event.raw), event.group, plane.id("group_id")) else { continue };
            let program = plane.stat(map, event.meta, event.raw, "program_id").map(|value| value.int().unwrap_or(0) as u64);
            found.push((group, (DEVICE_COMPUTE_32, Span { begin: line.absolute_ps(event.ts, plane.origin_ns), duration: event.dur }), program));
        }
    }
    found
}

fn attach_device_events(relatives: &Metadata, requests: &mut BTreeMap<i64, Request>, batches: &mut BTreeMap<i64, Batch>, device_events: Vec<DeviceEvent>) {
    for (group, event, program) in device_events {
        for target in targets(relatives, group) {
            if let Some(request) = requests.get_mut(&target) {
                request.events.push(event);
            }
        }
        if let Some(batch) = batches.get_mut(&group) {
            batch.events.push(event);
            if let Some(program) = program.filter(|program| !batch.detail.program_ids.contains(program)) {
                batch.detail.program_ids.push(program);
            }
        }
    }
}

fn stamp_host_events(events: &Events, relatives: &Metadata, requests: &mut BTreeMap<i64, Request>) {
    for (event, group) in events.grouped(&EXECUTE_EVENTS) {
        stamp(requests, relatives, group, &|stamps| stamps.execute = Some(event.ts));
    }
    for (event, group) in events.grouped(&LAUNCH_EVENTS) {
        stamp(requests, relatives, group, &|stamps| stamps.launch = Some(stamps.launch.map_or(event.ts, |launch| launch.min(event.ts))));
    }
    for (event, group) in events.grouped(&CALLBACK_EVENTS) {
        stamp(requests, relatives, group, &|stamps| stamps.callback = Some(event.ts));
    }
}

fn apply_batch_timing(requests: &mut BTreeMap<i64, Request>, batches: &mut BTreeMap<i64, Batch>) {
    for request in requests.values_mut() {
        if let Some(batch) = request.batches.iter().find_map(|id| batches.get(id)) {
            request.delay = batch.detail.start_time_ps.wrapping_sub(request.span.begin);
            if request.span.end() < batch.detail.end_time_ps {
                request.span = span(request.span.begin, batch.detail.end_time_ps);
            }
        }
    }
    for batch in batches.values_mut() {
        if let Some(first) = batch.detail.related_request_ids.iter().filter_map(|id| requests.get(id)).min_by_key(|request| request.span.begin) {
            batch.detail.batch_delay_ps = batch.detail.start_time_ps.wrapping_sub(first.span.begin);
        }
    }
}

fn add_host_phases(requests: &mut BTreeMap<i64, Request>) {
    for request in requests.values_mut() {
        let (begin, end, stamps) = (request.span.begin, request.span.end(), request.stamps.values());
        let runtimes: Vec<(u32, Span)> = stamps.clone().filter_map(|stamps| Some((HOST_RUNTIME, span(stamps.execute?, stamps.launch?)))).collect();
        let first_execute = stamps.clone().filter_map(|stamps| stamps.execute).min();
        let concat = stamps.clone().filter_map(|stamps| stamps.concat).min();
        let callback = stamps.clone().filter_map(|stamps| stamps.callback).max();
        let schedule = stamps.clone().find_map(|stamps| stamps.schedule);
        let device_end = request.events.iter().filter(|(kind, _)| *kind == DEVICE_COMPUTE_32).map(|(_, span)| span.end()).max().unwrap_or(0);
        request.events.extend(runtimes);
        request.events.extend(first_execute.map(|execute| (HOST_PREPROCESS, span(begin, execute))));
        match callback {
            Some(callback) => request.events.push((HOST_POSTPROCESS, span(callback, end))),
            None if device_end != 0 => request.events.push((HOST_POSTPROCESS, span(device_end, end))),
            None => {}
        }
        if let (Some(schedule), Some(concat)) = (schedule, concat) {
            request.events.push((HOST_BATCH_FORMATION, span(schedule, concat)));
        }
    }
}

fn tensor_patterns<'a>(events: &'a Events, host: &Plane, ids: &StatIds, requests: &mut BTreeMap<i64, Request<'a>>, batches: &mut BTreeMap<i64, Batch<'a>>) -> Vec<String> {
    for (event, group) in events.grouped(&TENSOR_EVENTS) {
        if let Some(request) = requests.get_mut(&group) {
            request.tensors.push(event);
        } else if let Some(batch) = batches.get_mut(&group) {
            batch.tensors.push(event);
        }
    }
    let pattern = |tensors: &[&HostEvent]| {
        let mut parts = Vec::new();
        for &event in tensors {
            let text = |id: Option<usize>| event_stat(event.raw, id).map(|value| host.text(&value));
            let shape = match (text(ids.shape), text(ids.dims)) {
                (Some(shape), _) => shape,
                (None, Some(dims)) => text(ids.kind).map_or(dims.clone(), |kind| format!("{}{dims}", kind.to_ascii_lowercase())),
                (None, None) => String::new(),
            };
            let Some(layout) = text(ids.layout).filter(|_| !shape.is_empty()) else { return String::new() };
            parts.push(format!("{} {shape} {layout}", event.name));
        }
        parts.sort();
        parts.join("<br>")
    };
    let mut patterns: HashMap<String, i32> = HashMap::new();
    let mut detail = |tensors: &[&HostEvent], owner: i32| {
        let text = Some(pattern(tensors)).filter(|text| !text.is_empty())?;
        let next = patterns.len() as i32;
        let tensor_pattern_index = *patterns.entry(text).or_insert(next);
        Some(TensorEventDetail { tensor_pattern_index, owner, linearize_delinearize_time_ps: tensors.iter().map(|event| event.dur).sum() })
    };
    for request in requests.values_mut() {
        request.details.extend(detail(&request.tensors, REQUEST_OWNER));
    }
    for batch in batches.values_mut() {
        batch.detail.tensor_event_detail = detail(&batch.tensors, BATCH_OWNER);
    }
    for batch in batches.values() {
        let Some(detail) = &batch.detail.tensor_event_detail else { continue };
        for id in &batch.detail.related_request_ids {
            if let Some(request) = requests.get_mut(id) {
                request.details.push(detail.clone());
            }
        }
    }
    let mut ordered: Vec<(String, i32)> = patterns.into_iter().collect();
    ordered.sort_by_key(|(_, index)| *index);
    ordered.into_iter().map(|(text, _)| text).collect()
}

fn host_details(host_id: i32, requests: &BTreeMap<i64, Request>, batches: BTreeMap<i64, Batch>) -> PerHostInferenceStats {
    let duration = |start: u64, end: u64| (span(start, end).duration, start);
    let total = |parts: &[(u32, Span)], kinds: &[u32]| parts.iter().filter(|(kind, _)| kinds.contains(kind)).map(|(_, span)| span.duration).sum::<u64>();
    let mut per_host = PerHostInferenceStats::default();
    for (&group, request) in requests.iter().filter(|(_, request)| request.span.duration != 0) {
        let parts = non_overlapped(&request.events);
        per_host.request_details.push(RequestDetail {
            request_id: group,
            model_id_index: request.model,
            start_time_ps: request.span.begin,
            end_time_ps: request.span.end(),
            device_time_ps: total(&parts, &[DEVICE_COMPUTE_16, DEVICE_COMPUTE_32]),
            write_to_device_time_ps: total(&parts, &[HOST_TO_DEVICE]),
            read_from_device_time_ps: total(&parts, &[DEVICE_TO_HOST]),
            batching_request_delay_ps: request.delay,
            related_batch_ids: request.batches.clone(),
            batching_request_size: request.size,
            host_preprocessing_ps: total(&parts, &[HOST_PREPROCESS]),
            host_batch_formation_ps: total(&parts, &[HOST_BATCH_FORMATION]),
            host_runtime_ps: total(&parts, &[HOST_RUNTIME]),
            host_postprocessing_ps: total(&parts, &[HOST_POSTPROCESS]),
            tensor_event_details: request.details.clone(),
            host_id,
            idle_time_ps: total(&parts, &[UNKNOWN_TIME]) as f64,
            ..Default::default()
        });
    }
    per_host.request_details.sort_by_key(|request| duration(request.start_time_ps, request.end_time_ps));
    for batch in batches.into_values() {
        let device_time_ps = total(&non_overlapped(&batch.events), &[DEVICE_COMPUTE_16, DEVICE_COMPUTE_32]);
        per_host.batch_details.push(BatchDetail { host_id, device_time_ps, ..batch.detail });
    }
    per_host.batch_details.sort_by_key(|batch| duration(batch.start_time_ps, batch.end_time_ps));
    per_host
}

fn batching_parameters(planes: &[Plane], map: &[u8], models: &mut ModelIdDatabase) {
    let Some((plane, line)) = planes.iter().find(|plane| plane.name == TFSTREAMZ_PLANE).and_then(|plane| Some((plane, plane.lines.first().filter(|line| line.events.len() == 2)?))) else { return };
    let mut params: HashMap<String, BatchingParameters> = HashMap::new();
    for stat in stats(slice(map, line.events[1].raw), 4, |_| true) {
        let Some(detail) = plane.stat_names.get(stat.id).and_then(|name| name.strip_prefix(BATCHING_PARAM_PREFIX)) else { continue };
        let (Some(open), Some(close)) = (detail.find('{'), detail.rfind('}')) else { continue };
        let labels = if open < close { &detail[open + 1..close] } else { continue };
        let found = labels.split(", ").find_map(|label| match label.split('=').collect::<Vec<_>>()[..] {
            ["model_name", model] => Some(model),
            _ => None,
        });
        let Some(model) = found else { continue };
        let Some(field) = BATCHING_PARAMS.iter().position(|param| detail.starts_with(param)) else { continue };
        let (entry, value) = (params.entry(model.to_string()).or_default(), int64(&stat.value));
        match field {
            0 => entry.num_batch_threads = value,
            1 => entry.batch_timeout_micros = value,
            2 => entry.max_batch_size = value,
            3 => entry.max_enqueued_batches = value,
            _ => entry.allowed_batch_sizes = plane.text(&stat.value),
        }
    }
    let mut sessions: HashMap<String, Vec<String>> = HashMap::new();
    for id in &models.ids {
        match id.rfind(':') {
            None => sessions.entry(id.clone()).or_default().push(id.clone()),
            Some(colon) if id[colon + 1..].trim_ascii().parse::<i64>().is_ok() => sessions.entry(id[..colon].to_string()).or_default().push(id.clone()),
            Some(_) => {}
        }
    }
    for (model, params) in params {
        for id in sessions.get(&model).into_iter().flatten() {
            models.id_to_batching_params.insert(id.clone(), params.clone());
        }
    }
}

pub fn generate(planes: &[Plane], map: &[u8], relatives: &Metadata, host_id: i32) -> InferenceStats {
    let mut result = InferenceStats::default();
    result.inference_stats_per_host.insert(host_id, PerHostInferenceStats::default());
    let Some(host) = planes.iter().find(|plane| plane.name == HOST_PLANE) else { return result };
    let names: Vec<Cow<str>> = host.meta.iter().map(|meta| meta.long_name(map)).collect();
    let ids = StatIds::new(host);
    let events = Events::collect(host, map, &names, &ids);
    let model_ids = model_ids(&events, host, &ids);
    let mut requests = build_requests(&events, relatives, &model_ids, &mut result.model_id_db);
    stamp_schedules(&events, relatives, &ids, &mut requests);
    let mut batches = build_batches(&events, relatives, &ids, &mut requests);
    attach_device_events(relatives, &mut requests, &mut batches, device_events(planes, map, &events, &ids));
    stamp_host_events(&events, relatives, &mut requests);
    apply_batch_timing(&mut requests, &mut batches);
    add_host_phases(&mut requests);
    result.tensor_patterns = tensor_patterns(&events, host, &ids, &mut requests, &mut batches);
    result.inference_stats_per_host.insert(host_id, host_details(host_id, &requests, batches));
    batching_parameters(planes, map, &mut result.model_id_db);
    result
}

fn combine(host_id: i32, src: InferenceStats, dst: &mut InferenceStats) {
    let update_models = if dst.model_id_db.ids.is_empty() {
        dst.model_id_db = src.model_id_db.clone();
        false
    } else {
        for (id, params) in &src.model_id_db.id_to_batching_params {
            dst.model_id_db.id_to_batching_params.entry(id.clone()).or_insert_with(|| params.clone());
        }
        let mut update = false;
        for id in &src.model_id_db.ids {
            if let Some(&index) = dst.model_id_db.id_to_index.get(id) {
                update |= index != src.model_id_db.id_to_index[id];
            } else {
                dst.model_id_db.id_to_index.insert(id.clone(), dst.model_id_db.ids.len() as i32);
                dst.model_id_db.ids.push(id.clone());
                update = true;
            }
        }
        update
    };
    let mut pattern_index: HashMap<String, i32> = dst.tensor_patterns.iter().enumerate().map(|(index, pattern)| (pattern.clone(), index as i32)).collect();
    let update_patterns = if dst.tensor_patterns.is_empty() {
        dst.tensor_patterns.clone_from(&src.tensor_patterns);
        false
    } else {
        let mut update = false;
        for (index, pattern) in src.tensor_patterns.iter().enumerate() {
            let next = pattern_index.len() as i32;
            match pattern_index.entry(pattern.clone()) {
                Entry::Occupied(entry) => update |= *entry.get() != index as i32,
                Entry::Vacant(entry) => {
                    entry.insert(next);
                    dst.tensor_patterns.push(pattern.clone());
                    update = true;
                }
            }
        }
        update
    };
    let remap = |detail: &mut TensorEventDetail| {
        if let Some(&index) = src.tensor_patterns.get(detail.tensor_pattern_index as usize).and_then(|pattern| pattern_index.get(pattern)) {
            detail.tensor_pattern_index = index;
        }
    };
    for (_, mut host) in src.inference_stats_per_host {
        for request in host.request_details.iter_mut().filter(|_| update_models || update_patterns) {
            if update_models && request.model_id_index != -1 {
                let Some(&index) = src.model_id_db.ids.get(request.model_id_index as usize).and_then(|id| dst.model_id_db.id_to_index.get(id)) else { continue };
                request.model_id_index = index;
            }
            if update_patterns {
                request.tensor_event_details.iter_mut().for_each(remap);
            }
        }
        if update_patterns {
            host.batch_details.iter_mut().for_each(|batch| remap(batch.tensor_event_detail.get_or_insert_with(TensorEventDetail::default)));
        }
        dst.inference_stats_per_host.entry(host_id).or_insert(host);
    }
}

fn throughput_and_latency(spans: &[(u64, u64)]) -> (f64, f64) {
    if spans.is_empty() {
        return (0.0, 0.0);
    }
    let begin = spans.iter().map(|span| span.0).min().unwrap_or_default();
    let end = spans.iter().map(|span| span.1).max().unwrap_or_default();
    let total = spans.iter().fold(0u64, |total, span| total.wrapping_add(span.1.wrapping_sub(span.0)));
    (spans.len() as f64 / (end.wrapping_sub(begin) as f64 / 1e12), total as f64 / 1e6 / spans.len() as f64)
}

fn add_request(sum: &mut RequestDetail, request: &RequestDetail) {
    sum.end_time_ps = sum.end_time_ps.wrapping_add(request.end_time_ps.wrapping_sub(request.start_time_ps));
    sum.device_time_ps += request.device_time_ps;
    sum.read_from_device_time_ps += request.read_from_device_time_ps;
    sum.write_to_device_time_ps += request.write_to_device_time_ps;
    sum.batching_request_delay_ps = sum.batching_request_delay_ps.wrapping_add(request.batching_request_delay_ps);
    sum.batching_request_size = sum.batching_request_size.wrapping_add(request.batching_request_size);
    sum.host_preprocessing_ps += request.host_preprocessing_ps;
    sum.host_batch_formation_ps += request.host_batch_formation_ps;
    sum.host_runtime_ps += request.host_runtime_ps;
    sum.host_postprocessing_ps += request.host_postprocessing_ps;
    sum.idle_time_ps += request.idle_time_ps;
}

fn average_request(sum: &RequestDetail, size: usize) -> RequestDetail {
    if size == 0 {
        return RequestDetail::default();
    }
    let size64 = size as u64;
    RequestDetail {
        start_time_ps: 0,
        end_time_ps: sum.end_time_ps / size64,
        device_time_ps: sum.device_time_ps / size64,
        write_to_device_time_ps: sum.write_to_device_time_ps / size64,
        read_from_device_time_ps: sum.read_from_device_time_ps / size64,
        batching_request_delay_ps: sum.batching_request_delay_ps / size64,
        batching_request_size: (i64::from(sum.batching_request_size) / size as i64) as i32,
        host_preprocessing_ps: sum.host_preprocessing_ps / size64,
        host_batch_formation_ps: sum.host_batch_formation_ps / size64,
        host_runtime_ps: sum.host_runtime_ps / size64,
        host_postprocessing_ps: sum.host_postprocessing_ps / size64,
        idle_time_ps: sum.idle_time_ps / size as f64,
        ..Default::default()
    }
}

fn add_batch(sum: &mut BatchDetail, batch: &BatchDetail) {
    sum.end_time_ps = sum.end_time_ps.wrapping_add(batch.end_time_ps.wrapping_sub(batch.start_time_ps));
    sum.batch_delay_ps = sum.batch_delay_ps.wrapping_add(batch.batch_delay_ps);
    sum.padding_amount = sum.padding_amount.wrapping_add(batch.padding_amount);
    sum.batch_size_after_padding = sum.batch_size_after_padding.wrapping_add(batch.batch_size_after_padding);
    sum.device_time_ps += batch.device_time_ps;
}

fn average_batch(sum: &BatchDetail, size: usize) -> BatchDetail {
    if size == 0 {
        return BatchDetail::default();
    }
    let size64 = size as u64;
    BatchDetail {
        start_time_ps: 0,
        end_time_ps: sum.end_time_ps / size64,
        batch_delay_ps: sum.batch_delay_ps / size64,
        padding_amount: (i64::from(sum.padding_amount) / size as i64) as i32,
        batch_size_after_padding: (i64::from(sum.batch_size_after_padding) / size as i64) as i32,
        device_time_ps: sum.device_time_ps / size64,
        ..Default::default()
    }
}

pub fn regroup(stats: &mut InferenceStats) {
    if stats.inference_stats_per_host.is_empty() {
        return;
    }
    let no_model = stats.model_id_db.ids.is_empty();
    let mut grouped = vec![PerModelInferenceStats::default(); if no_model { 1 } else { stats.model_id_db.ids.len() }];
    let bucket = |index: i32| if no_model { Some(0) } else { usize::try_from(index).ok() };
    for host in stats.inference_stats_per_host.values() {
        for request in &host.request_details {
            if let Some(model) = bucket(request.model_id_index).and_then(|index| grouped.get_mut(index)) {
                model.request_details.push(request.clone());
            }
        }
        for batch in &host.batch_details {
            if let Some(model) = bucket(batch.model_id_index).and_then(|index| grouped.get_mut(index)) {
                model.batch_details.push(batch.clone());
            }
        }
    }
    let duration = |start: u64, end: u64| (span(start, end).duration, start);
    for (index, mut model) in grouped.into_iter().enumerate() {
        model.request_details.sort_by_key(|request| duration(request.start_time_ps, request.end_time_ps));
        model.batch_details.sort_by_key(|batch| duration(batch.start_time_ps, batch.end_time_ps));
        let requests: Vec<(u64, u64)> = model.request_details.iter().map(|request| (request.start_time_ps, request.end_time_ps)).collect();
        let batches: Vec<(u64, u64)> = model.batch_details.iter().map(|batch| (batch.start_time_ps, batch.end_time_ps)).collect();
        (model.request_throughput, model.request_average_latency_us) = throughput_and_latency(&requests);
        (model.batch_throughput, model.batch_average_latency_us) = throughput_and_latency(&batches);
        let mut times: BTreeMap<i32, Vec<u64>> = BTreeMap::new();
        let owned = model.request_details.iter().flat_map(|request| &request.tensor_event_details).filter(|detail| detail.owner == REQUEST_OWNER);
        for detail in owned.chain(model.batch_details.iter().filter_map(|batch| batch.tensor_event_detail.as_ref())) {
            times.entry(detail.tensor_pattern_index).or_default().push(detail.linearize_delinearize_time_ps);
        }
        for (tensor_pattern_index, mut times) in times {
            times.sort_unstable();
            let linearize_delinearize_percentile_time = TENSOR_PERCENTILES.iter().map(|&percentile| (percentile, times[(percentile / 100.0 * times.len() as f64) as usize])).collect();
            model.tensor_pattern_results.push(TensorPatternResult { tensor_pattern_index, count: times.len() as u64, linearize_delinearize_percentile_time });
        }
        let sizes: HashMap<i64, i32> = model.batch_details.iter().map(|batch| (batch.batch_id, batch.batch_size_after_padding)).collect();
        let (mut requests_sum, mut batches_sum) = (RequestDetail::default(), BatchDetail::default());
        let mut per_size: BTreeMap<i32, (RequestDetail, usize, BatchDetail, usize)> = BTreeMap::new();
        for request in &model.request_details {
            add_request(&mut requests_sum, request);
            for size in request.related_batch_ids.iter().filter_map(|id| sizes.get(id)) {
                let info = per_size.entry(*size).or_default();
                add_request(&mut info.0, request);
                info.1 += 1;
            }
        }
        for batch in &model.batch_details {
            add_batch(&mut batches_sum, batch);
            let info = per_size.entry(batch.batch_size_after_padding).or_default();
            add_batch(&mut info.2, batch);
            info.3 += 1;
        }
        model.aggregated_request_detail = average_request(&requests_sum, model.request_details.len());
        model.aggregated_batch_detail = average_batch(&batches_sum, model.batch_details.len());
        for (batch_size, (requests, request_count, batches, batch_count)) in per_size {
            model.per_batch_size_aggregated_result.push(PerBatchSizeAggregatedResult {
                batch_size,
                aggregated_request_result: average_request(&requests, request_count),
                aggregated_batch_result: average_batch(&batches, batch_count),
                request_throughput: request_count as f64 * model.request_throughput / model.request_details.len() as f64,
                batch_throughput: batch_count as f64 * model.batch_throughput / model.batch_details.len() as f64,
            });
        }
        stats.inference_stats_per_model.insert(index as i32, model);
    }
    if no_model {
        stats.model_id_db.ids.push("ALL".into());
        stats.model_id_db.id_to_index.insert("ALL".into(), 0);
    }
    stats.inference_stats_per_host.clear();
}

fn efficiency(batch: &BatchDetail) -> f64 {
    safe_divide(f64::from(batch.batch_size_after_padding.wrapping_sub(batch.padding_amount)), f64::from(batch.batch_size_after_padding))
}

fn percentiles(count: usize) -> Vec<(usize, f64)> {
    let percentile = |index: usize| 100.0 * index as f64 / count as f64;
    if count <= WANTED_PERCENTILES.len() * PER_PERCENTILE {
        return (0..count).map(|index| (index, percentile(index))).collect();
    }
    let (mut next, mut selected) = (0, Vec::new());
    for (wanted, error) in WANTED_PERCENTILES {
        let mut taken = 0;
        for index in next..count {
            let value = percentile(index);
            if value >= wanted + error {
                next = index;
                break;
            }
            if value >= wanted && taken < PER_PERCENTILE {
                selected.push((index, value));
                taken += 1;
            }
        }
    }
    selected
}

pub fn sample(request_column: &str, batch_column: &str, stats: &InferenceStats) -> BTreeMap<i32, SampledPerModelInferenceStats> {
    let request_key = |request: &RequestDetail| -> (i128, f64) {
        match request_column {
            "Request delay for batching" => (request.batching_request_delay_ps.into(), 0.0),
            "Request size" => (request.batching_request_size.into(), 0.0),
            "Host preprocess" => (request.host_preprocessing_ps.into(), 0.0),
            "Host batch formation" => (request.host_batch_formation_ps.into(), 0.0),
            "Host runtime" => (request.host_runtime_ps.into(), 0.0),
            "Data transfer H2D" => (request.write_to_device_time_ps.into(), 0.0),
            "Data transfer D2H" => (request.read_from_device_time_ps.into(), 0.0),
            "Device compute" => (request.device_time_ps.into(), 0.0),
            "Host postprocess" => (request.host_postprocessing_ps.into(), 0.0),
            "Idle time" => (0, request.idle_time_ps),
            _ => (request.end_time_ps.wrapping_sub(request.start_time_ps).into(), 0.0),
        }
    };
    let batch_key = |batch: &BatchDetail| -> (i128, f64) {
        match batch_column {
            "Batching delay" => (batch.batch_delay_ps.into(), 0.0),
            "Padding amount" => (batch.padding_amount.into(), 0.0),
            "Batch size after padding" => (batch.batch_size_after_padding.into(), 0.0),
            "Batching efficiency" => (0, efficiency(batch)),
            _ => (batch.end_time_ps.wrapping_sub(batch.start_time_ps).into(), 0.0),
        }
    };
    let less = |a: (i128, f64), b: (i128, f64)| a.0 < b.0 || (a.0 == b.0 && a.1 < b.1);
    let mut sampled = BTreeMap::new();
    for (&index, model) in &stats.inference_stats_per_model {
        let mut requests: Vec<usize> = (0..model.request_details.len()).collect();
        if request_column != LATENCY_COLUMN {
            std_sort(&mut requests, &|a, b| less(request_key(&model.request_details[a]), request_key(&model.request_details[b])));
        }
        let mut batches: Vec<usize> = (0..model.batch_details.len()).collect();
        if batch_column != LATENCY_COLUMN {
            std_sort(&mut batches, &|a, b| less(batch_key(&model.batch_details[a]), batch_key(&model.batch_details[b])));
        }
        let sampled_requests = percentiles(requests.len()).into_iter().map(|(at, percentile)| RequestDetail { percentile, ..model.request_details[requests[at]].clone() }).collect();
        let sampled_batches = percentiles(batches.len()).into_iter().map(|(at, percentile)| BatchDetail { percentile, ..model.batch_details[batches[at]].clone() }).collect();
        sampled.insert(index, SampledPerModelInferenceStats { sampled_requests, sampled_batches });
    }
    sampled
}

fn request_table(model: &PerModelInferenceStats, sampled: &SampledPerModelInferenceStats, has_batching: bool) -> String {
    let mut table = Table::new(&[("percentile", "string", "Percentile"), ("request_id", "string", "Request ID"), ("latency_ms", "number", "Latency")]);
    table.prop("throughput", fixed(model.request_throughput, 1));
    table.prop("averageLatencyMs", fixed(model.request_average_latency_us / 1e3, 3));
    if has_batching {
        table.column("batching_request_size", "number", "Request size");
        table.column("host_batch_formation", "number", "Host batch formation");
        table.column("throughput", "string", "Throughput");
    }
    TPU_COLUMNS.iter().for_each(|&(id, kind, label)| table.column(id, kind, label));
    table.column("trace_viewer_url", "string", "Trace Viewer URL");
    let rows = sampled.sampled_requests.iter().map(|request| (request, fixed(request.percentile, 3), request.request_id.to_string(), "N/A".to_string()));
    let sizes = model.per_batch_size_aggregated_result.iter().map(|size| (&size.aggregated_request_result, format!("Batch size {}", size.batch_size), "N/A".into(), fixed(size.request_throughput, 1)));
    let average = once((&model.aggregated_request_detail, "Average".to_string(), "N/A".to_string(), fixed(model.request_throughput, 1)));
    for (request, percentile, id, throughput) in rows.chain(sizes).chain(average) {
        let row = table.row();
        row.extend([Cell::Text(percentile), Cell::Text(id), Cell::Number(ms(request.end_time_ps.wrapping_sub(request.start_time_ps)))]);
        if has_batching {
            row.extend([Cell::Number(request.batching_request_size.into()), Cell::Number(ms(request.batching_request_delay_ps)), Cell::Text(throughput)]);
        }
        let parts = [
            request.host_preprocessing_ps,
            request.host_runtime_ps,
            request.write_to_device_time_ps,
            request.read_from_device_time_ps,
            request.device_time_ps,
            request.host_postprocessing_ps,
            request.idle_time_ps as u64,
        ];
        row.extend(parts.into_iter().map(|ps| Cell::Number(ms(ps))));
        row.push(Cell::Text(String::new()));
    }
    table.json()
}

fn batch_table(model: &PerModelInferenceStats, sampled: &SampledPerModelInferenceStats, params: Option<&BatchingParameters>) -> String {
    let mut table = Table::new(&BATCH_COLUMNS);
    let rows = sampled.sampled_batches.iter().map(|batch| (batch, fixed(batch.percentile, 3), batch.batch_id.to_string(), "N/A".to_string()));
    let sizes = model.per_batch_size_aggregated_result.iter().map(|size| (&size.aggregated_batch_result, format!("Batch size {}", size.batch_size), "N/A".into(), fixed(size.batch_throughput, 1)));
    let aggregated = once((&model.aggregated_batch_detail, "Aggregated".to_string(), "N/A".to_string(), fixed(model.batch_throughput, 1)));
    for (batch, percentile, id, throughput) in rows.chain(sizes).chain(aggregated) {
        let programs = if batch.program_ids.is_empty() { "N/A".into() } else { batch.program_ids.iter().map(u64::to_string).collect::<Vec<_>>().join(", ") };
        table.row().extend([
            Cell::Text(percentile),
            Cell::Text(id),
            Cell::Number(ms(batch.end_time_ps.wrapping_sub(batch.start_time_ps))),
            Cell::Number(batch.padding_amount.into()),
            Cell::Number(batch.batch_size_after_padding.into()),
            Cell::Number(efficiency(batch)),
            Cell::Number(ms(batch.batch_delay_ps)),
            Cell::Text(throughput),
            Cell::Number(ms(batch.device_time_ps)),
            Cell::Text(programs),
            Cell::Text(String::new()),
        ]);
    }
    table.prop("throughput", fixed(model.batch_throughput, 1));
    table.prop("averageLatencyMs", fixed(model.batch_average_latency_us / 1e3, 3));
    table.prop("hasBatchingParam", params.is_some().to_string());
    if let Some(params) = params {
        table.prop("batchingParamNumBatchThreads", params.num_batch_threads.to_string());
        table.prop("batchingParamMaxBatchSize", params.max_batch_size.to_string());
        table.prop("batchingParamBatchTimeoutMicros", params.batch_timeout_micros.to_string());
        table.prop("batchingParamMaxEnqueuedBatches", params.max_enqueued_batches.to_string());
        table.prop("batchingParamAllowedBatchSizes", params.allowed_batch_sizes.clone());
    }
    table.json()
}

fn tensor_table(model: &PerModelInferenceStats, patterns: &[String]) -> String {
    let mut table =
        Table::new(&[("id", "number", "ID"), ("tensor_pattern", "string", "Tensor Pattern"), ("count", "number", "Number of Occurrence"), ("percentile", "string", "Linearize/Delinearize latency")]);
    for (index, result) in model.tensor_pattern_results.iter().enumerate() {
        let Some(pattern) = usize::try_from(result.tensor_pattern_index).ok().and_then(|index| patterns.get(index)) else { continue };
        let times: Vec<String> = result.linearize_delinearize_percentile_time.iter().map(|&(percentile, ps)| format!("{}%: {}ms", fixed(percentile, 2), fixed(ms(ps), 6))).collect();
        table.row().extend([Cell::Number(index as f64), Cell::Text(pattern.clone()), Cell::Number(result.count as f64), Cell::Text(times.join("<br>"))]);
    }
    table.json()
}

pub fn tables(stats: &InferenceStats) -> String {
    let mut ids = stats.model_id_db.ids.clone();
    ids.sort();
    let has_batching = stats.inference_stats_per_model.values().any(|model| !model.batch_details.is_empty());
    let mut meta = Table::new(&[("model_name", "string", "Model Name")]);
    meta.prop("hasBatching", has_batching.to_string());
    meta.prop("hasTensorPattern", "false");
    ids.iter().for_each(|id| meta.row().push(Cell::Text(id.clone())));
    let mut tables = vec![meta.json()];
    for id in &ids {
        let Some(index) = stats.model_id_db.id_to_index.get(id) else { continue };
        let (Some(model), Some(sampled)) = (stats.inference_stats_per_model.get(index), stats.sampled_inference_stats.get(index)) else { continue };
        tables.push(request_table(model, sampled, has_batching));
        if has_batching {
            tables.push(batch_table(model, sampled, stats.model_id_db.id_to_batching_params.get(id)));
        }
        if !stats.tensor_patterns.is_empty() {
            tables.push(tensor_table(model, &stats.tensor_patterns));
        }
    }
    format!("[{}]", tables.join(","))
}

pub fn load(paths: &[PathBuf]) -> Option<InferenceStats> {
    let maps: Vec<Vec<u8>> = paths.iter().map(|path| crate::read_file(path).ok().filter(|map| valid_space(map))).collect::<Option<_>>()?;
    let serving = maps.iter().any(|map| serves_requests(map));
    let hosts: Vec<InferenceStats> = maps
        .par_iter()
        .enumerate()
        .map(|(index, map)| {
            let mut planes = if serving { xplane::parse(map).ok()? } else { Vec::new() };
            planes.par_iter_mut().for_each(|plane| plane.add_threadpool_regions(map));
            let relatives = if serving && !is_grouped(&planes) { group(&mut planes, map).map(|groups| groups.relatives).unwrap_or_default() } else { Metadata::new() };
            Some(generate(&planes, map, &relatives, index as i32))
        })
        .collect::<Option<_>>()?;
    let mut stats = InferenceStats::default();
    for (index, host) in hosts.into_iter().enumerate() {
        combine(index as i32, host, &mut stats);
    }
    regroup(&mut stats);
    stats.sampled_inference_stats = sample("", "", &stats);
    Some(stats)
}

pub fn json(paths: &[PathBuf]) -> Option<String> {
    load(paths).map(|stats| tables(&stats))
}

#[cfg(test)]
#[path = "tests/inline/inference_profile.rs"]
mod tests;
