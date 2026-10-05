use crate::inference_profile::{
    BatchDetail, InferenceStats, ModelIdDatabase, PerBatchSizeAggregatedResult, PerHostInferenceStats, PerModelInferenceStats, RequestDetail, TensorEventDetail, TensorPatternResult,
};
use crate::opstats::{Db, OpStats};
use crate::xplane::{self, Ev, NONE_GROUP, Plane};
use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage, MapKey, MessageDescriptor, ReflectMessage, Value};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

static POOL: LazyLock<DescriptorPool> = LazyLock::new(|| DescriptorPool::decode(&include_bytes!("../../tests/data/test_descriptors.pb")[..]).unwrap());

pub const HOST: &str = "/host:CPU";
const UNMODELED_OP_METRICS: [&str; 4] = ["flops", "model_flops", "fingerprint", "precision_stats"];

#[derive(Clone, PartialEq, Message)]
pub struct XSpace {
    #[prost(message, repeated, tag = "1")]
    pub planes: Vec<XPlane>,
    #[prost(string, repeated, tag = "2")]
    pub errors: Vec<String>,
    #[prost(string, repeated, tag = "3")]
    pub warnings: Vec<String>,
    #[prost(string, repeated, tag = "4")]
    pub hostnames: Vec<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct XPlane {
    #[prost(int64, tag = "1")]
    pub id: i64,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "3")]
    pub lines: Vec<XLine>,
    #[prost(btree_map = "int64, message", tag = "4")]
    pub event_metadata: BTreeMap<i64, XEventMetadata>,
    #[prost(btree_map = "int64, message", tag = "5")]
    pub stat_metadata: BTreeMap<i64, XStatMetadata>,
    #[prost(message, repeated, tag = "6")]
    pub stats: Vec<XStat>,
}

#[derive(Clone, PartialEq, Message)]
pub struct XLine {
    #[prost(int64, tag = "1")]
    pub id: i64,
    #[prost(int64, tag = "10")]
    pub display_id: i64,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(string, tag = "11")]
    pub display_name: String,
    #[prost(int64, tag = "3")]
    pub timestamp_ns: i64,
    #[prost(int64, tag = "9")]
    pub duration_ps: i64,
    #[prost(message, repeated, tag = "4")]
    pub events: Vec<XEvent>,
}

#[derive(Clone, PartialEq, Message)]
pub struct XEvent {
    #[prost(int64, tag = "1")]
    pub metadata_id: i64,
    #[prost(int64, optional, tag = "2")]
    pub offset_ps: Option<i64>,
    #[prost(int64, optional, tag = "5")]
    pub num_occurrences: Option<i64>,
    #[prost(int64, tag = "3")]
    pub duration_ps: i64,
    #[prost(message, repeated, tag = "4")]
    pub stats: Vec<XStat>,
}

#[derive(Clone, PartialEq, Message)]
pub struct XStat {
    #[prost(int64, tag = "1")]
    pub metadata_id: i64,
    #[prost(double, optional, tag = "2")]
    pub double_value: Option<f64>,
    #[prost(uint64, optional, tag = "3")]
    pub uint64_value: Option<u64>,
    #[prost(int64, optional, tag = "4")]
    pub int64_value: Option<i64>,
    #[prost(string, optional, tag = "5")]
    pub str_value: Option<String>,
    #[prost(bytes = "vec", optional, tag = "6")]
    pub bytes_value: Option<Vec<u8>>,
    #[prost(uint64, optional, tag = "7")]
    pub ref_value: Option<u64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct XEventMetadata {
    #[prost(int64, tag = "1")]
    pub id: i64,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub display_name: String,
    #[prost(bytes = "vec", tag = "3")]
    pub metadata: Vec<u8>,
    #[prost(message, repeated, tag = "5")]
    pub stats: Vec<XStat>,
    #[prost(int64, repeated, tag = "6")]
    pub child_id: Vec<i64>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct XStatMetadata {
    #[prost(int64, tag = "1")]
    pub id: i64,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(string, tag = "3")]
    pub description: String,
}

#[derive(Clone)]
pub enum V {
    Int(i64),
    Uint(u64),
    Double(f64),
    Str(String),
    Ref(String),
    Bytes(Vec<u8>),
}

impl From<i32> for V {
    fn from(value: i32) -> Self {
        Self::Int(value.into())
    }
}

impl From<i64> for V {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<u64> for V {
    fn from(value: u64) -> Self {
        Self::Uint(value)
    }
}

impl From<f64> for V {
    fn from(value: f64) -> Self {
        Self::Double(value)
    }
}

impl From<&str> for V {
    fn from(value: &str) -> Self {
        Self::Str(value.into())
    }
}

impl From<String> for V {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

pub fn op_stats(spaces: &[XSpace]) -> Option<Arc<OpStats>> {
    let all: Vec<Option<Arc<OpStats>>> = spaces.iter().map(|space| crate::tests::with_file(&space.encode_to_vec(), crate::opstats::load)).collect();
    OpStats::combine(&all)
}

pub fn grouped(space: &XSpace) -> (Vec<u8>, Vec<Plane>, Option<HashMap<i64, String>>) {
    let (map, mut planes) = space.parsed();
    planes.iter_mut().for_each(|plane| plane.add_threadpool_regions(&map));
    let names = crate::group::group(&mut planes, &map).map(|groups| groups.names);
    (map, planes, names)
}

pub fn group_id(event: &Ev) -> Option<i64> {
    (event.group != NONE_GROUP).then_some(event.group)
}

pub fn name<'a>(plane: &'a Plane, event: &Ev) -> &'a str {
    &plane.meta[event.meta as usize].name
}

pub fn descriptor(name: &str) -> MessageDescriptor {
    POOL.get_message_by_name(name).unwrap_or_else(|| panic!("unknown message {name}"))
}

pub fn op_metrics_db(db: &Db) -> DynamicMessage {
    let mut message = DynamicMessage::new(descriptor("tensorflow.profiler.OpMetricsDb"));
    for metrics in &db.metrics {
        let mut op = DynamicMessage::new(descriptor("tensorflow.profiler.OpMetrics"));
        let strings =
            [("name", &metrics.name), ("long_name", &metrics.long_name), ("category", &metrics.category), ("provenance", &metrics.provenance), ("deduplicated_name", &metrics.deduplicated_name)];
        for (field, text) in strings {
            op.set_field_by_name(field, Value::String(text.to_string()));
        }
        let numbers = [
            ("hlo_module_id", metrics.module),
            ("time_ps", metrics.time_ps),
            ("normalized_time_ps", metrics.normalized_time_ps),
            ("min_time_ps", metrics.min_time_ps),
            ("self_time_ps", metrics.self_time_ps),
            ("dma_stall_ps", metrics.dma_stall_ps),
            ("bytes_accessed", metrics.bytes_accessed),
        ];
        for (field, number) in numbers {
            op.set_field_by_name(field, Value::U64(number));
        }
        op.set_field_by_name("occurrences", Value::U32(metrics.occurrences as u32));
        op.set_field_by_name("num_cores", Value::U32(metrics.num_cores));
        op.set_field_by_name("flops_v2", Value::F64(metrics.flops_v2));
        op.set_field_by_name("model_flops_v2", Value::F64(metrics.model_flops_v2));
        op.set_field_by_name("core_type", Value::EnumNumber(metrics.core_type.into()));
        op.set_field_by_name("autotuned", Value::Bool(metrics.autotuned));
        op.set_field_by_name("is_eager", Value::Bool(metrics.is_eager));
        if let Some(energy) = metrics.vdd_energy {
            op.set_field_by_name("vdd_energy_j", Value::F64(energy));
        }
        let breakdown: Vec<Value> = metrics
            .memory
            .iter()
            .map(|&(operation, space, bytes)| {
                let mut entry = DynamicMessage::new(descriptor("tensorflow.profiler.OpMetrics.MemoryAccessed"));
                entry.set_field_by_name("operation_type", Value::EnumNumber(operation.into()));
                entry.set_field_by_name("memory_space", Value::U64(space));
                entry.set_field_by_name("bytes_accessed", Value::U64(bytes));
                Value::Message(entry)
            })
            .collect();
        op.set_field_by_name("memory_accessed_breakdown", Value::List(breakdown));
        if let Some(source) = &metrics.source {
            let mut info = DynamicMessage::new(descriptor("tensorflow.profiler.SourceInfo"));
            info.set_field_by_name("file_name", Value::String(source.file.to_string()));
            info.set_field_by_name("line_number", Value::I32(source.line));
            info.set_field_by_name("stack_frame", Value::String(source.stack.to_string()));
            op.set_field_by_name("source_info", Value::Message(info));
        }
        if !metrics.children.metrics.is_empty() || metrics.children.total_time_ps != 0 {
            op.set_field_by_name("children", Value::Message(op_metrics_db(&metrics.children)));
        }
        message.get_field_by_name_mut("metrics_db").unwrap().as_list_mut().unwrap().push(Value::Message(op));
    }
    for (field, number) in [("total_time_ps", db.total_time_ps), ("total_op_time_ps", db.total_op_time_ps), ("normalized_total_op_time_ps", db.normalized_total_op_time_ps)] {
        message.set_field_by_name(field, Value::U64(number));
    }
    message
}

fn without_unmodeled(message: &mut DynamicMessage) {
    for field in UNMODELED_OP_METRICS {
        if message.descriptor().get_field_by_name(field).is_some() {
            message.clear_field_by_name(field);
        }
    }
    for name in ["metrics_db", "children"] {
        let Some(field) = message.descriptor().get_field_by_name(name).filter(|field| message.has_field(field)) else { continue };
        match message.get_field_mut(&field) {
            Value::List(items) => items.iter_mut().filter_map(|item| item.as_message_mut()).for_each(without_unmodeled),
            Value::Message(child) => without_unmodeled(child),
            _ => {}
        }
    }
}

pub fn assert_db(db: &Db, expected: &str) {
    let mut expected = DynamicMessage::parse_text_format(descriptor("tensorflow.profiler.OpMetricsDb"), expected).unwrap();
    without_unmodeled(&mut expected);
    let canonical = |message: &DynamicMessage| DynamicMessage::decode(message.descriptor(), &message.encode_to_vec()[..]).unwrap().to_text_format();
    assert_eq!(canonical(&op_metrics_db(db)), canonical(&expected));
}

pub fn proto_bytes(name: &str, text: &str) -> Vec<u8> {
    DynamicMessage::parse_text_format(descriptor(name), text).unwrap().encode_to_vec()
}

fn field(message: &DynamicMessage, name: &str) -> Value {
    message.get_field_by_name(name).unwrap().into_owned()
}

fn integer(message: &DynamicMessage, name: &str) -> i128 {
    match field(message, name) {
        Value::I32(value) | Value::EnumNumber(value) => value.into(),
        Value::I64(value) => value.into(),
        Value::U64(value) => value.into(),
        value => panic!("{name} is not an integer: {value:?}"),
    }
}

fn float(message: &DynamicMessage, name: &str) -> f64 {
    field(message, name).as_f64().unwrap()
}

fn messages(message: &DynamicMessage, name: &str) -> Vec<DynamicMessage> {
    field(message, name).as_list().unwrap().iter().map(|item| item.as_message().unwrap().clone()).collect()
}

fn child(message: &DynamicMessage, name: &str) -> DynamicMessage {
    field(message, name).as_message().unwrap().clone()
}

fn tensor_event(message: &DynamicMessage) -> TensorEventDetail {
    TensorEventDetail {
        tensor_pattern_index: integer(message, "tensor_pattern_index") as i32,
        owner: integer(message, "owner") as i32,
        linearize_delinearize_time_ps: integer(message, "linearize_delinearize_time_ps") as u64,
    }
}

fn request(message: &DynamicMessage) -> RequestDetail {
    let number = |name| integer(message, name) as u64;
    RequestDetail {
        request_id: integer(message, "request_id") as i64,
        model_id_index: integer(message, "model_id_index") as i32,
        start_time_ps: number("start_time_ps"),
        end_time_ps: number("end_time_ps"),
        device_time_ps: number("device_time_ps"),
        write_to_device_time_ps: number("write_to_device_time_ps"),
        read_from_device_time_ps: number("read_from_device_time_ps"),
        batching_request_delay_ps: number("batching_request_delay_ps"),
        related_batch_ids: field(message, "related_batch_ids").as_list().unwrap().iter().map(|id| id.as_i64().unwrap()).collect(),
        batching_request_size: integer(message, "batching_request_size") as i32,
        host_preprocessing_ps: number("host_preprocessing_ps"),
        host_batch_formation_ps: number("host_batch_formation_ps"),
        host_runtime_ps: number("host_runtime_ps"),
        host_postprocessing_ps: number("host_postprocessing_ps"),
        tensor_event_details: messages(message, "tensor_event_details").iter().map(tensor_event).collect(),
        host_id: integer(message, "host_id") as i32,
        percentile: float(message, "percentile"),
        idle_time_ps: float(message, "idle_time_ps"),
    }
}

fn batch(message: &DynamicMessage) -> BatchDetail {
    let number = |name| integer(message, name) as u64;
    BatchDetail {
        batch_id: integer(message, "batch_id") as i64,
        start_time_ps: number("start_time_ps"),
        end_time_ps: number("end_time_ps"),
        batch_delay_ps: number("batch_delay_ps"),
        related_request_ids: field(message, "related_request_ids").as_list().unwrap().iter().map(|id| id.as_i64().unwrap()).collect(),
        padding_amount: integer(message, "padding_amount") as i32,
        batch_size_after_padding: integer(message, "batch_size_after_padding") as i32,
        model_id_index: integer(message, "model_id_index") as i32,
        tensor_event_detail: message.has_field_by_name("tensor_event_detail").then(|| tensor_event(&child(message, "tensor_event_detail"))),
        host_id: integer(message, "host_id") as i32,
        percentile: float(message, "percentile"),
        device_time_ps: number("device_time_ps"),
        program_ids: field(message, "program_ids").as_list().unwrap().iter().map(|id| id.as_u64().unwrap()).collect(),
    }
}

fn tensor_results(message: &DynamicMessage) -> Vec<TensorPatternResult> {
    let percentile = |time: &DynamicMessage| (float(time, "percentile"), integer(time, "time_ps") as u64);
    let result = |result: &DynamicMessage| TensorPatternResult {
        tensor_pattern_index: integer(result, "tensor_pattern_index") as i32,
        count: integer(result, "count") as u64,
        linearize_delinearize_percentile_time: messages(result, "linearize_delinearize_percentile_time").iter().map(percentile).collect(),
    };
    messages(message, "tensor_pattern_results").iter().map(result).collect()
}

fn per_model(message: &DynamicMessage) -> PerModelInferenceStats {
    let per_batch_size = |result: &DynamicMessage| PerBatchSizeAggregatedResult {
        batch_size: integer(result, "batch_size") as i32,
        aggregated_request_result: request(&child(result, "aggregated_request_result")),
        aggregated_batch_result: batch(&child(result, "aggregated_batch_result")),
        request_throughput: float(result, "request_throughput"),
        batch_throughput: float(result, "batch_throughput"),
    };
    PerModelInferenceStats {
        request_details: messages(message, "request_details").iter().map(request).collect(),
        aggregated_request_detail: request(&child(message, "aggregated_request_detail")),
        request_throughput: float(message, "request_throughput"),
        request_average_latency_us: float(message, "request_average_latency_us"),
        batch_details: messages(message, "batch_details").iter().map(batch).collect(),
        aggregated_batch_detail: batch(&child(message, "aggregated_batch_detail")),
        batch_throughput: float(message, "batch_throughput"),
        batch_average_latency_us: float(message, "batch_average_latency_us"),
        tensor_pattern_results: tensor_results(&child(message, "tensor_transfer_aggregated_result")),
        per_batch_size_aggregated_result: messages(message, "per_batch_size_aggregated_result").iter().map(per_batch_size).collect(),
    }
}

fn keyed<T>(message: &DynamicMessage, name: &str, convert: impl Fn(&DynamicMessage) -> T) -> BTreeMap<i32, T> {
    let entries = field(message, name).as_map().unwrap().clone();
    entries.iter().map(|(key, value)| (key.as_i32().unwrap(), convert(value.as_message().unwrap()))).collect()
}

pub fn inference_stats(text: &str) -> InferenceStats {
    let message = DynamicMessage::parse_text_format(descriptor("tensorflow.profiler.InferenceStats"), text).unwrap();
    let models = child(&message, "model_id_db");
    let per_host = |host: &DynamicMessage| PerHostInferenceStats {
        request_details: messages(host, "request_details").iter().map(request).collect(),
        batch_details: messages(host, "batch_details").iter().map(batch).collect(),
    };
    let id_to_index = field(&models, "id_to_index").as_map().unwrap().clone();
    InferenceStats {
        inference_stats_per_host: keyed(&message, "inference_stats_per_host", per_host),
        inference_stats_per_model: keyed(&message, "inference_stats_per_model", per_model),
        model_id_db: ModelIdDatabase {
            ids: field(&models, "ids").as_list().unwrap().iter().map(|id| id.as_str().unwrap().to_string()).collect(),
            id_to_index: id_to_index.iter().map(|(key, value)| (if let MapKey::String(key) = key { key.clone() } else { String::new() }, value.as_i32().unwrap())).collect(),
            ..Default::default()
        },
        tensor_patterns: field(&child(&message, "tensor_pattern_db"), "tensor_pattern").as_list().unwrap().iter().map(|pattern| pattern.as_str().unwrap().to_string()).collect(),
        ..Default::default()
    }
}

pub fn tensor_transfer(text: &str) -> Vec<TensorPatternResult> {
    tensor_results(&DynamicMessage::parse_text_format(descriptor("tensorflow.profiler.TensorTransferAggregatedResult"), text).unwrap())
}

pub fn parse_text<T: Message + Default>(name: &str, text: &str) -> T {
    T::decode(&DynamicMessage::parse_text_format(descriptor(name), text).unwrap().encode_to_vec()[..]).unwrap()
}

impl XEvent {
    pub fn set_num_occurrences(&mut self, occurrences: i64) {
        (self.offset_ps, self.num_occurrences) = (None, Some(occurrences));
    }
}

impl XSpace {
    pub fn text(text: &str) -> Self {
        parse_text("tensorflow.profiler.XSpace", text)
    }

    pub fn plane(&mut self, name: &str) -> &mut XPlane {
        let index = self.planes.iter().position(|plane| plane.name == name).unwrap_or_else(|| {
            self.planes.push(XPlane { id: self.planes.len() as i64, name: name.into(), ..Default::default() });
            self.planes.len() - 1
        });
        &mut self.planes[index]
    }

    pub fn add_plane(&mut self) -> &mut XPlane {
        self.planes.push(XPlane::default());
        self.planes.last_mut().unwrap()
    }

    pub fn host(&mut self) -> &mut XPlane {
        self.plane(HOST)
    }

    pub fn tpu(&mut self, ordinal: i32, device_type: &str, peak_teraflops: f64, peak_hbm: f64, sparse_core: Option<i32>) -> &mut XPlane {
        let name = match sparse_core {
            Some(core) => format!("/device:TPU:{ordinal} SparseCore {core}"),
            None => format!("/device:TPU:{ordinal}"),
        };
        let plane = self.plane(&name);
        plane.add_stat("device_type_string", device_type);
        plane.add_stat("peak_teraflops_per_second", peak_teraflops);
        plane.add_stat("peak_hbm_bw_gigabytes_per_second", peak_hbm);
        plane
    }

    pub fn gpu(&mut self, ordinal: i32) -> &mut XPlane {
        self.plane(&format!("/device:GPU:{ordinal}"))
    }

    pub fn parsed(&self) -> (Vec<u8>, Vec<Plane>) {
        let map = self.encode_to_vec();
        let planes = xplane::parse(&map).unwrap();
        (map, planes)
    }
}

impl XPlane {
    pub fn event_metadata(&mut self, name: &str) -> &mut XEventMetadata {
        let id = self.event_metadata.values().find(|meta| meta.name == name).map_or(self.event_metadata.len() as i64 + 1, |meta| meta.id);
        self.event_metadata.entry(id).or_insert_with(|| XEventMetadata { id, name: name.into(), ..Default::default() })
    }

    pub fn stat_metadata(&mut self, name: &str) -> i64 {
        let id = self.stat_metadata.values().find(|meta| meta.name == name).map_or(self.stat_metadata.len() as i64 + 1, |meta| meta.id);
        self.stat_metadata.entry(id).or_insert_with(|| XStatMetadata { id, name: name.into(), ..Default::default() });
        id
    }

    pub fn stat(&mut self, name: &str, value: impl Into<V>) -> XStat {
        let mut stat = XStat { metadata_id: self.stat_metadata(name), ..Default::default() };
        match value.into() {
            V::Int(value) => stat.int64_value = Some(value),
            V::Uint(value) => stat.uint64_value = Some(value),
            V::Double(value) => stat.double_value = Some(value),
            V::Str(value) => stat.str_value = Some(value),
            V::Ref(value) => stat.ref_value = Some(self.stat_metadata(&value) as u64),
            V::Bytes(value) => stat.bytes_value = Some(value),
        }
        stat
    }

    pub fn stats(&mut self, stats: &[(&str, V)]) -> Vec<XStat> {
        stats.iter().map(|(name, value)| self.stat(name, value.clone())).collect()
    }

    pub fn add_stat(&mut self, name: &str, value: impl Into<V>) {
        let stat = self.stat(name, value);
        self.stats.retain(|existing| existing.metadata_id != stat.metadata_id);
        self.stats.push(stat);
    }

    pub fn set_pid_if_not_set(&mut self, pid: i32) {
        let id = self.stat_metadata("process_id");
        if !self.stats.iter().any(|stat| stat.metadata_id == id) {
            self.add_stat("process_id", pid);
        }
    }

    pub fn line(&mut self, id: i64) -> &mut XLine {
        let index = self.lines.iter().position(|line| line.id == id).unwrap_or_else(|| {
            self.lines.push(XLine { id, ..Default::default() });
            self.lines.len() - 1
        });
        &mut self.lines[index]
    }

    pub fn named_line(&mut self, id: i64, name: &str) -> &mut XLine {
        let line = self.line(id);
        line.name = name.into();
        line
    }

    pub fn event(&mut self, line: i64, name: &str, offset_ps: i64, duration_ps: i64, stats: &[(&str, V)]) -> &mut XEvent {
        let metadata_id = self.event_metadata(name).id;
        let stats = self.stats(stats);
        let line = self.line(line);
        line.events.push(XEvent { metadata_id, offset_ps: Some(offset_ps), duration_ps, stats, ..Default::default() });
        line.events.last_mut().unwrap()
    }

    pub fn metadata_stats(&mut self, name: &str, stats: &[(&str, V)]) {
        let stats = self.stats(stats);
        self.event_metadata(name).stats.extend(stats);
    }
}
