use crate::derive::is_derived;
use crate::framework_op_stats::parse_tf_op;
use crate::gpu_cost::{Cost, costs};
use crate::hlo::{Module, msg, protos};
use crate::hlo_text::{Printer, Style};
use crate::opstats::{Db, Perf, READ, Source, WRITE, combine_memory};
use crate::table::{Cell, Table};
use crate::xplane::{Ev, NONE_GROUP, Own, Plane, Value, slice, stats};
use arcstr::ArcStr;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::collections::HashMap;

pub const PREFIX: &str = "/device:GPU:";
pub const CORE: u32 = 1;
const MAX_EXPRESSION: usize = 1_000_000;
const MAX_KERNELS: usize = 1000;
const UNKNOWN_CATEGORY: &str = "unknown";
const CUDA_FLOPS: [(u64, [f64; 4]); 12] = [
    (2000, [64.0, 0.0, 0.0, 0.0]),
    (3000, [384.0, 0.0, 0.0, 0.0]),
    (5000, [256.0, 0.0, 0.0, 0.0]),
    (6000, [128.0, 256.0, 0.0, 0.0]),
    (6010, [256.0, 4.0, 0.0, 0.0]),
    (7000, [128.0, 256.0, 0.0, 1024.0]),
    (7050, [128.0, 256.0, 0.0, 1024.0]),
    (8000, [128.0, 512.0, 1024.0, 2048.0]),
    (8060, [256.0, 256.0, 256.0, 1024.0]),
    (8090, [256.0, 256.0, 512.0, 1024.0]),
    (9000, [256.0, 512.0, 2048.0, 4096.0]),
    (10000, [296.0, 592.0, 4096.0, 8192.0]),
];
const ROCM: [(&str, [f64; 4], f64); 4] = [
    ("gfx908", [128.0, 0.0, 256.0, 1024.0], 128.0),
    ("gfx90a", [128.0, 0.0, 256.0, 1024.0], 128.0),
    ("gfx942", [256.0, 256.0, 256.0, 2048.0], 128.0),
    ("gfx950", [256.0, 256.0, 256.0, 4096.0], 256.0),
];
const TENSOR_CORE_KERNELS: &str = "16816 c1688 conv1x1 conv2d_c1_k1 dgrad_1x1_stride_2x2 direct_group first_layer_wgrad_kernel h1688 h884 hmma i16832 i8816 s884 s1688 xmma_gemm xmma_implicit_gemm xmma_sparse_conv xmma_sparse_gemm xmma_warp_specialized_implicit_gemm";
const TENSOR_CORE_SUFFIXES: &str =
    "Conv2D Conv2DBackpropFilter Conv2DBackpropInput Conv3D DepthwiseConv2dNative DepthwiseConv2dNativeBackpropFilter DepthwiseConv2dNativeBackpropInput /MatMul FusedMatMul /CudnnRNN XlaDot XlaDotV2";
const TENSOR_CORE_INFIXES: &str = "BatchMatMul CudnnRNNV CudnnRNNForward CudnnRNNBackprop";
const KERNEL_COLUMNS: [(&str, &str, &str); 15] = [
    ("rank", "number", "Rank"),
    ("kernel_name", "string", "Kernel Name"),
    ("registers_per_thread", "number", "Registers per thread"),
    ("shmem_bytes", "number", "Shared Mem bytes"),
    ("block_dim", "string", "Block dim"),
    ("grid_dim", "string", "Grid dim"),
    ("occupancy_pct", "number", "Theoretical Occupancy %"),
    ("is_op_tensor_core_eligible", "boolean", "Op is TensorCore eligible"),
    ("is_kernel_using_tensor_core", "boolean", "Kernel uses TensorCore"),
    ("op_name", "string", "Op Name"),
    ("occurrences", "number", "Occurrences"),
    ("total_duration_us", "number", "Total Duration (\u{3bc}s)"),
    ("avg_duration_us", "number", "Avg Duration (\u{3bc}s)"),
    ("min_duration_us", "number", "Min Duration (\u{3bc}s)"),
    ("max_duration_us", "number", "Max Duration (\u{3bc}s)"),
];
const STATS: [&str; 14] = [
    "tf_op",
    "hlo_op",
    "program_id",
    "kernel_details",
    "correlation_id",
    "group_id",
    "is_eager",
    "equation",
    "cuda_graph_exec_id",
    "hlo_module",
    "scope_range_id",
    "cuda_graph_id",
    "flops",
    "bytes_accessed",
];
const UNWANTED: u8 = u8::MAX;
pub const TF_OP: usize = 0;
pub const HLO_OP: usize = 1;
pub const PROGRAM: usize = 2;
pub const KERNEL: usize = 3;
pub const CORRELATION: usize = 4;
pub const GROUP: usize = 5;
const EAGER: usize = 6;
const EQUATION: usize = 7;
pub const GRAPH_EXEC: usize = 8;
pub const MODULE: usize = 9;
pub const SCOPE: usize = 10;
pub const GRAPH: usize = 11;
const FLOPS: usize = 12;
const BYTES: usize = 13;

pub struct Info {
    pub category: ArcStr,
    pub provenance: ArcStr,
    pub deduplicated: ArcStr,
    pub expression: ArcStr,
    pub source: Source,
    pub cost: Option<Cost>,
}

pub type Infos = FxHashMap<u64, FxHashMap<String, Info>>;
type Names = FxHashMap<u64, FxHashSet<String>>;
type Launch<'a> = (u32, Cow<'a, str>, Cow<'a, str>, Cow<'a, str>, Option<u64>, Option<Cow<'a, str>>);

pub struct Scanner<'a> {
    plane: &'a Plane,
    map: &'a [u8],
    slots: Vec<u8>,
}

pub struct Stats<'a> {
    plane: &'a Plane,
    values: [Option<Value<'a>>; STATS.len()],
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct KernelKey {
    pub name: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub registers: u32,
    pub static_shmem: u32,
    pub dynamic_shmem: u32,
    pub tensor_core: bool,
    pub eligible: bool,
    pub op_name: String,
}

#[derive(Clone, Default)]
pub struct KernelReport {
    pub key: KernelKey,
    pub occupancy: f64,
    pub total_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub occurrences: u64,
}

#[derive(Default)]
struct Caps {
    clock_ghz: f64,
    cores: u64,
    bandwidth: u64,
    major: u64,
    minor: u64,
    vendor: String,
    name: String,
}

#[derive(Default)]
struct Tracker<'a> {
    info: Option<&'a Info>,
    name: String,
    program: u64,
    group: u64,
    eager: bool,
    occurrences: u64,
    duration: u64,
    flops: u64,
    bytes: u64,
    vdd: f64,
}

#[derive(Default)]
struct Builder<'a> {
    vdd: Option<usize>,
    db: Db,
    index: FxHashMap<(u64, ArcStr), usize>,
    current: Tracker<'a>,
}

#[derive(Default)]
struct Launches<'a> {
    index: FxHashMap<Launch<'a>, usize>,
    totals: Vec<(Launch<'a>, [u64; 4])>,
}

pub fn devices(planes: &[Plane]) -> Vec<&Plane> {
    if planes.iter().any(|plane| plane.name.starts_with("/device:TPU:")) {
        return Vec::new();
    }
    planes.iter().filter(|plane| plane.name.starts_with(PREFIX)).collect()
}

pub fn text<'a>(plane: &'a Plane, value: &Value<'a>) -> Cow<'a, str> {
    match value {
        Value::Str(bytes) => crate::xplane::lossy(bytes),
        Value::Ref(id) => Cow::Borrowed(plane.stat_names.get(*id as usize).map_or("", |name| name)),
        _ => Cow::Borrowed(""),
    }
}

pub fn event_name<'a>(plane: &'a Plane, map: &'a [u8], meta: u32) -> Cow<'a, str> {
    let meta = &plane.meta[meta as usize];
    if meta.display.is_empty() { Cow::Borrowed(&meta.name) } else { meta.long_name(map) }
}

impl<'a> Scanner<'a> {
    pub fn new(plane: &'a Plane, map: &'a [u8]) -> Scanner<'a> {
        let mut slots = vec![UNWANTED; plane.stat_names.len()];
        for (slot, name) in STATS.iter().enumerate() {
            if let Some(id) = plane.id(name) {
                slots[id] = slot as u8;
            }
        }
        Scanner { plane, map, slots }
    }

    pub fn scan(&self, event: &Ev) -> Stats<'a> {
        let mut values = [const { None }; STATS.len()];
        for stat in stats(slice(self.map, event.raw), 4, |id| self.slots.get(id).is_some_and(|&slot| slot != UNWANTED)) {
            values[self.slots[stat.id] as usize] = Some(stat.value);
        }
        Stats { plane: self.plane, values }
    }
}

impl<'a> Stats<'a> {
    pub fn has(&self, slot: usize) -> bool {
        self.values[slot].is_some()
    }

    pub fn text(&self, slot: usize) -> Cow<'a, str> {
        self.values[slot].as_ref().map_or(Cow::Borrowed(""), |value| text(self.plane, value))
    }

    pub fn int(&self, slot: usize) -> Option<i64> {
        self.values[slot].as_ref().and_then(Value::int)
    }

    pub fn signed(&self, slot: usize) -> Option<i64> {
        self.values[slot].as_ref().map(|value| if let Value::Int(value) = value { *value } else { 0 })
    }

    pub fn unsigned(&self, slot: usize) -> Option<u64> {
        self.values[slot].as_ref().map(|value| if let Value::Uint(value) = value { *value } else { 0 })
    }

    pub fn program(&self) -> Option<u64> {
        self.int(PROGRAM).map(|program| program as u64)
    }

    pub fn hlo_op(&self) -> Option<Cow<'a, str>> {
        self.has(HLO_OP).then(|| match self.text(HLO_OP) {
            Cow::Borrowed(text) => Cow::Borrowed(text.rsplit("::").next().unwrap_or_default()),
            Cow::Owned(text) => Cow::Owned(text.rsplit("::").next().unwrap_or_default().to_string()),
        })
    }
}

pub fn groups(plane: &Plane, map: &[u8]) -> Vec<Vec<Option<i64>>> {
    let id = plane.id("group_id");
    let own = |event: &Ev| {
        if event.group != NONE_GROUP { Some(event.group) } else { stats(slice(map, event.raw), 4, |stat| Some(stat) == id).next().and_then(|stat| stat.value.int()) }
    };
    let mut result: Vec<Vec<Option<i64>>> = plane.lines.iter().map(|line| line.events.iter().map(own).collect()).collect();
    if id.is_none() && result.iter().flatten().all(Option::is_none) {
        return result;
    }
    let mut order: Vec<(u64, u64, usize, usize)> = plane
        .lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !is_derived(line.id))
        .flat_map(|(line_index, line)| line.events.iter().enumerate().map(move |(index, event)| (event.ts, event.dur, line_index, index)))
        .collect();
    order.sort_by_key(|&(ts, dur, _, _)| (ts, dur));
    let mut current = None;
    for (_, _, line, index) in order {
        match result[line][index] {
            Some(group) => current = Some(group),
            None => result[line][index] = current,
        }
    }
    result
}

fn caps(plane: &Plane) -> Caps {
    let mut caps = Caps::default();
    for (name, value) in &plane.own {
        let number = if let Own::Int(number) = value { *number as u64 } else { 0 };
        let text = || if let Own::Text(text) = value { text.clone() } else { String::new() };
        match &**name {
            "clock_rate" => caps.clock_ghz = number as f64 / 1000000.0,
            "core_count" => caps.cores = number,
            "memory_bandwidth" => caps.bandwidth = number,
            "compute_cap_major" => caps.major = number,
            "compute_cap_minor" => caps.minor = number,
            "device_vendor" => caps.vendor = text(),
            "gpu_device_name" => caps.name = text(),
            _ => {}
        }
    }
    caps
}

fn gfx_version(caps: &Caps) -> &str {
    let name = caps.name.as_str();
    if name.strip_prefix("gfx").is_some_and(|suffix| suffix.len() >= 3 && suffix.bytes().all(|byte| byte.is_ascii_hexdigit())) {
        return name;
    }
    match (caps.major, caps.minor) {
        (9, 4) => "gfx942",
        (9, 5) => "gfx950",
        _ => "",
    }
}

fn max_scaled(values: [f64; 4], clock: f64) -> f64 {
    values.into_iter().map(|value| value * clock).fold(f64::MIN, f64::max)
}

fn flops_per_core(caps: &Caps) -> f64 {
    match caps.vendor.as_str() {
        "Nvidia" => {
            let wanted = caps.major * 1000 + caps.minor * 10;
            let entry = CUDA_FLOPS.iter().find(|(key, _)| *key == wanted).or_else(|| CUDA_FLOPS.iter().rev().find(|(key, _)| *key < wanted)).unwrap_or(&CUDA_FLOPS[0]);
            max_scaled(entry.1, caps.clock_ghz)
        }
        "AMD" => ROCM.iter().find(|(name, _, _)| *name == gfx_version(caps)).map_or(0.0, |(_, values, _)| max_scaled(*values, caps.clock_ghz)),
        _ => 0.0,
    }
}

fn shared_memory_per_core(caps: &Caps) -> f64 {
    match caps.vendor.as_str() {
        "Nvidia" => (if caps.major <= 2 { 32.0 * 4.0 / 2.0 } else { 32.0 * 8.0 }) * caps.clock_ghz * 1e9,
        "AMD" => match ROCM.iter().find(|(name, _, _)| *name == gfx_version(caps)) {
            Some((_, _, bytes)) if *bytes > 0.0 => bytes * caps.clock_ghz * 1e9,
            _ => 0.0,
        },
        _ => 0.0,
    }
}

pub fn model_name(plane: &Plane) -> String {
    let caps = caps(plane);
    match caps.vendor.as_str() {
        "Nvidia" => match (caps.major, caps.minor) {
            (2, _) => "Nvidia GPU (Fermi)",
            (3, _) => "Nvidia GPU (Kepler)",
            (5, _) => "Nvidia GPU (Maxwell)",
            (6, _) => "Nvidia GPU (Pascal)",
            (7, minor) if minor < 5 => "Nvidia GPU (Volta)",
            (7, _) => "Nvidia GPU (Turing)",
            (8, minor) if minor < 9 => "Nvidia GPU (Ampere)",
            (8, _) => "Nvidia GPU (Ada Lovelace)",
            (9, _) => "Nvidia GPU (Hopper)",
            (10, _) => "Nvidia GPU (Blackwell)",
            _ => "Nvidia GPU",
        }
        .to_string(),
        "AMD" => match gfx_version(&caps) {
            "" => match caps.major {
                9 => "AMD GPU - gfx-9XX series",
                10 => "AMD GPU - gfx-10XX series",
                11 => "AMD GPU - gfx-11XX series",
                _ => "AMD GPU",
            }
            .to_string(),
            gfx => format!("AMD GPU - {gfx}"),
        },
        _ => String::new(),
    }
}

pub fn perf_env(plane: &Plane) -> Perf {
    let caps = caps(plane);
    let mut peak_tera_flops = plane.own_double("peak_teraflops_per_second");
    if peak_tera_flops <= 0.0 {
        peak_tera_flops = caps.cores as f64 * (flops_per_core(&caps) / 1e3);
    }
    let hbm = caps.bandwidth as f64 / 1e9;
    let shared = caps.cores as f64 * (shared_memory_per_core(&caps) / 1e9);
    let sram = |name: &str| Some(plane.own_double(name)).filter(|&value| value > 0.0).unwrap_or(shared);
    let bandwidths = vec![hbm, sram("peak_sram_rd_bw_gigabytes_per_second"), sram("peak_sram_wr_bw_gigabytes_per_second")];
    Perf { peak_tera_flops, ridge_point: peak_tera_flops * 1e3 / hbm, bandwidths, cmem: false }
}

pub fn tf_op_fullname(kind: &str, name: &str) -> String {
    match (kind.is_empty(), name) {
        (true, "") => String::new(),
        (true, "XLA_Args" | "XLA_Retvals") => format!("{name}:{name}"),
        _ => format!("{name}:{kind}"),
    }
}

fn stack(module: &Module, mut frame: i32) -> String {
    let index = msg(&module.proto.stack_frame_index);
    let mut out = String::new();
    let get = |list: &[String], id: i32| usize::try_from(id - 1).ok().and_then(|id| list.get(id)).cloned().unwrap_or_default();
    while frame > 0 {
        let Some(entry) = index.stack_frames.get(frame as usize - 1) else { break };
        let Some(location) = usize::try_from(entry.file_location_id - 1).ok().and_then(|id| index.file_locations.get(id)) else { break };
        let (file, function) = (get(&index.file_names, location.file_name_id), get(&index.function_names, location.function_name_id));
        if file.is_empty() && function.is_empty() && location.line == 0 && location.column == 0 {
            break;
        }
        out.push_str(&format!("{file}:{}:{}\n", location.line, location.column));
        frame = entry.parent_frame_id;
    }
    out
}

fn source(module: &Module, metadata: &crate::hlo::xla::OpMetadata) -> Source {
    let frames = if metadata.stack_frame_id != 0 { stack(module, metadata.stack_frame_id) } else { String::new() };
    if !metadata.source_file.is_empty() {
        return Source { file: metadata.source_file.as_str().into(), line: metadata.source_line, stack: frames.into() };
    }
    let first = frames.split('\n').next().unwrap_or_default();
    let parts: Vec<&str> = first.split(':').collect();
    if !frames.is_empty() && parts.len() >= 2 {
        return Source { file: parts[0].into(), line: parts[1].parse().unwrap_or(-1), stack: frames.as_str().into() };
    }
    Source { line: -1, ..Default::default() }
}

fn wanted(devices: &[&Plane], map: &[u8]) -> Names {
    let found: Vec<Names> = devices
        .par_iter()
        .map(|plane| {
            let scanner = Scanner::new(plane, map);
            let mut found = Names::default();
            for event in plane.lines.iter().filter(|line| !is_derived(line.id)).flat_map(|line| &line.events) {
                let stats = scanner.scan(event);
                if let (Some(program), Some(name)) = (stats.program(), stats.hlo_op()) {
                    let names = found.entry(program).or_default();
                    if !names.contains(&*name) {
                        names.insert(name.into_owned());
                    }
                }
            }
            found
        })
        .collect();
    let mut merged = Names::default();
    for (program, names) in found.into_iter().flatten() {
        merged.entry(program).or_default().extend(names);
    }
    merged
}

fn info(module: &Module, printer: &Printer, node: usize) -> Info {
    let mut expression = String::new();
    let metadata = msg(&printer.instruction(node, &mut expression).metadata).into_owned();
    if expression.len() > MAX_EXPRESSION {
        expression.truncate(expression.floor_char_boundary(MAX_EXPRESSION));
    }
    Info {
        category: module.category(node).into(),
        provenance: tf_op_fullname(&metadata.op_type, &metadata.op_name).into(),
        deduplicated: metadata.deduplicated_name.as_str().into(),
        expression: expression.into(),
        source: source(module, &metadata),
        cost: None,
    }
}

pub fn infos(planes: &[Plane], map: &[u8], devices: &[&Plane]) -> Infos {
    let (wanted, modules) = rayon::join(|| wanted(devices, map), || protos(planes, map).into_par_iter().map(|(program, proto)| (program, Module::parse(Cow::Borrowed(proto)))).collect::<Vec<_>>());
    modules
        .into_par_iter()
        .filter_map(|(program, module)| Some((program, module, wanted.get(&program)?)))
        .map(|(program, module, names)| {
            if !module.valid {
                return (program, FxHashMap::default());
            }
            let mut index: HashMap<&str, usize> = HashMap::new();
            for (node, entry) in module.nodes.iter().enumerate() {
                index.entry(entry.name.as_str()).or_insert(node);
            }
            let nodes: Vec<(&String, usize)> = names.iter().filter_map(|name| Some((name, *index.get(name.as_str())?))).collect();
            let (costs, mut found) = rayon::join(
                || costs(&module),
                || {
                    let printer = Printer::new(&module, Style::Expression, false);
                    nodes.par_iter().map(|&(name, node)| (name.clone(), info(&module, &printer, node))).collect::<FxHashMap<String, Info>>()
                },
            );
            for (name, node) in nodes {
                found.get_mut(name).unwrap().cost = costs.as_ref().and_then(|costs| costs.get(node).cloned());
            }
            (program, found)
        })
        .collect()
}

impl<'a> Builder<'a> {
    fn enter(&mut self, op: &Tracker) {
        let Some(info) = op.info else { return };
        let metrics = self.db.entry(&mut self.index, op.program, &op.name.as_str().into());
        if metrics.occurrences == 0 && metrics.category.is_empty() && metrics.provenance.is_empty() {
            metrics.category = if info.category.is_empty() { UNKNOWN_CATEGORY.into() } else { info.category.clone() };
            metrics.provenance = info.provenance.clone();
            if !info.deduplicated.is_empty() {
                metrics.deduplicated_name = info.deduplicated.clone();
            }
            if !info.expression.is_empty() {
                metrics.long_name = info.expression.clone();
            }
            metrics.is_eager |= op.eager;
            metrics.source = Some(info.source.clone());
        }
        if op.vdd != 0.0 {
            metrics.vdd_energy = Some(metrics.vdd_energy.unwrap_or(0.0) + op.vdd);
        }
        metrics.num_cores = 1;
        metrics.occurrences += op.occurrences;
        metrics.time_ps += op.duration;
        metrics.self_time_ps += op.duration;
        let cost = info.cost.as_ref();
        let (flops, bytes) = (op.flops as i64, op.bytes as i64);
        let device_flops = cost.filter(|cost| cost.device_flops > 0).map_or(flops, |cost| cost.device_flops);
        let bytes = cost.filter(|cost| cost.bytes_accessed > 0).map_or(bytes, |cost| cost.bytes_accessed);
        let model = cost.filter(|cost| cost.model_flops > 0).map_or(flops, |cost| cost.model_flops);
        metrics.flops_v2 += device_flops as f64 * op.occurrences as f64;
        if model == 0 {
            metrics.model_flops_v2 = metrics.flops_v2;
        } else {
            metrics.model_flops_v2 += model as f64 * op.occurrences as f64;
        }
        metrics.bytes_accessed = metrics.bytes_accessed.wrapping_add((bytes as u64).wrapping_mul(op.occurrences));
        if let Some(cost) = cost {
            let breakdown: Vec<(u8, u64, u64)> =
                cost.memory.iter().map(|&(read, space, bytes)| (if read { READ } else { WRITE }, space, u64::try_from(bytes).unwrap_or(0).wrapping_mul(op.occurrences))).collect();
            combine_memory(&breakdown, &mut metrics.memory);
        }
        self.db.total_op_time_ps += op.duration;
    }

    fn flush(&mut self) {
        let current = std::mem::take(&mut self.current);
        self.enter(&current);
    }

    fn hlo_op(&mut self, info: &'a Info, name: Cow<str>, stats: &Stats, group: Option<i64>, map: &[u8], event: &Ev) {
        if *name != self.current.name || group.is_none_or(|group| group as u64 != self.current.group) {
            self.flush();
        }
        let current = &mut self.current;
        if current.occurrences == 0 {
            current.info = Some(info);
            current.name = name.into_owned();
            current.program = stats.program().unwrap_or(0);
            if let Some(group) = group {
                current.group = group as u64;
            }
            current.eager = event.eager.unwrap_or(stats.int(EAGER).unwrap_or(0) != 0);
        }
        current.occurrences += 1;
        current.duration += event.dur;
        if self.vdd.is_some() {
            for stat in crate::xplane::stats(slice(map, event.raw), 4, |id| Some(id) == self.vdd) {
                if let Value::Double(value) = stat.value {
                    current.vdd += value;
                }
            }
        }
        if stats.has(FLOPS) {
            current.flops = stats.int(FLOPS).unwrap_or(0) as u64;
        }
        if stats.has(BYTES) {
            current.bytes = stats.int(BYTES).unwrap_or(0) as u64;
        }
    }

    fn tf_op(&mut self, tf_op: &str, event_name: &str, stats: &Stats, event: &Ev) {
        self.flush();
        let op = parse_tf_op(tf_op);
        let info = Info { category: op.kind.into(), provenance: tf_op.into(), deduplicated: ArcStr::new(), expression: ArcStr::new(), source: Source { line: -1, ..Default::default() }, cost: None };
        let name = format!("{}/{event_name}", op.name);
        self.enter(&Tracker { info: Some(&info), name, eager: event.eager.unwrap_or(stats.int(EAGER).unwrap_or(0) != 0), occurrences: 1, duration: event.dur, ..Default::default() });
    }
}

impl<'a> Launches<'a> {
    fn add(&mut self, event: &Ev, stats: &Stats<'a>) {
        let (details, duration) = (stats.text(KERNEL), event.dur / 1000);
        if event.dur == 0 || details.is_empty() {
            return;
        }
        let launch = (event.meta, details, stats.text(TF_OP), stats.text(EQUATION), stats.program(), stats.hlo_op());
        let position = *self.index.entry(launch.clone()).or_insert_with(|| {
            self.totals.push((launch, [0, u64::MAX, 0, 0]));
            self.totals.len() - 1
        });
        let totals = &mut self.totals[position].1;
        *totals = [totals[0] + duration, totals[1].min(duration), totals[2].max(duration), totals[3] + 1];
    }
}

pub fn device_plane(plane: &Plane, map: &[u8], infos: &Infos, origin: i64) -> (Db, Vec<KernelReport>) {
    let (scanner, groups) = (Scanner::new(plane, map), groups(plane, map));
    let (mut builder, mut launches) = (Builder { vdd: plane.id("vdd_energy_j"), ..Default::default() }, Launches::default());
    let (mut first, mut last) = (i64::MAX, 0i64);
    for (line_index, line) in plane.lines.iter().enumerate().filter(|(_, line)| !is_derived(line.id)) {
        let base = (line.timestamp_ns - origin) * 1000;
        for (event, group) in line.events.iter().zip(&groups[line_index]) {
            let offset = event.ts as i64 - base;
            first = first.min(offset);
            last = last.max(offset + event.dur as i64);
            let stats = scanner.scan(event);
            launches.add(event, &stats);
            match stats.hlo_op() {
                Some(name) => {
                    if let Some(info) = stats.program().and_then(|program| infos.get(&program)?.get(&*name)) {
                        builder.hlo_op(info, name, &stats, *group, map, event);
                    }
                }
                None if !stats.text(TF_OP).is_empty() => builder.tf_op(&stats.text(TF_OP), &event_name(plane, map, event.meta), &stats, event),
                None => {}
            }
        }
        builder.flush();
    }
    (builder.db.with_idle(if last != 0 { (last - first) as u64 } else { 0 }), kernel_reports(plane, map, infos, launches.totals))
}

fn tensor_core_eligible(op_name: &str) -> bool {
    TENSOR_CORE_SUFFIXES.split(' ').any(|suffix| op_name.ends_with(suffix)) || TENSOR_CORE_INFIXES.split(' ').any(|part| op_name.contains(part))
}

fn einsum_eligible(equation: &str) -> bool {
    let parts: Vec<&str> = equation.split("->").collect();
    !equation.is_empty() && parts.len() == 2 && parts[0].split(',').count() == 2
}

pub(crate) fn launch_params(details: &str, key: &mut KernelKey, occupancy: &mut f64) {
    key.grid = [1; 3];
    key.block = [1; 3];
    for param in details.split([' ', '\n']) {
        let parts: Vec<&str> = param.split(':').collect();
        let [name, value] = parts[..] else { continue };
        let dims = || {
            let values: Vec<u32> = value.split(',').map_while(|part| part.parse().ok()).collect();
            <[u32; 3]>::try_from(values).ok().filter(|_| value.split(',').count() == 3)
        };
        match name {
            "regs" => key.registers = value.parse().unwrap_or(key.registers),
            "static_shared" => key.static_shmem = value.parse().unwrap_or(key.static_shmem),
            "dynamic_shared" => key.dynamic_shmem = value.parse().unwrap_or(key.dynamic_shmem),
            "block" => key.block = dims().unwrap_or(key.block),
            "grid" => key.grid = dims().unwrap_or(key.grid),
            "occ_pct" => *occupancy = value.parse().unwrap_or(*occupancy),
            _ => {}
        }
    }
}

fn kernel_reports(plane: &Plane, map: &[u8], infos: &Infos, launches: Vec<(Launch, [u64; 4])>) -> Vec<KernelReport> {
    let mut reports = Vec::with_capacity(launches.len());
    for ((meta, details, tf_op, equation, program, hlo_op), [total_ns, min_ns, max_ns, occurrences]) in launches {
        let name = event_name(plane, map, meta).into_owned();
        let mut key = KernelKey { tensor_core: TENSOR_CORE_KERNELS.split(' ').any(|pattern| name.contains(pattern)), name, ..Default::default() };
        let mut occupancy = 0.0;
        launch_params(&details, &mut key, &mut occupancy);
        if !tf_op.is_empty() {
            key.op_name = parse_tf_op(&tf_op).name;
            key.eligible = einsum_eligible(&equation) || tensor_core_eligible(&key.op_name) || key.tensor_core;
        }
        if let Some(info) = hlo_op.and_then(|name| infos.get(&program?)?.get(&*name)) {
            key.op_name = info.provenance.to_string();
            key.eligible = tensor_core_eligible(&key.op_name) || key.eligible;
        }
        reports.push(KernelReport { key, occupancy, total_ns, min_ns, max_ns, occurrences });
    }
    reports
}

pub fn device(planes: &[Plane], map: &[u8], gpus: &[&Plane]) -> (Db, Vec<KernelReport>) {
    let infos = infos(planes, map, gpus);
    let origin = crate::xplane::origin_ns(planes);
    let parts: Vec<(Db, Vec<KernelReport>)> = gpus.par_iter().map(|plane| device_plane(plane, map, &infos, origin)).collect();
    let db = Db::combined(parts.iter().map(|(part, _)| part), true);
    (db, top_kernels(parts.into_iter().flat_map(|(_, reports)| reports)))
}

pub fn top_kernels(reports: impl IntoIterator<Item = KernelReport>) -> Vec<KernelReport> {
    let mut merged: HashMap<KernelKey, KernelReport> = HashMap::new();
    for report in reports {
        match merged.get_mut(&report.key) {
            Some(existing) => {
                existing.total_ns += report.total_ns;
                existing.min_ns = existing.min_ns.min(report.min_ns);
                existing.max_ns = existing.max_ns.max(report.max_ns);
                existing.occurrences += report.occurrences;
            }
            None => _ = merged.insert(report.key.clone(), report),
        }
    }
    sorted_kernels(merged.into_values().collect())
}

pub fn sorted_kernels(mut reports: Vec<KernelReport>) -> Vec<KernelReport> {
    reports.sort_by(|a, b| b.total_ns.cmp(&a.total_ns).then_with(|| a.key.cmp(&b.key)));
    reports.truncate(MAX_KERNELS);
    reports
}

pub fn by_op_name(reports: &[KernelReport]) -> HashMap<&str, (bool, u64, u64)> {
    let mut result: HashMap<&str, (bool, u64, u64)> = HashMap::new();
    for report in reports {
        let entry = result.entry(report.key.op_name.as_str()).or_insert((report.key.eligible, 0, 0));
        entry.1 += report.total_ns;
        if report.key.tensor_core {
            entry.2 += report.total_ns;
        }
    }
    result
}

pub fn kernel_stats_json(stats: &crate::opstats::OpStats) -> String {
    let mut table = Table::new(&KERNEL_COLUMNS);
    for (rank, report) in stats.kernels.iter().enumerate() {
        let key = &report.key;
        let dims = |dims: &[u32; 3]| dims.map(|dim| dim.to_string()).join(",");
        let micros = |ns: u64| Cell::Number(ns as f64 / 1e3);
        table.rows.push(vec![
            Cell::Number((rank + 1) as f64),
            Cell::Text(key.name.clone()),
            Cell::Number(key.registers as f64),
            Cell::Number((key.static_shmem + key.dynamic_shmem) as f64),
            Cell::Text(dims(&key.block)),
            Cell::Text(dims(&key.grid)),
            Cell::Number(report.occupancy),
            Cell::Boolean(key.eligible),
            Cell::Boolean(key.tensor_core),
            Cell::Text(key.op_name.clone()),
            Cell::Number(report.occurrences as f64),
            micros(report.total_ns),
            micros(report.total_ns / report.occurrences),
            micros(report.min_ns),
            micros(report.max_ns),
        ]);
    }
    table.json()
}

#[cfg(test)]
#[path = "tests/inline/gpu.rs"]
pub mod tests;
