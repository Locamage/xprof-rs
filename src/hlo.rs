use crate::hlo_text::{Printer, Style};
use crate::opstats::{Db, Metrics};
use crate::xplane::{Field, Plane, Value, fields, nested, slice, stats};
use indexmap::IndexMap;
use itertools::Itertools;
use prost::Message;
use prost::encoding::encode_varint;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::SystemTime;

const LEGACY_QUEUES: &[u8] = b"\"wait_on_operation_queues\"";

#[allow(clippy::all, clippy::pedantic, dead_code)]
pub mod xla {
    include!(concat!(env!("OUT_DIR"), "/xla.rs"));
}

#[allow(clippy::all, clippy::pedantic, dead_code)]
pub mod profiler {
    include!(concat!(env!("OUT_DIR"), "/tensorflow.profiler.rs"));

    pub mod op_profile {
        include!(concat!(env!("OUT_DIR"), "/tensorflow.profiler.op_profile.rs"));
    }
}

pub use xla::{HloInstructionProto as Inst, ShapeProto as Shape};

const SUFFIX: &str = ".hlo_proto.pb";
const CACHED_MODULES: usize = 4;
const MAX_EXPRESSION: usize = 1_000_000;
pub const TUPLE: i32 = 13;
pub const OPAQUE: i32 = 14;
pub const TOKEN: i32 = 17;
const BUFFER: i32 = 34;
const TYPE_NAMES: &str = "primitive_type_invalid pred s8 s16 s32 s64 u8 u16 u32 u64 f16 f32 f64 tuple opaque c64 bf16 token c128 f8e5m2 f8e4m3fn s4 u4 f8e4m3b11fnuz f8e5m2fnuz f8e4m3fnuz s2 u2 f8e4m3 f8e3m4 s1 u1 f4e2m1fn f8e8m0fnu buffer f6e3m2fn f6e2m3fn";
const OPCODES: &str = "abs acos acosh add add-dependency after-all all-gather all-gather-done all-gather-start all-reduce all-reduce-done all-reduce-start all-to-all and asin asinh async-done async-start async-update atan2 atanh batch-norm-grad batch-norm-inference batch-norm-training bitcast bitcast-convert broadcast call cbrt ceil cholesky clamp count-leading-zeros collective-broadcast collective-permute collective-permute-done collective-permute-start collective-reduce compare complex concatenate conditional constant convert convolution copy copy-done copy-start cosine cosh custom-call divide domain dot dynamic-reshape dynamic-slice dynamic-update-slice erf exponential exponential-minus-one fft floor fusion gather get-dimension-size get-tuple-element imag infeed iota is-finite log log-plus-one logistic map maximum minimum mulhi multiply negate not opt-barrier or outfeed pad parameter partition-id popcnt power ragged-all-to-all ragged-dot real recv recv-done reduce reduce-precision reduce-scatter reduce-window remainder replica-id reshape reverse rng rng-bit-generator rng-get-and-update-state round-nearest-afz round-nearest-even rsqrt scaled-dot scan scatter select select-and-scatter send send-done set-dimension-size shift-left shift-right-arithmetic shift-right-logical sign sine sinh slice sort sqrt stochastic-convert subtract tan tanh topk transpose triangular-solve tuple while xor equal-to not-equal-to greater-than-or-equal-to greater-than less-than-or-equal-to less-than";
pub const ELEMENTWISE: &str = "abs acos acosh asin asinh atanh round-nearest-afz round-nearest-even ceil count-leading-zeros convert bitcast-convert copy cosine cosh erf exponential exponential-minus-one floor imag is-finite log log-plus-one not negate popcnt real reduce-precision rsqrt logistic sign sine sinh sqrt cbrt tan tanh add atan2 compare complex divide maximum minimum multiply mulhi power remainder subtract and or xor shift-left shift-right-arithmetic shift-right-logical stochastic-convert select clamp constant rng";

static TYPES: LazyLock<Vec<&str>> = LazyLock::new(|| TYPE_NAMES.split(' ').collect());
static KNOWN_OPCODES: LazyLock<std::collections::HashSet<&str>> = LazyLock::new(|| OPCODES.split(' ').collect());
type Cached = (PathBuf, SystemTime, Arc<Module<'static>>);

static CACHE: LazyLock<Mutex<Vec<Cached>>> = LazyLock::new(|| Mutex::new(Vec::new()));

pub fn type_name(kind: i32) -> &'static str {
    TYPES.get(kind as usize).copied().unwrap_or("primitive_type_invalid")
}

pub fn msg<T: Default + Clone>(value: &Option<T>) -> Cow<'_, T> {
    value.as_ref().map_or_else(|| Cow::Owned(T::default()), Cow::Borrowed)
}

pub fn last_bytes(buf: &[u8], number: u32) -> &[u8] {
    nested(buf, number).last().unwrap_or_default()
}

pub fn split(buf: &[u8], number: u32) -> (Vec<u8>, Vec<&[u8]>) {
    let (mut rest, mut matched) = (Vec::new(), Vec::new());
    let mut entries = fields(buf);
    let mut start = 0;
    while let Some((field, value)) = entries.next() {
        match value {
            Field::Bytes(_, body) if field == number => matched.push(body),
            _ => rest.extend_from_slice(&buf[start..entries.pos]),
        }
        start = entries.pos;
    }
    (rest, matched)
}

pub fn general(value: f64, precision: usize) -> String {
    let mut buffer = [0u8; 64];
    let length = unsafe { libc::snprintf(buffer.as_mut_ptr().cast(), buffer.len(), c"%.*g".as_ptr(), precision as libc::c_int, value) };
    crate::xplane::lossy(&buffer[..length as usize]).into_owned()
}

/// Writes `%.15g` when it gives back `value`, and `%.17g` if not. For a normal `value`, a shortest form of 15 or fewer digits is equal to `%.15g`.
pub fn round_trip(out: &mut String, value: f64) {
    use std::io::Write as _;
    if !value.is_normal() {
        let short = general(value, 15);
        return out.push_str(&if short.parse::<f64>().ok() == Some(value) { short } else { general(value, 17) });
    }
    let mut text = [0u8; 32];
    let free = {
        let mut cursor = &mut text[..];
        write!(cursor, "{:e}", value.abs()).unwrap();
        cursor.len()
    };
    let text = &text[..32 - free];
    let split = text.iter().position(|&byte| byte == b'e').unwrap();
    let digits: Vec<u8> = text[..split].iter().copied().filter(|&byte| byte != b'.').collect();
    if digits.len() > 15 {
        return out.push_str(&general(value, 17));
    }
    let exponent: i32 = std::str::from_utf8(&text[split + 1..]).unwrap().parse().unwrap();
    let digit = |index: usize| char::from(digits.get(index).copied().unwrap_or(b'0'));
    if value.is_sign_negative() {
        out.push('-');
    }
    if !(-4..15).contains(&exponent) {
        out.push(digit(0));
        if digits.len() > 1 {
            out.push('.');
            out.extend((1..digits.len()).map(digit));
        }
        out.push_str(if exponent < 0 { "e-" } else { "e+" });
        if exponent.abs() < 10 {
            out.push('0');
        }
        out.push_str(itoa::Buffer::new().format(exponent.abs()));
    } else if exponent >= 0 {
        let whole = exponent as usize + 1;
        out.extend((0..whole).map(digit));
        if digits.len() > whole {
            out.push('.');
            out.extend((whole..digits.len()).map(digit));
        }
    } else {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
        out.extend((0..digits.len()).map(digit));
    }
}

pub fn sanitize(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let mut out: Vec<u8> = name.bytes().collect();
    if !out[0].is_ascii_alphabetic() && out[0] != b'_' {
        out[0] = b'_';
    }
    for byte in out.iter_mut().skip(1) {
        if !byte.is_ascii_alphanumeric() && !matches!(*byte, b'_' | b'.' | b'-') {
            *byte = b'_';
        }
    }
    let mut out = String::from_utf8(out).unwrap();
    if TYPES.contains(&out.as_str()) && out != "tuple" && out != "buffer" && out != "primitive_type_invalid" {
        out.push('_');
    }
    if out.starts_with("__") && !out.starts_with("__xla_") {
        out.replace_range(0..1, "a");
    }
    out
}

impl Shape {
    pub fn is_array(&self) -> bool {
        !matches!(self.element_type, 0 | TUPLE | OPAQUE | TOKEN | BUFFER) && (self.element_type as usize) < TYPES.len()
    }

    pub fn is_tuple(&self) -> bool {
        self.element_type == TUPLE
    }

    pub fn elements(&self) -> i64 {
        self.dimensions.iter().product()
    }

    pub fn normalize(&mut self) {
        if self.is_array() {
            self.tuple_shapes.clear();
            self.is_dynamic_dimension.resize(self.dimensions.len(), false);
        } else {
            self.dimensions.clear();
            self.is_dynamic_dimension.clear();
            if self.element_type != BUFFER {
                self.layout = None;
            }
            if self.element_type != TUPLE && self.element_type != BUFFER {
                self.tuple_shapes.clear();
            }
        }
        self.tuple_shapes.iter_mut().for_each(Self::normalize);
    }

    pub fn print(&self, out: &mut String, layout: bool) {
        if self.is_tuple() || self.element_type == BUFFER {
            out.push_str(if self.is_tuple() { "(" } else { "b(" });
            for (index, element) in self.tuple_shapes.iter().take(if self.is_tuple() { usize::MAX } else { 1 }).enumerate() {
                match index {
                    0 => {}
                    index if index % 5 == 0 => write!(out, ", /*index={index}*/").unwrap(),
                    _ => out.push_str(", "),
                }
                element.print(out, layout);
            }
            out.push(')');
            return;
        }
        let dimensions: &[i64] = if self.is_array() { &self.dimensions } else { &[] };
        let texts = dimensions.iter().enumerate().map(|(index, &dimension)| match (self.is_dynamic_dimension.get(index) == Some(&true), dimension) {
            (true, i64::MIN) => "?".to_string(),
            (true, dimension) => format!("<={dimension}"),
            (false, dimension) => dimension.to_string(),
        });
        write!(out, "{}[{}]", type_name(self.element_type), texts.format(",")).unwrap();
        let Some(shape_layout) = self.layout.as_ref().filter(|_| layout && self.is_array()) else { return };
        let primitive = |kind: i32, symbol: char| match kind {
            0 => String::new(),
            2..=9 | 21 | 22 | 26 | 27 | 30 | 31 => format!("{symbol}({})", type_name(kind)),
            _ => format!("{symbol}(invalid)"),
        };
        let tile = |tile: &xla::TileProto| {
            let texts = tile.dimensions.iter().map(|&dimension| match dimension {
                0.. => dimension.to_string(),
                i64::MIN => "*".to_string(),
                _ => format!("Invalid value {dimension}"),
            });
            format!("({})", texts.format(","))
        };
        let splits: String = shape_layout.split_configs.iter().map(|split| format!("({}:{})", split.dimension, split.split_indices.iter().join(","))).collect();
        let sections = [
            if shape_layout.tiles.is_empty() { String::new() } else { format!("T{}", shape_layout.tiles.iter().map(tile).collect::<String>()) },
            if matches!(shape_layout.tail_padding_alignment_in_elements, 0 | 1) { String::new() } else { format!("L({})", shape_layout.tail_padding_alignment_in_elements) },
            primitive(shape_layout.index_primitive_type, '#'),
            primitive(shape_layout.pointer_primitive_type, '*'),
            if shape_layout.element_size_in_bits == 0 { String::new() } else { format!("E({})", shape_layout.element_size_in_bits) },
            if shape_layout.memory_space == 0 { String::new() } else { format!("S({})", shape_layout.memory_space) },
            if splits.is_empty() { String::new() } else { format!("SC{splits}") },
            shape_layout.physical_shape.as_ref().map_or_else(String::new, |physical| format!("P({})", physical.text(true))),
            if shape_layout.dynamic_shape_metadata_prefix_bytes > 0 { format!("M({})", shape_layout.dynamic_shape_metadata_prefix_bytes) } else { String::new() },
        ]
        .concat();
        let text = format!("{{{}{}{sections}}}", shape_layout.minor_to_major.iter().join(","), if sections.is_empty() { "" } else { ":" });
        if !self.dimensions.is_empty() || text != "{}" {
            out.push_str(&text);
        }
    }

    pub fn text(&self, layout: bool) -> String {
        let mut out = String::new();
        self.print(&mut out, layout);
        out
    }

    pub fn unpadded_bytes(&self) -> i64 {
        match self.element_type {
            TUPLE => 8 * self.tuple_shapes.len() as i64,
            BUFFER => self.tuple_shapes.first().map_or(0, Self::unpadded_bytes),
            OPAQUE => 8,
            _ if self.is_array() => match self.layout.as_ref().map_or(0, |layout| layout.element_size_in_bits) {
                0 => {
                    self.elements()
                        * match self.element_type {
                            3 | 7 | 10 | 16 => 2,
                            4 | 8 | 11 => 4,
                            5 | 9 | 12 | 15 => 8,
                            18 => 16,
                            _ => 1,
                        }
                }
                bits => (self.elements() * bits + 7).div_euclid(8),
            },
            _ => 0,
        }
    }
}

#[derive(Default)]
pub struct Node {
    pub raw: (usize, usize),
    pub name: String,
    pub opcode: String,
    pub shape: Shape,
    pub id: i64,
    pub parameter: i64,
    pub frame: i64,
    pub computation: usize,
    pub operands: Vec<usize>,
    pub users: Vec<usize>,
    pub called: Vec<usize>,
    pub predecessors: Vec<usize>,
}

pub struct Graph {
    pub name: String,
    pub id: i64,
    pub nodes: Vec<usize>,
    pub root: usize,
    pub root_id: i64,
    pub parameters: Vec<usize>,
    pub thread: String,
}

pub struct Module<'a> {
    pub data: Cow<'a, [u8]>,
    pub proto: xla::HloModuleProto,
    pub nodes: Vec<Node>,
    pub graphs: Vec<Graph>,
    pub entry: usize,
    pub valid: bool,
}

impl<'a> Module<'a> {
    pub fn post_order(&self, graph: usize) -> Vec<usize> {
        let module = self;
        let nodes = &module.graphs[graph].nodes;
        let first = nodes.first().copied().unwrap_or(0);
        let mut state = vec![0u8; nodes.len()];
        let mut order = Vec::with_capacity(nodes.len());
        let mut stack = Vec::new();
        for &start in nodes.iter().filter(|&&start| module.nodes[start].users.is_empty()) {
            stack.clear();
            if state[start - first] != 2 {
                stack.push(start);
            }
            while let Some(&current) = stack.last() {
                match state[current - first] {
                    0 => state[current - first] = 1,
                    visited => {
                        stack.pop();
                        if visited != 2 {
                            state[current - first] = 2;
                            order.push(current);
                        }
                        continue;
                    }
                }
                let entry = &module.nodes[current];
                stack.extend(entry.operands.iter().rev().chain(&entry.predecessors).filter(|&&next| state[next - first] != 2));
            }
        }
        order
    }

    pub fn parse(data: Cow<'a, [u8]>) -> Self {
        let bytes: &[u8] = &data;
        let (rest, bodies) = split(last_bytes(bytes, 1), 3);
        let proto = xla::HloModuleProto::decode(rest.as_slice()).unwrap_or_default();
        let parts: Vec<(xla::HloComputationProto, Vec<&[u8]>)> = bodies
            .par_iter()
            .map(|body| {
                let (rest, instructions) = split(body, 2);
                (xla::HloComputationProto::decode(rest.as_slice()).unwrap_or_default(), instructions)
            })
            .collect();
        let ids: HashMap<i64, usize> = parts.iter().enumerate().map(|(index, (computation, _))| (computation.id, index)).collect();
        let mut starts = vec![0];
        for (_, instructions) in &parts {
            starts.push(starts[starts.len() - 1] + instructions.len());
        }
        let spans: Vec<(usize, &[u8])> = parts.iter().enumerate().flat_map(|(index, (_, instructions))| instructions.iter().map(move |body| (index, *body))).collect();
        let mut decoded: Vec<(Node, bool, [Vec<i64>; 3])> = spans
            .par_iter()
            .map(|&(computation, body)| {
                let mut inst = xla::HloInstructionLite::decode(body).unwrap_or_default();
                let start = body.as_ptr() as usize - bytes.as_ptr() as usize;
                let sound =
                    !inst.name.is_empty() && KNOWN_OPCODES.contains(inst.opcode.as_str()) && inst.shape.as_ref().is_some_and(|shape| inst.opcode != "constant" || one_bit_leaf(shape).is_none());
                let node = Node {
                    raw: (start, start + body.len()),
                    name: sanitize(&inst.name),
                    opcode: std::mem::take(&mut inst.opcode),
                    shape: inst.shape.take().map_or_else(Shape::default, |mut shape| {
                        shape.normalize();
                        shape
                    }),
                    id: inst.id,
                    parameter: inst.parameter_number,
                    frame: inst.metadata.as_ref().map_or(0, |metadata| i64::from(metadata.stack_frame_id)),
                    computation,
                    ..Node::default()
                };
                (node, sound, [inst.operand_ids, inst.control_predecessor_ids, inst.called_computation_ids])
            })
            .collect();
        let linked: Vec<bool> = decoded
            .par_chunk_by_mut(|a, b| a.0.computation == b.0.computation)
            .map(|chunk| {
                let (index, first) = (chunk[0].0.computation, starts[chunk[0].0.computation]);
                let local: HashMap<i32, usize> = chunk.iter().enumerate().map(|(position, (node, _, _))| (node.id as i32, first + position)).collect();
                let mut valid = true;
                for position in 0..chunk.len() {
                    let [operand_ids, control_ids, called_ids] = &chunk[position].2;
                    let operands: Vec<usize> = operand_ids.iter().filter_map(|id| local.get(&(*id as i32)).copied()).collect();
                    let mut predecessors = Vec::new();
                    for id in control_ids {
                        match local.get(&(*id as i32)) {
                            Some(&target) if !predecessors.contains(&target) => predecessors.push(target),
                            Some(_) => {}
                            None => valid = false,
                        }
                    }
                    let called: Vec<usize> = called_ids.iter().filter_map(|id| ids.get(id).copied()).collect();
                    let earlier = |targets: &[usize]| targets.iter().all(|&target| target < first + position);
                    valid &= chunk[position].1 && earlier(&operands) && earlier(&predecessors);
                    valid &= operands.len() == operand_ids.len() && called.len() == called_ids.len() && called.iter().all(|&graph| graph < index);
                    for &target in &operands {
                        if chunk[target - first].0.users.last() != Some(&(first + position)) {
                            chunk[target - first].0.users.push(first + position);
                        }
                    }
                    let node = &mut chunk[position].0;
                    (node.operands, node.predecessors, node.called) = (operands, predecessors, called);
                }
                valid
            })
            .collect();
        let nodes: Vec<Node> = decoded.into_iter().map(|(node, _, _)| node).collect();
        let mut valid = fields(bytes).any(|(number, _)| number == 1)
            && proto.host_program_shape.as_ref().is_some_and(|program| program.parameters.len() == program.parameter_names.len())
            && linked.iter().all(|linked| *linked);
        let mut graphs = Vec::with_capacity(parts.len());
        for (index, (computation, _)) in parts.into_iter().enumerate() {
            let range = starts[index]..starts[index + 1];
            let root = range.clone().rev().find(|&node| nodes[node].id as i32 == computation.root_id as i32);
            let mut parameters: Vec<(i64, usize)> = range.clone().filter(|&node| nodes[node].opcode == "parameter").map(|node| (nodes[node].parameter, node)).collect();
            parameters.sort_unstable();
            valid &= root.is_some();
            graphs.push(Graph {
                name: sanitize(&computation.name),
                id: computation.id,
                nodes: range.clone().collect(),
                root: root.unwrap_or(range.start),
                root_id: computation.root_id,
                parameters: parameters.into_iter().map(|(_, node)| node).collect(),
                thread: computation.execution_thread,
            });
        }
        let entry = graphs.iter().rposition(|graph| graph.id == proto.entry_computation_id);
        Module { proto, nodes, graphs, valid: valid && entry.is_some(), entry: entry.unwrap_or(usize::MAX), data }
    }

    pub fn error(&self) -> String {
        if !fields(&self.data).any(|(number, _)| number == 1) {
            return "No HLO module found in the HLO proto".into();
        }
        let Some(program) = &self.proto.host_program_shape else { return "No program shape found in the proto".into() };
        if program.parameters.len() != program.parameter_names.len() {
            let message = format!("ProgramShapeProto has different numbers of parameters and parameter names: {} vs {}", program.parameters.len(), program.parameter_names.len());
            return format!("RET_CHECK failure (external/xla/xla/shape.cc:532) num_params == num_param_names {message}");
        }
        let failure = |place: &str, condition: &str, message: String| format!("RET_CHECK failure (external/xla/xla/hlo/ir/{place}) {condition} {message}");
        for (index, graph) in self.graphs.iter().enumerate() {
            let mut defined = std::collections::HashSet::new();
            for &node in &graph.nodes {
                let inst = self.inst(node);
                if inst.opcode.is_empty() {
                    return failure("hlo_instruction.cc:312", "!proto.opcode().empty()", String::new());
                }
                if !KNOWN_OPCODES.contains(inst.opcode.as_str()) {
                    return format!("Unknown opcode: {}", inst.opcode);
                }
                if inst.shape.is_none() {
                    return failure("hlo_instruction.cc:342", "proto.has_shape()", String::new());
                }
                if !inst.operand_ids.iter().all(|id| defined.contains(&(*id as i32))) {
                    let condition = "absl::c_all_of(proto.operand_ids(), [&](int64_t id) { return instruction_map.contains( CalculateLocalId(id)); })";
                    return failure("hlo_instruction.cc:385", condition, format!("{} instruction contains invalid operand id(s)", inst.name));
                }
                if !inst.called_computation_ids.iter().all(|id| self.graphs[..index].iter().any(|earlier| earlier.id == i64::from(*id as i32))) {
                    let condition = "absl::c_all_of(proto.called_computation_ids(), [&](int64_t id) { return computation_map.contains( CalculateLocalId(id)); })";
                    return failure("hlo_instruction.cc:391", condition, format!("{} instruction references invalid computation id(s)", inst.name));
                }
                if let Some(leaf) = inst.literal.as_ref().and_then(|literal| literal.shape.as_ref()).and_then(one_bit_leaf).filter(|_| inst.opcode == "constant") {
                    return format!("Is called on unsupported shape: {}", leaf.text(true));
                }
                if let Some(id) = inst.control_predecessor_ids.iter().find(|id| !defined.contains(&(**id as i32))) {
                    let message = format!("No instruction with id {id} (local id: {}) in computation {}", *id as i32, inst.name);
                    return failure("hlo_instruction.cc:1434", "ContainsKey(instruction_map, local_predecessor_id)", message);
                }
                if inst.name.is_empty() {
                    return failure("hlo_instruction.cc:1442", "!proto.name().empty()", String::new());
                }
                defined.insert(inst.id as i32);
            }
            let root = graph.root_id;
            if root == -1 {
                return failure("hlo_computation.cc:1371", "proto.root_id() != -1", String::new());
            }
            if !defined.contains(&(root as i32)) {
                return failure("hlo_computation.cc:1373", "ContainsKey(instruction_map, root_local_id)", "Root instruction not found in instruction map".into());
            }
        }
        if self.entry == usize::MAX {
            return failure("hlo_module.cc:1035", "entry != nullptr", String::new());
        }
        "Invalid HLO module".into()
    }

    pub fn inst(&self, node: usize) -> Inst {
        let (start, end) = self.nodes[node].raw;
        Inst::decode(&self.data[start..end]).unwrap_or_default()
    }

    pub fn backend_config<'b>(&'b self, inst: &'b Inst) -> Option<Cow<'b, [u8]>> {
        let raw: &[u8] = match inst.backend_config_payload.as_ref().map(|payload| &payload.payload_source) {
            None => &inst.backend_config,
            Some(Some(xla::payload::PayloadSource::Id(id))) => usize::try_from(*id).ok().and_then(|id| self.proto.payloads.get(id))?,
            Some(Some(xla::payload::PayloadSource::Value(value))) => value,
            Some(None) => &[],
        };
        let mut out = Cow::Borrowed(raw);
        let mut from = 0;
        while let Some(at) = out[from..].windows(LEGACY_QUEUES.len()).position(|window| window == LEGACY_QUEUES).map(|at| at + from) {
            let mut end = at + LEGACY_QUEUES.len();
            let matched = b":[],".iter().all(|&expected| {
                end += out[end..].iter().take_while(|byte| b" \t\n\r\x0c".contains(byte)).count();
                let found = out.get(end) == Some(&expected);
                end += 1;
                found
            });
            if matched {
                out.to_mut().drain(at..end);
                from = at;
            } else {
                from = at + 1;
            }
        }
        Some(out)
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        let name = name.strip_prefix('%').unwrap_or(name);
        self.nodes.iter().position(|node| node.name.eq_ignore_ascii_case(name))
    }

    pub fn find_graph(&self, name: &str) -> Option<usize> {
        self.graphs.iter().position(|graph| graph.name.eq_ignore_ascii_case(name))
    }

    pub fn category(&self, node: usize) -> String {
        match self.nodes[node].opcode.as_str() {
            "fusion" => fusion_category(&self.inst(node).fusion_kind).to_string(),
            "convolution" => {
                let inst = self.inst(node);
                let mut category = "convolution".to_string();
                if inst.window.iter().flat_map(|window| &window.dimensions).any(|dimension| dimension.base_dilation != 1) {
                    category.push_str(" base-dilated");
                }
                if inst.window.iter().flat_map(|window| &window.dimensions).any(|dimension| dimension.window_dilation != 1) {
                    category.push_str(" window-dilated");
                }
                category
            }
            "transpose" | "copy" | "reshape" | "dynamic-reshape" => "data formatting".to_string(),
            _ if self.is_elementwise(node) => "non-fusion elementwise".to_string(),
            opcode => opcode.to_string(),
        }
    }

    fn is_elementwise(&self, node: usize) -> bool {
        let entry = &self.nodes[node];
        match entry.opcode.as_str() {
            "bitcast-convert" => {
                let width = |kind: i32| match type_name(kind) {
                    "pred" | "s8" | "u8" | "s1" | "s2" | "s4" | "u1" | "u2" | "u4" => 8,
                    name => name.trim_start_matches(|c: char| c.is_ascii_alphabetic()).split(|c: char| !c.is_ascii_digit()).next().and_then(|bits| bits.parse().ok()).unwrap_or(8),
                };
                entry.operands.first().is_some_and(|&operand| width(entry.shape.element_type) == width(self.nodes[operand].shape.element_type))
            }
            "fusion" => entry.called.first().is_none_or(|&graph| self.graphs[graph].nodes.iter().all(|&child| self.nodes[child].opcode == "parameter" || self.is_elementwise(child))),
            "map" => {
                let dimensions = &self.inst(node).dimensions;
                dimensions.is_empty() || (dimensions.len() == entry.shape.dimensions.len() && dimensions.iter().enumerate().all(|(position, &dimension)| dimension == position as i64))
            }
            opcode => ELEMENTWISE.split(' ').any(|name| name == opcode),
        }
    }
}

fn one_bit_leaf(shape: &Shape) -> Option<&Shape> {
    match shape.element_type {
        TUPLE => shape.tuple_shapes.iter().find_map(one_bit_leaf),
        30 | 31 => Some(shape),
        _ => None,
    }
}

pub fn fusion_category(kind: &str) -> &'static str {
    match kind {
        "kLoop" => "loop fusion",
        "kInput" => "input fusion",
        "kOutput" => "output fusion",
        _ => "custom fusion",
    }
}

pub fn modules(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    entries.filter_map(|entry| entry.ok()?.file_name().to_str()?.strip_suffix(SUFFIX).map(str::to_string)).collect()
}

pub fn load(dir: &Path, module: &str) -> Option<Arc<Module<'static>>> {
    let path = dir.join(format!("{module}{SUFFIX}"));
    let modified = std::fs::metadata(&path).ok()?.modified().ok()?;
    if let Some((_, _, cached)) = CACHE.lock().unwrap().iter().find(|(cached, time, _)| *cached == path && *time == modified) {
        return Some(cached.clone());
    }
    let data = std::fs::read(&path).ok()?;
    let complete = |buf: &[u8]| {
        let mut walk = fields(buf);
        walk.by_ref().for_each(drop);
        walk.complete()
    };
    if !complete(&data) || !complete(last_bytes(&data, 1)) {
        return None;
    }
    let parsed = Arc::new(Module::parse(Cow::Owned(data)));
    let mut cache = CACHE.lock().unwrap();
    cache.retain(|(cached, _, _)| *cached != path);
    cache.insert(0, (path, modified, parsed.clone()));
    cache.truncate(CACHED_MODULES);
    Some(parsed)
}

pub fn by_options(dir: &Path, params: &HashMap<String, String>) -> Option<Arc<Module<'static>>> {
    match (params.get("module_name").filter(|name| !name.is_empty()), params.get("program_id").filter(|id| !id.is_empty())) {
        (Some(module), _) => load(dir, module),
        (None, Some(program)) => load(dir, modules(dir).iter().find(|module| module.contains(program.as_str()))?),
        (None, None) => None,
    }
}

/// Lists the modules of a session. When there are none, it extracts them first. Two requests do not extract at the same time.
pub fn extracted(dir: &Path, xspaces: &[PathBuf]) -> Option<Vec<String>> {
    static EXTRACTING: Mutex<()> = Mutex::new(());
    let _held = EXTRACTING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let existing = modules(dir);
    if !existing.is_empty() {
        return Some(existing);
    }
    extract(dir, xspaces)?;
    Some(modules(dir))
}

pub fn extract(dir: &Path, xspaces: &[PathBuf]) -> Option<()> {
    let mut modules: IndexMap<String, Vec<u8>> = IndexMap::new();
    for path in xspaces {
        let data = std::fs::read(path).ok()?;
        let mut metadata = Vec::new();
        for (tag, plane) in fields(&data) {
            if let Field::Bytes(_, plane) = plane
                && tag == 1
                && last_bytes(plane, 2) == b"/host:metadata"
            {
                metadata.push(10);
                encode_varint(plane.len() as u64, &mut metadata);
                metadata.extend_from_slice(plane);
            }
        }
        for (program, bytes) in protos(&crate::xplane::parse(&metadata).ok()?, &metadata) {
            let name = format!("{}({program})", module_name(bytes)).replace('/', "_");
            modules.entry(name).or_insert_with(|| bytes.to_vec());
        }
    }
    if modules.is_empty() {
        modules.insert("NO_MODULE".into(), Vec::new());
    }
    for (name, bytes) in modules {
        let (path, partial) = (dir.join(format!("{name}{SUFFIX}")), dir.join(format!(".{name}{SUFFIX}.partial")));
        std::fs::write(&partial, bytes).and_then(|()| std::fs::rename(&partial, path)).ok()?;
    }
    Some(())
}

pub fn protos<'a>(planes: &[Plane], map: &'a [u8]) -> Vec<(u64, &'a [u8])> {
    let mut found = IndexMap::new();
    for plane in planes.iter().filter(|plane| plane.name == "/host:metadata") {
        let Some(proto) = plane.id("HLO Proto").or_else(|| plane.id("Hlo Proto")) else { continue };
        let program = plane.id("program_id");
        for (index, meta) in plane.meta.iter().enumerate() {
            let raw = slice(map, meta.raw);
            let Some(Value::Bytes(body)) = stats(raw, 5, |stat| stat == proto).next().map(|stat| stat.value) else { continue };
            let named =
                meta.name.strip_suffix(')').and_then(|name| name.rsplit_once('(')).map(|(_, digits)| digits).filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()));
            let fixed = named.and_then(|digits| digits.parse::<u64>().ok()).unwrap_or(index as u64);
            let id = program.and_then(|program| stats(raw, 5, |stat| stat == program).next()).map_or(fixed, |stat| stat.value.int().unwrap_or(0) as u64);
            found.entry(id).or_insert(body);
        }
    }
    found.into_iter().collect()
}

pub fn module_name(proto: &[u8]) -> String {
    crate::xplane::lossy(last_bytes(last_bytes(proto, 1), 1)).into_owned()
}

fn fused_children(printer: &Printer, node: usize) -> Vec<Metrics> {
    let module = printer.module;
    let Some(&graph) = module.nodes[node].called.first().filter(|_| module.nodes[node].opcode == "fusion") else { return Vec::new() };
    module.graphs[graph]
        .nodes
        .iter()
        .filter(|&&child| !matches!(module.nodes[child].opcode.as_str(), "parameter" | "tuple"))
        .map(|&child| {
            let mut long_name = String::new();
            let metadata = printer.instruction(child, &mut long_name).metadata.unwrap_or_default();
            if long_name.len() > MAX_EXPRESSION {
                long_name.truncate(long_name.floor_char_boundary(MAX_EXPRESSION));
            }
            let mut metrics = Metrics {
                name: module.nodes[child].name.as_str().into(),
                category: module.category(child).into(),
                deduplicated_name: metadata.deduplicated_name.as_str().into(),
                provenance: format!("{}:{}", metadata.op_name, metadata.op_type).into(),
                num_cores: 1,
                occurrences: 1,
                long_name: long_name.into(),
                ..Default::default()
            };
            metrics.children.metrics = fused_children(printer, child);
            metrics
        })
        .collect()
}

pub fn parse_modules(protos: Vec<(u64, &[u8])>) -> Vec<(u64, Module<'_>)> {
    protos.into_par_iter().map(|(id, proto)| (id, Module::parse(Cow::Borrowed(proto)))).collect()
}

pub fn attach_fused(modules: &[(u64, Module)], db: &mut Db) {
    let mut wanted = FxHashMap::<u64, FxHashSet<&str>>::default();
    for metrics in &db.metrics {
        wanted.entry(metrics.module).or_default().insert(metrics.name.as_str());
    }
    let printers: HashMap<u64, (Printer, FxHashMap<&str, usize>)> = modules
        .iter()
        .filter(|(_, module)| module.valid)
        .map(|(id, module)| {
            let found: Vec<(&str, usize)> = match wanted.get(id) {
                Some(wanted) => module.nodes.par_iter().enumerate().filter(|(_, node)| wanted.contains(node.name.as_str())).map(|(index, node)| (node.name.as_str(), index)).collect(),
                None => Vec::new(),
            };
            let mut names = FxHashMap::default();
            for (name, index) in found {
                names.entry(name).or_insert(index);
            }
            (*id, (Printer::new(module, Style::Expression, false), names))
        })
        .collect();
    db.metrics.par_iter_mut().for_each(|metrics| {
        let Some((printer, names)) = printers.get(&metrics.module) else { return };
        let Some(&index) = names.get(metrics.name.as_str()) else { return };
        let children = fused_children(printer, index);
        metrics.children.metrics.extend(children);
    });
}
