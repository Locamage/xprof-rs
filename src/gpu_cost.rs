use crate::hlo::xla::hlo_instruction_proto::ReplicaGroupList;
use crate::hlo::{BUFFER, ELEMENTWISE, Inst, Module, OPAQUE, Shape, TOKEN, TUPLE, all_subshapes};
use rustc_hash::FxHashMap;
use std::cell::OnceCell;
use std::collections::HashMap;

const FMA_FLOPS: i64 = 2;
const DEFAULT_FLOPS_PER_ELEMENT: i64 = 3;
const IMMEDIATE_CONSTANT_MAX_ELEMENTS: i64 = 8;
const HBM: u64 = 1;
const PROFILE: [(&str, &[(i32, i64)]); 16] = [
    ("divide", &[(2, 370), (3, 367), (4, 306), (5, 918), (6, 306), (7, 302), (8, 115), (9, 838), (12, 2016), (15, 687), (18, 7941)]),
    ("power", &[(2, 392), (3, 396), (5, 601), (6, 388), (7, 399), (9, 604), (10, 975), (11, 925), (12, 5511), (15, 3351), (18, 18205)]),
    ("cbrt", &[(10, 925), (11, 867), (12, 6339)]),
    ("cosine", &[(10, 691), (11, 662), (12, 1717), (18, 6613)]),
    ("exponential", &[(10, 108), (11, 86), (12, 1652), (15, 1360), (18, 4028)]),
    ("exponential-minus-one", &[(10, 396), (11, 381), (12, 1900), (15, 1400), (18, 4161)]),
    ("log", &[(10, 266), (11, 244), (12, 608), (15, 950), (18, 7599)]),
    ("log-plus-one", &[(10, 284), (11, 262), (12, 2073), (15, 842), (18, 6962)]),
    ("logistic", &[(10, 226), (11, 176), (12, 2412)]),
    ("rsqrt", &[(10, 97), (11, 75), (12, 698), (15, 2383), (18, 11318)]),
    ("sqrt", &[(10, 97), (11, 75), (12, 986), (15, 3193), (18, 15606)]),
    ("tanh", &[(10, 212), (11, 190), (12, 1609), (18, 9939)]),
    ("sine", &[(11, 662), (12, 1789), (18, 5878)]),
    ("add", &[(12, 97), (18, 97)]),
    ("multiply", &[(12, 97), (18, 270)]),
    ("subtract", &[(12, 97), (18, 97)]),
];
pub const CUBLAS_LT: [&str; 4] = ["__cublas$lt$matmul", "__cublas$lt$matmul$f8", "__cublas$lt$matmul$mx", "__cublas$lt$groupedMatmul"];
pub const DNN_CONVOLUTION: [&str; 5] = ["__cudnn$convForward", "__cudnn$convForwardGraph", "__cudnn$convBackwardInput", "__cudnn$convBackwardFilter", "__cudnn$convBiasActivationForward"];
const CALL_MARKERS: [&str; 2] = ["__xla_internal_call_marker_before", "__xla_internal_call_marker_after"];
const QUOTES: [&str; 8] = ["\"", "22", "x22", "X22", "u0022", "U0022", "u00000022", ""];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cost {
    pub model_flops: i64,
    pub device_flops: i64,
    pub bytes_accessed: i64,
    pub memory: Vec<(bool, u64, i64)>,
}

type Status = Result<(), String>;

#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Flops,
    Transcendentals,
    Bytes,
    Optimal,
    Utilization,
    IrSize,
    ScaleRatio,
    NumDevices,
    Transferred,
    Adjustment,
    Operand(usize, Vec<i64>),
    OperandUtilization(usize),
    Output(Vec<i64>),
}

#[derive(Clone, Default)]
struct Props(FxHashMap<Key, f32>);

impl Props {
    fn get(&self, key: &Key) -> f32 {
        self.0.get(key).copied().unwrap_or(0.0)
    }

    fn slot(&mut self, key: Key) -> &mut f32 {
        self.0.entry(key).or_insert(0.0)
    }

    fn set(&mut self, key: Key, value: f32) {
        *self.slot(key) = value;
    }

    fn nonzero(&self) -> impl Iterator<Item = (&Key, f32)> {
        self.0.iter().filter(|(_, value)| **value != 0.0).map(|(key, value)| (key, *value))
    }

    fn merge(&mut self, other: &Self, combine: fn(f32, f32) -> f32) {
        for (key, value) in other.nonzero() {
            let slot = self.slot(key.clone());
            *slot = combine(*slot, value);
        }
    }
}

fn operand_bytes_key(operand: usize, index: &[i64]) -> Key {
    Key::Operand(operand, index.to_vec())
}

fn output_key(index: &[i64]) -> Key {
    Key::Output(index.to_vec())
}

pub(crate) fn bit_width(kind: i32) -> i64 {
    match kind {
        1 | 30 | 31 => 1,
        26 | 27 => 2,
        21 | 22 | 32 => 4,
        35 | 36 => 6,
        2 | 6 | 19 | 20 | 23 | 24 | 25 | 28 | 29 | 33 => 8,
        3 | 7 | 10 | 16 => 16,
        4 | 8 | 11 => 32,
        5 | 9 | 12 | 15 => 64,
        18 => 128,
        _ => 0,
    }
}

fn storage_bit_width(kind: i32) -> i64 {
    if kind == 1 { 8 } else { bit_width(kind) }
}

fn has_layout(shape: &Shape) -> bool {
    if shape.is_tuple() { shape.tuple_shapes.iter().all(has_layout) } else { !shape.is_array() || shape.layout.is_some() }
}

fn byte_size(shape: &Shape) -> i64 {
    if shape.element_type == BUFFER { 0 } else { shape.unpadded_bytes() }
}

fn shape_size(shape: &Shape) -> i64 {
    if has_layout(shape) { byte_size(shape) } else { 0 }
}

fn leaves(shape: &Shape) -> Vec<(Vec<i64>, &Shape)> {
    all_subshapes(shape).into_iter().filter(|(_, subshape)| !subshape.is_tuple()).collect()
}

fn ring_shape_size(shape: &Shape, skip: Option<i64>) -> i64 {
    leaves(shape).into_iter().filter(|(index, _)| skip.is_none_or(|skip| index.first() != Some(&skip))).filter(|(_, subshape)| subshape.is_array()).map(|(_, subshape)| byte_size(subshape)).sum()
}

fn memory_space(shape: &Shape) -> Option<i64> {
    shape.layout.as_ref().map(|layout| layout.memory_space)
}

fn valid(cost: i64) -> i64 {
    if cost == -1 { 0 } else { cost }
}

fn log2_floor(value: u64) -> i64 {
    if value == 0 { -1 } else { 63 - value.leading_zeros() as i64 }
}

fn log2_ceiling(value: u64) -> i64 {
    let floor = log2_floor(value);
    if value == 0 || value.is_power_of_two() { floor } else { floor + 1 }
}

fn adjusted(flops: f32, width: i64) -> f32 {
    let divisor = match width {
        8 => 2.0,
        4 => 4.0,
        _ => 1.0,
    };
    flops - flops / divisor
}

fn profile_flops(opcode: &str, kind: i32) -> i64 {
    PROFILE.iter().find(|entry| entry.0 == opcode).and_then(|entry| entry.1.iter().find(|entry| entry.0 == kind)).map_or(DEFAULT_FLOPS_PER_ELEMENT, |entry| entry.1)
}

fn quote_end(text: &str, at: usize) -> Vec<usize> {
    let mut ends = Vec::new();
    for slash in [true, false] {
        let start = if slash {
            if !text[at..].starts_with('\\') {
                continue;
            }
            at + 1
        } else {
            at
        };
        for quote in QUOTES {
            if text[start..].starts_with(quote) {
                ends.push(start + quote.len());
            }
        }
    }
    ends
}

fn skip_spaces(text: &str, at: usize) -> usize {
    at + text[at..].len() - text[at..].trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']).len()
}

fn cost_estimate(text: &str, key: &str) -> Option<i64> {
    for (position, _) in text.match_indices(key) {
        for first in quote_end(text, position + key.len()) {
            let separator = skip_spaces(text, first);
            if !text[separator..].starts_with(['=', ':']) {
                continue;
            }
            let value_start = skip_spaces(text, separator + 1);
            for second in quote_end(text, value_start) {
                let digits = text[second..].strip_prefix('-').map_or(&text[second..], |rest| rest);
                let length = digits.len() - digits.trim_start_matches(|c: char| c.is_ascii_digit()).len();
                if length == 0 {
                    continue;
                }
                let end = second + (text.len() - second - digits.len()) + length;
                return text[second..end].parse().ok();
            }
        }
    }
    None
}

#[allow(deprecated)]
fn replica_groups(inst: &Inst) -> &[crate::hlo::xla::ReplicaGroup] {
    &inst.replica_groups
}

fn json_dimensions(value: &serde_json::Value) -> Vec<i64> {
    value.as_array().map_or_else(Vec::new, |items| items.iter().filter_map(|item| item.as_i64().or_else(|| item.as_str().and_then(|text| text.parse().ok()))).collect())
}

struct Context<'m> {
    module: &'m Module<'m>,
    insts: Vec<OnceCell<Inst>>,
    successors: Vec<bool>,
}

impl<'m> Context<'m> {
    fn inst(&self, node: usize) -> &Inst {
        self.insts[node].get_or_init(|| self.module.inst(node))
    }

    fn shape(&self, node: usize) -> &'m Shape {
        &self.module.nodes[node].shape
    }

    fn opcode(&self, node: usize) -> &'m str {
        &self.module.nodes[node].opcode
    }

    fn operand(&self, node: usize, index: usize) -> usize {
        self.module.nodes[node].operands[index]
    }

    fn operands(&self, node: usize) -> &'m [usize] {
        &self.module.nodes[node].operands
    }

    fn users(&self, node: usize) -> &'m [usize] {
        &self.module.nodes[node].users
    }

    fn called(&self, node: usize, index: usize) -> Result<usize, String> {
        self.module.nodes[node].called.get(index).copied().ok_or_else(|| format!("missing called computation for {}", self.module.nodes[node].name))
    }

    fn root(&self, graph: usize) -> usize {
        self.module.graphs[graph].root
    }

    fn parameter(&self, graph: usize, index: usize) -> Result<usize, String> {
        self.module.graphs[graph].parameters.get(index).copied().ok_or_else(|| "missing fused parameter".to_string())
    }

    fn is_elementwise(&self, node: usize) -> bool {
        let opcode = self.opcode(node);
        match opcode {
            "bitcast-convert" => self.operands(node).first().is_some_and(|&operand| storage_bit_width(self.shape(node).element_type) == storage_bit_width(self.shape(operand).element_type)),
            "map" => {
                let dimensions = &self.inst(node).dimensions;
                dimensions.is_empty() || (dimensions.len() == self.shape(node).dimensions.len() && dimensions.iter().enumerate().all(|(position, &dimension)| dimension == position as i64))
            }
            "fusion" => {
                self.module.nodes[node].called.first().is_some_and(|&graph| self.module.graphs[graph].nodes.iter().all(|&child| self.opcode(child) == "parameter" || self.is_elementwise(child)))
            }
            _ => ELEMENTWISE.split(' ').any(|name| name == opcode),
        }
    }

    fn equal_ignoring_element_type(left: &Shape, right: &Shape) -> bool {
        let layout =
            |shape: &Shape| shape.layout.as_ref().map(|layout| (layout.minor_to_major.clone(), layout.tiles.iter().map(|tile| tile.dimensions.clone()).collect::<Vec<_>>(), layout.memory_space));
        left.dimensions == right.dimensions && left.is_dynamic_dimension == right.is_dynamic_dimension && left.is_array() == right.is_array() && layout(left) == layout(right)
    }

    fn transpose_is_bitcast(&self, node: usize) -> bool {
        let (input, output) = (self.shape(self.operand(node, 0)), self.shape(node));
        let (Some(input_layout), Some(output_layout)) = (input.layout.as_ref(), output.layout.as_ref()) else { return false };
        if input.element_type != output.element_type {
            return false;
        }
        let mapping = &self.inst(node).dimensions;
        let composed: Option<Vec<i64>> = output_layout.minor_to_major.iter().map(|&position| mapping.get(position as usize).copied()).collect();
        composed.is_some_and(|composed| composed == input_layout.minor_to_major)
    }
}

struct Analysis<'c, 'm> {
    context: &'c Context<'m>,
    current: Props,
    bottleneck: bool,
    properties: FxHashMap<usize, Props>,
    sum: Props,
    state: FxHashMap<usize, u8>,
    use_roots: FxHashMap<usize, Vec<usize>>,
    root_utilizations: FxHashMap<usize, f32>,
}

impl<'c, 'm> Analysis<'c, 'm> {
    fn new(context: &'c Context<'m>) -> Self {
        Analysis {
            context,
            current: Props::default(),
            bottleneck: true,
            properties: FxHashMap::default(),
            sum: Props::default(),
            state: FxHashMap::default(),
            use_roots: FxHashMap::default(),
            root_utilizations: FxHashMap::default(),
        }
    }

    fn property(&self, node: usize, key: &Key) -> f32 {
        self.properties.get(&node).map_or(0.0, |props| props.get(key))
    }

    fn slot(&mut self, node: usize, key: Key) -> &mut f32 {
        self.properties.entry(node).or_default().slot(key)
    }

    fn set_output(&mut self, index: &[i64], value: f32) {
        self.current.set(output_key(index), value);
    }

    fn replace_output(&mut self, size: f32) {
        *self.current.slot(Key::Bytes) -= self.current.get(&output_key(&[]));
        *self.current.slot(Key::Bytes) += size;
        self.set_output(&[], size);
    }

    fn set_operand_bytes(&mut self, operand: usize, index: &[i64], value: f32) {
        self.current.set(operand_bytes_key(operand, index), value);
    }

    fn set_operand_utilization(&mut self, operand: usize, value: f32) {
        self.current.set(Key::OperandUtilization(operand), value);
    }

    fn accept(&mut self, graph: usize) -> Status {
        let context = self.context;
        let module = context.module;
        let root = module.graphs[graph].root;
        let unreachable: Vec<usize> = module.graphs[graph].nodes.iter().copied().filter(|&node| context.users(node).is_empty() && node != root && !context.successors[node]).collect();
        for node in unreachable {
            self.dfs(node)?;
        }
        self.dfs(root)
    }

    fn dfs(&mut self, root: usize) -> Status {
        let context = self.context;
        let mut stack = vec![root];
        while let Some(&current) = stack.last() {
            match self.state.get(&current).copied().unwrap_or(0) {
                2 => {
                    stack.pop();
                    continue;
                }
                1 => {
                    stack.pop();
                    self.preprocess(current);
                    self.visit(current)?;
                    self.state.insert(current, 2);
                    self.postprocess(current)?;
                    continue;
                }
                _ => {}
            }
            self.state.insert(current, 1);
            let start = stack.len();
            for &child in context.operands(current).iter().chain(&context.module.nodes[current].predecessors) {
                match self.state.get(&child).copied().unwrap_or(0) {
                    1 => return Err(format!("cycle at {}", context.module.nodes[current].name)),
                    2 => {}
                    _ => stack.push(child),
                }
            }
            stack[start..].reverse();
        }
        Ok(())
    }

    fn preprocess(&mut self, node: usize) {
        let context = self.context;
        self.current = Props::default();
        self.bottleneck = true;
        let mut bytes = 0f32;
        for (_, leaf) in leaves(context.shape(node)) {
            bytes += shape_size(leaf) as f32;
        }
        self.set_output(&[], bytes);
        for (index, &operand) in context.operands(node).iter().enumerate() {
            let size = shape_size(context.shape(operand));
            bytes += size as f32;
            self.set_operand_bytes(index, &[], size as f32);
            self.set_operand_utilization(index, 1.0);
        }
        self.current.set(Key::Bytes, bytes);
        self.current.set(Key::IrSize, 1.0);
    }

    fn postprocess(&mut self, node: usize) -> Status {
        let context = self.context;
        if context.opcode(node) != "custom-call" {
            let width = context.operands(node).iter().map(|&operand| context.shape(operand).element_type).filter(|kind| !matches!(*kind, 0 | TUPLE | OPAQUE | TOKEN)).map(bit_width).max();
            self.current.set(Key::Adjustment, adjusted(self.current.get(&Key::Flops), width.unwrap_or(0)));
        }
        if self.bottleneck {
            self.current.set(Key::Optimal, 0.0);
        }
        let current = std::mem::take(&mut self.current);
        self.sum.merge(&current, |sum, value| sum + value);
        if self.properties.contains_key(&node) {
            return Err(format!("{} already exists in hlo_properties_", context.module.nodes[node].name));
        }
        self.properties.insert(node, current);
        Ok(())
    }

    fn subcomputation(&mut self, graph: usize) -> Result<Props, String> {
        let mut nested = Analysis::new(self.context);
        nested.accept(graph)?;
        self.properties.extend(nested.properties);
        Ok(nested.sum)
    }

    fn copy_scaled(&mut self, sub: &Props, factor: i64, accumulate: bool) {
        for (key, value) in sub.nonzero().filter(|(key, _)| matches!(key, Key::Flops | Key::Transcendentals | Key::Optimal | Key::ScaleRatio | Key::NumDevices | Key::Transferred | Key::Adjustment)) {
            let slot = self.current.slot(key.clone());
            *slot = if accumulate { *slot + value * factor as f32 } else { value * factor as f32 };
        }
    }

    fn transfers(&mut self, output: i64, operands: &[(usize, i64)]) {
        self.current.set(Key::Bytes, (output + operands.iter().map(|operand| operand.1).sum::<i64>()) as f32);
        self.set_output(&[], output as f32);
        for &(operand, bytes) in operands {
            self.set_operand_bytes(operand, &[], bytes as f32);
        }
    }

    fn zero_outputs(&mut self, node: usize, bottleneck: bool, operands: bool) {
        if !bottleneck {
            self.bottleneck = false;
        }
        self.current.set(Key::Bytes, 0.0);
        self.set_output(&[], 0.0);
        if operands {
            for index in 0..self.context.operands(node).len() {
                self.set_operand_bytes(index, &[], 0.0);
            }
        }
        self.current.set(Key::Optimal, 0.0);
    }

    fn visit(&mut self, node: usize) -> Status {
        let context = self.context;
        let opcode = context.opcode(node);
        let shape = context.shape(node);
        match opcode {
            _ if ELEMENTWISE.split(' ').any(|name| name == opcode) && !matches!(opcode, "copy" | "constant" | "rng") => self.elementwise(node),
            "parameter" | "constant" => self.zero_outputs(node, false, false),
            "get-tuple-element" => {
                self.zero_outputs(node, false, false);
                self.set_operand_bytes(0, &[], 0.0);
            }
            "domain" | "after-all" | "add-dependency" => self.zero_outputs(node, false, true),
            "bitcast" => self.bitcast(),
            "transpose" => {
                if context.transpose_is_bitcast(node) {
                    self.bitcast();
                }
            }
            "slice" | "dynamic-slice" | "dynamic-update-slice" => self.slicing(node, opcode),
            "tuple" => self.transfers(shape_size(shape), &(0..context.operands(node).len()).map(|index| (index, 0)).collect::<Vec<_>>()),
            "concatenate" => {
                let dimension = context.inst(node).dimensions.first().copied().unwrap_or(0);
                let size = context.shape(context.operand(node, 0)).dimensions.get(dimension as usize).copied().unwrap_or(0);
                let per_element = if dimension > 0 && size & 31 != 0 { 400 } else { 6 };
                self.current.set(Key::Flops, (per_element * shape.elements_recursive()) as f32);
            }
            "dot" | "scaled-dot" => {
                let numbers = context.inst(node).dot_dimension_numbers.clone().unwrap_or_default();
                self.current.set(Key::Flops, dot_flops(context.shape(context.operand(node, 0)), shape, &numbers.lhs_contracting_dimensions) as f32);
            }
            "ragged-dot" => {
                let numbers = context.inst(node).ragged_dot_dimension_numbers.clone().unwrap_or_default();
                let mut result = shape.clone();
                for index in 0..numbers.rhs_group_dimensions.len() {
                    if index < result.dimensions.len() {
                        result.dimensions.remove(index);
                    }
                }
                let lhs = numbers.dot_dimension_numbers.unwrap_or_default().lhs_contracting_dimensions;
                self.current.set(Key::Flops, dot_flops(context.shape(context.operand(node, 0)), &result, &lhs) as f32);
            }
            "infeed" | "outfeed" => self.feed(node, opcode),
            "map" => {
                let sub = self.subcomputation(context.called(node, 0)?)?;
                self.copy_scaled(&sub, shape.elements(), false);
            }
            "reduce" => self.reduce(node)?,
            "scan" => {
                let sub = self.subcomputation(context.called(node, 0)?)?;
                self.copy_scaled(&sub, context.shape(context.operand(node, 1)).elements(), false);
            }
            "reduce-window" => self.reduce_window(node)?,
            "select-and-scatter" => {
                let select = self.subcomputation(context.called(node, 0)?)?;
                let scatter = self.subcomputation(context.called(node, 1)?)?;
                let source = context.shape(context.operand(node, 1)).elements();
                let window: i64 = context.inst(node).window.as_ref().map_or(1, |window| window.dimensions.iter().map(|dimension| dimension.size).product());
                self.copy_scaled(&select, source * (window - 1), true);
                self.copy_scaled(&scatter, source, true);
            }
            "convolution" => {
                let flops = convolution_flops(context.inst(node), context.shape(context.operand(node, 0)), context.shape(context.operand(node, 1)), shape);
                self.current.set(Key::Flops, flops as f32);
            }
            "fft" => {
                let operand = context.shape(context.operand(node, 0));
                let real = if operand.is_tuple() { operand.tuple_shapes.first().cloned().unwrap_or_default() } else { operand.clone() };
                let factors: i64 = context.inst(node).fft_length.iter().map(|&length| log2_floor(length as u64)).product();
                self.current.set(Key::Flops, (FMA_FLOPS * 4 * factors * real.elements()) as f32);
            }
            "triangular-solve" | "cholesky" => self.dense_solver(node, opcode),
            "all-gather" => self.all_gather(node, None)?,
            "all-gather-start" => self.all_gather(node, Some(0))?,
            "all-reduce" => self.all_reduce(node)?,
            "all-reduce-start" => {
                let ranks = self.num_ranks(node)?;
                let transferred = ring_shape_size(shape, None);
                let root = context.root(context.called(node, 0)?);
                self.current.set(Key::Flops, (profile_flops(context.opcode(root), shape.element_type) * shape.elements_recursive()) as f32);
                self.current.set(Key::Bytes, transferred as f32);
                self.current.set(Key::Transferred, transferred as f32);
                self.ring(ranks, 2 * (ranks - 1));
            }
            "reduce-scatter" => self.reduce_scatter(node)?,
            "all-to-all" => self.current.set(Key::Transferred, ring_shape_size(shape, None) as f32),
            "collective-permute" | "collective-permute-start" => *self.current.slot(Key::Transferred) += byte_size(context.shape(context.operand(node, 0))) as f32,
            "async-start" => {
                let wrapped = context.root(context.called(node, 0)?);
                self.dfs(wrapped)?;
                match context.opcode(wrapped) {
                    "reduce-scatter" => self.reduce_scatter(wrapped)?,
                    "all-to-all" => self.current.set(Key::Transferred, ring_shape_size(context.shape(wrapped), None) as f32),
                    _ => {}
                }
            }
            "rng" => self.current.set(Key::Transcendentals, shape.elements() as f32),
            "rng-bit-generator" => self.current.set(Key::Transcendentals, shape.elements_recursive() as f32),
            "fusion" => self.fusion(node)?,
            "call" | "while" | "conditional" => self.control_flow(node, opcode)?,
            "custom-call" => self.custom_call(node)?,
            "sort" => {
                let count = context.shape(context.operand(node, 0)).elements();
                self.current.set(Key::Flops, (count * log2_ceiling(count as u64)) as f32);
            }
            "gather" => self.gather(node),
            "scatter" => self.scatter(node)?,
            _ => {}
        }
        Ok(())
    }

    fn slicing(&mut self, node: usize, opcode: &str) {
        let context = self.context;
        let shape = context.shape(node);
        let operand_size = |index: usize| shape_size(context.shape(context.operand(node, index)));
        if opcode == "dynamic-update-slice" {
            let update = operand_size(1);
            self.transfers(update, &[(0, 0), (1, update), (2, operand_size(2))]);
            let (updates, outputs) = (context.shape(context.operand(node, 1)).elements(), shape.elements());
            self.set_operand_utilization(0, ((outputs - updates) as f64 / outputs as f64) as f32);
            return;
        }
        let output = shape_size(shape);
        if opcode == "slice" {
            self.transfers(output, &[(0, output)]);
        } else {
            self.transfers(output, &[(0, output), (1, operand_size(1))]);
        }
        self.set_operand_utilization(0, (shape.elements() as f64 / context.shape(context.operand(node, 0)).elements() as f64) as f32);
    }

    fn feed(&mut self, node: usize, opcode: &str) {
        let context = self.context;
        if opcode == "infeed" {
            let mut size = 0i64;
            for (index, leaf) in leaves(context.shape(node)) {
                size += shape_size(leaf);
                self.set_output(&index, shape_size(leaf) as f32);
            }
            self.set_output(&[], size as f32);
            self.current.set(Key::Bytes, size as f32);
            return;
        }
        self.current.set(Key::Bytes, 0.0);
        for (position, &operand) in context.operands(node).iter().enumerate() {
            let mut size = 0i64;
            for (index, leaf) in leaves(context.shape(operand)) {
                size += shape_size(leaf);
                self.set_operand_bytes(position, &index, shape_size(leaf) as f32);
            }
            self.set_operand_bytes(position, &[], size as f32);
            *self.current.slot(Key::Bytes) += size as f32;
        }
    }

    fn dense_solver(&mut self, node: usize, opcode: &str) {
        let context = self.context;
        let shape = context.shape(node);
        let a = context.shape(context.operand(node, 0));
        if opcode == "cholesky" {
            let half = shape_size(a) as f32 / 2.0;
            self.set_output(&[], half);
            self.set_operand_bytes(0, &[], half);
            self.current.set(Key::Bytes, half + half);
            let count = a.dimensions.last().copied().unwrap_or(0) * a.elements();
            self.current.set(Key::Flops, (count / 3) as f32);
            return;
        }
        let b = context.shape(context.operand(node, 1));
        let mut bytes = shape_size(shape) as f32;
        self.set_output(&[], shape_size(shape) as f32);
        bytes += shape_size(a) as f32 / 2.0;
        self.set_operand_bytes(0, &[], shape_size(a) as f32 / 2.0);
        bytes += shape_size(b) as f32;
        self.set_operand_bytes(0, &[], shape_size(b) as f32);
        self.current.set(Key::Bytes, bytes);
        let count = a.dimensions.last().copied().unwrap_or(0) * b.elements();
        self.current.set(Key::Flops, (FMA_FLOPS * count) as f32);
    }

    fn control_flow(&mut self, node: usize, opcode: &str) -> Status {
        let context = self.context;
        match opcode {
            "call" => self.current = self.subcomputation(context.called(node, 0)?)?,
            "while" => {
                let body = self.subcomputation(context.called(node, 0)?)?;
                let condition = self.subcomputation(context.called(node, 1)?)?;
                self.current = Props::default();
                self.current.merge(&body, |sum, value| sum + value);
                self.current.merge(&condition, |sum, value| sum + value);
            }
            _ => {
                let branches = context.module.nodes[node].called.len();
                self.current = self.subcomputation(context.called(node, 0)?)?;
                for branch in 1..branches {
                    let props = self.subcomputation(context.called(node, branch)?)?;
                    self.current.merge(&props, f32::max);
                }
            }
        }
        self.bottleneck = false;
        Ok(())
    }

    fn bitcast(&mut self) {
        self.current.set(Key::Bytes, 0.0);
        self.set_output(&[], 0.0);
        self.set_operand_bytes(0, &[], 0.0);
        self.current.set(Key::Optimal, 0.0);
    }

    fn elementwise(&mut self, node: usize) {
        let shape = self.context.shape(node);
        self.current.set(Key::Flops, (profile_flops(self.context.opcode(node), shape.element_type) * shape.elements_recursive()) as f32);
    }

    fn ring(&mut self, ranks: i64, steps: i64) {
        self.current.set(Key::NumDevices, ranks as f32);
        self.current.set(Key::ScaleRatio, if steps > 0 { ranks as f32 / steps as f32 } else { 0.0 });
    }

    fn num_ranks(&self, node: usize) -> Result<i64, String> {
        let inst = self.context.inst(node);
        let (channel, global) = (inst.channel_id > 0, inst.use_global_device_ids);
        if !channel && global {
            return Err("Cannot have use_global_device_ids=true without channel_id".into());
        }
        let largest: Option<i64> = match (replica_groups(inst), &inst.replica_group_list) {
            (legacy, _) if !legacy.is_empty() => legacy.iter().map(|group| group.replica_ids.len() as i64).max(),
            (_, Some(ReplicaGroupList::IotaCollectiveDeviceList(list))) => (list.num_replica_groups > 0).then_some(list.num_devices_per_group),
            (_, Some(ReplicaGroupList::MeshAxesReplicaGroupList(list))) => {
                let axes = list.mesh.as_ref().map_or(&[][..], |mesh| &mesh.axes[..]);
                let devices = axes.iter().try_fold(1i64, |product, axis| product.checked_mul(axis.size));
                let size = list.axes.iter().try_fold(1i64, |product, axis| {
                    product.checked_mul(axis.sub_axis_info.map_or_else(|| axes.get(axis.mesh_axis_index as usize).map_or(1, |mesh_axis| mesh_axis.size), |sub| sub.size))
                });
                devices.zip(size).filter(|&(devices, size)| size > 0 && devices / size > 0).map(|(_, size)| size)
            }
            (_, Some(ReplicaGroupList::CollectiveDeviceList(list))) => list.replica_groups.iter().map(|group| group.replica_ids.len() as i64).max(),
            (_, None) => None,
        };
        if largest.is_none() && channel && global {
            return Err("RET_CHECK failure !replica_groups.empty() replica groups cannot be empty for kFlattenedID mode".into());
        }
        Ok(largest.map_or(1, |largest| largest.max(1)))
    }

    fn all_gather(&mut self, node: usize, skip: Option<i64>) -> Status {
        let ranks = self.num_ranks(node)?;
        let transferred = ring_shape_size(self.context.shape(node), skip);
        let rank_size = transferred / ranks;
        self.current.set(Key::Bytes, (rank_size * (2 * ranks - 1) + rank_size * ranks) as f32);
        self.current.set(Key::Transferred, transferred as f32);
        self.ring(ranks, ranks - 1);
        Ok(())
    }

    fn all_reduce(&mut self, node: usize) -> Status {
        let context = self.context;
        let ranks = self.num_ranks(node)?;
        let shape = context.shape(node);
        let output: i64 = all_subshapes(shape).into_iter().filter(|(_, subshape)| subshape.is_array()).map(|(_, subshape)| shape_size(subshape)).sum();
        let bytes = output + context.operands(node).iter().map(|&operand| shape_size(context.shape(operand))).sum::<i64>();
        self.set_output(&[], output as f32);
        self.current.set(Key::Transferred, output as f32);
        self.current.set(Key::Bytes, bytes as f32);
        let root = context.root(context.called(node, 0)?);
        self.current.set(Key::Flops, (profile_flops(context.opcode(root), shape.element_type) * shape.elements_recursive()) as f32);
        self.ring(ranks, 2 * (ranks - 1));
        Ok(())
    }

    fn reduce_scatter(&mut self, node: usize) -> Status {
        let context = self.context;
        let ranks = self.num_ranks(node)?;
        let transferred: i64 = context.operands(node).iter().map(|&operand| ring_shape_size(context.shape(operand), None)).sum();
        let rank_size = transferred / ranks;
        self.current.set(Key::Bytes, (rank_size * ranks + rank_size * (2 * ranks - 1)) as f32);
        self.current.set(Key::Transferred, transferred as f32);
        let root = context.root(context.called(node, 0)?);
        let shape = context.shape(node);
        self.current.set(Key::Flops, (profile_flops(context.opcode(root), shape.element_type) * shape.elements_recursive()) as f32);
        self.ring(ranks, ranks - 1);
        Ok(())
    }

    fn reduce(&mut self, node: usize) -> Status {
        let context = self.context;
        let sub = self.subcomputation(context.called(node, 0)?)?;
        let shape = context.shape(node);
        let output = if shape.is_array() { shape.clone() } else { shape.tuple_shapes.first().cloned().unwrap_or_default() };
        self.copy_scaled(&sub, context.shape(context.operand(node, 0)).elements() - output.elements(), false);
        let output_bytes: i64 = leaves(shape).into_iter().map(|(_, leaf)| shape_size(leaf)).sum();
        self.set_output(&[], output_bytes as f32);
        let inputs = context.operands(node).len() / 2;
        let mut bytes = output_bytes;
        for index in 0..inputs {
            bytes = (bytes as f32 + self.current.get(&operand_bytes_key(index, &[]))) as i64;
        }
        let output_elements = output.elements();
        for index in inputs..context.operands(node).len() {
            let size = output_elements * shape_size(context.shape(context.operand(node, index)));
            self.set_operand_bytes(index, &[], size as f32);
            self.set_operand_utilization(index, output_elements as f32);
            bytes += size;
        }
        self.current.set(Key::Bytes, bytes as f32);
        Ok(())
    }

    fn reduce_window(&mut self, node: usize) -> Status {
        let context = self.context;
        let sub = self.subcomputation(context.called(node, 0)?)?;
        let shape = context.shape(node);
        let window = context.inst(node).window.clone().unwrap_or_default();
        let mut window_elements: i64 = window.dimensions.iter().map(|dimension| dimension.size).product();
        let output_elements = (if shape.is_array() { shape } else { shape.tuple_shapes.first().unwrap_or(shape) }).elements();
        let mut count = (window_elements - 1) * output_elements;
        let reducing = window.dimensions.iter().filter(|dimension| dimension.size != 1).count();
        let padded = window.dimensions.iter().filter(|dimension| dimension.padding_low != 0 || dimension.padding_high != 0).count();
        if reducing == 1 && padded == 1 && shape.is_array() {
            let found = window.dimensions.iter().position(|dimension| {
                dimension.size != 1 && dimension.padding_low != 0 && dimension.padding_high != 0 && dimension.padding_low == dimension.padding_high && dimension.size == 2 * dimension.padding_low + 1
            });
            if let Some(position) = found
                && window.dimensions[position].padding_low == shape.dimensions[position] - 1
            {
                window_elements = shape.dimensions[position];
                count = output_elements / window_elements + (window_elements - 1);
            }
        }
        self.copy_scaled(&sub, count, false);
        Ok(())
    }

    fn gather(&mut self, node: usize) {
        let context = self.context;
        let shape = context.shape(node);
        let output = shape_size(shape);
        self.transfers(output, &[(0, output), (1, shape_size(context.shape(context.operand(node, 1))))]);
        self.set_operand_utilization(0, (shape.elements() as f64 / context.shape(context.operand(node, 0)).elements() as f64) as f32);
    }

    fn scatter(&mut self, node: usize) -> Status {
        let context = self.context;
        let count = (context.operands(node).len() - 1) / 2;
        let mut total = 0i64;
        for index in 0..count {
            let size = shape_size(context.shape(context.operand(node, count + 1 + index)));
            self.set_operand_bytes(index, &[], size as f32);
            self.set_operand_bytes(count + 1 + index, &[], size as f32);
            total += size;
        }
        let indices = shape_size(context.shape(context.operand(node, count)));
        self.set_operand_bytes(count, &[], indices as f32);
        self.current.set(Key::Bytes, (total * 3 + indices) as f32);
        self.set_output(&[], total as f32);
        let sub = self.subcomputation(context.called(node, 0)?)?;
        self.copy_scaled(&sub, context.shape(context.operand(node, count + 1)).elements(), false);
        Ok(())
    }

    fn custom_call(&mut self, node: usize) -> Status {
        let context = self.context;
        let inst = context.inst(node);
        let target = inst.custom_call_target.as_str();
        let shape = context.shape(node);
        let output = if shape.is_tuple() { shape.tuple_shapes.first().cloned().unwrap_or_default() } else { shape.clone() };
        if CUBLAS_LT.contains(&target) {
            let config = context.module.backend_config(inst).unwrap_or_default();
            let lhs = if config.is_empty() {
                Vec::new()
            } else {
                let value: serde_json::Value = serde_json::from_slice(&config).map_err(|error| error.to_string())?;
                json_dimensions(&value["gemm_backend_config"]["dot_dimension_numbers"]["lhs_contracting_dimensions"])
            };
            let operand = context.shape(context.operand(node, 0));
            let flops = dot_flops(operand, &output, &lhs) as f32;
            self.current.set(Key::Flops, flops);
            self.replace_output(byte_size(&output) as f32);
            self.current.set(Key::Adjustment, adjusted(flops, bit_width(operand.element_type)));
            return Ok(());
        }
        if DNN_CONVOLUTION.contains(&target) {
            let flops = convolution_flops(inst, context.shape(context.operand(node, 0)), context.shape(context.operand(node, 1)), &output);
            self.current.set(Key::Flops, flops as f32);
            if shape.is_tuple() {
                self.replace_output(byte_size(&output) as f32);
            }
            return Ok(());
        }
        let value = if CALL_MARKERS.contains(&target) { 0.0 } else { -1.0 };
        self.current.set(Key::Optimal, value);
        self.current.set(Key::Bytes, value);
        self.set_output(&[], value);
        for index in 0..context.operands(node).len() {
            self.set_operand_bytes(index, &[], value);
        }
        self.current.set(Key::Flops, value);
        self.bottleneck = false;
        let raw = crate::xplane::lossy(&context.module.backend_config(inst).unwrap_or_default()).into_owned();
        if raw.contains("cost_estimate_json") {
            if let Some(flops) = cost_estimate(&raw, "flops").filter(|value| *value >= 0) {
                self.current.set(Key::Flops, flops as f32);
            }
            if let Some(bytes) = cost_estimate(&raw, "bytes_accessed").filter(|value| *value >= 0) {
                self.current.set(Key::Bytes, bytes as f32);
            }
        }
        Ok(())
    }

    fn fusion(&mut self, node: usize) -> Status {
        let context = self.context;
        let graph = context.called(node, 0)?;
        if context.inst(node).fusion_kind == "kCustom" {
            for &child in &context.module.graphs[graph].nodes {
                match context.opcode(child) {
                    "gather" => {
                        self.gather(child);
                        return Ok(());
                    }
                    "scatter" => return self.scatter(child),
                    _ => {}
                }
            }
        }
        self.current = self.subcomputation(graph)?;
        self.current.set(Key::Bytes, 0.0);
        self.fusion_outputs(node, graph);
        self.fusion_utilizations(graph);
        for &child in &context.module.graphs[graph].nodes {
            if context.opcode(child) == "constant" && context.shape(child).elements() > IMMEDIATE_CONSTANT_MAX_ELEMENTS {
                let utilization = (self.property(child, &Key::Utilization) as f64).min(1.0) as f32;
                *self.current.slot(Key::Bytes) += shape_size(context.shape(child)) as f32 * utilization;
            }
        }
        for index in 0..context.module.graphs[graph].parameters.len() {
            let parameter = context.parameter(graph, index)?;
            let mut size = 0i64;
            if context.shape(parameter).is_tuple() {
                for (shape_index, _) in leaves(context.shape(parameter)) {
                    let mut current = parameter;
                    for &sub in &shape_index {
                        if let Some(&user) = context.users(current).iter().find(|&&user| context.opcode(user) == "get-tuple-element" && context.inst(user).tuple_index == sub) {
                            current = user;
                        }
                    }
                    let read = self.parameter_read_bytes(current);
                    size += read;
                    self.set_operand_bytes(index, &shape_index, read as f32);
                }
            } else {
                size = self.parameter_read_bytes(parameter);
            }
            *self.current.slot(Key::Bytes) += size as f32;
            self.set_operand_bytes(index, &[], size as f32);
            let utilization = self.property(parameter, &Key::Utilization);
            self.set_operand_utilization(index, utilization);
        }
        Ok(())
    }

    fn fusion_outputs(&mut self, node: usize, graph: usize) {
        let context = self.context;
        let fused_root = context.root(graph);
        for (index, subshape) in all_subshapes(context.shape(node)) {
            if !subshape.is_array() {
                continue;
            }
            let mut root = fused_root;
            if index.len() == 1 && context.opcode(root) == "tuple" {
                root = context.operand(root, index[0] as usize);
            }
            if context.opcode(root) == "dynamic-update-slice" {
                let size = shape_size(context.shape(context.operand(root, 1)));
                *self.current.slot(Key::Bytes) += size as f32;
                self.set_output(&index, size as f32);
                *self.slot(root, Key::OperandUtilization(0)) = 0.0;
                continue;
            }
            *self.current.slot(Key::Bytes) += shape_size(subshape) as f32;
            self.set_output(&index, shape_size(subshape) as f32);
        }
        if context.shape(node).is_tuple() {
            self.set_output(&[], 0.0);
            self.propagate_output(context.shape(node), &mut Vec::new());
        }
    }

    fn propagate_output(&mut self, shape: &Shape, index: &mut Vec<i64>) -> f32 {
        let key = output_key(index);
        let bytes = self.current.get(&key);
        if bytes != 0.0 || !shape.is_tuple() {
            self.current.slot(key);
            return bytes;
        }
        let mut total = bytes;
        for (position, subshape) in shape.tuple_shapes.iter().enumerate() {
            index.push(position as i64);
            total += self.propagate_output(subshape, index);
            index.pop();
        }
        self.current.set(key, total);
        total
    }

    fn fusion_utilizations(&mut self, graph: usize) {
        let context = self.context;
        let root = context.root(graph);
        let mut instructions = context.module.post_order(graph);
        instructions.reverse();
        let mut ir_sizes: HashMap<usize, i64> = HashMap::new();
        for &instruction in &instructions {
            *self.slot(instruction, Key::Utilization) = 0.0;
            *self.slot(instruction, Key::IrSize) = 0.0;
            self.use_roots.insert(instruction, Vec::new());
            self.root_utilizations.insert(instruction, 0.0);
        }
        self.root_utilizations.insert(root, 1.0);
        ir_sizes.insert(root, 1);
        self.use_roots.insert(root, vec![root]);
        self.current.set(Key::Flops, 0.0);
        self.current.set(Key::IrSize, 0.0);
        for &instruction in &instructions {
            let roots = self.use_roots.get(&instruction).cloned().unwrap_or_default();
            for &used in &roots {
                let utilization = self.root_utilizations.get(&used).copied().unwrap_or(0.0);
                let size = ir_sizes.get(&used).copied().unwrap_or(0);
                *self.slot(instruction, Key::Utilization) += utilization;
                *self.slot(instruction, Key::IrSize) += size as f32;
            }
            let utilization = self.property(instruction, &Key::Utilization);
            let emitted = self.property(instruction, &Key::IrSize);
            let flops = self.property(instruction, &Key::Flops);
            *self.current.slot(Key::Flops) += utilization * flops;
            *self.current.slot(Key::IrSize) += emitted;
            let opcode = context.opcode(instruction);
            let passes = context.is_elementwise(instruction)
                || (opcode == "bitcast" && Context::equal_ignoring_element_type(context.shape(context.operand(instruction, 0)), context.shape(instruction)))
                || opcode == "tuple"
                || opcode == "get-tuple-element";
            for (index, &operand) in context.operands(instruction).iter().enumerate() {
                if passes {
                    let entry = self.use_roots.entry(operand).or_default();
                    for &used in &roots {
                        if !entry.contains(&used) {
                            entry.push(used);
                        }
                    }
                } else {
                    let entry = self.use_roots.entry(operand).or_default();
                    if !entry.contains(&operand) {
                        entry.push(operand);
                    }
                    let mut operand_utilization = utilization * self.property(instruction, &Key::OperandUtilization(index));
                    let count = context.shape(operand).elements_recursive();
                    operand_utilization = if count == 0 { 0.0 } else { (operand_utilization * count as f32).ceil() / count as f32 };
                    *self.root_utilizations.entry(operand).or_insert(0.0) += operand_utilization;
                    let size = ir_sizes.entry(operand).or_insert(0);
                    *size = (*size as f32 + emitted) as i64;
                }
            }
        }
    }

    fn parameter_read_bytes(&self, node: usize) -> i64 {
        let utilization = (self.property(node, &Key::Utilization) as f64).min(1.0) as f32;
        (shape_size(self.context.shape(node)) as f32 * utilization).round() as i64
    }
}

fn dot_flops(lhs: &Shape, result: &Shape, contracting: &[i64]) -> i64 {
    let width: i64 = contracting.iter().map(|&dimension| lhs.dimensions.get(dimension as usize).copied().unwrap_or(1)).product();
    FMA_FLOPS * result.elements() * width
}

fn convolution_flops(inst: &Inst, lhs: &Shape, rhs: &Shape, result: &Shape) -> i64 {
    let numbers = inst.convolution_dimension_numbers.clone().unwrap_or_default();
    let dimension = |shape: &Shape, index: i64| shape.dimensions.get(index as usize).copied().unwrap_or(0);
    let (input_feature, output_feature, batch) = (dimension(lhs, numbers.input_feature_dimension), dimension(result, numbers.output_feature_dimension), dimension(lhs, numbers.input_batch_dimension));
    let mut window = inst.window.as_deref().cloned().unwrap_or_default().dimensions;
    let (mut kernels, mut outputs, mut inputs) = (Vec::new(), Vec::new(), Vec::new());
    if window.is_empty() {
        window.push(crate::hlo::xla::WindowDimension { size: 1, stride: 1, window_dilation: 1, base_dilation: 1, ..Default::default() });
        (kernels, outputs, inputs) = (vec![1], vec![1], vec![1]);
    } else {
        for spatial in 0..window.len() {
            let at = |list: &[i64]| list.get(spatial).copied().unwrap_or(0);
            kernels.push(dimension(rhs, at(&numbers.kernel_spatial_dimensions)));
            outputs.push(dimension(result, at(&numbers.output_spatial_dimensions)));
            inputs.push(dimension(lhs, at(&numbers.input_spatial_dimensions)));
        }
    }
    let mut positions = Vec::new();
    for (spatial, window) in window.iter().enumerate() {
        let (input, output, kernel) = (inputs[spatial], outputs[spatial], kernels[spatial]);
        if input == output
            && kernel == output
            && input == window.base_dilation
            && window.window_dilation == 1
            && (input - 1).max(1) == window.stride
            && window.padding_low == 0
            && window.padding_high == 0
        {
            positions.push(input);
            continue;
        }
        if input == 1 && kernel == output && window.window_dilation == 1 && window.base_dilation == 1 && window.stride == 1 && window.padding_high == output - 1 && window.padding_low == output - 1 {
            positions.push(output);
            continue;
        }
        let mut count = 0i64;
        for kernel_index in 0..kernel {
            if window.stride == 1 && window.base_dilation == 1 {
                let base = window.padding_low - kernel_index * window.window_dilation;
                count += ((input + base).min(output) - base.max(0)).max(0);
                continue;
            }
            for output_index in 0..output {
                let undilated = output_index * window.stride - window.padding_low + kernel_index * window.window_dilation;
                let spatial_index = if window.base_dilation > 1 { undilated / window.base_dilation } else { undilated };
                if undilated != spatial_index * window.base_dilation || spatial_index < 0 || spatial_index >= input {
                    continue;
                }
                count += 1;
            }
        }
        positions.push(count);
    }
    let (features, batches) = (inst.feature_group_count.max(1), inst.batch_group_count.max(1));
    (input_feature / features) * output_feature * (batch / batches) * positions.iter().product::<i64>() * FMA_FLOPS
}

pub fn costs(module: &Module) -> Option<Vec<Cost>> {
    if !module.valid {
        return None;
    }
    let mut successors = vec![false; module.nodes.len()];
    for node in &module.nodes {
        for &predecessor in &node.predecessors {
            successors[predecessor] = true;
        }
    }
    let context = Context { module, insts: (0..module.nodes.len()).map(|_| OnceCell::new()).collect(), successors };
    let mut analysis = Analysis::new(&context);
    analysis.accept(module.entry).ok()?;
    Some(
        (0..module.nodes.len())
            .map(|node| {
                let props = analysis.properties.get(&node);
                let get = |key: Key| props.map_or(0.0, |props| props.get(&key));
                let flops = get(Key::Flops) as i64;
                let bytes_accessed = valid(get(Key::Bytes) as i64);
                let mut memory = Vec::new();
                if bytes_accessed > 0 {
                    let mut read = 0i64;
                    for (operand, &input) in module.nodes[node].operands.iter().enumerate() {
                        for (index, leaf) in leaves(&module.nodes[input].shape) {
                            if memory_space(leaf) == Some(0) {
                                read += get(operand_bytes_key(operand, &index)) as i64;
                            }
                        }
                    }
                    let written: i64 = leaves(&module.nodes[node].shape).into_iter().filter(|(_, leaf)| memory_space(leaf) == Some(0)).map(|(index, _)| get(output_key(&index)) as i64).sum();
                    for (is_read, value) in [(true, valid(read)), (false, valid(written))] {
                        if value > 0 {
                            memory.push((is_read, HBM, value));
                        }
                    }
                }
                Cost { model_flops: valid(flops), device_flops: valid(flops - get(Key::Adjustment) as i64), bytes_accessed, memory }
            })
            .collect(),
    )
}

#[cfg(test)]
#[path = "tests/inline/gpu_cost.rs"]
pub mod tests;
