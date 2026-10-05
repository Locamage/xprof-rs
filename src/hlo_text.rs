use crate::hlo::xla::hlo_instruction_proto::ReplicaGroupList;
use crate::hlo::xla::{self, LiteralProto, MeshAxesReplicaGroupListProto, OpMetadata, OpSharding, PrecisionConfig, WindowDimension};
use crate::hlo::{Inst, Module, Shape, all_subshapes, general, msg};
use crate::pbtext::{c_escape, enum_name};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use prost::Message;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::hash::Hash;
use std::sync::Mutex;

const INDEX_INTERVAL: usize = 5;
const MAX_COMPACT_ELEMENTS: i64 = 10;
/// The printers allocate, and the literal printer recurses, once for each dimension.
const MAX_RANK: usize = 1024;
/// The literal printer prints each empty sub-array.
const MAX_EMPTY_ARRAYS: usize = 1 << 20;
const CALLS_TO_APPLY: [&str; 11] = ["call", "map", "reduce-window", "reduce", "all-reduce", "reduce-scatter", "collective-reduce", "all-reduce-start", "scatter", "sort", "scan"];
const CHANNEL_OPS: [&str; 4] = ["send", "send-done", "recv", "recv-done"];
const COLLECTIVES: [&str; 9] = ["all-gather", "all-gather-start", "all-reduce", "all-reduce-start", "reduce-scatter", "collective-reduce", "all-to-all", "ragged-all-to-all", "collective-broadcast"];
const GATHER: [&str; 5] = ["offset_dims", "collapsed_slice_dims", "start_index_map", "operand_batching_dims", "start_indices_batching_dims"];
const SCATTER: [&str; 5] = ["update_window_dims", "inserted_window_dims", "scatter_dims_to_operand_dims", "input_batching_dims", "scatter_indices_batching_dims"];
const WINDOW_FIELDS: [WindowField; 6] = [
    ("size", |_| true, |dimension| dimension.size.to_string()),
    ("stride", |dimension| dimension.stride != 1, |dimension| dimension.stride.to_string()),
    ("pad", |dimension| dimension.padding_low != 0 || dimension.padding_high != 0, |dimension| format!("{}_{}", dimension.padding_low, dimension.padding_high)),
    ("lhs_dilate", |dimension| dimension.base_dilation != 1, |dimension| dimension.base_dilation.to_string()),
    ("rhs_dilate", |dimension| dimension.window_dilation != 1, |dimension| dimension.window_dilation.to_string()),
    ("rhs_reversal", |dimension| dimension.window_reversal, |dimension| i64::from(dimension.window_reversal).to_string()),
];
const SUBGROUPS: [&str; 7] = ["replicated", "maximal", "error_type.", "error_type.", "manual", "error_type.", "unreduced"];

type WindowField = (&'static str, fn(&WindowDimension) -> bool, fn(&WindowDimension) -> String);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Short,
    Long,
    Graph,
    Expression,
}

#[derive(Default)]
pub struct Frames {
    pub files: IndexSet<String>,
    pub functions: IndexSet<String>,
    pub locations: IndexSet<[i64; 6]>,
    pub frames: IndexSet<(i64, i64)>,
    pub of: Vec<i64>,
}

pub struct Printer<'a> {
    pub module: &'a Module<'a>,
    style: Style,
    metadata: bool,
    pub(crate) operand_shapes: bool,
    pub(crate) large_constants: bool,
    async_graphs: HashSet<usize>,
    failed: Mutex<Option<String>>,
    pub frames: Frames,
}

fn escaped(text: &str) -> String {
    let mut out = String::new();
    c_escape(&mut out, text.as_bytes());
    out
}

fn intern<K: Hash + Eq>(set: &mut IndexSet<K>, key: K) -> i64 {
    set.insert_full(key).0 as i64 + 1
}

impl Frames {
    fn new(module: &Module) -> Self {
        let stack = msg(&module.proto.stack_frame_index);
        let mut frames = Self { of: vec![0; module.nodes.len()], ..Self::default() };
        let mut mapping: HashMap<i64, i64> = HashMap::new();
        for node in 0..module.nodes.len() {
            let (mut current, mut path, mut seen) = (module.nodes[node].frame, Vec::new(), HashSet::new());
            if current == 0 {
                continue;
            }
            while current != 0 && !mapping.contains_key(&current) {
                if current < 0 || current as usize > stack.stack_frames.len() || !seen.insert(current) {
                    break;
                }
                path.push(current);
                current = i64::from(stack.stack_frames[current as usize - 1].parent_frame_id);
            }
            if current != 0 && !mapping.contains_key(&current) {
                continue;
            }
            let mut id = if current == 0 { 0 } else { mapping[&current] };
            while let Some(old) = path.pop() {
                let get = |list: &[String], id: i32| list.get(usize::try_from(id - 1).ok()?).cloned();
                let location = usize::try_from(stack.stack_frames[old as usize - 1].file_location_id - 1).ok().and_then(|index| stack.file_locations.get(index));
                let Some((location, file, function)) =
                    location.and_then(|location| Some((location, get(&stack.file_names, location.file_name_id)?, get(&stack.function_names, location.function_name_id)?)))
                else {
                    id = 0;
                    break;
                };
                let file = intern(&mut frames.files, file);
                let function = intern(&mut frames.functions, function);
                let place = [file, function, i64::from(location.line), i64::from(location.column), i64::from(location.end_line), i64::from(location.end_column)];
                let location = intern(&mut frames.locations, place);
                id = intern(&mut frames.frames, (location, id));
                mapping.insert(old, id);
            }
            frames.of[node] = id;
        }
        frames
    }
}

fn skip_space(bytes: &[u8], mut pos: usize) -> Option<usize> {
    loop {
        match bytes.get(pos) {
            Some(b' ' | b'\t' | b'\n' | b'\r') => pos += 1,
            Some(b'/') if bytes.get(pos + 1) == Some(&b'*') => pos += bytes[pos + 2..].windows(2).position(|pair| pair == b"*/")? + 4,
            Some(b'/') if bytes.get(pos + 1) == Some(&b'/') => {
                while !matches!(bytes.get(pos), None | Some(b'\n' | b'\r')) {
                    if bytes[pos] == 0 {
                        return None;
                    }
                    pos += 1;
                }
            }
            _ => return Some(pos),
        }
    }
}

pub fn lexes_as_json_dict(text: &str) -> bool {
    let bytes = text.as_bytes();
    let Some(mut pos) = skip_space(bytes, 0) else { return false };
    if bytes.get(pos) != Some(&b'{') {
        return false;
    }
    pos += 1;
    let mut depth = 1;
    while pos < bytes.len() && depth > 0 {
        match bytes[pos] {
            b'"' => {
                pos += 1;
                loop {
                    match bytes.get(pos) {
                        None => return false,
                        Some(b'"') => break,
                        Some(b'\\') => {
                            if matches!(bytes.get(pos + 1), None | Some(b'\n')) {
                                return false;
                            }
                            pos += 2;
                        }
                        Some(_) => pos += 1,
                    }
                }
            }
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        pos += 1;
    }
    depth == 0 && skip_space(bytes, pos) == Some(bytes.len())
}

pub fn frontend_attributes(attributes: &HashMap<String, String>) -> String {
    let items = attributes.iter().sorted().map(|(key, value)| if lexes_as_json_dict(value) { format!("{key}={value}") } else { format!("{key}=\"{}\"", escaped(value)) });
    format!("{{{}}}", items.format(","))
}

#[allow(deprecated)]
fn metadata_text(metadata: &OpMetadata, frame: i64, payloads: &[Vec<u8>]) -> String {
    let quoted = |key: &str, value: &str| (!value.is_empty()).then(|| format!("{key}=\"{}\"", escaped(value)));
    let number = |key: &str, value: i64| (value != 0).then(|| format!("{key}={value}"));
    let payload = match metadata.metadata_payload.as_ref().and_then(|payload| payload.payload_source.as_ref()) {
        Some(xla::payload::PayloadSource::Value(value)) => Some(value.as_slice()),
        Some(xla::payload::PayloadSource::Id(id)) => usize::try_from(*id).ok().and_then(|id| payloads.get(id)).map(Vec::as_slice),
        None => None,
    };
    let payload = payload.map(|value| format!("metadata_payload=\"{}\"", escaped(std::str::from_utf8(value).unwrap_or(""))));
    let parts = [
        quoted("op_type", &metadata.op_type),
        quoted("op_name", &metadata.op_name),
        quoted("source_file", &metadata.source_file),
        number("source_line", metadata.source_line.into()),
        number("source_end_line", metadata.source_end_line.into()),
        number("source_column", metadata.source_column.into()),
        number("source_end_column", metadata.source_end_column.into()),
        (!metadata.profile_type.is_empty()).then(|| format!("profile_type={{{}}}", metadata.profile_type.iter().join(","))),
        quoted("deduplicated_name", &metadata.deduplicated_name),
        (!metadata.scheduling_name.is_empty()).then(|| format!("scheduling_name=\"{}\"", metadata.scheduling_name)),
        number("stack_frame_id", frame),
        payload,
    ];
    parts.into_iter().flatten().join(" ")
}

fn canonicalize_iota(dims: &mut Vec<i64>, perm: &mut Vec<i64>) {
    if dims.len() <= 1 || perm.len() != dims.len() || !(0..dims.len() as i64).all(|dimension| perm.contains(&dimension)) {
        return;
    }
    loop {
        let kept: Vec<usize> = (0..dims.len()).filter(|&index| dims[index] != 1).collect();
        *perm = perm.iter().filter_map(|&dim| kept.iter().position(|&old| old as i64 == dim).map(|new| new as i64)).collect();
        *dims = kept.iter().map(|&index| dims[index]).collect();
        let (mut base, mut changed) = (0, false);
        for index in 1..dims.len() {
            let (base_dim, dim) = (perm[base] as usize, perm[index] as usize);
            if base_dim + (index - base) == dim {
                dims[base_dim] *= dims[dim];
                dims[dim] = 1;
                changed = true;
            } else {
                base = index;
            }
        }
        if !changed {
            break;
        }
    }
}

pub fn sparsity_config_text(config: &xla::SparsityConfig) -> String {
    let side = |name: &str, side: &xla::sparsity_config::TensorSparsityConfig| {
        format!("{name}={{sparsity={}x{} dimension={} stride={} idx={}}}", side.num_non_zero, side.block_size, side.dimension, side.stride, side.idx)
    };
    [config.lhs.map(|lhs| side("lhs", &lhs)), config.rhs.map(|rhs| side("rhs", &rhs))].into_iter().flatten().join(" ")
}

pub fn block_scaling_config_text(config: &xla::BlockScalingConfig) -> String {
    let side = |name: &str, side: &xla::block_scaling_config::TensorBlockScalingConfig| {
        let mut text = format!("{name}={{scale_idx={}", side.scale_idx);
        if let Some(zero) = side.zero_idx {
            write!(text, " zero_idx={zero}").unwrap();
        }
        if !side.strides.is_empty() {
            write!(text, " strides={}", side.strides.iter().join("x")).unwrap();
        }
        if !side.steps.is_empty() {
            write!(text, " steps={}", side.steps.iter().join("x")).unwrap();
        }
        text + "}"
    };
    [config.lhs.as_ref().map(|lhs| side("lhs", lhs)), config.rhs.as_ref().map(|rhs| side("rhs", rhs))].into_iter().flatten().join(" ")
}

pub fn iota_text(dims: &[i64], reshape: &[i64], perm: &[i64]) -> String {
    let (mut reshape, mut perm) = (reshape.to_vec(), perm.to_vec());
    canonicalize_iota(&mut reshape, &mut perm);
    if reshape.is_empty() {
        reshape = vec![1];
        perm = vec![0];
    }
    let mut out = format!("[{}]<=[{}]", dims.iter().join(","), reshape.iter().join(","));
    if reshape.len() > 1 {
        write!(out, "T({})", perm.iter().join(",")).unwrap();
    }
    out
}

fn leaf_count(shape: &Shape) -> usize {
    if shape.is_tuple() { shape.tuple_shapes.iter().map(leaf_count).sum() } else { 1 }
}

pub(crate) fn separate(out: &mut String, index: usize, interval: usize) {
    match index {
        0 => {}
        index if interval != 0 && index % interval == 0 => write!(out, ", /*index={index}*/").unwrap(),
        _ => out.push_str(", "),
    }
}

fn dim_labels(numbers: &xla::ConvolutionDimensionNumbers) -> String {
    let labels = |batch: i64, feature: i64, spatial: &[i64], letters: [&str; 2]| {
        let length = (batch.max(feature).max(spatial.iter().copied().max().unwrap_or(0)) + 1).clamp(0, MAX_RANK as i64);
        let mut out = vec!["?".to_string(); length as usize];
        let labels = [(batch, letters[0].to_string()), (feature, letters[1].to_string())].into_iter().chain(spatial.iter().enumerate().map(|(index, &dimension)| (dimension, index.to_string())));
        for (dimension, label) in labels {
            if let Some(slot) = usize::try_from(dimension).ok().and_then(|dimension| out.get_mut(dimension)) {
                *slot = label;
            }
        }
        out.concat()
    };
    format!(
        "{}_{}->{}",
        labels(numbers.input_batch_dimension, numbers.input_feature_dimension, &numbers.input_spatial_dimensions, ["b", "f"]),
        labels(numbers.kernel_input_feature_dimension, numbers.kernel_output_feature_dimension, &numbers.kernel_spatial_dimensions, ["i", "o"]),
        labels(numbers.output_batch_dimension, numbers.output_feature_dimension, &numbers.output_spatial_dimensions, ["b", "f"])
    )
}

fn dot_numbers(numbers: &xla::DotDimensionNumbers) -> String {
    let mut parts = Vec::new();
    for (side, batch, contracting) in [("lhs", &numbers.lhs_batch_dimensions, &numbers.lhs_contracting_dimensions), ("rhs", &numbers.rhs_batch_dimensions, &numbers.rhs_contracting_dimensions)] {
        if !batch.is_empty() {
            parts.push(format!("{side}_batch_dims={{{}}}", batch.iter().join(",")));
        }
        parts.push(format!("{side}_contracting_dims={{{}}}", contracting.iter().join(",")));
    }
    parts.join(", ")
}

fn dimension_numbers(names: [&str; 5], lists: [&Vec<i64>; 5], index_vector_dim: i64) -> String {
    let mut parts: Vec<String> = names
        .iter()
        .zip(lists)
        .enumerate()
        .filter(|(position, (_, values))| *position < 3 || !values.is_empty())
        .map(|(_, (name, values))| format!("{name}={{{}}}", values.iter().join(",")))
        .collect();
    parts.push(format!("index_vector_dim={index_vector_dim}"));
    parts.join(", ")
}

fn flag(attributes: &mut Vec<String>, field: &str, value: bool) {
    if value {
        attributes.push(format!("{field}=true"));
    }
}

fn precision(config: &PrecisionConfig, attributes: &mut Vec<String>) {
    if config.operand_precision.iter().any(|precision| *precision != 0) {
        let names = config.operand_precision.iter().map(|&precision| enum_name("xla.PrecisionConfig.Precision", precision).to_ascii_lowercase()).join(",");
        attributes.push(format!("operand_precision={{{names}}}"));
    }
    if config.algorithm != 0 {
        attributes.push(format!("algorithm={}", enum_name("xla.PrecisionConfig.Algorithm", config.algorithm).trim_start_matches("ALG_").to_ascii_lowercase()));
    }
}

fn aliasing(inst: &Inst, attributes: &mut Vec<String>) {
    let entries = &inst.output_operand_aliasing;
    if !entries.is_empty() {
        let items = entries.iter().map(|entry| format!("{{{}}}: ({}, {{{}}})", entry.output_shape_index.iter().join(","), entry.operand_index, entry.operand_shape_index.iter().join(","))).join(", ");
        attributes.push(format!("output_to_operand_aliasing={{{items}}}"));
    }
}

fn group_counts(inst: &Inst, attributes: &mut Vec<String>) {
    for (field, count) in [("feature_group_count", inst.feature_group_count), ("batch_group_count", inst.batch_group_count)] {
        if count > 1 {
            attributes.push(format!("{field}={count}"));
        }
    }
}

fn window(inst: &Inst) -> String {
    let dimensions: Vec<&WindowDimension> = inst.window.iter().flat_map(|window| &window.dimensions).collect();
    let fields = WINDOW_FIELDS.iter().filter(|(_, shown, _)| dimensions.iter().any(|dimension| shown(dimension)));
    format!("window={{{}}}", fields.map(|(heading, _, text)| format!("{heading}={}", dimensions.iter().map(|dimension| text(dimension)).join("x"))).join(" "))
}

fn mesh_text(mesh: &xla::MeshProto) -> String {
    let devices = match (&mesh.iota_transform, mesh.device_ids.is_empty()) {
        (_, false) => Some(mesh.device_ids.iter().join(",")),
        (Some(iota), true) => {
            let (mut dims, mut perm) = (iota.reshape_dims.clone(), iota.transpose_perm.clone());
            canonicalize_iota(&mut dims, &mut perm);
            (dims.len() > 1).then(|| format!("[{}]T({})", dims.iter().join(","), perm.iter().join(",")))
        }
        (None, true) => None,
    };
    let names = mesh.axes.iter().map(|axis| format!("'{}'={}", axis.name, axis.size)).join(",");
    format!("mesh[{names}]{}", devices.map_or_else(String::new, |devices| format!(", device_ids=({devices})")))
}

fn mesh_groups(list: &MeshAxesReplicaGroupListProto) -> String {
    let mesh = msg(&list.mesh);
    let declared = if mesh.axes.is_empty() { format!("maximal_mesh[device_id={}]", mesh.device_ids.first().copied().unwrap_or(0)) } else { mesh_text(&mesh) };
    let axis_text = |axis: &xla::AxisRefProto| {
        let index = axis.mesh_axis_index;
        let mut text = usize::try_from(index).ok().and_then(|index| mesh.axes.get(index)).map_or_else(|| index.to_string(), |axis| format!("'{}'", axis.name));
        if let Some(sub) = &axis.sub_axis_info {
            write!(text, ":({}){}", sub.pre_size, sub.size).unwrap();
        }
        text
    };
    format!("{declared} {{{}}}", list.axes.iter().map(axis_text).join(","))
}

#[allow(deprecated)]
fn replica_groups(inst: &Inst) -> String {
    match &inst.replica_group_list {
        Some(ReplicaGroupList::IotaCollectiveDeviceList(list)) => {
            let dims = [list.num_replica_groups, list.num_devices_per_group];
            if list.iota_reshape_dims.is_empty() {
                iota_text(&dims, &[dims[0] * dims[1]], &[0])
            } else {
                iota_text(&dims, &list.iota_reshape_dims, &list.iota_transpose_perm.iter().map(|&value| i64::from(value)).collect::<Vec<_>>())
            }
        }
        Some(ReplicaGroupList::MeshAxesReplicaGroupList(list)) => mesh_groups(list),
        list => {
            let groups = match (inst.replica_groups.is_empty(), list) {
                (true, Some(ReplicaGroupList::CollectiveDeviceList(list))) => &list.replica_groups,
                _ => &inst.replica_groups,
            };
            format!("{{{}}}", groups.iter().map(|group| format!("{{{}}}", group.replica_ids.iter().join(","))).join(","))
        }
    }
}

#[allow(deprecated)]
fn collective(opcode: &str, inst: &Inst, channel_id: Option<i64>, first_dimension: String, attributes: &mut Vec<String>) {
    let reduce = matches!(opcode, "all-reduce" | "all-reduce-start" | "reduce-scatter" | "collective-reduce");
    let gather = matches!(opcode, "all-gather" | "all-gather-start");
    let channel = if reduce && inst.all_reduce_id > 0 { Some(inst.all_reduce_id) } else { channel_id };
    attributes.extend(channel.map(|channel| format!("channel_id={channel}")));
    attributes.push(format!("replica_groups={}", replica_groups(inst)));
    flag(attributes, "constrain_layout", inst.constrain_layout && opcode != "collective-broadcast");
    if gather {
        attributes.push(first_dimension.clone());
    }
    flag(attributes, "use_global_device_ids", (reduce || gather) && inst.use_global_device_ids);
    match opcode {
        "reduce-scatter" => attributes.push(first_dimension),
        "collective-reduce" | "collective-broadcast" => attributes.push(format!("has_dynamic_root={}", inst.has_dynamic_root)),
        "all-to-all" if !inst.dimensions.is_empty() => attributes.push(first_dimension),
        _ => {}
    }
}

impl<'a> Printer<'a> {
    pub fn new(module: &'a Module<'a>, style: Style, metadata: bool) -> Self {
        let async_graphs = module.nodes.iter().filter(|node| node.opcode == "async-start").flat_map(|node| node.called.iter().copied()).collect();
        let frames = if metadata { Frames::new(module) } else { Frames { of: vec![0; module.nodes.len()], ..Frames::default() } };
        Printer { module, style, metadata, operand_shapes: style == Style::Expression, large_constants: style == Style::Long, async_graphs, failed: Mutex::new(None), frames }
    }

    pub fn fail(&self, reason: &str) {
        self.failed.lock().unwrap().get_or_insert_with(|| reason.to_string());
    }

    pub fn unsupported(&self) -> bool {
        let failed = self.failed.lock().unwrap();
        if let Some(reason) = failed.as_ref() {
            eprintln!("xprof-rs cannot render this HLO module: {reason}");
        }
        failed.is_some()
    }

    fn name(&self, name: &str) -> String {
        if self.style == Style::Short { name.to_string() } else { format!("%{name}") }
    }

    fn graph_name(&self, graph: usize) -> String {
        self.name(&self.module.graphs[graph].name)
    }

    fn can_expand(&self, graph: usize) -> bool {
        let graph = &self.module.graphs[graph];
        graph.nodes.iter().all(|&node| node == graph.root || self.module.nodes[node].opcode == "parameter")
    }

    pub fn wrapped(&self, node: usize) -> Option<usize> {
        let module = self.module;
        let mut current = node;
        for _ in 0..module.nodes.len() {
            let entry = &module.nodes[current];
            if let Some(&graph) = entry.called.first() {
                return Some(graph);
            }
            let &operand = entry.operands.first()?;
            if !matches!(module.nodes[operand].opcode.as_str(), "async-start" | "async-update") {
                return None;
            }
            current = operand;
        }
        None
    }

    fn sugar(&self, node: usize) -> Option<usize> {
        if !matches!(self.module.nodes[node].opcode.as_str(), "async-start" | "async-update" | "async-done") {
            return None;
        }
        self.wrapped(node).filter(|&graph| self.can_expand(graph))
    }

    fn opcode(&self, node: usize) -> String {
        let entry = &self.module.nodes[node];
        match self.sugar(node) {
            Some(graph) => format!("{}{}", self.module.nodes[self.module.graphs[graph].root].opcode, &entry.opcode[5..]),
            None => entry.opcode.clone(),
        }
    }

    fn named_sharding(&self, sharding: &xla::NamedShardingProto) -> String {
        let mesh = msg(&sharding.mesh);
        let axis_size = |index: i64| usize::try_from(index).ok().and_then(|index| mesh.axes.get(index)).map_or(1, |axis| axis.size);
        let axis = |axis: &xla::AxisRefProto| {
            let name = usize::try_from(axis.mesh_axis_index).ok().and_then(|index| mesh.axes.get(index)).map_or("", |axis| axis.name.as_str());
            let sub = axis.sub_axis_info.as_ref().map_or_else(String::new, |sub| format!(":({}){}", sub.pre_size, sub.size));
            format!("'{name}'{sub}")
        };
        let axes = |list: &[xla::AxisRefProto]| list.iter().map(axis).join(", ");
        let in_order = |list: &[xla::AxisRefProto]| {
            let mut position = 0;
            for index in 0..mesh.axes.len() as i64 {
                match list.get(position) {
                    Some(axis) if axis.mesh_axis_index == index && axis.sub_axis_info.is_some() => return false,
                    Some(axis) if axis.mesh_axis_index == index => position += 1,
                    _ if axis_size(index) != 1 => return false,
                    _ => {}
                }
            }
            position == list.len()
        };
        let metadata = if self.metadata && !sharding.metadata.is_empty() {
            format!(", metadata={{{}}}", sharding.metadata.iter().map(|entry| format!("{{{}}}", metadata_text(entry, i64::from(entry.stack_frame_id), &self.module.proto.payloads))).join(", "))
        } else {
            String::new()
        };
        let maximal = mesh.axes.is_empty() && mesh.device_ids.len() == 1;
        let mut text = if maximal { format!("{{maximal_mesh[device_id={}]", mesh.device_ids[0]) } else { format!("{{{}", mesh_text(&mesh)) };
        let no_dimensions = sharding.dim_shardings.is_empty();
        let reduction = ["", "max", "min"].get(sharding.reduction_op as usize).copied().unwrap_or("");
        if maximal {
            return text + &metadata + "}";
        }
        if sharding.unreduced_axes.is_empty() && sharding.manual_axes.is_empty() && no_dimensions && sharding.replicated_axes.is_empty() {
            text.push_str(", replicated");
        } else if !sharding.unreduced_axes.is_empty() && in_order(&sharding.unreduced_axes) && no_dimensions {
            text.push_str(", unreduced");
            if !reduction.is_empty() {
                write!(text, "={reduction}").unwrap();
            }
            if sharding.unreduced_axes.len() < mesh.axes.len() {
                write!(text, "{{{}}}", axes(&sharding.unreduced_axes)).unwrap();
            }
        } else if !sharding.manual_axes.is_empty() && in_order(&sharding.manual_axes) && no_dimensions {
            text.push_str(", manual");
        } else {
            let dimensions = sharding.dim_shardings.iter().map(|dimension| {
                let open = match (dimension.is_closed, dimension.axes.is_empty()) {
                    (true, _) => "",
                    (false, true) => "?",
                    (false, false) => ", ?",
                };
                format!("{{{}{open}}}", axes(&dimension.axes))
            });
            write!(text, ", [{}]", dimensions.format(", ")).unwrap();
            if !sharding.replicated_axes.is_empty() {
                write!(text, ", replicated={{{}}}", axes(&sharding.replicated_axes)).unwrap();
            }
            if !sharding.unreduced_axes.is_empty() {
                write!(text, ", unreduced={reduction}{{{}}}", axes(&sharding.unreduced_axes)).unwrap();
            }
            if !sharding.manual_axes.is_empty() {
                write!(text, ", manual={{{}}}", axes(&sharding.manual_axes)).unwrap();
            }
        }
        text + &metadata + "}"
    }

    fn sharding(&self, sharding: &OpSharding, out: &mut String) {
        if let Some(named) = &sharding.named_sharding {
            out.push_str(&self.named_sharding(named));
            return;
        }
        if sharding.metadata.iter().any(|entry| entry.op_type == "sdy::reduction_op") {
            self.fail("sharding reduction metadata");
            return;
        }
        let mut suffix = String::new();
        if sharding.is_shard_group {
            let kind = if sharding.shard_group_type == 0 { "shard_as" } else { "shard_like" };
            write!(suffix, " {kind} {}", sharding.shard_group_id).unwrap();
        }
        if self.metadata && !sharding.metadata.is_empty() {
            let items = sharding.metadata.iter().map(|entry| metadata_text(entry, i64::from(entry.stack_frame_id), &self.module.proto.payloads)).join("}, {");
            match sharding.metadata.len() {
                1 => write!(suffix, " metadata={{{items}}}").unwrap(),
                _ => write!(suffix, " metadata={{{{{items}}}}}").unwrap(),
            }
        }
        let (devices, reshape, dims, subgroups) = (&sharding.tile_assignment_devices, &sharding.iota_reshape_dims, &sharding.tile_assignment_dimensions, &sharding.last_tile_dims);
        let replicated = (!reshape.is_empty() && reshape.iter().all(|&dim| dim == 1)) || (reshape.is_empty() && devices.len() == 1);
        let subgroup = !subgroups.is_empty()
            && subgroups.iter().all(|&kind| kind == subgroups[0])
            && dims[..dims.len().saturating_sub(subgroups.len())].iter().product::<i64>() == 1
            && matches!(subgroups[0], 0 | 4 | 6);
        match sharding.r#type {
            2 => {
                out.push('{');
                for (index, element) in sharding.tuple_shardings.iter().enumerate() {
                    separate(out, index, INDEX_INTERVAL);
                    self.sharding(element, out);
                }
                out.push('}');
            }
            1 => write!(out, "{{maximal device={}{suffix}}}", devices.first().copied().unwrap_or(0)).unwrap(),
            kind @ (0 | 4 | 5 | 6) => write!(out, "{{{}{suffix}}}", ["replicated", "", "", "", "manual", "unknown", "unreduced"][kind as usize]).unwrap(),
            _ if replicated => write!(out, "{{replicated{suffix}}}").unwrap(),
            _ if subgroup => write!(out, "{{{}{suffix}}}", SUBGROUPS[subgroups[0] as usize]).unwrap(),
            _ => {
                out.push_str("{devices=");
                if reshape.is_empty() {
                    write!(out, "[{}]{}", dims.iter().join(","), devices.iter().join(",")).unwrap();
                } else {
                    out.push_str(&iota_text(dims, reshape, &sharding.iota_transpose_perm.iter().map(|&value| i64::from(value)).collect::<Vec<_>>()));
                }
                if sharding.replicate_on_last_tile_dim {
                    out.push_str(" last_tile_dim_replicate");
                }
                if !subgroups.is_empty() {
                    let names = subgroups.iter().map(|&kind| SUBGROUPS.get(kind as usize).copied().unwrap_or("error_type.")).join(", ");
                    write!(out, " last_tile_dims={{{names}}}").unwrap();
                }
                write!(out, "{suffix}}}").unwrap();
            }
        }
    }

    pub fn literal(&self, literal: &LiteralProto, oneline: bool, out: &mut String) {
        let mut shape = literal.shape.clone().unwrap_or_default();
        shape.normalize();
        if shape.is_tuple() {
            out.push_str(if oneline { "( " } else { "(\n" });
            for (index, element) in literal.tuple_literals.iter().enumerate() {
                if index > 0 {
                    out.push_str(if oneline { ", " } else { ",\n" });
                }
                self.literal(element, oneline, out);
            }
            out.push_str(if oneline { " )" } else { "\n)" });
            return;
        }
        if shape.element_type == 17 {
            out.push_str("token");
            return;
        }
        if !shape.is_array() || shape.is_dynamic_dimension.contains(&true) {
            self.fail("dynamic or non-array literal");
            return;
        }
        let Some(values) = literal_values(literal, shape.element_type) else {
            self.fail("literal element type");
            return;
        };
        let rank = shape.dimensions.len();
        let prefixes: Vec<Option<i64>> = shape
            .dimensions
            .iter()
            .scan(Some(1i64), |product, &dimension| {
                *product = product.and_then(|product| product.checked_mul(dimension)).filter(|_| dimension >= 0);
                Some(*product)
            })
            .collect();
        let bounded = prefixes.iter().all(|prefix| prefix.is_some_and(|prefix| prefix <= values.len().max(MAX_EMPTY_ARRAYS) as i64));
        if rank > MAX_RANK || !bounded || prefixes.last().copied().unwrap_or(Some(1)) != Some(values.len() as i64) {
            self.fail("literal size");
            return;
        }
        let minor_to_major = match &shape.layout {
            Some(layout) if layout.minor_to_major.len() == rank && (0..rank as i64).all(|dimension| layout.minor_to_major.contains(&dimension)) => layout.minor_to_major.clone(),
            _ => (0..rank as i64).rev().collect(),
        };
        let mut strides = vec![1i64; rank];
        let mut stride = 1;
        for &dimension in &minor_to_major {
            strides[dimension as usize] = stride;
            stride *= shape.dimensions[dimension as usize];
        }
        print_array(out, &shape.dimensions, oneline, &mut Vec::new(), &|indices: &[i64]| {
            let linear: i64 = indices.iter().zip(&strides).map(|(index, stride)| index * stride).sum();
            let value = &values[linear as usize];
            if shape.element_type == 1 && rank > 0 { if value == "true" { "1".to_string() } else { "0".to_string() } } else { value.clone() }
        });
    }

    fn literal_with_shapes(&self, literal: &LiteralProto) -> String {
        let shape = msg(&literal.shape);
        if shape.is_tuple() {
            return format!("( {} )", literal.tuple_literals.iter().map(|element| self.literal_with_shapes(element)).join(", "));
        }
        let mut text = format!("{} ", shape.text(true));
        self.literal(literal, true, &mut text);
        text
    }

    fn operands(&self, node: usize, inst: &Inst, out: &mut String) {
        let module = self.module;
        let entry = &module.nodes[node];
        match entry.opcode.as_str() {
            "constant" => match &inst.literal {
                Some(literal) if self.large_constants || (entry.shape.is_array() && entry.shape.elements() <= MAX_COMPACT_ELEMENTS) => self.literal(literal, true, out),
                _ => out.push_str("{...}"),
            },
            "parameter" => write!(out, "{}", entry.parameter).unwrap(),
            _ => {
                for (index, &operand) in entry.operands.iter().enumerate() {
                    separate(out, index, if self.style == Style::Short { 0 } else { INDEX_INTERVAL });
                    if self.operand_shapes {
                        module.nodes[operand].shape.print(out, true);
                        out.push(' ');
                    }
                    out.push_str(&self.name(&module.nodes[operand].name));
                }
            }
        }
    }

    #[allow(deprecated)]
    fn implementation(&self, node: usize, inst: &Inst, attributes: &mut Vec<String>) {
        let module = self.module;
        let entry = &module.nodes[node];
        let opcode = entry.opcode.as_str();
        let channel_id = (inst.channel_id > 0).then_some(inst.channel_id);
        let dimensions = &inst.dimensions;
        let first_dimension = format!("dimensions={{{}}}", dimensions.first().copied().unwrap_or(0));
        let has_window = inst.window.as_ref().is_some_and(|window| !window.dimensions.is_empty());
        match opcode {
            "fft" => attributes.extend([format!("fft_type={}", enum_name("xla.FftType", inst.fft_type)), format!("fft_length={{{}}}", inst.fft_length.iter().join(","))]),
            "async-start" => {
                if !inst.async_execution_thread.is_empty() && inst.async_execution_thread != "main" {
                    attributes.push(format!("async_execution_thread=\"{}\"", inst.async_execution_thread));
                }
                if let Some(graph) = self.sugar(node) {
                    let root = module.graphs[graph].root;
                    self.extra(root, &module.inst(root), msg(&inst.frontend_attributes).map.is_empty(), attributes);
                }
                aliasing(inst, attributes);
            }
            "async-update" | "call" => aliasing(inst, attributes),
            "copy-start" => match inst.optional_cross_program_prefetch_index {
                Some(xla::hlo_instruction_proto::OptionalCrossProgramPrefetchIndex::CrossProgramPrefetchIndex(index)) => attributes.push(format!("cross_program_prefetch_index={index}")),
                None if inst.is_cross_program_prefetch => attributes.push("cross_program_prefetch_index=0".into()),
                None => {}
            },
            "compare" => {
                attributes.push(format!("direction={}", inst.comparison_direction));
                let default = match entry.operands.first().map_or(0, |&operand| module.nodes[operand].shape.element_type) {
                    10..=12 | 15 | 16 | 18..=20 | 23..=25 | 28 | 29 | 32 | 33 | 35 | 36 => "FLOAT",
                    2..=5 | 21 | 26 | 30 => "SIGNED",
                    _ => "UNSIGNED",
                };
                if !inst.comparison_type.is_empty() && inst.comparison_type != default {
                    attributes.push(format!("type={}", inst.comparison_type));
                }
            }
            "topk" => attributes.extend([format!("k={}", inst.k), format!("largest={}", inst.largest), format!("is_stable={}", inst.is_stable)]),
            opcode if CHANNEL_OPS.contains(&opcode) => {
                attributes.extend(channel_id.map(|channel| format!("channel_id={channel}")));
                flag(attributes, "is_host_transfer", inst.is_host_transfer);
            }
            opcode if COLLECTIVES.contains(&opcode) => collective(opcode, inst, channel_id, first_dimension, attributes),
            "collective-permute" | "collective-permute-start" => {
                attributes.extend(channel_id.map(|channel| format!("channel_id={channel}")));
                let pairs = inst.source_target_pairs.iter().map(|pair| format!("{{{},{}}}", pair.source, pair.target)).join(",");
                attributes.push(format!("source_target_pairs={{{pairs}}}"));
                if !inst.dynamic_slice_sizes.is_empty() {
                    match slice_size_groups(module, &entry.operands, &inst.dynamic_slice_sizes) {
                        Some(groups) => attributes.push(format!("slice_sizes={{{}}}", groups.iter().map(|group| format!("{{{}}}", group.iter().join(","))).join(","))),
                        None => self.fail("dynamic collective-permute"),
                    }
                }
            }
            "reverse" | "concatenate" | "reduce" | "transpose" | "broadcast" | "map" => attributes.push(format!("dimensions={{{}}}", dimensions.iter().join(","))),
            "scan" => {
                attributes.extend([format!("dimensions={{{}}}", dimensions.iter().join(",")), format!("num_carries={}", inst.num_carries)]);
                flag(attributes, "is_reverse", inst.is_reverse);
                if inst.is_associative != 0 {
                    attributes.push(format!("is_associative={}", inst.is_associative == 1));
                }
            }
            "sort" => {
                attributes.push(format!("dimensions={{{}}}", dimensions.iter().join(",")));
                flag(attributes, "is_stable", inst.is_stable);
            }
            "reshape" if !dimensions.is_empty() && dimensions[0] != -1 => attributes.push(format!("inferred_dimension={}", dimensions[0])),
            "slice" => {
                let omit_stride = inst.slice_dimensions.iter().all(|slice| slice.stride == 1);
                let stride = |slice: &xla::hlo_instruction_proto::SliceDimensions| if omit_stride { String::new() } else { format!(":{}", slice.stride) };
                let items = inst.slice_dimensions.iter().map(|slice| format!("[{}:{}{}]", slice.start, slice.limit, stride(slice))).join(", ");
                attributes.push(format!("slice={{{items}}}"));
            }
            "fusion" => {
                attributes.push(format!("kind={}", inst.fusion_kind));
                aliasing(inst, attributes);
            }
            "rng" => attributes.push(format!("distribution={}", enum_name("xla.RandomDistribution", inst.distribution).to_ascii_lowercase())),
            "parameter" => {
                let replication = &msg(&inst.parameter_replication).replicated_at_leaf_buffers;
                if !replication.is_empty() {
                    attributes.push(format!("parameter_replication={{{}}}", replication.iter().join(",")));
                }
            }
            "get-tuple-element" => attributes.push(format!("index={}", inst.tuple_index)),
            "reduce-precision" => attributes.extend([format!("exponent_bits={}", inst.exponent_bits), format!("mantissa_bits={}", inst.mantissa_bits)]),
            "convolution" => Self::convolution(inst, has_window, attributes),
            "reduce-window" | "select-and-scatter" if has_window => attributes.push(window(inst)),
            "custom-call" => self.custom_call(inst, attributes),
            "pad" => {
                let config = msg(&inst.padding_config);
                let interior = config.dimensions.iter().any(|dim| dim.interior_padding != 0);
                let suffix = |dim: &xla::padding_config::PaddingConfigDimension| if interior { format!("_{}", dim.interior_padding) } else { String::new() };
                attributes.push(format!("padding={}", config.dimensions.iter().map(|dim| format!("{}_{}{}", dim.edge_padding_low, dim.edge_padding_high, suffix(dim))).join("x")));
            }
            "dynamic-slice" => attributes.push(format!("dynamic_slice_sizes={{{}}}", inst.dynamic_slice_sizes.iter().join(","))),
            "gather" => {
                let numbers = msg(&inst.gather_dimension_numbers);
                let lists = [&numbers.offset_dims, &numbers.collapsed_slice_dims, &numbers.start_index_map, &numbers.operand_batching_dims, &numbers.start_indices_batching_dims];
                attributes.push(dimension_numbers(GATHER, lists, numbers.index_vector_dim));
                attributes.push(format!("slice_sizes={{{}}}", inst.gather_slice_sizes.iter().join(",")));
                flag(attributes, "indices_are_sorted", inst.indices_are_sorted);
            }
            "scatter" => {
                let numbers = msg(&inst.scatter_dimension_numbers);
                let lists = [&numbers.update_window_dims, &numbers.inserted_window_dims, &numbers.scatter_dims_to_operand_dims, &numbers.input_batching_dims, &numbers.scatter_indices_batching_dims];
                attributes.push(dimension_numbers(SCATTER, lists, numbers.index_vector_dim));
                flag(attributes, "indices_are_sorted", inst.indices_are_sorted);
                flag(attributes, "unique_indices", inst.unique_indices);
            }
            "iota" => attributes.push(format!("iota_dimension={}", dimensions.first().copied().unwrap_or(0))),
            "dot" | "scaled-dot" => {
                attributes.push(dot_numbers(&msg(&inst.dot_dimension_numbers)));
                precision(&msg(&inst.precision_config), attributes);
            }
            "ragged-dot" => {
                let numbers = msg(&inst.ragged_dot_dimension_numbers);
                let mut parts = vec![dot_numbers(&msg(&numbers.dot_dimension_numbers))];
                for (label, values) in [("lhs_ragged_dims", &numbers.lhs_ragged_dimensions), ("rhs_group_dims", &numbers.rhs_group_dimensions)] {
                    if !values.is_empty() {
                        parts.push(format!("{label}={{{}}}", values.iter().join(",")));
                    }
                }
                attributes.push(parts.join(", "));
                precision(&msg(&inst.precision_config), attributes);
            }
            "batch-norm-training" | "batch-norm-inference" | "batch-norm-grad" => {
                attributes.extend([format!("epsilon={}", general(f64::from(inst.epsilon), 6)), format!("feature_index={}", inst.feature_index)]);
            }
            "triangular-solve" => {
                let options = msg(&inst.triangular_solve_options);
                flag(attributes, "left_side", options.left_side);
                flag(attributes, "lower", options.lower);
                flag(attributes, "unit_diagonal", options.unit_diagonal);
                if options.transpose_a != 0 {
                    attributes.push(format!("transpose_a={}", enum_name("xla.TriangularSolveOptions.Transpose", options.transpose_a)));
                }
            }
            "cholesky" => flag(attributes, "lower", msg(&inst.cholesky_options).lower),
            "infeed" if !inst.infeed_config.is_empty() => attributes.push(format!("infeed_config=\"{}\"", escaped(&crate::xplane::lossy(&inst.infeed_config)))),
            "outfeed" => {
                attributes.push(format!("outfeed_shape={}", msg(&inst.outfeed_shape).text(true)));
                if !inst.outfeed_config.is_empty() {
                    attributes.push(format!("outfeed_config=\"{}\"", escaped(&crate::xplane::lossy(&inst.outfeed_config))));
                }
            }
            "domain" => {
                let (mut entry, mut exit) = (String::new(), String::new());
                self.sharding(&msg(&inst.domain_exit_sharding), &mut entry);
                self.sharding(&msg(&inst.domain_entry_sharding), &mut exit);
                attributes.push(format!("domain={{kind=\"sharding\", entry={entry}, exit={exit}}}"));
            }
            "get-dimension-size" | "set-dimension-size" => attributes.push(first_dimension),
            "rng-get-and-update-state" => attributes.push(format!("delta={}", inst.delta)),
            "rng-bit-generator" => attributes.push(format!("algorithm={}", enum_name("xla.RandomAlgorithm", inst.rng_algorithm).to_ascii_lowercase())),
            _ => {}
        }
    }

    fn convolution(inst: &Inst, has_window: bool, attributes: &mut Vec<String>) {
        if has_window {
            attributes.push(window(inst));
        }
        attributes.push(format!("dim_labels={}", dim_labels(&msg(&inst.convolution_dimension_numbers))));
        group_counts(inst, attributes);
        match inst.conv_kind {
            0 => {}
            kind @ 1..=3 => attributes.push(format!("convolution_kind={}", ["fprop", "dgrad", "wgrad"][kind as usize - 1])),
            kind => attributes.push(format!("convolution_kind=unknown({kind})")),
        }
        precision(&msg(&inst.precision_config), attributes);
        if let Some(config) = inst.sparsity_config.as_ref().filter(|config| config.lhs.is_some() || config.rhs.is_some()) {
            attributes.push(format!("sparsity_config={{{}}}", sparsity_config_text(config)));
        }
        if let Some(config) = inst.block_scaling_config.as_ref().filter(|config| config.lhs.is_some() || config.rhs.is_some()) {
            attributes.push(format!("block_scaling_config={{{}}}", block_scaling_config_text(config)));
        }
    }

    fn custom_call(&self, inst: &Inst, attributes: &mut Vec<String>) {
        if inst.window.is_some() {
            attributes.push(window(inst));
        }
        if let Some(numbers) = &inst.convolution_dimension_numbers {
            attributes.push(format!("dim_labels={}", dim_labels(numbers)));
        }
        group_counts(inst, attributes);
        precision(&msg(&inst.precision_config), attributes);
        if inst.padding_type != 0 {
            attributes.push(format!("padding_type={}", enum_name("xla.PaddingType", inst.padding_type)));
        }
        attributes.push(format!("custom_call_target=\"{}\"", escaped(&inst.custom_call_target)));
        if inst.constrain_layout {
            attributes.push(format!("operand_layout_constraints={{{}}}", inst.operand_shapes_with_layout.iter().map(|shape| shape.text(true)).join(", ")));
        }
        flag(attributes, "custom_call_has_side_effect", inst.custom_call_has_side_effect);
        if let Some(literal) = &inst.literal {
            attributes.push(format!("literal={}", self.literal_with_shapes(literal)));
        }
        aliasing(inst, attributes);
        if inst.custom_call_schedule != 0 {
            attributes.push(format!("schedule={}", enum_name("xla.CustomCallSchedule", inst.custom_call_schedule)));
        }
        if inst.custom_call_api_version != 1 {
            attributes.push(format!("api_version={}", enum_name("xla.CustomCallApiVersion", inst.custom_call_api_version)));
        }
    }

    pub fn extra(&self, node: usize, inst: &Inst, print_frontend: bool, attributes: &mut Vec<String>) {
        let module = self.module;
        let entry = &module.nodes[node];
        self.implementation(node, inst, attributes);
        let names = |graphs: &[usize]| graphs.iter().map(|&graph| self.graph_name(graph)).join(", ");
        let called: &[usize] = if self.style == Style::Graph { &[] } else { &entry.called };
        match entry.opcode.as_str() {
            "while" if called.len() == 2 => attributes.extend([format!("condition={}", self.graph_name(called[1])), format!("body={}", self.graph_name(called[0]))]),
            "select-and-scatter" if called.len() == 2 => attributes.extend([format!("select={}", self.graph_name(called[0])), format!("scatter={}", self.graph_name(called[1]))]),
            "conditional" if self.style != Style::Graph => {
                if entry.operands.first().map(|&operand| module.nodes[operand].shape.element_type) == Some(1) && called.len() == 2 {
                    attributes.extend([format!("true_computation={}", self.graph_name(called[0])), format!("false_computation={}", self.graph_name(called[1]))]);
                } else {
                    attributes.push(format!("branch_computations={{{}}}", names(called)));
                }
            }
            opcode if CALLS_TO_APPLY.contains(&opcode) => {
                if let Some(&graph) = called.first() {
                    attributes.push(format!("to_apply={}", self.graph_name(graph)));
                }
                if opcode == "call" && inst.is_composite && self.style != Style::Graph {
                    attributes.push("is_composite=true".into());
                }
            }
            "custom-call" if !called.is_empty() => attributes.push(format!("called_computations={{{}}}", names(called))),
            "async-start" if self.sugar(node).is_none() && self.style != Style::Graph => {
                if let Some(graph) = self.wrapped(node) {
                    attributes.push(format!("calls={}", self.graph_name(graph)));
                }
            }
            "custom-call" | "async-start" | "async-update" | "async-done" => {}
            _ if !called.is_empty() => attributes.push(format!("calls={}", names(called))),
            _ => {}
        }
        if let Some(sharding) = &inst.sharding {
            let mut text = String::from("sharding=");
            if entry.shape.is_tuple() && sharding.r#type != 2 {
                text.push('{');
                for index in 0..leaf_count(&entry.shape).max(1) {
                    separate(&mut text, index, INDEX_INTERVAL);
                    self.sharding(sharding, &mut text);
                }
                text.push('}');
            } else {
                self.sharding(sharding, &mut text);
            }
            attributes.push(text);
        }
        let frontend = &msg(&inst.frontend_attributes).map;
        if print_frontend && !frontend.is_empty() {
            attributes.push(format!("frontend_attributes={}", frontend_attributes(frontend)));
        }
        if self.style != Style::Short && !entry.predecessors.is_empty() {
            let items = entry.predecessors.iter().map(|&other| self.name(&module.nodes[other].name)).join(", ");
            attributes.push(format!("control-predecessors={{{items}}}"));
        }
        if let Some(viz) = inst.statistics_viz.as_ref().filter(|viz| !viz.statistics.is_empty()) {
            let statistics = viz.statistics.iter().map(|statistic| format!("{}={}", statistic.stat_name, general(statistic.stat_val, 6))).join(",");
            attributes.push(format!("statistics={{visualizing_index={},{statistics}}}", viz.stat_index_to_visualize));
        }
        match msg(&inst.result_accuracy).specs {
            Some(xla::result_accuracy::Specs::Tolerance(tolerance)) => {
                attributes.push(format!("result_accuracy={{tolerance={{atol={},rtol={},ulps={}}}}}", general(tolerance.atol, 6), general(tolerance.rtol, 6), tolerance.ulps));
            }
            Some(xla::result_accuracy::Specs::Mode(mode)) if mode != 0 => attributes.push(format!("result_accuracy={{mode={}}}", if mode == 1 { "highest" } else { "unknown" })),
            _ => {}
        }
    }

    /// Writes the text of an instruction, and gives the instruction that it decoded.
    pub fn instruction(&self, node: usize, out: &mut String) -> Inst {
        let module = self.module;
        let entry = &module.nodes[node];
        write!(out, "{} = {} {}(", self.name(&entry.name), entry.shape.text(true), self.opcode(node)).unwrap();
        let inst = module.inst(node);
        self.operands(node, &inst, out);
        out.push(')');
        let mut attributes = Vec::new();
        self.extra(node, &inst, true, &mut attributes);
        attributes.iter().for_each(|attribute| write!(out, ", {attribute}").unwrap());
        if let Some(original) = &inst.original_value {
            write!(out, ", origin={{{}}}", original_value(original, &entry.shape, &mut Vec::new())).unwrap();
        }
        let metadata = msg(&inst.metadata);
        let frame = self.frames.of[node];
        if self.metadata
            && ([&metadata.op_type, &metadata.op_name, &metadata.source_file, &metadata.scheduling_name].iter().any(|field| !field.is_empty()) || frame != 0 || metadata.metadata_payload.is_some())
        {
            write!(out, ", metadata={{{}}}", metadata_text(&metadata, frame, &module.proto.payloads)).unwrap();
        }
        if self.style == Style::Expression {
            return inst;
        }
        match module.backend_config(&inst) {
            None => self.fail("backend config payload id"),
            Some(config) if !config.is_empty() => {
                out.push_str(", backend_config=");
                let text = crate::xplane::lossy(&config);
                if lexes_as_json_dict(&text) {
                    out.push_str(&text);
                } else {
                    out.push('"');
                    c_escape(out, &config);
                    out.push('"');
                }
            }
            Some(_) => {}
        }
        inst
    }

    fn computation(&self, graph: usize, schedule: &HashMap<usize, Vec<usize>>, out: &mut String) {
        let module = self.module;
        let entry = &module.graphs[graph];
        if graph == module.entry {
            out.push_str("ENTRY ");
        }
        write!(out, "{} ", self.graph_name(graph)).unwrap();
        if self.style != Style::Short {
            out.push('(');
            for (index, &parameter) in entry.parameters.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                let name = &module.nodes[parameter].name;
                write!(out, "{}: {}", if name.is_empty() { "(unknown)" } else { name }, module.nodes[parameter].shape.text(false)).unwrap();
            }
            write!(out, ") -> {} ", module.nodes[entry.root].shape.text(false)).unwrap();
        }
        out.push_str("{\n");
        for node in schedule.get(&graph).cloned().unwrap_or_else(|| self.module.post_order(graph)) {
            out.push_str(if node == entry.root { "  ROOT " } else { "  " });
            self.instruction(node, out);
            out.push('\n');
        }
        out.push('}');
        if !entry.thread.is_empty() && entry.thread != "main" {
            write!(out, ", execution_thread=\"{}\"", entry.thread).unwrap();
        }
    }

    fn computation_order(&self) -> Vec<usize> {
        let module = self.module;
        let mut visited = vec![false; module.graphs.len()];
        module.nodes.iter().flat_map(|node| &node.called).for_each(|&graph| visited[graph] = true);
        let roots: Vec<usize> = (0..module.graphs.len()).filter(|&graph| !visited[graph]).collect();
        visited.fill(false);
        let (mut order, mut stack) = (Vec::new(), Vec::new());
        let mut visit = |stack: &mut Vec<(usize, usize)>, node: usize| {
            for &called in module.nodes[node].called.iter().rev() {
                if !visited[called] {
                    visited[called] = true;
                    stack.push((called, 0));
                }
            }
        };
        for root in roots {
            for &node in &module.graphs[root].nodes {
                visit(&mut stack, node);
                while let Some(&(current, position)) = stack.last() {
                    let nodes = &module.graphs[current].nodes;
                    if position == nodes.len() {
                        stack.pop();
                        order.push(current);
                        continue;
                    }
                    stack.last_mut().unwrap().1 += 1;
                    visit(&mut stack, nodes[position]);
                }
            }
            order.push(root);
        }
        order
    }

    pub fn module_text(&self) -> Option<String> {
        let module = self.module;
        let proto = &module.proto;
        let mut schedule = HashMap::new();
        for (key, sequence) in &msg(&proto.schedule).sequences {
            let Some(graph) = module.graphs.iter().position(|graph| graph.id == *key) else { continue };
            let ids: HashMap<i64, usize> = module.graphs[graph].nodes.iter().map(|&node| (module.nodes[node].id, node)).collect();
            let mut order = Vec::new();
            for id in &sequence.instruction_ids {
                match ids.get(id) {
                    Some(&node) => order.push(node),
                    None => self.fail("schedule"),
                }
            }
            schedule.insert(graph, order);
        }
        let mut out = format!("HloModule {}", proto.name);
        if proto.schedule.is_some() {
            out.push_str(", is_scheduled=true");
        }
        let mut aliases: IndexMap<Vec<i64>, String> = IndexMap::new();
        for alias in &msg(&proto.input_output_alias).entries {
            let kind = if alias.kind == 1 { "may-alias" } else { "must-alias" };
            let text = format!("{{{}}}: ({}, {{{}}}, {kind})", alias.output_shape_index.iter().join(","), alias.parameter_number, alias.parameter_shape_index.iter().join(","));
            aliases.insert(alias.output_shape_index.clone(), text);
        }
        if !aliases.is_empty() {
            let indices = all_subshapes(&module.nodes[module.graphs[module.entry].root].shape);
            let pieces: Vec<&String> = indices.iter().filter_map(|(index, _)| aliases.get(index)).collect();
            if pieces.len() != aliases.len() {
                return None;
            }
            write!(out, ", input_output_alias={{ {} }}", pieces.iter().join(", ")).unwrap();
        }
        let donors = proto.buffer_donor.iter().flat_map(|donor| &donor.entries).map(|entry| (entry.parameter_number, &entry.parameter_shape_index)).sorted().dedup();
        let donors = donors.map(|(number, index)| format!("({number}, {{{}}})", index.iter().join(","))).join(", ");
        if !donors.is_empty() {
            write!(out, ", buffer_donor={{ {donors} }}").unwrap();
        }
        let program = proto.host_program_shape.as_ref()?;
        out.push_str(", entry_computation_layout={(");
        for (index, parameter) in program.parameters.iter().enumerate() {
            separate(&mut out, index, INDEX_INTERVAL);
            parameter.print(&mut out, true);
        }
        write!(out, ")->{}}}", msg(&program.result).text(true)).unwrap();
        let attributes = &msg(&proto.frontend_attributes).map;
        if !attributes.is_empty() {
            write!(out, ", frontend_attributes={}", frontend_attributes(attributes)).unwrap();
        }
        if let Some(table) = proto.original_value_recovery_table.as_ref().filter(|table| !table.entries.is_empty()) {
            out.push_str(", origin_recovery_table={\n");
            let mut entries: Vec<_> = table.entries.iter().collect();
            entries.sort_by_key(|entry| array_key(&entry.old_original_array));
            for entry in entries {
                write!(out, "  {{{}}} : {{{}}}", original_array(&msg(&entry.old_original_array)), original_array(&msg(&entry.new_original_array))).unwrap();
                if let Some(recovery) = &entry.recovery_module {
                    let recovery = Module::parse(Cow::Owned(xla::HloProto { hlo_module: Some(recovery.clone()), ..Default::default() }.encode_to_vec()));
                    let text = Printer::new(&recovery, Style::Long, false).module_text()?;
                    let text = text.replace('\\', "\\\\").replace('"', "\\\"").lines().map(|line| if line.is_empty() { String::new() } else { format!("    {line}") }).join("\n");
                    write!(out, ",\n  \"\n{text}\n\n  \"").unwrap();
                }
                out.push('\n');
            }
            out.push_str("}\n");
        }
        let mut debug: Vec<_> = proto.debug_attributes.iter().filter(|entry| !entry.debug_attributes.is_empty()).collect();
        debug.sort_by_key(|entry| array_key(&entry.original_array));
        if !debug.is_empty() {
            let mut lines = debug.iter().map(|entry| {
                let mut attributes = entry.debug_attributes.iter().map(|attribute| {
                    let mode = match attribute.log_mode {
                        0 => None,
                        1 => Some("default"),
                        2 => Some("fusion_debugger"),
                        _ => Some(""),
                    };
                    let parts = mode.map(|mode| format!("log_mode={mode}")).into_iter().chain((attribute.callback_id != 0).then(|| format!("callback_id={}", attribute.callback_id)));
                    let mut parts = parts.chain(attribute.partitioned.then(|| "partitioned=true".to_string())).chain((attribute.op_id != 0).then(|| format!("op_id={}", attribute.op_id)));
                    format!("{{{}}}", parts.join(","))
                });
                format!("  {{{}}}:({})", original_array(&msg(&entry.original_array)), attributes.join(","))
            });
            write!(out, ",\ndebug_attributes={{\n{}\n}}", lines.join(",\n")).unwrap();
        }
        let frames = &self.frames;
        if self.metadata && !frames.frames.is_empty() {
            for (title, names) in [("\n\nFileNames\n", &frames.files), ("\nFunctionNames\n", &frames.functions)] {
                out.push_str(title);
                for (position, name) in names.iter().enumerate() {
                    writeln!(out, "{} \"{}\"", position + 1, escaped(name)).unwrap();
                }
            }
            out.push_str("\nFileLocations\n");
            for (position, [file, function, line, column, end_line, end_column]) in frames.locations.iter().enumerate() {
                writeln!(out, "{} {{file_name_id={file} function_name_id={function} line={line} end_line={end_line} column={column} end_column={end_column}}}", position + 1).unwrap();
            }
            out.push_str("\nStackFrames\n");
            for (position, (location, parent)) in frames.frames.iter().enumerate() {
                writeln!(out, "{} {{file_location_id={location} parent_frame_id={}}}", position + 1, parent + 1).unwrap();
            }
        }
        out.push_str("\n\n");
        let order = self.computation_order();
        if order.is_empty() {
            return None;
        }
        let texts: Vec<String> = order
            .par_iter()
            .filter(|graph| !(self.async_graphs.contains(graph) && self.can_expand(**graph)))
            .map(|&graph| {
                let mut text = String::new();
                self.computation(graph, &schedule, &mut text);
                text.push_str("\n\n");
                text
            })
            .collect();
        out.extend(texts);
        (!self.unsupported()).then_some(out)
    }
}

fn print_array(out: &mut String, dimensions: &[i64], oneline: bool, indices: &mut Vec<i64>, value: &dyn Fn(&[i64]) -> String) {
    let rank = dimensions.len();
    let remaining = &dimensions[indices.len()..];
    let linebreak = if oneline { " " } else { "\n" };
    if remaining.is_empty() {
        out.push_str(&value(indices));
        return;
    }
    let spaced = if remaining[0] <= 1 { "" } else { " " };
    match (rank, remaining.len()) {
        (1, _) => out.push('{'),
        (_, 1) => write!(out, "{}{{{spaced}", if oneline { "" } else { "  " }).unwrap(),
        _ if rank > 3 && !indices.is_empty() => write!(out, "{{ /*i{}={}*/{}", indices.len() - 1, indices.last().unwrap(), if remaining[0] > 0 { linebreak } else { "" }).unwrap(),
        _ => write!(out, "{{{linebreak}").unwrap(),
    }
    for index in 0..remaining[0] {
        indices.push(index);
        print_array(out, dimensions, oneline, indices, value);
        indices.pop();
        if index < remaining[0] - 1 {
            out.push(',');
            out.push_str(if remaining.len() > 1 { linebreak } else { " " });
        }
    }
    match (rank, remaining.len()) {
        (1, _) => out.push('}'),
        (_, 1) => write!(out, "{spaced}}}").unwrap(),
        _ => write!(out, "{linebreak}}}").unwrap(),
    }
}

fn array_key(array: &Option<xla::OriginalArrayProto>) -> (String, Vec<i64>) {
    let array = msg(array);
    (array.instruction_name.clone(), array.shape_index.clone())
}

fn slice_size_groups(module: &Module, operands: &[usize], sizes: &[i64]) -> Option<Vec<Vec<i64>>> {
    let (input, starts) = (&module.nodes[*operands.first()?].shape, &module.nodes[*operands.get(2)?].shape);
    let mut remaining = sizes;
    let mut take = |rank: usize| -> Option<Vec<i64>> {
        let (head, tail) = remaining.split_at_checked(rank)?;
        remaining = tail;
        Some(head.to_vec())
    };
    let rank = |shape: &Shape| shape.dimensions.len();
    let mut groups = Vec::new();
    if input.is_tuple() {
        let nested = starts.tuple_shapes.first()?.tuple_shapes.first()?.is_array();
        for (index, element) in input.tuple_shapes.iter().enumerate() {
            let count = if nested { 1 } else { starts.tuple_shapes.get(index)?.tuple_shapes.len() };
            for _ in 0..count {
                groups.push(take(rank(element))?);
            }
        }
    } else if starts.tuple_shapes.first()?.is_tuple() {
        for _ in 0..starts.tuple_shapes.len() {
            groups.push(take(rank(input))?);
        }
    } else {
        groups.push(take(rank(input))?);
    }
    Some(groups)
}

fn original_array(array: &xla::OriginalArrayProto) -> String {
    let index = if array.shape_index.is_empty() { String::new() } else { format!(" {{{}}}", array.shape_index.iter().join(",")) };
    format!("\"{}\"{index}", array.instruction_name)
}

fn original_value(original: &xla::OriginalValueProto, shape: &Shape, index: &mut Vec<i64>) -> String {
    if original.is_synthetic_call {
        return "[synthetic_call]".into();
    }
    if !shape.is_tuple() {
        let leaf = original.elements.iter().find(|element| element.shape_index == *index).and_then(|element| element.original_array.as_ref());
        return leaf.map_or_else(|| "{}".into(), |array| format!("{{{}}}", original_array(array)));
    }
    let mut children = shape.tuple_shapes.iter().enumerate().map(|(position, element)| {
        index.push(position as i64);
        let text = original_value(original, element, index);
        index.pop();
        text
    });
    if shape.tuple_shapes.is_empty() { "()".into() } else { format!("({})", children.join(", ")) }
}

fn float_text(value: f64, digits: usize, payload: Option<(u64, u32)>) -> String {
    let mut text = general(value, digits);
    if let Some((bits, payload_bits)) = payload.filter(|_| value.is_nan()) {
        let mask = (1u64 << payload_bits) - 1;
        if bits & mask != 1 << (payload_bits - 1) {
            write!(text, "(0x{:x})", bits & mask).unwrap();
        }
    }
    text
}

fn small_float(bits: u32, exponent_bits: u32, mantissa_bits: u32, bias: i32, kind: u8) -> f64 {
    let sign = if bits >> (exponent_bits + mantissa_bits) & 1 == 1 { -1.0 } else { 1.0 };
    let exponent = (bits >> mantissa_bits) & ((1 << exponent_bits) - 1);
    let mantissa = bits & ((1 << mantissa_bits) - 1);
    let all = (1 << exponent_bits) - 1;
    match kind {
        0 if exponent == all => return if mantissa == 0 { sign * f64::INFINITY } else { f64::NAN.copysign(sign) },
        1 if exponent == all && mantissa == (1 << mantissa_bits) - 1 => return f64::NAN.copysign(sign),
        2 if bits == 1 << (exponent_bits + mantissa_bits) => return f64::NAN.copysign(if bias == 11 { -1.0 } else { 1.0 }),
        _ => {}
    }
    match exponent {
        0 => sign * f64::from(mantissa) * 2f64.powi(1 - bias - mantissa_bits as i32),
        _ => sign * (1.0 + f64::from(mantissa) / f64::from(1u32 << mantissa_bits)) * 2f64.powi(exponent as i32 - bias),
    }
}

fn literal_values(literal: &LiteralProto, element_type: i32) -> Option<Vec<String>> {
    let small = |bytes: &[u8], exponent_bits: u32, mantissa_bits: u32, bias: i32, kind: u8, digits: usize, payload: Option<u32>| {
        bytes.iter().map(|&byte| float_text(small_float(u32::from(byte), exponent_bits, mantissa_bits, bias, kind), digits, payload.map(|bits| (u64::from(byte), bits)))).collect()
    };
    let narrow = |bytes: &[u8], bits: u32, signed: bool| -> Vec<String> {
        bytes
            .iter()
            .map(|&byte| {
                let value = i64::from(byte) & ((1 << bits) - 1);
                if signed && value >> (bits - 1) == 1 { (value - (1 << bits)).to_string() } else { value.to_string() }
            })
            .collect()
    };
    let halves = |bytes: &[u8]| bytes.as_chunks::<2>().0.iter().map(|&pair| u16::from_le_bytes(pair)).collect::<Vec<u16>>();
    Some(match element_type {
        1 => literal.preds.iter().map(bool::to_string).collect(),
        30 => narrow(&literal.s1s, 1, true),
        26 => narrow(&literal.s2s, 2, true),
        21 => narrow(&literal.s4s, 4, true),
        31 => narrow(&literal.u1s, 1, false),
        27 => narrow(&literal.u2s, 2, false),
        22 => narrow(&literal.u4s, 4, false),
        2 => literal.s8s.iter().map(|&byte| (byte as i8).to_string()).collect(),
        6 => literal.u8s.iter().map(u8::to_string).collect(),
        3 => halves(&literal.s16s).into_iter().map(|half| (half as i16).to_string()).collect(),
        7 => halves(&literal.u16s).into_iter().map(|half| half.to_string()).collect(),
        4 => literal.s32s.iter().map(i32::to_string).collect(),
        5 => literal.s64s.iter().map(i64::to_string).collect(),
        8 => literal.u32s.iter().map(u32::to_string).collect(),
        9 => literal.u64s.iter().map(u64::to_string).collect(),
        11 => literal
            .f32s
            .iter()
            .map(|&value| float_text(f64::from(value), if general(f64::from(value), 6).parse::<f32>() == Ok(value) { 6 } else { 9 }, Some((u64::from(value.to_bits()), 23))))
            .collect(),
        15 => literal.c64s.as_chunks::<2>().0.iter().map(|pair| format!("({}, {})", float_text(f64::from(pair[0]), 6, None), float_text(f64::from(pair[1]), 6, None))).collect(),
        18 => literal.c128s.as_chunks::<2>().0.iter().map(|pair| format!("({}, {})", float_text(pair[0], 15, None), float_text(pair[1], 15, None))).collect(),
        12 => literal.f64s.iter().map(|&value| float_text(value, if general(value, 15).parse::<f64>() == Ok(value) { 15 } else { 17 }, Some((value.to_bits(), 52)))).collect(),
        16 => halves(&literal.bf16s).into_iter().map(|bits| float_text(f64::from(f32::from_bits(u32::from(bits) << 16)), 4, Some((u64::from(bits), 7)))).collect(),
        10 => halves(&literal.f16s).into_iter().map(|bits| float_text(small_float(u32::from(bits), 5, 10, 15, 0), 5, Some((u64::from(bits), 10)))).collect(),
        19 => small(&literal.f8e5m2s, 5, 2, 15, 0, 2, Some(2)),
        28 => small(&literal.f8e4m3s, 4, 3, 7, 0, 3, Some(3)),
        29 => small(&literal.f8e3m4s, 3, 4, 3, 0, 3, Some(4)),
        20 => small(&literal.f8e4m3fns, 4, 3, 7, 1, 3, None),
        23 => small(&literal.f8e4m3b11fnuzs, 4, 3, 11, 2, 3, None),
        24 => small(&literal.f8e5m2fnuzs, 5, 2, 16, 2, 2, None),
        25 => small(&literal.f8e4m3fnuzs, 4, 3, 8, 2, 3, None),
        32 => small(&literal.f4e2m1fns, 2, 1, 1, 3, 2, None),
        35 => small(&literal.f6e3m2fns, 3, 2, 3, 3, 2, None),
        36 => small(&literal.f6e2m3fns, 2, 3, 1, 3, 3, None),
        33 => literal.f8e8m0fnus.iter().map(|&byte| float_text(if byte == u8::MAX { f64::NAN } else { 2f64.powi(i32::from(byte) - 127) }, 2, None)).collect(),
        _ => return None,
    })
}
