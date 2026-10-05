use crate::hlo::{self, Module, Shape};
use crate::hlo_text::{Printer, Style};
use itertools::Itertools;
use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor, ReflectMessage};
use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;
use std::sync::LazyLock;

const VIEWS: [&str; 7] = ["graph", "json", "pb", "pbtxt", "short_txt", "long_txt", "adj_nodes"];
const DOWNLOAD: &str = "application/octet-stream";
const ACTIVATION_MODES: [&str; 9] = ["none", "sigmoid", "relu", "relu6", "reluX", "tanh", "bandpass", "elu", "leakyrelu"];

static GPU_BACKEND_CONFIG: LazyLock<MessageDescriptor> =
    LazyLock::new(|| DescriptorPool::decode(&include_bytes!("gpu_backend_descriptors.pb")[..]).unwrap().get_message_by_name("xla.gpu.GpuBackendConfig").unwrap());
const GRAPH_HTML: &str = include_str!("graph.html");
const TRIVIAL: &str = "add=add multiply=multiply minimum=min maximum=max xor=xor and=and or=or LE=less-or-equal GE=greater-or-equal GT=greater-than LT=less-than EQ=equal-to NE=not-equal-to";
const SCHEMES: &str = "blue:filled:#bbdefb:#8aacc8:black brown:filled:#bcaaa4:#8c7b75:black dark_blue:filled:#1565c0:#003c8f:white dark_green:filled:#2e7d32:#005005:white dark_orange:filled:#ffb74d:#c88719:black dark_red:filled:#b71c1c:#7f0000:white gray:filled:#cfd8dc:#9ea7aa:black green:filled:#c8e6c9:#97b498:black orange:filled:#ffe0b2:#cbae82:black purple:filled:#e1bee7:#af8eb5:black red:filled:#ffcdd2:#cb9ca1:black white:filled:white:#9e9e9e:black yellow:filled:#fff9c4:#cbc693:black";
const STATISTIC_FILLS: &str = "#f7d4cc #f8b2a3 #f9a28f #fa917b #fb8066 #fc7052 #fd5f3d #fd4e29 #fe3e14 #ff2d00";
const BROWN: &str = "all-gather all-gather-start all-gather-done all-reduce reduce-scatter all-reduce-start all-reduce-done all-to-all collective-broadcast collective-reduce collective-permute collective-permute-start collective-permute-done infeed outfeed partition-id ragged-all-to-all recv recv-done send send-done replica-id";

pub fn wrap_dot_html(dot: &str, engine: &str) -> String {
    let mut escaped = String::with_capacity(dot.len());
    for (index, character) in dot.char_indices() {
        let rest = &dot.as_bytes()[index + 1..];
        match character {
            '\\' | '`' => escaped.push('\\'),
            '$' if rest.starts_with(b"{") => escaped.push('\\'),
            '<' if rest.starts_with(b"!--") || rest.get(..7).is_some_and(|tag| tag.eq_ignore_ascii_case(b"/script")) => {
                escaped.push_str("<\\");
                continue;
            }
            _ => {}
        }
        escaped.push(character);
    }
    GRAPH_HTML.replace("$LAYOUT_ENGINE", engine).replace("$DOT", &escaped)
}

fn adjacent(module: &Module, name: &str) -> Option<String> {
    fn walk(module: &Module, node: usize, forward: bool, out: &mut Vec<String>) {
        let next = |node: usize| if forward { module.nodes[node].users.iter() } else { module.nodes[node].operands.iter() };
        let mut stack = vec![next(node)];
        while let Some(top) = stack.last_mut() {
            let Some(&other) = top.next() else {
                stack.pop();
                continue;
            };
            if module.nodes[other].name.starts_with("get-tuple-element") {
                stack.push(next(other));
            } else {
                out.push(module.nodes[other].name.clone());
            }
        }
    }
    let node = module.find(name)?;
    let mut operands = Vec::new();
    walk(module, node, false, &mut operands);
    let mut consumers = Vec::new();
    walk(module, node, true, &mut consumers);
    Some(format!("{{\"consumer_names\":{},\"operand_names\":{}}}", serde_json::to_string(&consumers).ok()?, serde_json::to_string(&operands).ok()?))
}

pub fn serve(dir: &Path, params: &HashMap<String, String>) -> Result<(Vec<u8>, &'static str), String> {
    let text = |key: &str| params.get(key).map(String::as_str).filter(|value| !value.is_empty());
    let kind = params.get("type").map(String::as_str);
    let positional = matches!(kind, Some("graph" | "adj_nodes"));
    let node = if positional { text("node_name").unwrap_or("") } else { "" };
    let module = if let Some(module) = hlo::by_options(dir, params) {
        module
    } else {
        let unparsable = |module: &str| format!("Can't parse {} as binary proto", dir.join(format!("{module}.hlo_proto.pb")).display());
        let missing = match (text("module_name"), text("program_id")) {
            (Some(module), _) if dir.join(format!("{module}.hlo_proto.pb")).is_file() => unparsable(module),
            (Some(module), _) => format!("{}; No such file or directory", dir.join(format!("{module}.hlo_proto.pb")).display()),
            (None, Some(program)) => match hlo::modules(dir).into_iter().find(|module| module.contains(program)) {
                Some(module) => unparsable(&module),
                None => format!("HLO proto file containing program ID {program} not found in {}", dir.display()),
            },
            (None, None) => "Can not load hlo proto from options.".into(),
        };
        let valid = kind.is_some_and(|kind| VIEWS.contains(&kind));
        if !valid || node.is_empty() {
            return Err(missing);
        }
        let by_node = hlo::modules(dir).iter().filter_map(|module| hlo::load(dir, module)).find(|module| (0..module.nodes.len()).any(|index| module.inst(index).name == node));
        by_node.ok_or_else(|| format!("HLO proto file containing node name {node} not found in {}", dir.display()))?
    };
    let kind = kind.ok_or("Graph viewer must provide a type option.")?;
    if !VIEWS.contains(&kind) {
        return Err(format!("Unknown graph viewer type option: {kind}"));
    }
    let flag = |key: &str| params.get(key).is_some_and(|value| value == "true");
    let not_found = || format!("Couldn't find HloInstruction or HloComputation named {node}.");
    match kind {
        "pb" => Ok((module.data.to_vec(), DOWNLOAD)),
        "pbtxt" => {
            let mut out = String::new();
            crate::pbtext::print_hlo(&mut out, &module.data);
            Ok((out.into_bytes(), DOWNLOAD))
        }
        "json" => Err("Not implemented".into()),
        _ if !module.valid => Err(module.error()),
        "short_txt" | "long_txt" => {
            let style = if kind == "short_txt" { Style::Short } else { Style::Long };
            let text = Printer::new(&module, style, flag("show_metadata")).module_text().ok_or("Unsupported HLO instruction")?;
            Ok((text.into_bytes(), DOWNLOAD))
        }
        "graph" | "adj_nodes" if node.is_empty() => Err("node_name should not be empty".into()),
        "graph" => {
            let width = text("graph_width").filter(|value| value.bytes().all(|byte| byte.is_ascii_digit())).map_or(Some(3), |value| value.parse().ok()).unwrap_or(3);
            if module.find_graph(node).is_none() && module.find(node).is_none() {
                return Err(not_found());
            }
            render(&module, node, width, flag("show_metadata"), !flag("merge_fusion"), params.get("format").map_or("url", String::as_str))
        }
        _ => {
            if module.find(node).is_none() {
                return Err(if module.find_graph(node).is_some() { "GetAdjacentNodes is not implemented for HloComputation.".into() } else { not_found() });
            }
            Ok((adjacent(&module, node).ok_or("Unsupported HLO instruction")?.into_bytes(), "text/plain"))
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Show {
    Normal,
    Hide,
    Highlight,
    SomeOperandsOmitted,
    OmitOperands,
    SomeUsersOmitted,
}

struct Dumper<'a> {
    printer: Printer<'a>,
    module: &'a Module<'a>,
    computation: usize,
    label: String,
    filter: Option<HashMap<usize, Show>>,
    backend_config: bool,
    fusions: bool,
    node_ids: HashMap<usize, i64>,
    edge_counts: HashMap<(usize, Option<usize>), usize>,
    cluster_ids: HashMap<usize, i64>,
    edges: Vec<(usize, Option<usize>, String)>,
    callers: Vec<Vec<usize>>,
}

fn sanitize_html(text: &str) -> String {
    text.replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn statistic_attributes(scheme: &str, statistic: Option<(&str, &str)>) -> String {
    let entry = SCHEMES.split(' ').find(|entry| entry.split(':').next() == Some(scheme)).unwrap_or("dashed:filled,dashed:white:#757575:#757575");
    let parts: Vec<&str> = entry.split(':').collect();
    let (fill, stroke, font) = statistic.map_or((parts[2], parts[3], parts[4]), |(fill, font)| (fill, "#c2c2c2", font));
    format!("style=\"{}\", fontcolor=\"{font}\", color=\"{stroke}\", fillcolor=\"{fill}\"", parts[1])
}

fn statistic_colors(viz: &crate::hlo::xla::StatisticsViz) -> Option<(&'static str, &'static str)> {
    if viz.statistics.is_empty() {
        return (viz.stat_index_to_visualize == -1).then_some(("#f5f5f5", "black"));
    }
    let value = usize::try_from(viz.stat_index_to_visualize).ok().and_then(|index| viz.statistics.get(index))?.stat_val;
    let step = (0..9u8).find(|&step| value < f64::from(step + 1) * 10.0).map_or(9, usize::from);
    let fill = if value == 0.0 { "#f5f5f5" } else { STATISTIC_FILLS.split(' ').nth(step).unwrap_or_default() };
    Some((fill, if value < 60.0 { "black" } else { "white" }))
}

fn effective_scalar(shape: &Shape) -> bool {
    shape.is_array() && shape.dimensions.iter().all(|dimension| *dimension == 1)
}

fn elements_recursive(shape: &Shape) -> i64 {
    match shape.is_tuple() {
        true => shape.tuple_shapes.iter().map(elements_recursive).sum(),
        false if shape.is_array() => shape.dimensions.iter().product(),
        false => 0,
    }
}

fn has_type(shape: &Shape, kinds: &[i32]) -> bool {
    kinds.contains(&shape.element_type) || shape.tuple_shapes.iter().any(|element| has_type(element, kinds))
}

fn is_small(module: &Module, node: usize) -> bool {
    let shape = &module.nodes[node].shape;
    has_type(shape, &[14, 17]) || elements_recursive(shape) < 4096
}

fn instruction_id(node: usize) -> String {
    (node + 1).to_string()
}

fn computation_id(graph: usize) -> String {
    format!("cluster_{}", graph + 1)
}

impl Dumper<'_> {
    fn show(&self, node: usize) -> Show {
        let Some(filter) = &self.filter else { return Show::Normal };
        if let Some(&result) = filter.get(&node) {
            return result;
        }
        if self.module.nodes[node].computation != self.computation && !self.acf_parameter(node) { Show::Normal } else { Show::Hide }
    }

    fn shown(&self, node: usize) -> bool {
        self.show(node) != Show::Hide
    }

    fn fusion_of(&self, graph: usize) -> Option<usize> {
        self.callers[graph].iter().copied().find(|&caller| self.module.nodes[caller].opcode == "fusion")
    }

    fn fused(&self, node: usize) -> bool {
        self.fusion_of(self.module.nodes[node].computation).is_some()
    }

    fn acf_parameter(&self, node: usize) -> bool {
        let module = self.module;
        let entry = &module.nodes[node];
        if entry.opcode != "parameter" || entry.users.len() != 1 {
            return false;
        }
        let graph = entry.computation;
        let Some(fusion) = self.fusion_of(graph) else { return false };
        let Some(&operand) = module.nodes[fusion].operands.get(module.nodes[node].parameter as usize) else { return false };
        if module.nodes[operand].opcode != "get-tuple-element" {
            return false;
        }
        let tuple = module.nodes[operand].operands[0];
        if module.nodes[tuple].opcode != "fusion" {
            return false;
        }
        let fused = module.nodes[tuple].called[0];
        let source = module.graphs[fused].root;
        let custom_call = |node: usize, target: &str| module.nodes[node].opcode == "custom-call" && module.inst(node).custom_call_target == target;
        let user = entry.users[0];
        (module.graphs[graph].name.starts_with("async_collective_fusion") && custom_call(source, "AsyncCollectiveStart") && user == module.graphs[graph].root)
            || (custom_call(user, "AsyncCollectiveDone") && module.graphs[fused].name.starts_with("async_collective_fusion"))
    }

    fn trivial(&self, graph: usize) -> Option<&'static str> {
        let module = self.module;
        let computation = &module.graphs[graph];
        if computation.nodes.len() != 3 {
            return None;
        }
        let root = &module.nodes[computation.root];
        if root.operands.len() != 2 || !effective_scalar(&root.shape) {
            return None;
        }
        let parameter = |node: usize, number: i64| module.nodes[node].opcode == "parameter" && module.nodes[node].parameter == number && effective_scalar(&module.nodes[node].shape);
        let (first, second) = (root.operands[0], root.operands[1]);
        if !((parameter(first, 0) && parameter(second, 1)) || (parameter(first, 1) && parameter(second, 0))) {
            return None;
        }
        let key = if root.opcode == "compare" { module.inst(computation.root).comparison_direction } else { root.opcode.clone() };
        TRIVIAL.split(' ').filter_map(|pair| pair.split_once('=')).find(|(name, _)| *name == key).map(|(_, kind)| kind)
    }

    fn broadcast_of_scalar_constant(&self, node: usize) -> bool {
        let module = self.module;
        let entry = &module.nodes[node];
        self.fusion_of(entry.computation).is_some()
            && entry.opcode == "broadcast"
            && entry.operands.len() == 1
            && module.nodes[entry.operands[0]].opcode == "constant"
            && effective_scalar(&module.nodes[entry.operands[0]].shape)
    }

    fn show_subcomputation(&self, graph: usize) -> bool {
        if let Some(fusion) = self.fusion_of(graph) {
            if !self.shown(fusion) || matches!(self.show(fusion), Show::SomeOperandsOmitted | Show::OmitOperands) || !self.fusions {
                return false;
            }
        } else if self.trivial(graph).is_some() {
            return false;
        }
        self.module.graphs[graph].nodes.iter().any(|&node| self.shown(node))
    }

    fn show_fusion(&self, node: usize) -> bool {
        self.show_subcomputation(self.module.nodes[node].called[0])
    }

    fn parameter_constant(&self, node: usize) -> Option<usize> {
        let module = self.module;
        let entry = &module.nodes[node];
        if entry.opcode != "parameter" {
            return None;
        }
        let fusion = self.fusion_of(entry.computation)?;
        let &operand = module.nodes[fusion].operands.get(module.nodes[node].parameter as usize)?;
        (module.nodes[operand].opcode == "constant").then_some(operand)
    }

    fn merge_into_users(&self, node: usize) -> bool {
        let module = self.module;
        let entry = &module.nodes[node];
        if (entry.opcode == "get-tuple-element" && node != module.graphs[entry.computation].root) || self.parameter_constant(node).is_some() {
            return true;
        }
        entry.opcode == "parameter"
            && entry.shape.is_tuple()
            && !self.fused(node)
            && entry.users.iter().filter(|&&user| self.shown(user)).count() > 3
            && entry.users.iter().all(|&user| !self.shown(user) || module.nodes[user].opcode == "get-tuple-element")
    }

    fn node_for_edge(&self, mut node: usize) -> usize {
        let module = self.module;
        if module.nodes[node].opcode == "get-tuple-element" {
            node = module.nodes[node].operands[0];
        }
        while module.nodes[node].opcode == "fusion" && self.show_fusion(node) {
            node = module.graphs[module.nodes[node].called[0]].root;
        }
        node
    }

    fn color(&self, node: usize) -> &'static str {
        let module = self.module;
        let entry = &module.nodes[node];
        let parameter_color = if is_small(module, node) { "orange" } else { "dark_orange" };
        if entry.operands.iter().any(|&operand| module.nodes[operand].opcode == "parameter" && self.merge_into_users(operand) && self.parameter_constant(operand).is_none()) {
            return parameter_color;
        }
        match entry.opcode.as_str() {
            "broadcast" | "dynamic-update-slice" => "yellow",
            "concatenate" | "dynamic-slice" | "reshape" | "dynamic-reshape" | "reverse" | "transpose" | "copy" | "copy-start" | "copy-done" => "green",
            "bitcast" if self.fused(node) => "green",
            "async-start" | "async-update" | "async-done" => match self.printer.wrapped(node) {
                Some(graph) => self.color(module.graphs[graph].root),
                None => "white",
            },
            "convolution" | "dot" | "ragged-dot" | "scaled-dot" | "fft" | "triangular-solve" | "cholesky" => "dark_blue",
            "parameter" => parameter_color,
            "batch-norm-grad" | "batch-norm-inference" | "batch-norm-training" | "reduce" | "reduce-window" | "scan" | "scatter" | "select-and-scatter" | "gather" => "purple",
            "domain" | "fusion" | "map" | "get-dimension-size" | "set-dimension-size" => "gray",
            opcode if BROWN.split(' ').any(|name| name == opcode) => "brown",
            "call" | "conditional" | "custom-call" | "while" => "dark_green",
            _ => "white",
        }
    }

    fn label(&self, node: usize) -> String {
        let entry = &self.module.nodes[node];
        if entry.opcode == "parameter" {
            return format!("<b>Parameter {}</b>", self.module.nodes[node].parameter);
        }
        if entry.name.starts_with(&entry.opcode) {
            return format!("<b>{}</b>", sanitize_html(&entry.name));
        }
        let mut extended = entry.opcode.clone();
        if entry.opcode == "fusion" {
            extended.push(':');
            extended.push_str(&self.module.inst(node).fusion_kind);
        }
        format!("<b>{}</b><br/>{}", sanitize_html(&entry.name), sanitize_html(&extended))
    }

    fn metadata(&self, node: usize) -> String {
        let module = self.module;
        let metadata = module.inst(node).metadata.unwrap_or_default();
        let mut lines: Vec<String> = [
            (!metadata.op_name.is_empty()).then(|| sanitize_html(&metadata.op_name)),
            (!metadata.op_type.is_empty()).then(|| format!("op_type: {}", sanitize_html(&metadata.op_type))),
            (!metadata.source_file.is_empty() && metadata.source_line != 0).then(|| format!("source: {}:{}", metadata.source_file, metadata.source_line)),
        ]
        .into_iter()
        .flatten()
        .collect();
        let frames = &self.printer.frames;
        let mut frame = frames.of[node];
        while frame > 0 && frame as usize <= frames.frames.len() {
            let (location, parent) = frames.frames[frame as usize - 1];
            let place = frames.locations[location as usize - 1];
            frame = parent;
            let column = if place[3] == 0 { String::new() } else { format!(":{}", place[3]) };
            lines.push(format!("{}:{}:{}{column}", frames.files[place[0] as usize - 1], frames.functions[place[1] as usize - 1], place[2]));
        }
        lines.join("\n")
    }

    fn backend_config(&self, node: usize) -> String {
        let inst = self.module.inst(node);
        let config = self.module.backend_config(&inst).unwrap_or_default();
        if let Some(properties) = gpu_properties(&self.module.nodes[node].opcode, &inst.custom_call_target, self.module.nodes[node].shape.element_type, &config) {
            return properties;
        }
        if !self.backend_config || config.is_empty() {
            return String::new();
        }
        format!("backend_config=\"{}\"", crate::xplane::lossy(&config))
    }

    fn extra_info(&mut self, node: usize) -> String {
        let module = self.module;
        let entry = &module.nodes[node];
        let mut lines = Vec::new();
        let graph = entry.computation;
        let calls: Vec<usize> = self.callers[graph].iter().copied().filter(|&caller| module.nodes[caller].opcode == "call").collect();
        if calls.len() == 1 && entry.opcode == "parameter" {
            let mut producer = module.nodes[calls[0]].operands[module.nodes[node].parameter as usize];
            let mut indices = Vec::new();
            loop {
                let current = &module.nodes[producer];
                match current.opcode.as_str() {
                    "bitcast" | "copy" => producer = current.operands[0],
                    "get-tuple-element" => {
                        indices.push(module.inst(producer).tuple_index);
                        producer = current.operands[0];
                    }
                    "tuple" if !indices.is_empty() => {
                        let index = *indices.last().unwrap() as usize;
                        if index >= current.operands.len() {
                            break;
                        }
                        producer = current.operands[index];
                        indices.pop();
                    }
                    "call" => producer = module.graphs[current.called[0]].root,
                    _ => break,
                }
            }
            let parent = module.nodes[producer].computation;
            let place = if parent == module.entry { "the ENTRY computation".to_string() } else { sanitize_html(&module.graphs[parent].name) };
            lines.push(format!("<i>from {} in {place}</i>", sanitize_html(&module.nodes[producer].name)));
        }
        let mut attributes = Vec::new();
        self.printer.extra(node, &module.inst(node), true, &mut attributes);
        for line in attributes {
            if (line.starts_with("replica_groups=") || line.starts_with("source_target_pairs=") || line.starts_with("control-predecessors=")) && line.len() > 128 {
                lines.push(sanitize_html(&format!("{}...", crate::xplane::lossy(&line.as_bytes()[..125]))));
            } else if line.starts_with("feature_group_count=") {
                lines.push(format!("<b>{}</b>", sanitize_html(&line)));
            } else {
                lines.push(sanitize_html(&line));
            }
        }
        if entry.opcode != "fusion" || !self.show_fusion(node) {
            fn multidim(shape: &Shape) -> bool {
                (shape.is_array() && shape.dimensions.len() > 1) || shape.tuple_shapes.iter().any(multidim)
            }
            let mut text = entry.shape.text(entry.opcode != "tuple" && multidim(&entry.shape));
            if text.len() > 64 {
                text = format!("{}...", &text[..61]);
            }
            lines.push(sanitize_html(&text));
        }
        lines.join("<br/>")
    }

    fn constant_text(&mut self, constant: usize, shape: &Shape) -> String {
        let module = self.module;
        let entry = &module.nodes[constant];
        if shape.is_array() && shape.dimensions.contains(&0) {
            return format!("{{}} ({})", entry.shape.text(false));
        }
        if let Some(proto) = module.inst(constant).literal.filter(|_| shape.is_array() && entry.shape.elements() <= 8) {
            let mut literal = String::new();
            self.printer.literal(&proto, false, &mut literal);
            if literal.len() <= 64 {
                return format!("{} {literal}", shape.text(false));
            }
        }
        let name = if entry.name.starts_with("constant") { entry.name.clone() } else { format!("constant {}", entry.name) };
        format!("{name} {}", shape.text(false))
    }

    fn inlined_operands(&mut self, node: usize) -> String {
        let module = self.module;
        let entry = &module.nodes[node];
        let mut lines = Vec::new();
        let count = entry.operands.len();
        for (index, &operand) in entry.operands.iter().enumerate() {
            let other = &module.nodes[operand];
            let text = if other.opcode == "constant" {
                Some(self.constant_text(operand, &other.shape))
            } else if self.broadcast_of_scalar_constant(operand) {
                Some(self.constant_text(other.operands[0], &other.shape))
            } else if self.merge_into_users(operand) {
                Some(match other.opcode.as_str() {
                    "parameter" => match self.parameter_constant(operand) {
                        Some(constant) => self.constant_text(constant, &module.nodes[constant].shape),
                        None => format!("Parameter {}", module.nodes[operand].parameter),
                    },
                    "get-tuple-element" => format!("tuple-element {} of {} {}", module.inst(operand).tuple_index, module.nodes[other.operands[0]].name, other.shape.text(true)),
                    _ => other.name.clone(),
                })
            } else {
                None
            };
            if let Some(text) = text {
                let label = if count > 1 { format!(" {index}") } else { String::new() };
                lines.push(format!("<b>operand{label}</b> = {text}"));
            }
            if lines.len() == 32 && index < count - 1 {
                lines.push("...".into());
                break;
            }
        }
        if entry.opcode == "parameter"
            && let Some(fusion) = self.fusion_of(entry.computation)
        {
            let index = module.nodes[fusion].operands[module.nodes[node].parameter as usize];
            let input = &module.nodes[index];
            if input.opcode == "get-tuple-element" {
                lines.push(format!("tuple-element {} of {} {}", module.inst(index).tuple_index, module.nodes[input.operands[0]].name, input.shape.text(true)));
            }
        }
        lines.join("<br/>")
    }

    fn trivial_text(&self, node: usize) -> String {
        let entry = &self.module.nodes[node];
        if entry.opcode == "fusion" {
            return String::new();
        }
        let mut lines = Vec::new();
        for (index, &graph) in entry.called.iter().enumerate() {
            let Some(kind) = self.trivial(graph) else { continue };
            let label = if entry.called.len() == 1 { String::new() } else { format!(" {index}") };
            lines.push(format!("Subcomputation{label}: <b>{}</b>", sanitize_html(kind)));
        }
        lines.join("<br/>")
    }

    fn add_edge(&mut self, from: usize, to: usize, operand: usize, control: bool, operand_count: usize) {
        let module = self.module;
        if self.edge_counts.get(&(from, Some(to))).copied().unwrap_or(0) > 64 {
            return;
        }
        let from = self.node_for_edge(from);
        if !self.shown(from) || module.nodes[from].opcode == "constant" || self.broadcast_of_scalar_constant(from) || self.merge_into_users(from) {
            return;
        }
        *self.edge_counts.entry((from, Some(to))).or_insert(0) += 1;
        let label = match (control, operand_count > 1) {
            (true, _) => "style=\"dotted\" color=\"gray\" label=\"ctrl\"".to_string(),
            (false, true) => format!(" headlabel=\"{operand}\", labeldistance=2"),
            (false, false) => String::new(),
        };
        let text = format!(
            "{} -> {} [arrowhead={} tooltip=\"{} -> {}\" {label}];",
            instruction_id(from),
            instruction_id(to),
            if is_small(module, from) { "empty" } else { "normal" },
            module.nodes[from].name,
            module.nodes[to].name
        );
        self.edges.push((from, Some(to), text));
    }

    fn instruction(&mut self, node: usize) -> String {
        let module = self.module;
        let entry = &module.nodes[node];
        let root = module.graphs[entry.computation].root;
        if (entry.opcode == "constant" || self.broadcast_of_scalar_constant(node)) && node != root {
            return String::new();
        }
        if self.merge_into_users(node) {
            return String::new();
        }
        if entry.opcode == "fusion" && self.show_fusion(node) {
            return String::new();
        }
        self.node_ids.insert(node, self.node_ids.len() as i64 + 1);
        let mut shape = if entry.opcode == "while" { "ellipse" } else { "rect" };
        let label = self.label(node);
        let metadata = self.metadata(node);
        let backend_config = self.backend_config(node);
        let extra = self.extra_info(node);
        let inlined = self.inlined_operands(node);
        let trivial = self.trivial_text(node);
        if entry.opcode == "parameter" && self.fused(node) {
            if entry.computation != self.computation {
                let fusion = self.fusion_of(entry.computation).unwrap();
                let input = module.nodes[fusion].operands[module.nodes[node].parameter as usize];
                self.add_edge(input, node, 0, false, entry.operands.len());
            }
        } else {
            for (index, &operand) in entry.operands.iter().enumerate() {
                self.add_edge(operand, node, index, false, entry.operands.len());
            }
            for &predecessor in &entry.predecessors {
                self.add_edge(predecessor, node, 0, true, entry.operands.len());
            }
        }
        let mut scheme = self.color(node);
        if matches!(self.show(node), Show::OmitOperands | Show::SomeOperandsOmitted | Show::SomeUsersOmitted) {
            scheme = "dashed";
        }
        if self.show(node) == Show::Highlight {
            shape = "diamond";
            scheme = "dark_red";
        }
        let mut body = label;
        for part in [trivial, extra, inlined, backend_config].iter().filter(|part| !part.is_empty()) {
            write!(body, "<br/>{part}").unwrap();
        }
        let statistic = statistic_colors(&module.inst(node).statistics_viz.unwrap_or_default());
        format!("{} [label=<{body}>, shape={shape}, tooltip=\"{metadata}\", {}];\n", instruction_id(node), statistic_attributes(scheme, statistic))
    }

    fn subcomputation(&mut self, graph: usize, parent: usize) -> String {
        let module = self.module;
        let parent_entry = &module.nodes[parent];
        if parent_entry.opcode != "fusion" {
            let from = self.node_for_edge(module.graphs[graph].root);
            *self.edge_counts.entry((from, Some(parent))).or_insert(0) += 1;
            let text = format!(
                "{} -> {} [ltail=\"{}\", style=\"dashed\" tooltip=\"{} -> {}\"];",
                instruction_id(from),
                instruction_id(parent),
                computation_id(graph),
                module.graphs[graph].name,
                parent_entry.name
            );
            self.edges.push((from, Some(parent), text));
        }
        if self.cluster_ids.contains_key(&graph) {
            return String::new();
        }
        self.cluster_ids.insert(graph, self.cluster_ids.len() as i64 + 1);
        let (label, style) = if parent_entry.opcode == "fusion" {
            let category = hlo::fusion_category(&module.inst(parent).fusion_kind);
            let mut label = format!("Fused expression for <b>{}</b><br/>{}", sanitize_html(&parent_entry.name), sanitize_html(category));
            for part in [self.extra_info(parent), self.backend_config(parent)].iter().filter(|part| !part.is_empty()) {
                write!(label, "<br/>{part}").unwrap();
            }
            let highlight = self.show(parent) == Show::Highlight;
            let statistic = statistic_colors(&module.inst(parent).statistics_viz.unwrap_or_default()).filter(|_| !highlight);
            let (fill, stroke) = statistic.map_or(if highlight { ("#ffcdd2", "#b71c1c") } else { ("#f5f5f5", "#c2c2c2") }, |(fill, _)| (fill, "#c2c2c2"));
            (label, format!("style=\"rounded,filled,bold\"; fillcolor=\"{fill}\"; color=\"{stroke};\""))
        } else {
            (format!("Subcomputation for <b>{}</b><br/>{}", sanitize_html(&parent_entry.name), sanitize_html(&module.graphs[graph].name)), "style=rounded; color=black;".to_string())
        };
        let body = self.computation_body(graph);
        let id = computation_id(graph);
        format!("subgraph {id} {{\n{style}\nlabel = <{label}>;\nlabelloc = t;\ntooltip = \" \";\n{body}\n}}  // {id}\n\n")
    }

    fn computation_body(&mut self, graph: usize) -> String {
        let module = self.module;
        let mut out = String::new();
        for &node in &module.graphs[graph].nodes {
            if !self.shown(node) {
                continue;
            }
            for &called in &module.nodes[node].called {
                if self.show_subcomputation(called) {
                    let text = self.subcomputation(called, node);
                    out.push_str(&text);
                }
            }
            let text = self.instruction(node);
            out.push_str(&text);
        }
        out
    }

    fn root_tag(&mut self) -> String {
        let module = self.module;
        let from = self.node_for_edge(module.graphs[self.computation].root);
        if !self.shown(from) || module.nodes[from].opcode == "constant" || self.broadcast_of_scalar_constant(from) {
            return String::new();
        }
        let to = computation_id(self.computation);
        self.edges.push((from, None, format!("{} -> {to} [tooltip=\" \"];", instruction_id(from))));
        format!("{to} [label=<ROOT>, shape=circle, tooltip=\" \", {}];\n", statistic_attributes("brown", None))
    }

    fn header(&self) -> Option<String> {
        let module = self.module;
        let graph = &module.graphs[self.computation];
        let mut label = format!("{}<br/>Computation {}", self.label, graph.name);
        if let Some(fusion) = self.fusion_of(self.computation) {
            write!(label, " (in fusion instruction {})", module.nodes[fusion].name).unwrap();
        } else if self.computation == module.entry {
            label.push_str("<br/>ENTRY computation");
        } else if !self.callers[self.computation].is_empty() {
            let names = self.callers[self.computation].iter().map(|&caller| &module.nodes[caller].name).join(", ");
            write!(label, "<br/>Caller instructions: {names}").unwrap();
        }
        let mut rules = String::new();
        let mut rule = |kind: &str, id: i64, edge: i64, color: &str| {
            if !rules.is_empty() {
                rules.push('\n');
            }
            write!(
                rules,
                "  %23{kind}{id}:hover ~ %23edge{edge} text {{ fill: {color}; }}\n  %23{kind}{id}:hover ~ %23edge{edge} path {{ stroke: {color}; stroke-width: .2em; }}\n  %23{kind}{id}:hover ~ %23edge{edge} polygon {{ fill: {color}; stroke: {color}; stroke-width: .2em; }}\n"
            )
            .unwrap();
        };
        for (index, &(from, to, _)) in self.edges.iter().enumerate() {
            let edge = index as i64 + 1;
            let from_id = *self.node_ids.get(&from)?;
            let to_id = match to {
                Some(to) => *self.node_ids.get(&to)?,
                None => self.node_ids.len() as i64 + 1,
            };
            rule("node", from_id, edge, "%231976d2");
            rule("node", to_id, edge, "%23d32f2f");
            if let Some(to) = to {
                let from_graph = module.nodes[from].computation;
                if self.fusion_of(from_graph).is_some() && module.graphs[from_graph].root == from {
                    rule("clust", *self.cluster_ids.get(&from_graph)?, edge, "%231976d2");
                }
                let to_graph = module.nodes[to].computation;
                if self.fusion_of(to_graph).is_some() && module.nodes[to].opcode == "parameter" {
                    rule("clust", *self.cluster_ids.get(&to_graph)?, edge, "%23d32f2f");
                }
            }
        }
        Some(format!(
            "digraph G {{\nrankdir = TB;\ncompound = true;\nlabel = <<b>{label}</b>>;\nlabelloc = t;\n// Disable the tooltip.  Interestingly, \"\" doesn't work!\ntooltip = \" \";\n// DOT graphs accept a stylesheet as a URI.  So naturally, an inline\n// stylesheet is a data URI!\nstylesheet=<\n  data:text/css,\n  @import url(https://fonts.googleapis.com/css?family=Roboto:400,700);\n  svg text {{\n    font-family: 'Roboto';\n    font-size: 12px;\n  }}\n\n  {rules}\n>\n\n"
        ))
    }

    fn dump(&mut self) -> Option<String> {
        let body = self.computation_body(self.computation);
        let root = self.root_tag();
        let header = self.header()?;
        if self.printer.unsupported() {
            return None;
        }
        Some(format!("{header}{body}{root}{}\n}}", self.edges.iter().map(|(_, _, text)| text).join("\n")))
    }
}

fn radius_filter(module: &Module, dumper: &Dumper, root: usize, radius: i64) -> HashMap<usize, Show> {
    let mut nodes: HashMap<usize, Show> = HashMap::new();
    let mut worklist = std::collections::VecDeque::from([(root, 0i64)]);
    while let Some((node, depth)) = worklist.pop_front() {
        nodes.insert(node, if dumper.acf_parameter(node) { Show::Hide } else { Show::Normal });
        if depth == radius {
            continue;
        }
        let entry = &module.nodes[node];
        if node == root || entry.opcode != "tuple" {
            for &operand in &entry.operands {
                if !nodes.contains_key(&operand) {
                    let next = if module.nodes[operand].opcode == "bitcast" || entry.opcode == "bitcast" { depth } else { depth + 1 };
                    worklist.push_back((operand, next));
                }
            }
        }
        for &graph in &entry.called {
            worklist.push_back((module.graphs[graph].root, depth + 1));
        }
        if entry.opcode == "constant" {
            continue;
        }
        if entry.users.len() > 16 {
            nodes.insert(node, Show::SomeUsersOmitted);
            continue;
        }
        for &user in &entry.users {
            if !nodes.contains_key(&user) {
                worklist.push_back((user, depth + 1));
            }
        }
    }
    let root_graph = module.nodes[root].computation;
    let displayed = |node: usize| nodes.contains_key(&node) || module.nodes[node].opcode == "constant" || module.nodes[node].computation != root_graph;
    let mut result = nodes.clone();
    for (&node, show) in &mut result {
        let operands = &module.nodes[node].operands;
        if operands.iter().any(|&operand| displayed(operand)) && !operands.iter().all(|&operand| displayed(operand)) {
            *show = Show::SomeOperandsOmitted;
        } else if !operands.is_empty() && !operands.iter().any(|&operand| displayed(operand)) {
            *show = Show::OmitOperands;
        }
        if *show == Show::SomeUsersOmitted && module.nodes[node].users.iter().all(|&user| displayed(user)) {
            *show = Show::Normal;
        }
    }
    result.insert(root, Show::Highlight);
    result
}

fn gpu_properties(opcode: &str, target: &str, element_type: i32, config: &[u8]) -> Option<String> {
    let convolution = crate::gpu_cost::DNN_CONVOLUTION.contains(&target);
    if opcode != "custom-call" || !(convolution || crate::gpu_cost::CUBLAS_LT.contains(&target)) {
        return None;
    }
    let mut message = DynamicMessage::new(GPU_BACKEND_CONFIG.clone());
    if !config.is_empty() {
        let mut deserializer = serde_json::Deserializer::from_slice(config);
        message = DynamicMessage::deserialize(GPU_BACKEND_CONFIG.clone(), &mut deserializer).ok()?;
        deserializer.end().ok()?;
    }
    let value = |message: &DynamicMessage, name: &str| message.get_field_by_name(name).unwrap().into_owned();
    let child = |message: &DynamicMessage, name: &str| value(message, name).as_message().unwrap().clone();
    let number = |message: &DynamicMessage, name: &str| value(message, name).as_f64().unwrap();
    let mut properties: Vec<(&str, String)> = Vec::new();
    if convolution {
        let conv = child(&message, "cudnn_conv_backend_config");
        let (scale, side, activation) = (number(&conv, "conv_result_scale"), number(&conv, "side_input_scale"), value(&conv, "activation_mode").as_enum_number().unwrap());
        if scale != 1.0 {
            properties.push(("conv_result_scale", hlo::general(scale, 6)));
        }
        if side != 0.0 && side != 1.0 {
            properties.push(("side_input_scale", hlo::general(side, 6)));
        }
        if activation == 8 {
            properties.push(("leakyrelu_alpha", hlo::general(number(&conv, "leakyrelu_alpha"), 6)));
        }
        properties.push(("activation_mode", ACTIVATION_MODES.get(activation as usize).map_or_else(|| format!("unknown: {activation}"), std::string::ToString::to_string)));
        let algorithm = child(&conv, "algorithm");
        let knobs: std::collections::BTreeMap<i64, i64> = value(&algorithm, "tuning_knobs").as_map().unwrap().iter().map(|(key, value)| (key.as_i64().unwrap(), value.as_i64().unwrap())).collect();
        properties.push(("algo", format!("eng{}{{{}}}", value(&algorithm, "algo_id").as_i64().unwrap(), knobs.iter().map(|(key, value)| format!("k{key}={value}")).join(","))));
    } else {
        let gemm = child(&message, "gemm_backend_config");
        let (alpha_real, alpha_imag, beta) = (number(&gemm, "alpha_real"), number(&gemm, "alpha_imag"), number(&gemm, "beta"));
        if matches!(crate::hlo::type_name(element_type), "c64" | "c128") {
            if alpha_real != 1.0 || alpha_imag != 1.0 {
                properties.extend([("alpha_real", hlo::general(alpha_real, 6)), ("alpha_imag", hlo::general(alpha_real, 6))]);
            }
        } else if alpha_real != 1.0 {
            properties.push(("alpha", hlo::general(alpha_real, 6)));
        }
        if beta != 0.0 && beta != 1.0 {
            properties.push(("beta", hlo::general(beta, 6)));
        }
        let numbers = child(&gemm, "dot_dimension_numbers");
        let mut parts = Vec::new();
        for (name, field, optional) in [
            ("lhs_batch_dims", "lhs_batch_dimensions", true),
            ("lhs_contracting_dims", "lhs_contracting_dimensions", false),
            ("rhs_batch_dims", "rhs_batch_dimensions", true),
            ("rhs_contracting_dims", "rhs_contracting_dimensions", false),
        ] {
            let values = value(&numbers, field).as_list().unwrap().iter().map(|item| item.as_i64().unwrap()).join(",");
            if !optional || !values.is_empty() {
                parts.push(format!("{name}={{{values}}}"));
            }
        }
        properties.push(("", parts.join("<br/>")));
        if gemm.has_field_by_name("selected_algorithm") {
            properties.push(("algorithm", value(&gemm, "selected_algorithm").as_i64().unwrap().to_string()));
        }
        let epilogue = value(&gemm, "epilogue").as_enum_number().unwrap();
        if epilogue != 0 {
            let name = gemm.descriptor().get_field_by_name("epilogue").unwrap().kind().as_enum().unwrap().get_value(epilogue).map_or_else(String::new, |value| value.name().to_string());
            properties.push(("epilogue", name));
        }
    }
    let joined: Vec<String> = properties.iter().map(|(key, value)| if key.is_empty() { value.clone() } else { format!("{key}={value}") }).collect();
    Some(format!("{}{}", if properties.len() > 1 { "<br/>" } else { "" }, joined.join("<br/>")))
}

fn render(module: &Module, node_name: &str, radius: i64, backend_config: bool, fusions: bool, format: &str) -> Result<(Vec<u8>, &'static str), String> {
    const UNSUPPORTED: &str = "Unsupported HLO instruction";
    if module.nodes.iter().any(|node| matches!(node.opcode.as_str(), "infeed" | "outfeed")) {
        return Err(UNSUPPORTED.into());
    }
    let mut callers: Vec<Vec<usize>> = vec![Vec::new(); module.graphs.len()];
    for (index, node) in module.nodes.iter().enumerate() {
        for &graph in &node.called {
            if !callers[graph].contains(&index) {
                callers[graph].push(index);
            }
        }
    }
    let graph = module.find_graph(node_name);
    let instruction = module.find(node_name);
    let (computation, label, root) = match (graph, instruction) {
        (Some(graph), _) => (graph, String::new(), None),
        (None, Some(node)) => (module.nodes[node].computation, format!("Neighborhood of {radius} nodes around {}", module.nodes[node].name), Some(node)),
        (None, None) => return Err(format!("Couldn't find HloInstruction or HloComputation named {node_name}.")),
    };
    let mut dumper = Dumper {
        printer: Printer::new(module, Style::Graph, true),
        module,
        computation,
        label,
        filter: None,
        backend_config,
        fusions,
        node_ids: HashMap::new(),
        edge_counts: HashMap::new(),
        cluster_ids: HashMap::new(),
        edges: Vec::new(),
        callers,
    };
    if let Some(root) = root {
        dumper.filter = Some(radius_filter(module, &dumper, root, radius));
    }
    let dot = dumper.dump().ok_or(UNSUPPORTED)?;
    match format {
        "html" => Ok((wrap_dot_html(&dot, "dot").into_bytes(), "text/html")),
        "dot" => Ok((dot.into_bytes(), "text/html")),
        _ => Err("Can't render as URL; no URL renderer was registered.".into()),
    }
}

#[cfg(test)]
#[path = "tests/inline/graph_viewer.rs"]
mod tests;
