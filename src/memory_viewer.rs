use crate::graph_viewer::wrap_dot_html;
use crate::hlo::profiler::{BufferAllocation, BufferBlockProto, BufferSpan, HeapObject, LogicalBuffer, PreprocessResult, SourceInfo, heap_object::Color};
use crate::hlo::xla::{BufferAllocationProto as Allocation, BufferAssignmentProto, HloInstructionLite, LogicalBufferProto, OpMetadata, StackFrameIndexProto};
use crate::hlo::{Module, Shape, last_bytes, msg, split};
use indexmap::IndexMap;
use itertools::Itertools;
use prost::Message;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::BTreeMap;
use std::sync::LazyLock;

const SMALL_BUFFER: i64 = 16 * 1024;
const MIB: f64 = (1u64 << 20) as f64;
const POINTS_PER_INCH: f64 = 72.0;
const MARGIN_POINTS: f64 = 0.02 * POINTS_PER_INCH;
const GRAPH_SIZE: f64 = 4096.0;
const NO_TIMELINE: &str =
    "<html><body style=\"font-family: sans-serif; padding: 20px;\"><h2>No memory allocation timeline available</h2><p>There is no memory activity data to display for this timeline.</body></html>";
const COLOR_NAMES: &str = "#e91e63 #2196f3 #81c784 #4dd0e1 #3f51b5 #e53935 #ff9100 #b39ddb #90a4ae #26c6da #ad1457 #03a9f4 #2196f3 #c2185b #795548 #f9a825 #00bfa5 #880e4f #d500f9 #ce93d8 #ec407a #4caf50 #ff8f00 #ffca28 #ab47bc #00e5ff #ff9800 #40c4ff #1e88e5 #9fa8da #bf360c #00b8d4 #f57f17 #64b5f6 #e040fb #ffab91 #4caf50 #01579b #66bb6a #ef9a9a #558b2f #fb8c00 #ff4081 #00e676 #388e3c #424242 #6d4c41 #c62828 #616161 #00897b #448aff #0d47a1 #607d8b #673ab7 #00c853 #2e7d32 #ffa726 #5e35b1 #ba68c8 #8d6e63 #00bcd4 #ff6f00 #f4511e #ff1744 #9e9e9e #d81b60 #4a148c #26a69a #689f38 #7b1fa2 #b0bec5 #304ffe #f48fb1 #ffd600 #ffb74d #8bc34a #303f9f #5d4037 #80cbc4 #ffcc80 #00acc1 #3e2723 #ff5252 #ff7043 #e91e63 #ea80fc #e65100 #d84315 #212121 #ff5722 #1976d2 #2962ff #bdbdbd #3949ab #69f0ae #d50000 #ffd740 #c0ca33 #ff6e40 #00b0ff #2979ff #e64a19 #7c4dff #607d8b #009688 #ffb300 #c51162 #ffc400 #29b6f6 #3d5afe #76ff03 #cddc39 #b388ff #5c6bc0 #9e9d24 #7cb342 #ef5350 #fdd835 #ef6c00 #4fc3f7 #6200ea #004d40 #ff8a65 #ffab00 #80deea #0097a7 #7e57c2 #ff6d00 #1565c0 #455a64 #ffc107 #4527a0 #ff5722 #f44336 #f57c00 #827717 #a5d6a7 #82b1ff #9c27b0 #ff80ab #e1bee7 #78909c #311b92 #00695c #4e342e #3f51b5 #651fff #9e9e9e #81d4fa #f8bbd0 #b71c1c #0091ea #673ab7 #a1887f #4db6ac #ffa000 #6a1b9a #43a047 #bcaaa4 #546e7a #aeea00 #e57373 #ffccbc #006064 #fbc02d #ffeb3b #8bc34a #039be5 #8e24aa #80d8ff #009688 #9ccc65 #512da8 #ffc107 #757575 #0277bd #ff3d00 #33691e #03a9f4 #00838f #ff8a80 #283593 #f50057 #1a237e #90caf9 #9c27b0 #aa00ff #aed581 #afb42b #9575cd #d32f2f #64dd17 #f44336 #795548 #cddc39 #ff9e80 #7986cb #dd2c00 #0288d1 #ff9800 #263238 #00796b #42a5f5 #8c9eff #1b5e20 #ffab40 #536dfe #00bcd4 #f06292";

static COLORS: LazyLock<Vec<&str>> = LazyLock::new(|| COLOR_NAMES.split(' ').collect());

struct Logical {
    id: i64,
    size: i64,
    node: usize,
    shape_index: Vec<i64>,
    color: i64,
    name: String,
    metadata: OpMetadata,
    shape: Shape,
    unpadded: i64,
}

struct Buffer<'a> {
    logical: &'a Logical,
    allocation: usize,
    offset: i64,
    span: Option<(i64, i64)>,
    refs: i64,
    canonical: Option<usize>,
}

fn mib(bytes: i64) -> f64 {
    bytes as f64 / MIB
}

fn indefinite(allocation: &Allocation) -> bool {
    allocation.is_thread_local || allocation.is_entry_computation_parameter || allocation.is_constant || allocation.maybe_live_out
}

fn category(allocation: &Allocation) -> &'static str {
    match allocation {
        allocation if allocation.is_entry_computation_parameter => "Parameter",
        allocation if allocation.maybe_live_out => "Output",
        allocation if allocation.is_thread_local => "Thread-local",
        allocation if allocation.is_constant => "Constant",
        _ => "Temporary",
    }
}

pub fn std_sort(items: &mut [usize], less: &impl Fn(usize, usize) -> bool) {
    if items.len() > 1 {
        introsort(items, 2 * (usize::BITS - 1 - items.len().leading_zeros()) as usize, less);
    }
    let sorted = items.len().min(16);
    for index in 1..items.len() {
        if index < sorted && less(items[index], items[0]) {
            items[..=index].rotate_right(1);
            continue;
        }
        let mut position = index;
        while less(items[position], items[position - 1]) {
            items.swap(position, position - 1);
            position -= 1;
        }
    }
}

fn introsort(mut items: &mut [usize], mut depth: usize, less: &impl Fn(usize, usize) -> bool) {
    while items.len() > 16 {
        if depth == 0 {
            let length = items.len();
            crate::op_profile::partial_sort(items, length, less);
            return;
        }
        depth -= 1;
        let (middle, last) = (items.len() / 2, items.len() - 1);
        let median = match (less(items[1], items[middle]), less(items[middle], items[last]), less(items[1], items[last])) {
            (true, true, _) => middle,
            (true, false, true) => last,
            (true, false, false) | (false, _, true) => 1,
            (false, _, false) if less(items[middle], items[last]) => last,
            (false, _, false) => middle,
        };
        items.swap(0, median);
        let (mut left, mut right) = (1, items.len());
        loop {
            while less(items[left], items[0]) {
                left += 1;
            }
            right -= 1;
            while less(items[0], items[right]) {
                right -= 1;
            }
            if left >= right {
                break;
            }
            items.swap(left, right);
            left += 1;
        }
        let (head, tail) = items.split_at_mut(left);
        introsort(tail, depth, less);
        items = head;
    }
}

fn source_info(stack: &StackFrameIndexProto, metadata: &OpMetadata) -> Option<SourceInfo> {
    let mut frames = Vec::new();
    let mut frame = metadata.stack_frame_id;
    while frame > 0 && frame as usize <= stack.stack_frames.len() {
        let entry = stack.stack_frames[frame as usize - 1];
        frame = entry.parent_frame_id;
        let (mut file, mut line, mut column) = ("", -1, -1);
        if let Some(location) = usize::try_from(entry.file_location_id - 1).ok().and_then(|index| stack.file_locations.get(index)) {
            (line, column) = (location.line, location.column);
            if location.file_name_id > 0 && location.file_name_id as usize <= stack.file_names.len() {
                file = &stack.file_names[location.file_name_id as usize - 1];
            }
        }
        frames.push(format!("{file}:{line}:{column}"));
    }
    let frames = frames.join("\n");
    if !metadata.source_file.is_empty() {
        return Some(SourceInfo { file_name: metadata.source_file.clone(), line_number: metadata.source_line, stack_frame: frames });
    }
    let parts: Vec<&str> = frames.split('\n').next().unwrap_or("").split(':').collect();
    (!frames.is_empty() && parts.len() >= 2 && !parts[0].is_empty()).then(|| SourceInfo {
        file_name: parts[0].to_string(),
        line_number: parts[1].trim_ascii().parse().unwrap_or(-1),
        stack_frame: frames.clone(),
    })
}

impl Buffer<'_> {
    fn name_with_index(&self) -> String {
        if self.logical.shape_index.is_empty() { self.logical.name.clone() } else { format!("{}{{{}}}", self.logical.name, self.logical.shape_index.iter().join(",")) }
    }
}

struct Model<'a> {
    module: &'a Module<'a>,
    stack: std::borrow::Cow<'a, StackFrameIndexProto>,
    allocations: &'a [Allocation],
    traces: &'a [crate::hlo::xla::HeapSimulatorTrace],
    buffers: IndexMap<i64, Buffer<'a>, FxBuildHasher>,
    by_index: BTreeMap<i64, (usize, Option<usize>)>,
}

impl<'a> Model<'a> {
    fn new(module: &'a Module<'a>, assignment: &'a BufferAssignmentProto, logicals: &'a [Logical]) -> Self {
        let logicals: HashMap<i64, &Logical> = logicals.iter().map(|logical| (logical.id, logical)).collect();
        let stack = msg(&module.proto.stack_frame_index);
        let (allocations, traces) = (&assignment.buffer_allocations[..], &assignment.heap_simulator_traces[..]);
        let mut model = Model { module, stack, allocations, traces, buffers: IndexMap::default(), by_index: BTreeMap::new() };
        for (position, allocation) in allocations.iter().enumerate() {
            model.by_index.insert(allocation.index, (position, None));
            for assigned in &allocation.assigned {
                let Some(&logical) = logicals.get(&assigned.logical_buffer_id) else { continue };
                model.buffers.insert(assigned.logical_buffer_id, Buffer { logical, allocation: position, offset: assigned.offset, span: None, refs: 0, canonical: None });
            }
        }
        for (trace, events) in traces.iter().enumerate() {
            let Some(index) = events.events.first().and_then(|event| model.buffers.get_index_of(&event.buffer_id)) else { continue };
            if let Some(entry) = model.by_index.get_mut(&allocations[model.buffers[index].allocation].index) {
                entry.1 = Some(trace);
            }
        }
        model
    }

    fn colored(&self, color: i64) -> impl Iterator<Item = (&Allocation, Option<usize>)> + '_ {
        self.by_index.values().map(|&(allocation, trace)| (&self.allocations[allocation], trace)).filter(move |(allocation, _)| allocation.color == color)
    }

    fn root(&self, mut index: usize) -> usize {
        while let Some(next) = self.buffers[index].canonical {
            index = next;
        }
        index
    }

    fn trace_id(&self, color: i64) -> Option<usize> {
        if let Some(trace) = self.colored(color).filter(|(allocation, _)| !indefinite(allocation)).find_map(|(_, trace)| trace) {
            return Some(trace);
        }
        let mut best = (None, 0);
        for (index, trace) in self.traces.iter().enumerate() {
            let count = trace.events.iter().filter(|event| self.buffers.get(&event.buffer_id).is_some_and(|buffer| buffer.logical.color == color)).count();
            if count > best.1 {
                best = (Some(index), count);
            }
        }
        best.0
    }

    fn heap_object(&self, buffer: &Buffer, numbered: usize) -> HeapObject {
        let shape = buffer.logical.shape.text(true);
        let metadata = &buffer.logical.metadata;
        HeapObject {
            color: Some(Color::Numbered(numbered as i32)),
            label: format!("{}: {} # {}", buffer.logical.name, shape, metadata.op_name),
            logical_buffer_id: buffer.logical.id as i32,
            logical_buffer_size_mib: mib(buffer.logical.size),
            unpadded_shape_mib: mib(buffer.logical.unpadded),
            instruction_name: buffer.name_with_index(),
            shape_string: shape,
            tf_op_name: metadata.op_name.clone(),
            group_name: category(&self.allocations[buffer.allocation]).into(),
            op_code: self.module.nodes[buffer.logical.node].opcode.clone(),
            source_info: source_info(&self.stack, metadata),
        }
    }
}

#[derive(Default)]
struct Simulation {
    heap: i64,
    unpadded: i64,
    peak: i64,
    peak_unpadded: i64,
    live: Vec<i64>,
    peak_live: Vec<i64>,
    timeline: Vec<i64>,
    unpadded_timeline: Vec<i64>,
    names: Vec<String>,
    peak_position: i64,
    seen: Vec<usize>,
    seen_allocations: HashSet<i64>,
    display: HashMap<i64, i64>,
    peak_display: HashMap<i64, i64>,
    events: i64,
}

impl Simulation {
    fn increase(&mut self, buffer: &mut Buffer, init: bool) {
        self.live.push(buffer.logical.id);
        self.heap += buffer.logical.size;
        self.unpadded += buffer.logical.unpadded;
        if self.heap > self.peak {
            self.peak = self.heap;
            self.peak_position = self.timeline.len() as i64 - 1;
            self.peak_unpadded = self.unpadded;
            self.peak_live = self.live.clone();
            self.peak_display = self.display.clone();
        }
        if init {
            buffer.span = Some((self.timeline.len() as i64 - 1, self.events - 1));
        }
    }
}

fn simulate(model: &mut Model, color: i64) -> Option<Simulation> {
    let mut stats = Simulation::default();
    let Some(trace) = model.trace_id(color).filter(|trace| *trace < model.traces.len()) else { return Some(stats) };
    let events = &model.traces[trace].events;
    stats.events = events.len() as i64;
    let mut seen = HashSet::default();
    for event in events {
        stats.timeline.push(stats.heap);
        stats.unpadded_timeline.push(stats.unpadded);
        stats.names.push(event.instruction_name.clone());
        let Some(index) = model.buffers.get_index_of(&event.buffer_id) else { continue };
        if seen.insert(index) {
            stats.seen.push(index);
        }
        stats.seen_allocations.insert(model.allocations[model.buffers[index].allocation].index);
        match event.kind {
            0 => {
                let root = model.root(index);
                model.buffers[root].refs += 1;
                stats.increase(&mut model.buffers[index], true);
            }
            1 => {
                let root = model.root(index);
                model.buffers[root].refs -= 1;
                let refs = model.buffers[root].refs;
                if refs < 0 {
                    return None;
                }
                if refs == 0 {
                    let buffer = &mut model.buffers[root];
                    stats.live.retain(|live| *live != buffer.logical.id);
                    stats.heap -= buffer.logical.size;
                    if stats.heap < 0 {
                        return None;
                    }
                    stats.unpadded -= buffer.logical.unpadded;
                    if let Some(span) = &mut buffer.span {
                        span.1 = stats.timeline.len() as i64 - 1;
                    }
                }
                if model.buffers[index].canonical.is_some()
                    && let Some(span) = &mut model.buffers[index].span
                {
                    span.1 = stats.timeline.len() as i64 - 1;
                }
            }
            2 => {
                let Some(canonical) = model.buffers.get_index_of(&event.share_with_canonical_id) else { continue };
                model.buffers[index].canonical = Some(canonical);
                let root = model.root(canonical);
                model.buffers[root].refs += 1;
                if model.buffers[root].refs == 1 {
                    stats.display.insert(model.buffers[root].logical.id, model.buffers[index].logical.id);
                    model.buffers[index].span = Some((stats.timeline.len() as i64 - 1, stats.events - 1));
                    stats.increase(&mut model.buffers[root], false);
                }
            }
            _ => return None,
        }
    }
    stats.timeline.push(stats.heap);
    stats.unpadded_timeline.push(stats.unpadded);
    stats.names.push(String::new());
    (stats.seen_allocations.len() == 1).then_some(stats)
}

fn scoped_vmem(module: &Module, color: i64) -> (i64, String) {
    if color != 1 {
        return (0, String::new());
    }
    let get = |entry: &serde_json::Value, key: &str| match entry.get(key) {
        Some(serde_json::Value::Number(number)) => number.as_i64().or_else(|| number.as_u64().map(|value| value as i64)).unwrap_or_else(|| number.as_f64().unwrap_or(0.0) as i64),
        Some(serde_json::Value::String(text)) => text.trim_ascii().parse().unwrap_or(0),
        _ => 0,
    };
    let best = (0..module.nodes.len()).into_par_iter().filter_map(|node| {
        let inst = module.inst(node);
        let json = serde_json::from_slice::<serde_json::Value>(&module.backend_config(&inst)?).ok()?;
        let size = json.get("used_scoped_memory_configs")?.as_array()?.iter().filter(|entry| get(entry, "memory_space") == color).map(|entry| get(entry, "size")).max()?;
        Some((size, std::cmp::Reverse(node), inst.name))
    });
    best.max().filter(|(size, _, _)| *size > 0).map_or((0, String::new()), |(size, _, name)| (size, name))
}

fn fitting_label(label: &str, width: f64, height: f64, fontsize: f64) -> String {
    let (usable_height, usable_width) = (height - 2.0 * MARGIN_POINTS, width - 2.0 * MARGIN_POINTS);
    let char_width = 0.55 * fontsize;
    if label.is_empty() || usable_height < 1.2 * fontsize || usable_width < char_width {
        return String::new();
    }
    let max_chars = (usable_width / char_width) as i64;
    if label.len() as i64 <= max_chars {
        return label.to_string();
    }
    if max_chars >= 4 { format!("{}...", crate::xplane::lossy(&label.as_bytes()[..max_chars as usize - 3])) } else { String::new() }
}

fn logical_buffers(module: &Module, bodies: &[&[u8]]) -> Option<Vec<Logical>> {
    let nodes: HashMap<i64, usize> = module.nodes.iter().enumerate().map(|(index, node)| (node.id, index)).collect();
    bodies
        .par_iter()
        .map(|body| {
            let buffer = LogicalBufferProto::decode(*body).unwrap_or_default();
            let defined = msg(&buffer.defined_at);
            let node = *nodes.get(&defined.instruction_id)?;
            let shape_index = defined.shape_index.clone();
            let (start, end) = module.nodes[node].raw;
            let lite = HloInstructionLite::decode(&module.data[start..end]).unwrap_or_default();
            let mut shape = lite.shape.unwrap_or_default();
            if let Some(last) = usize::try_from(*shape_index.last().unwrap_or(&-1)).ok().filter(|last| *last < shape.tuple_shapes.len()) {
                shape = shape.tuple_shapes.swap_remove(last);
            }
            shape.normalize();
            Some(Logical {
                id: buffer.id,
                size: buffer.size,
                node,
                color: buffer.color,
                name: lite.name,
                metadata: lite.metadata.unwrap_or_default(),
                unpadded: shape.unpadded_bytes(),
                shape,
                shape_index,
            })
        })
        .collect()
}

impl Model<'_> {
    fn heap_objects(&self, color: i64, small_buffer: i64, stats: &Simulation) -> (Vec<HeapObject>, HashMap<i64, usize>, i64) {
        let mut objects: Vec<HeapObject> = Vec::new();
        let mut colors: HashMap<i64, usize> = HashMap::default();
        let (mut small, mut indefinite_bytes) = (0, 0);
        let mut add = |buffer: &Buffer, objects: &mut Vec<HeapObject>| {
            if buffer.logical.size < small_buffer {
                small += buffer.logical.size;
            } else {
                let numbered = objects.len();
                colors.insert(buffer.logical.id, numbered);
                objects.push(self.heap_object(buffer, numbered));
            }
        };
        for (allocation, _) in self.colored(color).filter(|(allocation, _)| indefinite(allocation)) {
            let buffers = allocation.assigned.iter().filter_map(|assigned| self.buffers.get(&assigned.logical_buffer_id));
            let best = buffers.fold(None, |best: Option<&Buffer>, buffer| if buffer.logical.size > best.map_or(0, |best| best.logical.size) { Some(buffer) } else { best });
            let Some(best) = best else { continue };
            indefinite_bytes += self.allocations[best.allocation].size;
            if color == 0 {
                add(best, &mut objects);
            }
        }
        let complete = stats.peak_live.iter().all(|id| self.buffers.get(stats.peak_display.get(id).unwrap_or(id)).map(|buffer| add(buffer, &mut objects)).is_some());
        if complete && small != 0 {
            let label = format!("small (<{small_buffer} bytes)");
            objects.push(HeapObject { color: Some(Color::Numbered(objects.len() as i32)), label, logical_buffer_id: -1, logical_buffer_size_mib: mib(small), ..Default::default() });
        }
        (objects, colors, indefinite_bytes)
    }

    fn layout(&self, color: i64, timeline: bool, colors: &HashMap<i64, usize>) -> Option<(Vec<BufferBlockProto>, Vec<String>, bool)> {
        let mut blocks = Vec::new();
        let mut rects: Vec<String> = Vec::new();
        let allocations: Vec<&Allocation> = self.colored(color).filter(|(allocation, _)| !indefinite(allocation)).map(|(allocation, _)| allocation).collect();
        let mut offsets = Vec::new();
        let (mut total_y, mut total_x) = (0i64, 0usize);
        for (allocation, trace) in self.colored(color).filter(|(allocation, _)| !indefinite(allocation)) {
            let Some(trace) = trace else { continue };
            offsets.push(total_y);
            total_y += allocation.size;
            total_x = total_x.max(self.traces[trace].events.len());
        }
        let has_timeline = total_y != 0 && total_x != 0;
        if !has_timeline {
            return Some((blocks, rects, false));
        }
        let (scale_x, scale_y) = (GRAPH_SIZE / total_x as f64, GRAPH_SIZE / total_y as f64);
        let mut node = 0usize;
        let mut rect = |x: i64, y: i64, width: i64, height: i64, description: &dyn Fn() -> String, color: &str, label: &str, node: &mut usize| {
            let (center_x, center_y) = (x as f64 + width as f64 / 2.0, y as f64 + height as f64 / 2.0);
            let (rect_width, rect_height) = (width as f64 * scale_x, height as f64 * scale_y);
            if rect_width < 1.0 || rect_height < 1.0 {
                return;
            }
            let fontsize = (rect_height * 0.6).clamp(8.0, 14.0);
            *node += 1;
            if !timeline {
                return;
            }
            rects.push(format!(
                "\"{}\" [tooltip=\"{}\", pos=\"{:.2},{:.2}!\", width=\"{:.4}\", height=\"{:.4}\", fixedsize=true, color=\"{color}\", fontsize={fontsize:.1}, label=\"{}\"];",
                *node - 1,
                description(),
                center_x * scale_x,
                center_y * scale_y,
                rect_width / POINTS_PER_INCH,
                rect_height / POINTS_PER_INCH,
                fitting_label(label, rect_width, rect_height, fontsize)
            ));
        };
        for (position, allocation) in allocations.into_iter().enumerate() {
            let offset = *offsets.get(position)?;
            let name = category(allocation).to_string();
            if !timeline {
                blocks.push(BufferBlockProto {
                    logical_buffer_id: -1,
                    name: name.clone(),
                    offset: offset as f64,
                    size: allocation.size as f64,
                    end_step: total_x as i32,
                    category: name,
                    color: "#ffffff".into(),
                    ..Default::default()
                });
            }
            let description = || format!("buffer_allocation_id:{}\nsize:{}\nbuffer_counts:{}\n", allocation.index, allocation.size, allocation.assigned.len());
            rect(0, offset, total_x as i64, allocation.size, &description, "#ffffffff", "", &mut node);
            for assigned in &allocation.assigned {
                let Some(buffer) = self.buffers.get(&assigned.logical_buffer_id) else { continue };
                let Some((start, end)) = buffer.span.filter(|_| buffer.canonical.is_none()) else { continue };
                let y = offset + buffer.offset;
                let shape = buffer.logical.shape.text(true);
                let metadata = &buffer.logical.metadata;
                let block_color = COLORS[colors.get(&buffer.logical.id).map_or(node % COLORS.len(), |numbered| numbered % COLORS.len())];
                if !timeline {
                    blocks.push(BufferBlockProto {
                        logical_buffer_id: buffer.logical.id as i32,
                        name: buffer.name_with_index(),
                        offset: y as f64,
                        size: buffer.logical.size as f64,
                        start_step: start as i32,
                        end_step: end as i32,
                        tf_op_name: metadata.op_name.clone(),
                        category: category(&self.allocations[buffer.allocation]).into(),
                        source_info: source_info(&self.stack, metadata),
                        shape_string: shape.clone(),
                        unpadded_size: buffer.logical.unpadded as f64,
                        color: block_color.into(),
                    });
                }
                let description = || {
                    format!(
                        "buffer_id:{}\nhlo_op:{}\nshape:{}\nsize:{}\nunpadded_size:{}\noffset:{}\nspan:({},{})",
                        buffer.logical.id, buffer.logical.name, shape, buffer.logical.size, buffer.logical.unpadded, buffer.offset, start, end
                    )
                };
                rect(start, y, end - start, buffer.logical.size, &description, block_color, &buffer.logical.name, &mut node);
            }
        }
        Some((blocks, rects, true))
    }

    fn indefinite_lifetimes(&self, color: i64) -> Vec<BufferAllocation> {
        let shape_of = |id: &i64| self.buffers.get(id).map(|buffer| buffer.logical);
        self.colored(color)
            .filter(|(allocation, _)| indefinite(allocation))
            .map(|(allocation, _)| {
                let flags = [
                    (allocation.is_entry_computation_parameter, "entry computation parameter"),
                    (allocation.maybe_live_out, "may-be live out"),
                    (!allocation.is_thread_local && !allocation.is_tuple, "reusable"),
                ];
                let shapes: Vec<String> = allocation.assigned.iter().map(|assigned| shape_of(&assigned.logical_buffer_id).map_or(String::new(), |logical| logical.shape.text(true))).collect();
                BufferAllocation {
                    id: allocation.index,
                    size_mib: mib(allocation.size),
                    attributes: flags.into_iter().filter(|(on, _)| *on).map(|(_, name)| name.to_string()).collect(),
                    logical_buffers: allocation
                        .assigned
                        .iter()
                        .map(|assigned| {
                            let logical = shape_of(&assigned.logical_buffer_id);
                            LogicalBuffer {
                                id: assigned.logical_buffer_id,
                                shape: logical.map_or(String::new(), |logical| logical.shape.text(true)),
                                size_mib: mib(assigned.size),
                                hlo_name: logical.map_or(String::new(), |logical| logical.name.clone()),
                                shape_index: logical.map_or(Vec::new(), |logical| logical.shape_index.clone()),
                            }
                        })
                        .collect(),
                    common_shape: if shapes.iter().all_equal() { shapes.first().cloned().unwrap_or_default() } else { String::new() },
                }
            })
            .collect()
    }
}

pub fn render(module: &Module, color: i64, small_buffer: i64, timeline: bool) -> Option<(String, &'static str)> {
    let (rest, bodies) = split(last_bytes(&module.data, 3), 1);
    let assignment = BufferAssignmentProto::decode(rest.as_slice()).unwrap_or_default();
    let logicals = logical_buffers(module, &bodies)?;
    let mut model = Model::new(module, &assignment, &logicals);
    let stats = simulate(&mut model, color)?;
    let (objects, colors, indefinite_bytes) = model.heap_objects(color, small_buffer, &stats);
    let (blocks, rects, has_timeline) = model.layout(color, timeline, &colors)?;
    if timeline {
        if !has_timeline {
            return Some((NO_TIMELINE.to_string(), "text/html"));
        }
        let dot = format!("graph G {{\n epsilon=0.5 \n inputscale=72 \n node [shape=box,style=filled,fontname=\"Arial\",margin=\"0.02,0.02\"];\n  {}\n}}", rects.join("\n"));
        return Some((wrap_dot_html(&dot, "neato"), "text/html"));
    }
    let mut by_size: Vec<usize> = (0..objects.len()).collect();
    std_sort(&mut by_size, &|a, b| objects[a].logical_buffer_size_mib > objects[b].logical_buffer_size_mib);
    let mut to_by_size = vec![0; objects.len()];
    for (position, &object) in by_size.iter().enumerate() {
        to_by_size[object] = position as i32;
    }
    let add_mib = mib(indefinite_bytes);
    let total_mib = |keep: &dyn Fn(&Allocation) -> bool| mib(model.allocations.iter().filter(|allocation| allocation.color == color && keep(allocation)).map(|allocation| allocation.size).sum());
    let indexed_mib = |keep: &dyn Fn(&Allocation) -> bool| mib(model.colored(color).filter(|(allocation, _)| keep(allocation)).map(|(allocation, _)| allocation.size).sum());
    let (scoped, scoped_name) = scoped_vmem(module, color);
    let result = PreprocessResult {
        heap_sizes: stats.timeline.iter().map(|bytes| mib(*bytes) + add_mib).collect(),
        unpadded_heap_sizes: stats.unpadded_timeline.iter().map(|bytes| mib(*bytes) + add_mib).collect(),
        max_heap_by_size: by_size.iter().map(|&index| objects[index].clone()).collect(),
        max_heap: objects,
        logical_buffer_spans: stats
            .seen
            .iter()
            .filter_map(|&index| model.buffers[index].span.map(|(start, limit)| (model.buffers[index].logical.id as i32, BufferSpan { start: start as i32, limit: limit as i32 })))
            .collect(),
        max_heap_to_by_size: to_by_size,
        by_size_to_max_heap: by_size.iter().map(|&index| index as i32).collect(),
        module_name: module.proto.name.clone(),
        entry_computation_name: module.proto.entry_computation_name.clone(),
        peak_heap_mib: mib(stats.peak) + add_mib,
        peak_unpadded_heap_mib: mib(stats.peak_unpadded) + add_mib,
        peak_heap_size_position: stats.peak_position as i32,
        entry_computation_parameters_mib: indexed_mib(&|allocation| allocation.is_entry_computation_parameter),
        non_reusable_mib: indexed_mib(&|allocation| allocation.is_thread_local || allocation.is_tuple),
        maybe_live_out_mib: indexed_mib(&|allocation| allocation.maybe_live_out),
        indefinite_lifetimes: model.indefinite_lifetimes(color),
        total_buffer_allocation_mib: total_mib(&|_| true),
        indefinite_buffer_allocation_mib: total_mib(&indefinite),
        hlo_instruction_names: stats.names,
        max_scoped_vmem_allocation_mib: mib(scoped),
        max_scoped_vmem_instruction_name: scoped_name,
        buffer_blocks: blocks,
        ..Default::default()
    };
    Some((crate::pbtext::json("tensorflow.profiler.PreprocessResult", &result, false), "application/json"))
}

pub fn serve(dir: &std::path::Path, params: &std::collections::HashMap<String, String>) -> Option<(String, &'static str)> {
    let module = crate::hlo::by_options(dir, params)?;
    let color = params.get("memory_space").map_or(Some(0), |value| value.trim_ascii().parse::<i32>().ok()).unwrap_or(0);
    render(&module, i64::from(color), SMALL_BUFFER, params.get("view_memory_allocation_timeline").is_some_and(|value| !value.is_empty()))
}
