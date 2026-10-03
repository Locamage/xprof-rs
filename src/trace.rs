use crate::xplane::{Line, Links, NONE_GROUP, Plane, Step, slice};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::AtomicU8;

pub const NONE_RESOURCE: u32 = u32::MAX;
const COUNTERS: &str = "_counters_";
pub const NONE_FLOW: u64 = u64::MAX;
pub const DERIVED_META: u32 = u32::MAX;
pub const FLOW_START: u8 = 1;
pub const FLOW_MID: u8 = 2;
pub const FLOW_END: u8 = 3;
const LAYER_PS: [u64; 13] = [1_000_000_000_000, 100_000_000_000, 10_000_000_000, 1_000_000_000, 100_000_000, 10_000_000, 1_000_000, 100_000, 10_000, 1_000, 100, 10, 1];
const SPLIT: usize = LAYER_PS.len() - 1;
const STREAMING_THRESHOLD: usize = 500_000;
pub const MAX_SERIAL: u32 = 256;
const BAD_TIMESTAMP: u64 = u64::MAX / 2;
const LEVEL_CHUNK: usize = 1 << 16;

#[derive(Clone, Copy, Default)]
pub struct Event {
    pub ts: u64,
    pub dur: u64,
    pub flow: u64,
    pub group: i64,
    pub raw: (u32, u32),
    pub name: u32,
    pub device: u32,
    pub resource: u32,
    pub track: u32,
    pub serial: u32,
    pub meta: u32,
    pub eager: Option<bool>,
    pub level: u8,
    pub plane: u16,
    pub flow_entry: u8,
    pub flow_cat: u8,
}

#[derive(Default)]
pub struct Device {
    pub name: String,
    pub resources: BTreeMap<u32, String>,
}

pub struct Trace {
    pub devices: BTreeMap<u32, Device>,
    pub names: Vec<Box<str>>,
    pub events: Vec<Event>,
    pub min_ps: u64,
    pub max_ps: u64,
    pub levels: Vec<Vec<u32>>,
    pub tpu_devices: HashSet<u32>,
    pub dma_devices: HashSet<u32>,
    pub long_names: HashMap<u32, Box<str>>,
    pub steps: HashMap<(u32, u32), Step>,
    pub tracks: usize,
    pub flow_ids: Vec<u64>,
    pub args: Vec<Vec<String>>,
}

pub struct Options {
    pub start_ms: f64,
    pub end_ms: f64,
    pub resolution: f64,
    pub full_dma: bool,
}

#[derive(Clone, Default)]
pub struct Row {
    last_end: Vec<u64>,
    last_flow: Option<u64>,
    last_counter: Option<u64>,
}

impl Row {
    fn depth(&self, begin: u64) -> usize {
        self.last_end.partition_point(|&end| end > begin)
    }

    fn push(&mut self, depth: usize, end: u64) {
        self.last_end.truncate(depth);
        self.last_end.push(self.last_end.last().map_or(end, |&outer| outer.min(end)));
    }

    fn counter(&mut self, ts: u64, resolution: u64) -> bool {
        let visible = self.last_counter.is_none_or(|last| ts.wrapping_sub(last) >= resolution);
        if visible {
            self.last_counter = Some(ts);
        }
        visible
    }

    fn probe(&self, ts: u64, dur: u64, resolution: u64) -> (bool, usize) {
        let depth = self.depth(ts);
        (dur >= resolution || self.last_end.get(depth).is_none_or(|last| ts.wrapping_sub(*last) >= resolution), depth)
    }

    fn set(&mut self, ts: u64, dur: u64, is_counter: bool, is_flow: bool) {
        if is_counter {
            self.last_counter = Some(ts);
        } else {
            if is_flow {
                self.last_flow = Some(ts + dur);
            }
            let depth = self.depth(ts);
            self.push(depth, ts + dur);
        }
    }
}

pub struct Visibility {
    pub resolution: u64,
    pub rows: Vec<Row>,
    pub flows: Vec<Option<bool>>,
}

impl Visibility {
    pub fn visible_at_resolution(&mut self, event: &Event) -> bool {
        let (end, resolution) = (event.ts + event.dur, self.resolution);
        let row = &mut self.rows[event.track as usize];
        if event.resource == NONE_RESOURCE {
            return row.counter(event.ts, resolution);
        }
        let (mut visible, depth) = row.probe(event.ts, event.dur, resolution);
        if event.flow != NONE_FLOW {
            let flow = &mut self.flows[event.flow as usize];
            let first = flow.is_none();
            if first {
                *flow = Some(visible);
            }
            if !visible {
                if first {
                    *flow = Some(row.last_flow.is_none_or(|last| end.wrapping_sub(last) >= resolution));
                }
                visible = *flow == Some(true);
            }
            if event.flow_entry == FLOW_END {
                *flow = None;
            }
            if visible {
                row.last_flow = Some(end);
            }
        }
        if visible {
            row.push(depth, end);
        }
        visible
    }

    fn set_visible_at_resolution(&mut self, event: &Event) {
        if let Some(flow) = self.flows.get_mut(event.flow as usize).filter(|_| event.resource != NONE_RESOURCE) {
            *flow = if event.flow_entry == FLOW_END { None } else { flow.or(Some(true)) };
        }
        self.rows[event.track as usize].set(event.ts, event.dur, event.resource == NONE_RESOURCE, event.flow != NONE_FLOW);
    }
}

pub fn is_tpu_core_device_name(name: &str) -> bool {
    name.contains("/device:TPU:") || name.contains("TensorNode") || name.contains("TPU Core")
}

pub fn maybe_tpu_non_core_device_name(name: &str) -> bool {
    name.contains("#Chip") && ["Host Interface", "HBM", "ICI Router"].iter().any(|part| name.contains(part))
}

fn device_planes(planes: &[Plane]) -> Vec<(usize, u32)> {
    let with_prefix = |prefix: &str| (0..planes.len()).filter(|&index| planes[index].name.starts_with(prefix)).collect::<Vec<_>>();
    let mut hosts = with_prefix("/host:CPU");
    hosts.sort_by(|&a, &b| planes[a].name.cmp(&planes[b].name));
    let mut selected: Vec<(usize, u32)> = hosts.into_iter().zip(701..).collect();
    let gpus = with_prefix("/device:GPU:");
    let devices = if gpus.is_empty() { with_prefix("/device:TPU:") } else { gpus };
    selected.extend(devices.into_iter().map(|index| (index, if 1 + planes[index].id > 500 { 1 } else { 1 + planes[index].id as u32 })));
    selected.extend(with_prefix("/device:CUSTOM:").into_iter().map(|index| (index, 501 + planes[index].id as u32)));
    selected
}

fn compare(names: &[Box<str>], a: &Event, b: &Event) -> Ordering {
    a.ts.cmp(&b.ts)
        .then(b.dur.cmp(&a.dur))
        .then(a.device.cmp(&b.device))
        .then((a.resource == NONE_RESOURCE).cmp(&(b.resource == NONE_RESOURCE)))
        .then(a.resource.cmp(&b.resource))
        .then_with(|| if a.resource == NONE_RESOURCE { names[a.name as usize].cmp(&names[b.name as usize]) } else { Ordering::Equal })
}

struct Job<'a> {
    plane: usize,
    line: &'a Line,
    device: u32,
    track: u32,
    label_base: u32,
}

struct Layout<'a> {
    devices: BTreeMap<u32, Device>,
    jobs: Vec<Job<'a>>,
    tracks: usize,
    names: Vec<Box<str>>,
    bases: Vec<u32>,
    long_names: HashMap<u32, Box<str>>,
    steps: HashMap<(u32, u32), Step>,
}

fn layout<'a>(planes: &'a [Plane], host: &str) -> Layout<'a> {
    let per_plane: Vec<Vec<Box<str>>> = planes.par_iter().map(|plane| plane.meta.iter().map(|meta| if meta.display.is_empty() { meta.name.clone() } else { meta.display.clone() }).collect()).collect();
    let bases: Vec<u32> = per_plane.iter().scan(0, |base, names| Some(std::mem::replace(base, *base + names.len() as u32))).collect();
    let mut names: Vec<Box<str>> = per_plane.into_iter().flatten().collect();
    let (mut devices, mut jobs, mut track_ids) = (BTreeMap::<u32, Device>::new(), Vec::new(), FxHashMap::<(u32, u32), u32>::default());
    let (mut long_names, mut steps) = (HashMap::new(), HashMap::new());
    for (plane_index, device) in device_planes(planes) {
        let plane = &planes[plane_index];
        if plane.lines.is_empty() {
            continue;
        }
        devices.entry(device).or_default().name = format!("{host} {}", plane.name);
        for line in plane.lines.iter().filter(|line| !line.events.is_empty()) {
            if line.name != COUNTERS {
                devices.get_mut(&device).unwrap().resources.insert(line.resource_id(), (if line.display_name.is_empty() { &line.name } else { &line.display_name }).clone());
            }
            steps.extend(line.steps.iter().map(|(&index, step)| (line.events[index].raw, step.clone())));
            let label_base = names.len() as u32;
            names.extend(line.labels.iter().cloned());
            long_names.extend(line.longs.iter().enumerate().filter(|(_, long)| !long.is_empty()).map(|(index, long)| (label_base + index as u32, long.clone())));
            let next = track_ids.len() as u32;
            jobs.push(Job { plane: plane_index, line, device, track: *track_ids.entry((device, line.resource_id())).or_insert(next), label_base });
        }
    }
    Layout { devices, jobs, tracks: track_ids.len(), names, bases, long_names, steps }
}

fn convert(planes: &[Plane], map: &[u8], layout: &Layout) -> (Vec<Event>, Vec<Box<str>>, Vec<Vec<String>>) {
    let (extra, extra_base, args) = (Mutex::new(Vec::<Box<str>>::new()), layout.names.len() as u32, Mutex::new(Vec::<Vec<String>>::new()));
    let mut events: Vec<Event> = (0..layout.jobs.iter().map(|job| job.line.events.len()).sum()).into_par_iter().map(|_| Event::default()).collect();
    let (mut rest, mut outputs) = (events.as_mut_slice(), Vec::new());
    for job in &layout.jobs {
        let (head, tail) = rest.split_at_mut(job.line.events.len());
        outputs.push(head);
        rest = tail;
    }
    layout.jobs.par_iter().zip(outputs).for_each(|(&Job { plane: plane_index, line, device, track, label_base }, out)| {
        let (plane, derived) = (&planes[plane_index], !line.labels.is_empty());
        let ordinal = crate::group::ordinal(plane, line);
        out.par_iter_mut().zip(line.events.par_iter()).enumerate().for_each(|(index, (slot, event))| {
            if !derived && plane.meta[event.meta as usize].internal {
                slot.ts = u64::MAX;
                return;
            }
            let links = if derived { Links::default() } else { plane.links(event.meta, slice(map, event.raw), ordinal) };
            let mut name = if derived { label_base + event.meta } else { layout.bases[plane_index] + event.meta };
            if let Some(step) = line.steps.get(&index).map(|step| step.name.as_str()).or(links.step.as_deref()) {
                let mut extra = extra.lock().unwrap();
                extra.push(step.into());
                name = extra_base + extra.len() as u32 - 1;
            }
            let (flow, flow_entry, flow_cat) =
                links.flow.map_or((NONE_FLOW, 0, 0), |encoded| ((encoded >> 2) & ((1 << 56) - 1), [0, FLOW_END, FLOW_START, FLOW_MID][(encoded & 3) as usize], (encoded >> 58) as u8));
            let raw = match line.args.get(&index) {
                Some(extra) => {
                    let mut args = args.lock().unwrap();
                    args.push(extra.clone());
                    (u32::MAX, args.len() as u32 - 1)
                }
                None => event.raw,
            };
            *slot = Event {
                ts: event.ts,
                dur: event.dur,
                flow,
                group: if event.group != NONE_GROUP { event.group } else { links.group.unwrap_or(NONE_GROUP) },
                raw,
                name,
                device,
                resource: if line.name == COUNTERS || (links.asynchronous.is_some() && flow != NONE_FLOW) { NONE_RESOURCE } else { line.resource_id() },
                track,
                meta: if derived { DERIVED_META } else { event.meta },
                eager: event.eager,
                plane: plane_index as u16,
                flow_entry,
                flow_cat,
                ..Default::default()
            };
        });
    });
    (events, extra.into_inner().unwrap(), args.into_inner().unwrap())
}

impl Trace {
    pub fn build(planes: &[Plane], host: &str, map: &[u8]) -> Trace {
        let layout = layout(planes, host);
        let (events, extra, args) = convert(planes, map, &layout);
        let Layout { devices, tracks, mut names, long_names, steps, .. } = layout;
        names.extend(extra);
        let mut keys: Vec<(u64, u32)> = events.par_iter().enumerate().map(|(index, event)| (event.ts, index as u32)).collect();
        keys.par_sort_by(|a, b| a.0.cmp(&b.0).then_with(|| compare(&names, &events[a.1 as usize], &events[b.1 as usize])));
        keys.truncate(keys.partition_point(|key| key.0 != u64::MAX));
        let mut events: Vec<Event> = keys.par_iter().map(|key| events[key.1 as usize]).collect();
        drop(keys);
        let (mut async_tracks, mut by_track) = (FxHashMap::<(u32, &str), u32>::default(), vec![Vec::new(); tracks]);
        let (mut flow_index, mut flow_ids, mut previous) = (FxHashMap::<u64, u64>::default(), Vec::new(), (u64::MAX, 0));
        for (index, event) in events.iter_mut().enumerate() {
            previous = (event.ts, if event.ts == previous.0 { previous.1 + 1 } else { 0 });
            event.serial = previous.1;
            if event.resource == NONE_RESOURCE {
                event.track = *async_tracks.entry((event.device, &*names[event.name as usize])).or_insert_with(|| {
                    by_track.push(Vec::new());
                    by_track.len() as u32 - 1
                });
            }
            if event.flow != NONE_FLOW {
                event.flow = *flow_index.entry(event.flow).or_insert_with(|| {
                    flow_ids.push(event.flow);
                    flow_ids.len() as u64 - 1
                });
            }
            by_track[event.track as usize].push(index as u32);
        }
        let span = (
            events.iter().map(|event| event.ts).find(|&ts| ts <= BAD_TIMESTAMP).unwrap_or(0),
            events.par_iter().map(|event| event.ts.saturating_add(event.dur)).filter(|&end| end <= BAD_TIMESTAMP).max().unwrap_or(0),
        );
        let assigned = assign_levels(&events, &by_track, flow_ids.len());
        let chunks: Vec<Vec<Vec<u32>>> = events
            .par_chunks_mut(LEVEL_CHUNK)
            .zip(assigned.par_chunks(LEVEL_CHUNK))
            .enumerate()
            .map(|(chunk, (events, assigned))| {
                let mut levels = vec![Vec::new(); LAYER_PS.len()];
                for (offset, (event, &level)) in events.iter_mut().zip(assigned).enumerate() {
                    event.level = level;
                    if event.serial < MAX_SERIAL {
                        levels[level as usize].push((chunk * LEVEL_CHUNK + offset) as u32);
                    }
                }
                levels
            })
            .collect();
        let levels = (0..LAYER_PS.len()).into_par_iter().map(|level| chunks.iter().flat_map(|chunk| chunk[level].iter().copied()).collect()).collect();
        let tpu_devices: HashSet<u32> = devices.iter().filter(|(_, device)| is_tpu_core_device_name(&device.name)).map(|(id, _)| *id).collect();
        let dma_devices =
            devices.iter().filter(|(_, device)| !tpu_devices.is_empty() && (is_tpu_core_device_name(&device.name) || maybe_tpu_non_core_device_name(&device.name))).map(|(id, _)| *id).collect();
        Trace { devices, names, events, min_ps: span.0, max_ps: span.1, levels, tpu_devices, dma_devices, long_names, steps, tracks: by_track.len(), flow_ids, args }
    }

    fn is_dma_flow(&self, index: u32) -> bool {
        let event = &self.events[index as usize];
        event.flow != NONE_FLOW && event.flow_entry == FLOW_MID && self.dma_devices.contains(&event.device)
    }

    pub fn load(&self, options: &Options) -> Vec<u32> {
        let ms_to_ps = |ms: f64| (ms * 1e9) as u64;
        let (mut begin, mut end) = (ms_to_ps(options.start_ms), ms_to_ps(options.end_ms));
        if !self.events.is_empty() {
            begin = begin.max(self.min_ps);
            if end == 0 || end > self.max_ps {
                end = self.max_ps;
            }
        }
        end = end.max(begin);
        let resolution = if self.events.len() >= STREAMING_THRESHOLD { options.resolution } else { 0.0 };
        let resolution_ps = if resolution == 0.0 { 0 } else { ((end - begin) as f64 / resolution).round() as u64 };
        let layer = |level: usize| LAYER_PS.get(level).copied().unwrap_or(0);
        let mut sources: Vec<&[u32]> = Vec::new();
        for (level, indices) in self.levels.iter().enumerate() {
            let first_ps = if level > 0 && begin > layer(level - 1) { begin - layer(level - 1) } else { 0 };
            let (start, stop) = (indices.partition_point(|&index| self.events[index as usize].ts < first_ps), indices.partition_point(|&index| self.events[index as usize].ts <= end));
            sources.push(&indices[start..stop.max(start)]);
            if resolution_ps >= layer(level) {
                break;
            }
        }
        let mut loaded = self.merge(sources);
        let mut visibility = Visibility { resolution: resolution_ps, rows: vec![Row::default(); self.tracks], flows: vec![None; self.flow_ids.len()] };
        loaded.retain(|&index| {
            let event = &self.events[index as usize];
            if self.is_dma_flow(index) {
                options.full_dma
            } else {
                begin == end || (event.ts <= end && begin <= event.ts + event.dur && (resolution_ps == 0 || visibility.visible_at_resolution(event)))
            }
        });
        loaded
    }

    fn merge(&self, sources: Vec<&[u32]>) -> Vec<u32> {
        let mut out = Vec::with_capacity(sources.iter().map(|source| source.len()).sum());
        let mut heap: Vec<&[u32]> = sources.into_iter().filter(|source| !source.is_empty()).collect();
        let after = |a: &[u32], b: &[u32]| compare(&self.names, &self.events[b[0] as usize], &self.events[a[0] as usize]) == Ordering::Less;
        let length = heap.len();
        if length >= 2 {
            for parent in (0..=(length - 2) / 2).rev() {
                let (value, top) = (heap[parent], parent);
                let (mut hole, mut child) = (parent, parent);
                while child < (length - 1) / 2 {
                    child = 2 * (child + 1);
                    if after(heap[child], heap[child - 1]) {
                        child -= 1;
                    }
                    heap[hole] = heap[child];
                    hole = child;
                }
                if length.is_multiple_of(2) && child == (length - 2) / 2 {
                    child = 2 * (child + 1);
                    heap[hole] = heap[child - 1];
                    hole = child - 1;
                }
                while hole > top && after(heap[(hole - 1) / 2], value) {
                    heap[hole] = heap[(hole - 1) / 2];
                    hole = (hole - 1) / 2;
                }
                heap[hole] = value;
            }
        }
        while let Some(&front) = heap.first() {
            out.push(front[0]);
            heap[0] = &front[1..];
            if heap[0].is_empty() {
                if heap.len() == 1 {
                    break;
                }
                heap.swap_remove(0);
            }
            let (value, mut hole) = (heap[0], 0);
            loop {
                let mut child = 2 * hole + 1;
                if child + 1 < heap.len() && after(heap[child], heap[child + 1]) {
                    child += 1;
                }
                if child >= heap.len() || !after(value, heap[child]) {
                    break;
                }
                heap[hole] = heap[child];
                hole = child;
            }
            heap[hole] = value;
        }
        out
    }

    pub fn search(&self, prefix: &str, full_dma: bool) -> Vec<u32> {
        let mut found: Vec<u32> = (0..self.events.len() as u32)
            .filter(|&index| self.events[index as usize].serial < MAX_SERIAL && self.names[self.events[index as usize].name as usize].starts_with(prefix) && (full_dma || !self.is_dma_flow(index)))
            .collect();
        found.sort_by_key(|&index| (self.events[index as usize].level, index));
        found
    }
}

fn assign_levels(events: &[Event], by_track: &[Vec<u32>], flow_count: usize) -> Vec<u8> {
    let mut global: Vec<Visibility> = LAYER_PS[..SPLIT].iter().map(|&resolution| Visibility { resolution, rows: vec![Row::default(); by_track.len()], flows: vec![None; flow_count] }).collect();
    let mut flow_visible = vec![vec![None; flow_count]; SPLIT];
    for event in events.iter().filter(|event| event.flow != NONE_FLOW) {
        let mut level = 0;
        while level < SPLIT {
            let visible = global[level].visible_at_resolution(event);
            flow_visible[level][event.flow as usize].get_or_insert(visible);
            if visible {
                break;
            }
            level += 1;
        }
        for deeper in level + 1..SPLIT {
            global[deeper].set_visible_at_resolution(event);
            flow_visible[deeper][event.flow as usize].get_or_insert(true);
        }
    }
    let levels: Vec<AtomicU8> = (0..events.len()).map(|_| AtomicU8::new(0)).collect();
    by_track.par_iter().for_each(|indices| {
        let compact: Vec<(u64, u64, u8)> = indices
            .iter()
            .map(|&index| (events[index as usize].ts, events[index as usize].dur, ((events[index as usize].flow != NONE_FLOW) as u8) << 1 | (events[index as usize].resource == NONE_RESOURCE) as u8))
            .collect();
        let mut assigned: Vec<usize> = indices
            .iter()
            .map(|&index| {
                let event = &events[index as usize];
                if event.flow == NONE_FLOW { usize::MAX } else { (0..SPLIT).find(|&level| flow_visible[level][event.flow as usize] == Some(true) || event.dur >= LAYER_PS[level]).unwrap_or(SPLIT) }
            })
            .collect();
        for (level, &resolution) in LAYER_PS[..SPLIT].iter().enumerate() {
            let mut row = Row::default();
            for (position, &(ts, dur, kind)) in compact.iter().enumerate() {
                if assigned[position] < level || (kind & 2 != 0 && assigned[position] == level) {
                    row.set(ts, dur, kind & 1 != 0, false);
                } else if kind & 2 == 0 && assigned[position] == usize::MAX {
                    let (visible, depth) = if kind & 1 != 0 { (row.counter(ts, resolution), 0) } else { row.probe(ts, dur, resolution) };
                    if visible {
                        assigned[position] = level;
                        if kind & 1 == 0 {
                            row.push(depth, ts + dur);
                        }
                    }
                }
            }
        }
        indices.iter().zip(assigned).for_each(|(&index, level)| levels[index as usize].store(level.min(SPLIT) as u8, std::sync::atomic::Ordering::Relaxed));
    });
    levels.into_iter().map(AtomicU8::into_inner).collect()
}
