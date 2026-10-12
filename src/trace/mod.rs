pub mod delta;
pub mod json;
pub mod legacy;

use crate::xplane::{Ev, Links, NONE_GROUP, Plane, Step, slice};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicU8, AtomicUsize};
use std::sync::{Condvar, Mutex};

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
/// From this event count, and from the share of the events of one thread, the level passes of a track run at the same time, a part of `PART` events apart. The tests use small values, and no share.
const PIPELINE: usize = if cfg!(test) { 2 } else { 1 << 20 };
const PART: usize = if cfg!(test) { 4 } else { 1 << 16 };
const UNASSIGNED: u8 = u8::MAX;
/// The events of a bucket of the sort, and of a part of the conversion. The tests use small values, so that they have many buckets and parts.
const BUCKET: usize = if cfg!(test) { 4 } else { 1 << 16 };
const MAX_BUCKETS: usize = 1024;
const UNIT: usize = if cfg!(test) { 8 } else { 1 << 16 };
/// The units that the conversion writes before it frees the input events that it has converted.
const WAVE: usize = if cfg!(test) { 2 } else { 128 };

/// Strings with dense ids in the order of their first use.
#[derive(Default)]
pub struct Interner(indexmap::IndexSet<String, rustc_hash::FxBuildHasher>);

impl Interner {
    pub fn intern(&mut self, text: &str) -> u32 {
        self.0.get_index_of(text).unwrap_or_else(|| self.0.insert_full(text.to_string()).0) as u32
    }

    pub fn get(&self, id: u32) -> &str {
        &self.0[id as usize]
    }
}

#[derive(Clone, Copy, Default)]
pub struct Event {
    pub ts: u64,
    pub dur: u64,
    pub flow: u64,
    pub group: i64,
    pub raw: (u32, u32),
    pub name: u32,
    pub track: u32,
    pub serial: u32,
    pub meta: u32,
    pub plane: u32,
    pub eager: Option<bool>,
    pub level: u8,
    /// The flow entry in the low 2 bits, and the flow category above them.
    pub flow_kind: u8,
    /// The event has no resource. After the build, its track is an async track.
    pub unbound: bool,
}

impl Event {
    pub fn flow_entry(&self) -> u8 {
        self.flow_kind & 3
    }

    pub fn flow_cat(&self) -> u8 {
        self.flow_kind >> 2
    }
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
    /// Bit `index` is 1 when event `index` compares equal to the event before it.
    pub ties: Vec<u64>,
    pub tpu_devices: HashSet<u32>,
    pub dma_devices: HashSet<u32>,
    pub long_names: HashMap<u32, Box<str>>,
    pub steps: HashMap<(u32, u32), Step>,
    pub tracks: usize,
    /// The device and the resource of each track. An async track has the resource `NONE_RESOURCE`.
    pub owners: Vec<(u32, u32)>,
    pub flow_ids: Vec<u64>,
    pub args: Vec<Vec<String>>,
    pub stack_frames: String,
    pub frame_keys: Vec<u64>,
}

pub struct Options {
    pub start_ms: f64,
    pub end_ms: f64,
    pub resolution: f64,
    pub full_dma: bool,
}

/// `absl::SimpleAtod`: one `+` sign is permitted, and `0x` starts a hexadecimal number.
pub fn atod(text: &str) -> Option<f64> {
    let text = text.trim_matches(|c| matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r'));
    let (plus, text) = text.strip_prefix('+').map_or((false, text), |rest| (true, rest));
    let (negative, body) = text.strip_prefix('-').map_or((false, text), |rest| (true, rest));
    if (plus && negative) || body.starts_with(['+', '-']) {
        return None;
    }
    let value = if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        hex_float(hex)?
    } else if body.len() > 5 && body[..4].eq_ignore_ascii_case("nan(") && body.ends_with(')') {
        body[4..body.len() - 1].bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_').then_some(f64::NAN)?
    } else {
        body.parse::<f64>().ok()?
    };
    Some(if negative { -value } else { value })
}

fn hex_float(text: &str) -> Option<f64> {
    let (mantissa, exponent) = text.split_once(['p', 'P']).unwrap_or((text, "0"));
    let (sign, digits) = exponent.strip_prefix('-').map_or((1, exponent.strip_prefix('+').unwrap_or(exponent)), |rest| (-1, rest));
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) || whole.len() + fraction.len() == 0 {
        return None;
    }
    let mut exponent = sign * digits.bytes().fold(0i64, |sum, byte| (sum * 10 + i64::from(byte - b'0')).min(1 << 20)) - 4 * fraction.len() as i64;
    let (mut value, mut sticky) = (0u128, false);
    for byte in whole.bytes().chain(fraction.bytes()) {
        let digit = u128::from(char::from(byte).to_digit(16)?);
        if value >> 120 == 0 {
            value = value << 4 | digit;
        } else {
            sticky |= digit != 0;
            exponent += 4;
        }
    }
    let shift = (128 - value.leading_zeros()).saturating_sub(55);
    let top = (value >> shift) as u64 | u64::from(sticky || value & ((1 << shift) - 1) != 0);
    let exponent = (exponent + i64::from(shift)).clamp(-4000, 4000) as i32;
    Some(if top == 0 { 0.0 } else { top as f64 * 2f64.powi(exponent / 2) * 2f64.powi(exponent - exponent / 2) })
}

/// The conversion of a C++ `double` to `uint64_t` on x86-64: a negative value wraps, and a value out of range gives 2^63.
pub fn to_u64(value: f64) -> u64 {
    if (0.0..(1u128 << 64) as f64).contains(&value) {
        value as u64
    } else if (-((1u64 << 63) as f64)..0.0).contains(&value) {
        value as i64 as u64
    } else {
        1 << 63
    }
}

/// `absl::SimpleAtoi` for `uint64_t`, else `absl::SimpleAtod` and a cast to `uint64_t`.
pub fn atou(text: &str) -> Option<u64> {
    let trimmed = text.trim_matches(|c| matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r'));
    trimmed.parse::<u64>().ok().or_else(|| atod(trimmed).map(to_u64))
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
            self.push(self.depth(ts), ts + dur);
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
        if event.unbound {
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
            if event.flow_entry() == FLOW_END {
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
    selected.extend(devices.into_iter().map(|index| (index, Some(1u32.wrapping_add(planes[index].id as u32)).filter(|&pid| pid <= 500).unwrap_or(1))));
    selected.extend(with_prefix("/device:CUSTOM:").into_iter().map(|index| (index, 501u32.wrapping_add(planes[index].id as u32))));
    selected
}

fn compare(names: &[Box<str>], owners: &[(u32, u32)], a: &Event, b: &Event) -> Ordering {
    let owner = |event: &Event| {
        let (device, resource) = owners[event.track as usize];
        (device, event.unbound, if event.unbound { NONE_RESOURCE } else { resource })
    };
    a.ts.cmp(&b.ts).then(b.dur.cmp(&a.dur)).then_with(|| owner(a).cmp(&owner(b))).then_with(|| if a.unbound { names[a.name as usize].cmp(&names[b.name as usize]) } else { Ordering::Equal })
}

struct Job {
    plane: usize,
    line: usize,
    track: u32,
    label_base: u32,
}

struct Layout {
    devices: BTreeMap<u32, Device>,
    jobs: Vec<Job>,
    owners: Vec<(u32, u32)>,
    names: Vec<Box<str>>,
    bases: Vec<u32>,
    long_names: HashMap<u32, Box<str>>,
    steps: HashMap<(u32, u32), Step>,
}

fn layout(planes: &[Plane], host: &str) -> Layout {
    let per_plane: Vec<Vec<Box<str>>> = planes.par_iter().map(|plane| plane.meta.iter().map(|meta| if meta.display.is_empty() { meta.name.clone() } else { meta.display.clone() }).collect()).collect();
    let bases: Vec<u32> = per_plane.iter().scan(0, |base, names| Some(std::mem::replace(base, *base + names.len() as u32))).collect();
    let mut names: Vec<Box<str>> = per_plane.into_iter().flatten().collect();
    let (mut devices, mut jobs, mut owners, mut track_ids) = (BTreeMap::<u32, Device>::new(), Vec::new(), Vec::new(), FxHashMap::<(u32, u32), u32>::default());
    let (mut long_names, mut steps) = (HashMap::new(), HashMap::new());
    for (plane_index, device) in device_planes(planes) {
        let plane = &planes[plane_index];
        if plane.lines.is_empty() {
            continue;
        }
        let entry = devices.entry(device).or_default();
        entry.name = format!("{host} {}", plane.name);
        for (line_index, line) in plane.lines.iter().enumerate().filter(|(_, line)| !line.events.is_empty()) {
            if line.name != COUNTERS {
                entry.resources.insert(line.resource_id(), (if line.display_name.is_empty() { &line.name } else { &line.display_name }).clone());
            }
            steps.extend(line.steps.iter().map(|(&index, step)| (line.events[index].raw, step.clone())));
            let label_base = names.len() as u32;
            names.extend(line.labels.iter().cloned());
            long_names.extend(line.longs.iter().enumerate().filter(|(_, long)| !long.is_empty()).map(|(index, long)| (label_base + index as u32, long.clone())));
            let owner = (device, line.resource_id());
            let track = *track_ids.entry(owner).or_insert_with(|| {
                owners.push(owner);
                owners.len() as u32 - 1
            });
            jobs.push(Job { plane: plane_index, line: line_index, track, label_base });
        }
    }
    Layout { devices, jobs, owners, names, bases, long_names, steps }
}

/// The slots of a vector that threads fill in parallel. Each input index has a different slot.
struct Slots<T>(*mut MaybeUninit<T>);

// SAFETY: the threads write to different slots.
unsafe impl<T: Send> Sync for Slots<T> {}

impl<T> Slots<T> {
    /// # Safety
    /// Only one thread writes to `index`, and `index` is in the allocation.
    unsafe fn write(&self, index: usize, value: T) {
        unsafe { self.0.add(index).write(MaybeUninit::new(value)) };
    }
}

/// Changes the counts of each group in each unit into the first slot of the group in the unit. The groups come one after the other, and in a group, the units keep their order. Gives the start of each group, and the total at the end.
fn into_cursors(units: &mut [Vec<u32>], groups: usize) -> Vec<usize> {
    let (mut bounds, mut next) = (Vec::with_capacity(groups + 1), 0);
    for group in 0..groups {
        bounds.push(next as usize);
        for unit in units.iter_mut() {
            next += std::mem::replace(&mut unit[group], next);
        }
    }
    bounds.push(next as usize);
    bounds
}

/// Calls `visit` with the index of the job, the index in the line and the event, for the input indices in `range`.
fn each_event<'a>(inputs: &'a [Vec<Ev>], starts: &[usize], range: Range<usize>, mut visit: impl FnMut(usize, usize, &'a Ev)) {
    let (mut job, mut global) = (starts.partition_point(|&start| start <= range.start) - 1, range.start);
    while global < range.end {
        let events = &inputs[job];
        let first = global - starts[job];
        for (index, event) in events[first..(range.end - starts[job]).min(events.len())].iter().enumerate() {
            visit(job, first + index, event);
            global += 1;
        }
        job += 1;
    }
}

/// The events, the extra names, the extra arguments, and the start of each bucket.
type Converted = (Vec<Event>, Vec<Box<str>>, Vec<Vec<String>>, Vec<usize>);

/// Gives the events in buckets of time, and the start of each bucket. All events of a bucket come before the events of the next bucket, and in a bucket, the events keep their input order.
/// `inputs` has the events of each job. The conversion frees the events of a job when it has all of them, so the input and the output are not both full at the same time.
fn convert(planes: &[Plane], mut inputs: Vec<Vec<Ev>>, map: &[u8], layout: &Layout) -> Converted {
    let (extra, extra_base, args) = (Mutex::new(Vec::<Box<str>>::new()), layout.names.len() as u32, Mutex::new(Vec::<Vec<String>>::new()));
    let jobs = &layout.jobs;
    let line_of = |job: usize| &planes[jobs[job].plane].lines[jobs[job].line];
    let starts: Vec<usize> = inputs.iter().scan(0, |next, events| Some(std::mem::replace(next, *next + events.len()))).collect();
    let total = inputs.iter().map(Vec::len).sum::<usize>();
    let ordinals: Vec<Option<u64>> = (0..jobs.len()).map(|job| crate::xplane::group::ordinal(&planes[jobs[job].plane], line_of(job))).collect();
    let internal = |job: usize, event: &Ev| line_of(job).labels.is_empty() && planes[jobs[job].plane].meta[event.meta as usize].internal;
    let key = |job: usize, event: &Ev| if internal(job, event) { u64::MAX } else { event.ts };
    let buckets = (total / BUCKET).clamp(1, MAX_BUCKETS);
    let splitters: Vec<u64> = if buckets == 1 {
        Vec::new()
    } else {
        let mut samples: Vec<u64> = (0..total)
            .step_by((total / (buckets * 16)).max(1))
            .map(|global| {
                let job = starts.partition_point(|&start| start <= global) - 1;
                key(job, &inputs[job][global - starts[job]])
            })
            .collect();
        samples.par_sort_unstable();
        (1..buckets).map(|bucket| samples[bucket * samples.len() / buckets]).collect()
    };
    // Most events of a line come in time order, so the search first checks the bucket of the event before.
    let bucket = |last: &mut usize, job: usize, event: &Ev| {
        let key = key(job, event);
        if (*last > 0 && splitters[*last - 1] > key) || splitters.get(*last).is_some_and(|&splitter| splitter <= key) {
            *last = splitters.partition_point(|&splitter| splitter <= key);
        }
        *last
    };
    let range = |unit: usize| unit * UNIT..((unit + 1) * UNIT).min(total);
    let mut cursors: Vec<Vec<u32>> = (0..total.div_ceil(UNIT))
        .into_par_iter()
        .map(|unit| {
            let (mut counts, mut last) = (vec![0; buckets], 0);
            each_event(&inputs, &starts, range(unit), |job, _, event| counts[bucket(&mut last, job, event)] += 1);
            counts
        })
        .collect();
    let bounds = into_cursors(&mut cursors, buckets);
    // An event without link stats has the links of its metadata, thus `plain` keeps these links for each job and metadata.
    let converted = |job: usize, index: usize, event: &Ev, plain: &mut FxHashMap<(usize, u32), Links>| {
        let Job { plane: plane_index, track, label_base, .. } = jobs[job];
        let line = line_of(job);
        let (plane, derived) = (&planes[plane_index], !line.labels.is_empty());
        let fresh;
        let links = if derived {
            fresh = Links::default();
            &fresh
        } else if event.linked {
            fresh = plane.links(event.meta, slice(map, event.raw), ordinals[job]);
            &fresh
        } else {
            &*plain.entry((job, event.meta)).or_insert_with(|| plane.links(event.meta, &[], ordinals[job]))
        };
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
        Event {
            ts: event.ts,
            dur: event.dur,
            flow,
            group: if event.group == NONE_GROUP { links.group.unwrap_or(NONE_GROUP) } else { event.group },
            raw,
            name,
            track,
            meta: if derived { DERIVED_META } else { event.meta },
            plane: plane_index as u32,
            eager: event.eager,
            flow_kind: flow_entry | flow_cat << 2,
            unbound: line.name == COUNTERS || (links.asynchronous.is_some() && flow != NONE_FLOW),
            ..Default::default()
        }
    };
    let mut events: Vec<Event> = Vec::with_capacity(total);
    let slots = Slots(events.spare_capacity_mut().as_mut_ptr());
    let mut freed = 0;
    for (wave, cursors) in cursors.chunks_mut(WAVE).enumerate() {
        cursors.par_iter_mut().enumerate().for_each(|(offset, cursors)| {
            let (mut last, mut plain) = (0, FxHashMap::default());
            each_event(&inputs, &starts, range(wave * WAVE + offset), |job, index, event| {
                let slot = &mut cursors[bucket(&mut last, job, event)];
                let out = if internal(job, event) { Event { ts: u64::MAX, ..Default::default() } } else { converted(job, index, event, &mut plain) };
                // SAFETY: the cursors of the units give each input index a different slot below `total`.
                unsafe { slots.write(*slot as usize, out) };
                *slot += 1;
            });
        });
        let done = ((wave + 1) * WAVE * UNIT).min(total);
        while freed < inputs.len() && starts[freed] + inputs[freed].len() <= done {
            crate::discard(std::mem::take(&mut inputs[freed]));
            freed += 1;
        }
    }
    // SAFETY: the units write all `total` slots.
    unsafe { events.set_len(total) };
    (events, extra.into_inner().unwrap(), args.into_inner().unwrap(), bounds)
}

impl Trace {
    /// Moves the lines out of `planes` when they are converted.
    pub fn build(planes: &mut [Plane], host: &str, map: &[u8]) -> Self {
        let layout = layout(planes, host);
        let inputs = layout.jobs.iter().map(|job| std::mem::take(&mut planes[job.plane].lines[job.line].events)).collect();
        let (mut events, extra, args, bounds) = convert(planes, inputs, map, &layout);
        let Layout { devices, mut owners, mut names, long_names, steps, .. } = layout;
        let tracks = owners.len();
        crate::release(planes.iter_mut().map(|plane| std::mem::take(&mut plane.lines)).collect::<Vec<_>>());
        let planes = &*planes;
        names.extend(extra);
        let (mut buckets, mut rest) = (Vec::with_capacity(bounds.len()), events.as_mut_slice());
        for pair in bounds.windows(2) {
            let (bucket, tail) = rest.split_at_mut(pair[1] - pair[0]);
            buckets.push(bucket);
            rest = tail;
        }
        buckets.into_par_iter().for_each(|bucket| {
            let mut keys: Vec<(u64, u32)> = bucket.iter().enumerate().map(|(index, event)| (event.ts, index as u32)).collect();
            keys.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| compare(&names, &owners, &bucket[a.1 as usize], &bucket[b.1 as usize])));
            let sorted: Vec<Event> = keys.iter().map(|key| bucket[key.1 as usize]).collect();
            bucket.copy_from_slice(&sorted);
        });
        events.truncate(events.partition_point(|event| event.ts != u64::MAX));
        let part = events.len().div_ceil(rayon::current_num_threads() * 4).max(1);
        let run_starts: Vec<usize> = (0..events.len()).step_by(part).map(|start| events.partition_point(|event| event.ts < events[start].ts)).collect();
        // Each part gives its new async tracks and flows in the order of their first event, and these get ids in the order of the parts. A part counts the events of each track, and of each of its new async tracks.
        let units: Vec<_> = events
            .par_chunks_mut(part)
            .zip(run_starts)
            .enumerate()
            .map(|(unit, (events, mut run))| {
                let (mut counts, mut async_counts, mut flows, mut flowing) = (vec![0; tracks], Vec::new(), Vec::new(), Vec::new());
                let (mut local, mut seen_flows) = (indexmap::IndexMap::<(u32, &str), (), rustc_hash::FxBuildHasher>::default(), FxHashSet::default());
                let mut previous = events[0].ts;
                for (offset, event) in events.iter_mut().enumerate() {
                    let index = unit * part + offset;
                    if event.ts != previous {
                        (run, previous) = (index, event.ts);
                    }
                    event.serial = (index - run) as u32;
                    if event.unbound {
                        let id = local.insert_full((owners[event.track as usize].0, &*names[event.name as usize]), ()).0;
                        if id == async_counts.len() {
                            async_counts.push(0);
                        }
                        async_counts[id] += 1;
                    } else {
                        counts[event.track as usize] += 1;
                    }
                    if event.flow != NONE_FLOW {
                        flowing.push(index as u32);
                        if seen_flows.insert(event.flow) {
                            flows.push(event.flow);
                        }
                    }
                }
                (counts, local.into_keys().zip(async_counts).collect::<Vec<_>>(), flows, flowing)
            })
            .collect();
        let (mut async_tracks, mut flow_index, mut flow_ids) = (FxHashMap::<(u32, &str), u32>::default(), FxHashMap::<u64, u64>::default(), Vec::new());
        for (_, unit_tracks, unit_flows, _) in &units {
            for &(key, _) in unit_tracks {
                async_tracks.entry(key).or_insert_with(|| {
                    owners.push((key.0, NONE_RESOURCE));
                    owners.len() as u32 - 1
                });
            }
            for &flow in unit_flows {
                flow_index.entry(flow).or_insert_with(|| {
                    flow_ids.push(flow);
                    flow_ids.len() as u64 - 1
                });
            }
        }
        let tracks = owners.len();
        let (mut cursors, mut flowing) = (Vec::with_capacity(units.len()), Vec::new());
        for (mut counts, unit_tracks, _, unit_flowing) in units {
            counts.resize(tracks, 0);
            for (key, count) in unit_tracks {
                counts[async_tracks[&key] as usize] += count;
            }
            cursors.push(counts);
            flowing.extend(unit_flowing);
        }
        let bounds = into_cursors(&mut cursors, tracks);
        let mut order: Vec<u32> = Vec::with_capacity(events.len());
        let slots = Slots(order.spare_capacity_mut().as_mut_ptr());
        events.par_chunks_mut(part).zip(cursors).enumerate().for_each(|(unit, (events, mut cursors))| {
            for (offset, event) in events.iter_mut().enumerate() {
                if event.unbound {
                    event.track = async_tracks[&(owners[event.track as usize].0, &*names[event.name as usize])];
                }
                if event.flow != NONE_FLOW {
                    event.flow = flow_index[&event.flow];
                }
                let slot = &mut cursors[event.track as usize];
                // SAFETY: the cursors of the units give each event a different slot below `events.len()`.
                unsafe { slots.write(*slot as usize, (unit * part + offset) as u32) };
                *slot += 1;
            }
        });
        // SAFETY: the units write all slots.
        unsafe { order.set_len(events.len()) };
        let by_track: Vec<&[u32]> = bounds.windows(2).map(|pair| &order[pair[0]..pair[1]]).collect();
        let span = (
            events.iter().map(|event| event.ts).find(|&ts| ts <= BAD_TIMESTAMP).unwrap_or(0),
            events.par_iter().map(|event| event.ts.saturating_add(event.dur)).filter(|&end| end <= BAD_TIMESTAMP).max().unwrap_or(0),
        );
        let ((stack_frames, frame_keys), (assigned, ties)) = rayon::join(
            || crate::trace::json::stack_frames(planes, map, &events, &long_names),
            || {
                let assigned = assign_levels(&events, &by_track, &flowing, flow_ids.len());
                let tie = |index: usize| index > 0 && index < events.len() && compare(&names, &owners, &events[index - 1], &events[index]).is_eq();
                (assigned, (0..events.len().div_ceil(64)).into_par_iter().map(|word| (0..64).filter(|bit| tie(word * 64 + bit)).fold(0, |bits, bit| bits | 1 << bit)).collect())
            },
        );
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
        Self {
            devices,
            names,
            events,
            min_ps: span.0,
            max_ps: span.1,
            levels,
            ties,
            tpu_devices,
            dma_devices,
            long_names,
            steps,
            tracks: by_track.len(),
            owners,
            flow_ids,
            args,
            stack_frames,
            frame_keys,
        }
    }

    fn is_dma_flow(&self, index: u32) -> bool {
        let event = &self.events[index as usize];
        event.flow != NONE_FLOW && event.flow_entry() == FLOW_MID && self.dma_devices.contains(&self.device(event))
    }

    pub fn device(&self, event: &Event) -> u32 {
        self.owners[event.track as usize].0
    }

    pub fn resource(&self, event: &Event) -> u32 {
        if event.unbound { NONE_RESOURCE } else { self.owners[event.track as usize].1 }
    }

    pub fn load(&self, options: &Options) -> Vec<u32> {
        let mut begin = to_u64(options.start_ms * 1e9);
        let mut end = to_u64(options.end_ms * 1e9).max(begin);
        if !self.events.is_empty() {
            begin = begin.max(self.min_ps);
            if end == 0 || end > self.max_ps {
                end = self.max_ps;
            }
            end = end.max(begin);
        }
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
        // The events are in order, so an event before another compares less, unless the two are equal.
        let tied = |index: u32| self.ties[index as usize / 64] >> (index % 64) & 1 != 0;
        let after = |a: &[u32], b: &[u32]| b[0] < a[0] && (!tied(a[0]) || compare(&self.names, &self.owners, &self.events[b[0] as usize], &self.events[a[0] as usize]) == Ordering::Less);
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
        let matching: Vec<bool> = self.names.par_iter().map(|name| name.starts_with(prefix)).collect();
        let mut found: Vec<u32> = self
            .events
            .par_iter()
            .enumerate()
            .filter(|(index, event)| event.serial < MAX_SERIAL && matching[event.name as usize] && (full_dma || !self.is_dma_flow(*index as u32)))
            .map(|(index, _)| index as u32)
            .collect();
        found.par_sort_unstable_by_key(|&index| (self.events[index as usize].level, index));
        found
    }
}

/// `flowing` has the indices of the events with a flow, in order.
fn assign_levels(events: &[Event], by_track: &[&[u32]], flowing: &[u32], flow_count: usize) -> Vec<u8> {
    let mut global: Vec<Visibility> = LAYER_PS[..SPLIT].iter().map(|&resolution| Visibility { resolution, rows: vec![Row::default(); by_track.len()], flows: vec![None; flow_count] }).collect();
    let mut flow_visible = vec![vec![None; flow_count]; SPLIT];
    for event in flowing.iter().map(|&index| &events[index as usize]) {
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
            let resolution = &mut global[deeper];
            if let Some(flow) = resolution.flows.get_mut(event.flow as usize).filter(|_| !event.unbound) {
                *flow = if event.flow_entry() == FLOW_END { None } else { flow.or(Some(true)) };
            }
            resolution.rows[event.track as usize].set(event.ts, event.dur, event.unbound, event.flow != NONE_FLOW);
            flow_visible[deeper][event.flow as usize].get_or_insert(true);
        }
    }
    let levels: Vec<AtomicU8> = (0..events.len()).map(|_| AtomicU8::new(0)).collect();
    // The large tracks start first, so that a large track does not run alone at the end.
    let mut order = by_track.to_vec();
    order.sort_unstable_by_key(|indices| std::cmp::Reverse(indices.len()));
    order.into_par_iter().with_max_len(1).for_each(|indices| {
        let track = || indices.iter().map(|&index| &events[index as usize]);
        let compact: Vec<(u64, u64, u8)> = track().map(|event| (event.ts, event.dur, u8::from(event.flow != NONE_FLOW) << 1 | u8::from(event.unbound))).collect();
        let assigned: Vec<AtomicU8> = track()
            .map(|event| {
                AtomicU8::new(if event.flow == NONE_FLOW {
                    UNASSIGNED
                } else {
                    (0..SPLIT).find(|&level| flow_visible[level][event.flow as usize] == Some(true) || event.dur >= LAYER_PS[level]).unwrap_or(SPLIT) as u8
                })
            })
            .collect();
        let remaining = AtomicUsize::new(assigned.iter().filter(|level| level.load(Relaxed) == UNASSIGNED).count());
        let (done, lock, wake): (Vec<AtomicUsize>, _, _) = ((0..SPLIT).map(|_| AtomicUsize::new(0)).collect(), Mutex::new(()), Condvar::new());
        // The pass of a level reads the levels that the pass before it gave, so it stays behind that pass. The passes after all events have a level change nothing.
        let pass = |level: usize| {
            let (mut row, resolution) = (Row::default(), LAYER_PS[level]);
            for start in (0..compact.len()).step_by(PART) {
                let end = (start + PART).min(compact.len());
                if level > 0 {
                    let ready = || done[level - 1].load(Acquire) >= end || remaining.load(Relaxed) == 0;
                    drop(wake.wait_while(lock.lock().unwrap(), |()| !ready()).unwrap());
                }
                if remaining.load(Relaxed) == 0 {
                    break;
                }
                let mut found = 0;
                for (position, &(ts, dur, kind)) in compact.iter().enumerate().take(end).skip(start) {
                    let current = assigned[position].load(Relaxed);
                    if usize::from(current) < level || (kind & 2 != 0 && usize::from(current) == level) {
                        row.set(ts, dur, kind & 1 != 0, false);
                    } else if kind & 2 == 0 && current == UNASSIGNED {
                        let (visible, depth) = if kind & 1 != 0 { (row.counter(ts, resolution), 0) } else { row.probe(ts, dur, resolution) };
                        if visible {
                            assigned[position].store(level as u8, Relaxed);
                            found += 1;
                            if kind & 1 == 0 {
                                row.push(depth, ts + dur);
                            }
                        }
                    }
                }
                remaining.fetch_sub(found, Relaxed);
                done[level].store(end, Release);
                drop(lock.lock().unwrap());
                wake.notify_all();
            }
        };
        if compact.len() < PIPELINE || (!cfg!(test) && compact.len() < events.len() / rayon::current_num_threads()) {
            (0..SPLIT).for_each(pass);
        } else {
            std::thread::scope(|scope| {
                for level in 1..SPLIT {
                    scope.spawn(move || pass(level));
                }
                pass(0);
            });
        }
        indices.iter().zip(assigned).for_each(|(&index, level)| levels[index as usize].store(level.into_inner().min(SPLIT as u8), Relaxed));
    });
    levels.into_iter().map(AtomicU8::into_inner).collect()
}
