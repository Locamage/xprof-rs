use crate::hlo::profiler::{
    ActiveAllocation, HloModule, MemoryActivityMetadata as Activity, MemoryAggregationStats as Stats, MemoryProfile, MemoryProfileSnapshot, MemoryProfileSummary, PerAllocatorMemoryProfile,
};
use crate::xplane::{Field, NONE_GROUP, Plane, Value, fields, slice, stats};
use std::collections::{BTreeMap, HashMap};

const HOST_PLANE: &str = "/host:CPU";
const TPU_PREFIX: &str = "/device:TPU:";
const MODULE_LINE: &str = "XLA Modules";
const MAX_SNAPSHOTS: usize = 1000;
const ALLOCATION: i32 = 1;
const DEALLOCATION: i32 = 2;
const DATA_TYPES: &str = "INVALID float double int32 uint8 int16 int8 string complex64 int64 bool qint8 quint8 qint32 bfloat16 qint16 quint16 uint16 complex128 half resource variant uint32 uint64 float8_e5m2 float8_e4m3fn float8_e4m3fnuz float8_e4m3b11fnuz float8_e5m2fnuz int4 uint4";

#[derive(Clone)]
struct Snapshot {
    time: i64,
    stats: Stats,
    meta: Activity,
}

#[derive(Default)]
struct Summary {
    lifetime: i64,
    peak: Stats,
    peak_time: i64,
    capacity: i64,
}

#[derive(Default)]
struct Allocator {
    snapshots: Vec<Snapshot>,
    summary: Summary,
    active: Vec<ActiveAllocation>,
    special: Vec<Activity>,
    sampled: Vec<Snapshot>,
}

fn data_type(value: i64) -> String {
    let name = |value: i64| DATA_TYPES.split(' ').nth(value as usize).filter(|_| value >= 0).map_or_else(|| format!("unknown dtype enum ({value})"), str::to_string);
    if value > 100 { format!("{}_ref", name(value - 100)) } else { name(value) }
}

fn offset(raw: &[u8]) -> i64 {
    fields(raw).find_map(|(tag, field)| if let (2, Field::Num(offset)) = (tag, field) { Some(offset as i64) } else { None }).unwrap_or(0)
}

fn generate(plane: &Plane, map: &[u8]) -> BTreeMap<String, Allocator> {
    let mut allocators: BTreeMap<String, Allocator> = BTreeMap::new();
    for event in plane.lines.iter().flat_map(|line| &line.events) {
        let activity = match &*plane.meta[event.meta as usize].name {
            "MemoryAllocation" => ALLOCATION,
            "MemoryDeallocation" => DEALLOCATION,
            _ => continue,
        };
        let raw = slice(map, event.raw);
        let (mut stats_, mut meta, mut memory_id) = (Stats::default(), Activity { memory_activity: activity, step_id: -1, ..Default::default() }, String::new());
        for stat in stats(raw, 4, |_| true) {
            let Some(name) = plane.stat_names.get(stat.id) else { continue };
            let int = if let Value::Int(value) = stat.value { value } else { 0 };
            match &**name {
                "index_on_host" | "device_ordinal" => memory_id = int.to_string(),
                "allocator_name" => memory_id = plane.text(&stat.value),
                "bytes_reserved" => stats_.stack_reserved_bytes = int,
                "bytes_allocated" => stats_.heap_allocated_bytes = int,
                "bytes_available" => stats_.free_memory_bytes = int,
                "fragmentation" => stats_.fragmentation = if let Value::Double(value) = stat.value { value } else { 0.0 },
                "peak_bytes_in_use" => stats_.peak_bytes_in_use = int,
                "requested_bytes" => meta.requested_bytes = int,
                "allocation_bytes" => meta.allocation_bytes = int,
                "addr" => meta.address = int as u64,
                "tf_op" => meta.tf_op_name = plane.text(&stat.value),
                "group_id" => meta.step_id = int,
                "region_type" => meta.region_type = plane.text(&stat.value),
                "data_type" => meta.data_type = data_type(int),
                "shape" => meta.tensor_shape = plane.text(&stat.value),
                _ => {}
            }
        }
        if event.group != NONE_GROUP {
            meta.step_id = event.group;
        }
        let time = offset(raw);
        let allocator = allocators.entry(memory_id).or_default();
        let summary = &mut allocator.summary;
        summary.lifetime = stats_.peak_bytes_in_use;
        let in_use = stats_.stack_reserved_bytes.wrapping_add(stats_.heap_allocated_bytes);
        if in_use >= summary.peak.peak_bytes_in_use {
            summary.peak = Stats { peak_bytes_in_use: in_use, ..stats_ };
            summary.peak_time = time;
            summary.capacity = in_use.wrapping_add(stats_.free_memory_bytes);
        }
        allocator.snapshots.push(Snapshot { time, stats: stats_, meta });
    }
    allocators
}

fn key(meta: &Activity) -> (i64, i64, &str, &str, &str, &str) {
    (meta.allocation_bytes.wrapping_neg(), meta.requested_bytes.wrapping_neg(), &meta.tf_op_name, &meta.region_type, &meta.data_type, &meta.tensor_shape)
}

fn process(allocator: &mut Allocator) {
    let snapshots = &mut allocator.snapshots;
    snapshots.sort_by_key(|snapshot| snapshot.time);
    let mut last_step = -1i64;
    for snapshot in snapshots.iter_mut() {
        if snapshot.meta.step_id == -1 {
            snapshot.meta.step_id = last_step.wrapping_add(1);
        } else {
            last_step = snapshot.meta.step_id;
        }
    }
    let mut allocations: HashMap<u64, usize> = HashMap::new();
    for index in 0..snapshots.len() {
        let address = snapshots[index].meta.address;
        if snapshots[index].meta.memory_activity == DEALLOCATION {
            if let Some(source) = allocations.remove(&address) {
                let source = snapshots[source].meta.clone();
                let meta = &mut snapshots[index].meta;
                (meta.tf_op_name, meta.region_type, meta.data_type, meta.tensor_shape) = (source.tf_op_name, source.region_type, source.data_type, source.tensor_shape);
            }
        } else {
            allocations.entry(address).or_insert(index);
        }
    }
    let count = snapshots.len();
    let in_use = |snapshot: &Snapshot| snapshot.stats.heap_allocated_bytes.wrapping_add(snapshot.stats.stack_reserved_bytes);
    allocator.sampled = if count > MAX_SNAPSHOTS {
        let width = count / MAX_SNAPSHOTS;
        let first = MAX_SNAPSHOTS * (width + 1) - count;
        let mut sampled = Vec::with_capacity(MAX_SNAPSHOTS);
        for (width, samples, start) in [(width, first, 0), (width + 1, MAX_SNAPSHOTS - first, width * first)] {
            for begin in (0..samples).map(|sample| start + width * sample) {
                sampled.push(snapshots[(begin..begin + width).rev().max_by_key(|&index| in_use(&snapshots[index])).unwrap()].clone());
            }
        }
        sampled
    } else {
        snapshots.clone()
    };
    let summary = &allocator.summary;
    let peak_step = snapshots.iter().rfind(|snapshot| in_use(snapshot) == summary.peak.peak_bytes_in_use).map_or(0, |snapshot| snapshot.meta.step_id);
    let (mut unmapped, mut unmapped_free) = (summary.peak.heap_allocated_bytes, 0i64);
    let mut active: HashMap<u64, usize> = HashMap::new();
    for (index, snapshot) in snapshots.iter().enumerate() {
        if snapshot.time > summary.peak_time {
            break;
        }
        if snapshot.meta.step_id != peak_step {
            continue;
        }
        if snapshot.meta.memory_activity == ALLOCATION {
            active.insert(snapshot.meta.address, index);
            unmapped = unmapped.wrapping_sub(snapshot.meta.allocation_bytes);
        } else {
            if active.remove(&snapshot.meta.address).is_none() {
                unmapped_free = unmapped_free.wrapping_add(snapshot.meta.allocation_bytes);
            }
            unmapped = unmapped.wrapping_add(snapshot.meta.allocation_bytes);
        }
    }
    unmapped = unmapped.wrapping_sub(unmapped_free);
    let mut order: Vec<usize> = active.into_values().collect();
    order.sort_unstable();
    let mut entries: Vec<(i64, &Activity)> = order.into_iter().map(|index| (index as i64, &snapshots[index].meta)).collect();
    let special = |name: &str, bytes: i64, region: &str| Activity {
        memory_activity: ALLOCATION,
        requested_bytes: bytes,
        allocation_bytes: bytes,
        tf_op_name: name.to_string(),
        step_id: peak_step,
        region_type: region.to_string(),
        data_type: "INVALID".to_string(),
        tensor_shape: "unknown".to_string(),
        ..Default::default()
    };
    let mut specials = Vec::new();
    if unmapped > 0 {
        specials.push(special("unused preallocated device memory", unmapped, "persist/dynamic"));
    }
    if summary.peak.stack_reserved_bytes > 0 {
        specials.push(special("stack", summary.peak.stack_reserved_bytes, "stack"));
    }
    entries.extend(specials.iter().enumerate().map(|(position, meta)| (-(position as i64) - 1, meta)));
    entries.sort_by_key(|(_, meta)| key(meta));
    let mut active: Vec<ActiveAllocation> = entries
        .chunk_by(|a, b| key(a.1) == key(b.1))
        .map(|group| ActiveAllocation { snapshot_index: group[0].0, special_index: if group[0].0 < 0 { -group[0].0 - 1 } else { -1 }, num_occurrences: group.len() as i64 })
        .collect();
    let kept: Vec<Snapshot> = active.iter().filter(|entry| entry.snapshot_index >= 0).map(|entry| snapshots[entry.snapshot_index as usize].clone()).collect();
    for (next, entry) in active.iter_mut().filter(|entry| entry.snapshot_index >= 0).enumerate() {
        entry.snapshot_index = next as i64;
    }
    allocator.snapshots = kept;
    allocator.active = active;
    allocator.special = specials;
}

fn snapshot(snapshot: Snapshot) -> MemoryProfileSnapshot {
    MemoryProfileSnapshot { time_offset_ps: snapshot.time, aggregation_stats: Some(snapshot.stats), activity_metadata: Some(snapshot.meta) }
}

/// The memory profile of a file. A file that is not a valid `XSpace` gives `None`.
pub fn load(path: &std::path::Path) -> anyhow::Result<Option<String>> {
    from_map(&crate::read_file(path)?)
}

/// The memory profile uses the groups, but not the op statistics.
pub fn from_map(map: &[u8]) -> anyhow::Result<Option<String>> {
    let Some(mut planes) = crate::parse_checked(map)? else { return Ok(None) };
    crate::finish(&mut planes, map, false);
    Ok(Some(json(&planes, map)))
}

pub fn json(planes: &[Plane], map: &[u8]) -> String {
    let Some(host) = planes.iter().find(|plane| plane.name == HOST_PLANE) else { return String::new() };
    let mut allocators = generate(host, map);
    allocators.values_mut().for_each(process);
    let memory_ids = allocators.iter().filter(|(_, allocator)| !allocator.snapshots.is_empty()).map(|(id, _)| id.clone()).collect();
    let hlo_modules = planes
        .iter()
        .filter(|plane| plane.name.starts_with(TPU_PREFIX))
        .flat_map(|plane| plane.lines.iter().filter(|line| line.name == MODULE_LINE).flat_map(|line| &line.events).map(move |event| (plane, event)))
        .enumerate()
        .map(|(id, (plane, event))| {
            let start = offset(slice(map, event.raw));
            HloModule {
                name: plane.meta[event.meta as usize].name.to_string(),
                start_time_ps: start,
                end_time_ps: start.saturating_add(event.dur as i64),
                id: id as i64,
                plane_name: plane.name.strip_prefix("/device:").unwrap_or(&plane.name).to_string(),
            }
        })
        .collect();
    let memory_profile_per_allocator = allocators
        .into_iter()
        .map(|(id, allocator)| {
            let summary = allocator.summary;
            let profile = PerAllocatorMemoryProfile {
                memory_profile_snapshots: allocator.snapshots.into_iter().map(snapshot).collect(),
                profile_summary: Some(MemoryProfileSummary {
                    peak_bytes_usage_lifetime: summary.lifetime,
                    peak_stats: Some(summary.peak),
                    peak_stats_time_ps: summary.peak_time,
                    memory_capacity: summary.capacity,
                }),
                active_allocations: allocator.active,
                special_allocations: allocator.special,
                sampled_timeline_snapshots: allocator.sampled.into_iter().map(snapshot).collect(),
            };
            (id, profile)
        })
        .collect();
    crate::hlo::proto_text::json("tensorflow.profiler.MemoryProfile", &MemoryProfile { memory_profile_per_allocator, num_hosts: 1, memory_ids, version: 1, hlo_modules }, true)
}

#[cfg(test)]
#[path = "../tests/inline/tools/memory_profile.rs"]
mod tests;
