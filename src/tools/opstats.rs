use crate::tools::roofline::add_scaled;
use crate::xplane::derive::is_tensor_core;
use crate::xplane::{Ev, Field, Plane, Stat, Value, fields, slice, stat, stats, varint};
use arcstr::ArcStr;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

pub const IDLE: &str = "IDLE";
const CUSTOM_CALL: &str = "custom-call";
const EVENT_STATS: [&str; 9] =
    ["vdd_energy_j", "min_duration_ps", "dma_stall_duration_ps", "bytes_accessed", "model_flops", "flops", "device_offset_ps", "device_duration_ps", "Time Scale Multiplier"];
const VDD_ENERGY: u8 = 0;
const MIN_DURATION: u8 = 1;
const DMA_STALL: u8 = 2;
const BYTES_ACCESSED: u8 = 3;
const MODEL_FLOPS: u8 = 4;
const FLOPS: u8 = 5;
const DEVICE_OFFSET: u8 = 6;
const DEVICE_DURATION: u8 = 7;
const TIME_SCALE: u8 = 8;
const UNREAD: u8 = u8::MAX;
/// The code that reads the child tree is recursive, so a deeper tree can overflow the stack.
const MAX_DEPTH: usize = 256;
pub const TENSOR_CORE: u8 = 1;
pub const SPARSE_CORE: u8 = 2;
pub const HBM: u64 = 1;
pub const CMEM: u64 = 2;
pub const VMEM: u64 = 3;
pub const READ: u8 = 1;
pub const WRITE: u8 = 2;
pub const GIBI_IN_GIGA: f64 = (1u64 << 30) as f64 / 1.0e9;

macro_rules! add {
    ($to:expr, $from:expr, $($field:ident),*) => { $($to.$field += $from.$field;)* };
}
pub(crate) use add;

#[derive(Clone, Default)]
pub struct Source {
    pub file: ArcStr,
    pub line: i32,
    pub stack: ArcStr,
}

#[derive(Clone, Default)]
pub struct Metrics {
    pub module: u64,
    pub name: ArcStr,
    pub long_name: ArcStr,
    pub category: ArcStr,
    pub provenance: ArcStr,
    pub deduplicated_name: ArcStr,
    pub occurrences: u64,
    pub time_ps: u64,
    pub normalized_time_ps: u64,
    pub min_time_ps: u64,
    pub self_time_ps: u64,
    pub dma_stall_ps: u64,
    pub flops_v2: f64,
    pub model_flops_v2: f64,
    pub bytes_accessed: u64,
    pub memory: Vec<(u8, u64, u64)>,
    pub vdd_energy: Option<f64>,
    pub num_cores: u32,
    pub core_type: u8,
    pub autotuned: bool,
    pub is_eager: bool,
    pub source: Option<Source>,
    pub children: Db,
}

#[derive(Clone, Default)]
pub struct Db {
    pub metrics: Vec<Metrics>,
    pub total_time_ps: u64,
    pub total_op_time_ps: u64,
    pub normalized_total_op_time_ps: u64,
}

pub type Template = Option<(Option<(u64, u64)>, Metrics)>;

#[derive(Default)]
pub struct EventStats {
    occurrences: u64,
    min_time_ps: Option<u64>,
    dma_stall_ps: u64,
    vdd_energy: Option<f64>,
    custom: [u64; 3],
    offset: Option<u64>,
    duration: Option<u64>,
    scale: Option<f64>,
}

#[derive(Clone, Default)]
pub struct Perf {
    pub peak_tera_flops: f64,
    pub bandwidths: Vec<f64>,
    pub ridge_point: f64,
    pub cmem: bool,
}

#[derive(Clone)]
pub struct OpStats {
    pub db: Db,
    pub perf: Perf,
    pub tpu: bool,
    pub host: Db,
    /// The sums of the durations of the enqueues of infeed, and of the times between them, on the host.
    pub infeed_enqueue: (u64, u64),
    pub extra: Arc<crate::xplane::steps::Extra>,
    pub programs: HashMap<u64, String>,
    pub kernels: Vec<crate::xplane::gpu::KernelReport>,
}

pub fn safe_divide(dividend: f64, divisor: f64) -> f64 {
    if (-1e-10..1e-10).contains(&divisor) { 0.0 } else { dividend / divisor }
}

pub fn pico_to_micro(ps: u64) -> f64 {
    ps as f64 / 1e6
}

pub fn pico_to_nano(ps: u64) -> f64 {
    ps as f64 / 1e3
}

pub fn giga_to_gibi(giga: f64) -> f64 {
    giga / GIBI_IN_GIGA
}

fn children(raw: &[u8]) -> Vec<usize> {
    let mut ids = Vec::new();
    for (_, field) in fields(raw).filter(|(tag, _)| *tag == 6) {
        match field {
            Field::Num(id) => ids.push(id as usize),
            Field::Bytes(_, packed) => {
                let mut position = 0;
                while position < packed.len() {
                    let Some(id) = varint(packed, &mut position) else { break };
                    ids.push(id as usize);
                }
            }
        }
    }
    ids
}

fn breakdown(data: &[u8]) -> Vec<(u8, u64, u64)> {
    fields(data)
        .filter_map(|(tag, field)| match (tag, field) {
            (1, Field::Bytes(_, body)) => {
                let mut entry = [0u64; 3];
                for (tag, field) in fields(body) {
                    if let (1..=3, Field::Num(number)) = (tag, field) {
                        entry[tag as usize - 1] = number;
                    }
                }
                Some((entry[0] as u8, entry[1], entry[2]))
            }
            _ => None,
        })
        .collect()
}

fn from_metadata(plane: &Plane, map: &[u8], meta: usize, path: &mut Vec<usize>) -> (Option<(u64, u64)>, Metrics) {
    path.push(meta);
    let (meta, mut out, mut key) = (&plane.meta[meta], Metrics::default(), (None, None));
    if meta.display.is_empty() {
        out.name = meta.name.as_ref().into();
    } else {
        out.name = meta.display.as_ref().into();
        out.long_name = meta.long_name(map).as_ref().into();
    }
    let raw = slice(map, meta.raw);
    for stat in stats(raw, 5, |_| true) {
        let Some(name) = plane.stat_names.get(stat.id) else { continue };
        let text = || ArcStr::from(plane.text(&stat.value));
        let number = stat.value.int().unwrap_or(0) as u64;
        match &**name {
            "program_id" => (out.module, key.0) = (number, Some(number)),
            "symbol_id" => key.1 = Some(number),
            "hlo_category" => out.category = text(),
            "tf_op" => out.provenance = text(),
            "flops" => out.flops_v2 = number as f64,
            "model_flops" => out.model_flops_v2 = number as f64,
            "bytes_accessed" => out.bytes_accessed = number,
            "memory_access_breakdown" => {
                if let Value::Bytes(data) = stat.value {
                    out.memory = breakdown(data);
                }
            }
            "deduplicated_name" => out.deduplicated_name = text(),
            "source" => {
                if let Some((file, line)) = plane.text(&stat.value).split_once(':').and_then(|(file, line)| Some((file.into(), line.parse::<i32>().ok()?))) {
                    let source = out.source.get_or_insert_with(Source::default);
                    (source.file, source.line) = (file, line);
                }
            }
            "source_stack" => out.source.get_or_insert_with(Source::default).stack = text(),
            _ => {}
        }
    }
    for child in children(raw) {
        if child < plane.meta.len() && path.len() < MAX_DEPTH && !path.contains(&child) {
            out.children.metrics.push(Metrics { occurrences: 1, ..from_metadata(plane, map, child, path).1 });
        }
    }
    path.pop();
    (key.0.zip(key.1), out)
}

/// The templates of the operations of one plane. Each operation with a key has a slot, and the slots are in the order of the keys.
#[derive(Default)]
pub struct Templates {
    metas: Vec<Template>,
    slots: Vec<u32>,
    keys: usize,
}

pub fn templates(plane: &Plane, map: &[u8]) -> Templates {
    let mut used = vec![false; plane.meta.len()];
    for event in plane.lines.iter().flat_map(|line| &line.events) {
        if let Some(slot) = used.get_mut(event.meta as usize) {
            *slot = true;
        }
    }
    let metas: Vec<Template> = used.par_iter().enumerate().map(|(meta, &used)| used.then(|| from_metadata(plane, map, meta, &mut Vec::new()))).collect();
    let key = |template: &Template| template.as_ref().and_then(|(key, _)| key.filter(|key| key.1 != 0));
    let mut keys: Vec<(u64, u64)> = metas.iter().filter_map(key).collect();
    keys.sort_unstable();
    keys.dedup();
    let slots = metas.iter().map(|template| key(template).map_or(u32::MAX, |key| keys.binary_search(&key).unwrap() as u32)).collect();
    Templates { metas, slots, keys: keys.len() }
}

pub struct EventReader(Vec<u8>);

impl EventReader {
    pub fn new(plane: &Plane) -> Self {
        Self(plane.stat_names.iter().map(|name| EVENT_STATS.iter().position(|known| *known == &**name).map_or(UNREAD, |kind| kind as u8)).collect())
    }

    pub fn read(&self, map: &[u8], event: &Ev) -> EventStats {
        let mut out = EventStats::default();
        for (tag, field) in fields(slice(map, event.raw)) {
            match (tag, field) {
                (4, Field::Bytes(_, body)) => self.add(body, &mut out),
                (5, Field::Num(count)) => out.occurrences = count,
                _ => {}
            }
        }
        out
    }

    fn add(&self, body: &[u8], out: &mut EventStats) {
        let Some(Stat { id, value }) = stat(body, |id| self.0.get(id).is_some_and(|&kind| kind != UNREAD)) else { return };
        let number = value.int().unwrap_or(0) as u64;
        match (self.0[id], &value) {
            (VDD_ENERGY, Value::Double(value)) => out.vdd_energy = Some(out.vdd_energy.unwrap_or(0.0) + value),
            (MIN_DURATION, _) => out.min_time_ps = Some(number),
            (DMA_STALL, _) => out.dma_stall_ps = number,
            (BYTES_ACCESSED, _) => out.custom[0] += number,
            (MODEL_FLOPS, _) => out.custom[1] += number,
            (FLOPS, _) => out.custom[2] += number,
            (DEVICE_OFFSET, _) => _ = out.offset.get_or_insert(number),
            (DEVICE_DURATION, _) => _ = out.duration.get_or_insert(number),
            (TIME_SCALE, value) => _ = out.scale.get_or_insert(if let Value::Double(factor) = value { *factor } else { 1.0 }),
            _ => {}
        }
    }

    pub fn span(&self, map: &[u8], event: &Ev) -> (u64, u64) {
        self.read(map, event).span(event)
    }
}

impl EventStats {
    pub fn span(&self, event: &Ev) -> (u64, u64) {
        self.offset.zip(self.duration).unwrap_or((event.ts, event.dur))
    }
}

/// The sums of one operation. It keeps only numbers, so that many of them fit in the cache.
#[derive(Clone, Copy, Default)]
struct Totals {
    occurrences: u64,
    time_ps: u64,
    self_time_ps: u64,
    normalized_time_ps: u64,
    min_time_ps: u64,
    dma_stall_ps: u64,
    flops_v2: f64,
    model_flops_v2: f64,
    bytes_accessed: u64,
    vdd_energy: Option<f64>,
    core_type: u8,
}

struct Accumulator<'a> {
    template: &'a Metrics,
    custom: bool,
    totals: Totals,
}

impl Accumulator<'_> {
    /// The flops, model flops and bytes of all occurrences.
    fn counts(&self) -> (f64, f64, u64) {
        let totals = &self.totals;
        if self.custom {
            return (totals.flops_v2, totals.model_flops_v2, totals.bytes_accessed);
        }
        let occurrences = totals.occurrences;
        let flops = totals.flops_v2 * occurrences as f64;
        (flops, if totals.model_flops_v2 > 0.0 { totals.model_flops_v2 * occurrences as f64 } else { flops }, totals.bytes_accessed.saturating_mul(occurrences))
    }

    fn adjusted(self) -> Metrics {
        let (flops_v2, model_flops_v2, bytes_accessed) = self.counts();
        let occurrences = self.totals.occurrences;
        let memory = self.template.memory.iter().map(|&(operation, space, bytes)| (operation, space, bytes.saturating_mul(occurrences))).collect();
        let Totals { time_ps, self_time_ps, normalized_time_ps, min_time_ps, dma_stall_ps, vdd_energy, core_type, .. } = self.totals;
        Metrics { occurrences, time_ps, normalized_time_ps, min_time_ps, self_time_ps, dma_stall_ps, flops_v2, model_flops_v2, bytes_accessed, memory, vdd_energy, core_type, ..Default::default() }
    }
}

/// A builder keeps only the operations that it saw, so a builder for one step is small.
pub struct Builder<'a> {
    templates: &'a Templates,
    positions: Vec<u32>,
    entries: Vec<(u32, Accumulator<'a>)>,
}

impl<'a> Builder<'a> {
    pub fn new(templates: &'a Templates) -> Self {
        Builder { templates, positions: vec![u32::MAX; templates.keys], entries: Vec::new() }
    }

    pub fn add(&mut self, event: &Ev, stats: &EventStats, (time_ps, self_time_ps): (u64, u64), tensor_core: bool) {
        let (meta, min_time_ps) = (event.meta as usize, stats.min_time_ps.unwrap_or(event.dur));
        let slot = self.templates.slots[meta];
        let (Some(position), Some((_, template))) = (self.positions.get_mut(slot as usize), &self.templates.metas[meta]) else { return };
        let (custom, fresh) = (template.category == CUSTOM_CALL, *position == u32::MAX);
        if fresh {
            *position = self.entries.len() as u32;
            self.entries.push((slot, Accumulator { template, custom, totals: Totals { min_time_ps, ..Default::default() } }));
        }
        let entry = &mut self.entries[*position as usize].1;
        let totals = &mut entry.totals;
        totals.occurrences += stats.occurrences.max(1);
        totals.time_ps += time_ps;
        totals.self_time_ps += self_time_ps;
        totals.dma_stall_ps += stats.dma_stall_ps;
        totals.normalized_time_ps += if tensor_core { (time_ps as f64 * stats.scale.unwrap_or(1.0)) as u64 } else { 0 };
        totals.core_type = if tensor_core { TENSOR_CORE } else { 0 };
        totals.min_time_ps = totals.min_time_ps.min(min_time_ps);
        if stats.vdd_energy.is_some() || totals.vdd_energy.is_some() {
            totals.vdd_energy = Some(stats.vdd_energy.unwrap_or(0.0) + totals.vdd_energy.unwrap_or(0.0));
        }
        if fresh || entry.custom {
            let custom = if custom { stats.custom } else { [0; 3] };
            totals.flops_v2 += template.flops_v2 + custom[2] as f64;
            totals.model_flops_v2 += template.model_flops_v2 + custom[1] as f64;
            totals.bytes_accessed += template.bytes_accessed + custom[0];
        }
    }

    fn sorted(mut self) -> impl Iterator<Item = Accumulator<'a>> {
        self.entries.sort_unstable_by_key(|&(slot, _)| slot);
        self.entries.into_iter().map(|(_, entry)| entry)
    }

    pub fn finish(self) -> Db {
        let mut db = Db::default();
        for entry in self.sorted() {
            let template = entry.template;
            let metrics = Metrics {
                module: template.module,
                name: template.name.clone(),
                long_name: template.long_name.clone(),
                category: template.category.clone(),
                provenance: template.provenance.clone(),
                deduplicated_name: template.deduplicated_name.clone(),
                num_cores: 1,
                autotuned: template.autotuned,
                is_eager: template.is_eager,
                source: template.source.clone(),
                children: template.children.clone(),
                ..entry.adjusted()
            };
            db.total_op_time_ps += metrics.self_time_ps;
            db.normalized_total_op_time_ps += (metrics.self_time_ps as f64 * safe_divide(metrics.normalized_time_ps as f64, metrics.time_ps as f64)) as u64;
            db.metrics.push(metrics);
        }
        db
    }

    pub fn program(self) -> (u64, ([Metrics; 2], u64)) {
        let (mut total_op_time_ps, mut program) = (0, Default::default());
        for entry in self.sorted() {
            total_op_time_ps += entry.totals.self_time_ps;
            add_scaled(&mut program, entry.template, (entry.totals.core_type, entry.totals.time_ps), entry.counts(), &entry.template.memory, entry.totals.occurrences);
        }
        (total_op_time_ps, program)
    }
}

pub fn combine_memory(source: impl IntoIterator<Item = impl Borrow<(u8, u64, u64)>>, destination: &mut Vec<(u8, u64, u64)>) {
    for entry in source {
        let &(operation, space, bytes) = entry.borrow();
        match destination.iter_mut().find(|entry| entry.0 == operation && entry.1 == space) {
            Some(entry) => entry.2 += bytes,
            None => destination.push((operation, space, bytes)),
        }
    }
}

fn absorb(destination: &mut Metrics, metrics: &Metrics, update_cores: bool) {
    for (to, from) in [
        (&mut destination.long_name, &metrics.long_name),
        (&mut destination.category, &metrics.category),
        (&mut destination.provenance, &metrics.provenance),
        (&mut destination.deduplicated_name, &metrics.deduplicated_name),
    ] {
        if to.is_empty() {
            to.clone_from(from);
        }
    }
    if destination.children.metrics.is_empty() && destination.children.total_time_ps == 0 {
        destination.children = metrics.children.clone();
    }
    if destination.source.is_none() {
        destination.source.clone_from(&metrics.source);
    }
    if destination.core_type == 0 {
        destination.core_type = metrics.core_type;
    }
    destination.min_time_ps = if destination.occurrences == 0 { metrics.min_time_ps } else { destination.min_time_ps.min(metrics.min_time_ps) };
    destination.is_eager |= metrics.is_eager;
    destination.autotuned |= metrics.autotuned;
    if update_cores {
        destination.num_cores += metrics.num_cores;
    }
    add!(destination, metrics, occurrences, time_ps, self_time_ps, normalized_time_ps, flops_v2, model_flops_v2, bytes_accessed, dma_stall_ps);
    combine_memory(&metrics.memory, &mut destination.memory);
    if metrics.vdd_energy.is_some() || destination.vdd_energy.is_some() {
        destination.vdd_energy = Some(metrics.vdd_energy.unwrap_or(0.0) + destination.vdd_energy.unwrap_or(0.0));
    }
}

impl Db {
    pub fn combined<'a>(parts: impl IntoIterator<Item = &'a Self>, update_cores: bool) -> Self {
        let (mut db, mut index, parts) = (Self::default(), FxHashMap::default(), parts.into_iter().collect::<Vec<_>>());
        let Some((first, rest)) = parts.split_first() else { return db };
        db.merge(&mut index, first, update_cores);
        // Parts of one trace list the same operations in the same order, so they merge position by position in parallel.
        let same = |part: &&&Self| part.metrics.len() == db.metrics.len() && part.metrics.par_iter().zip(&db.metrics).all(|(a, b)| a.module == b.module && a.name == b.name);
        let (aligned, rest) = rest.split_at(rest.iter().take_while(same).count());
        for part in aligned {
            add!(db, part, total_time_ps, total_op_time_ps, normalized_total_op_time_ps);
        }
        db.metrics.par_iter_mut().enumerate().for_each(|(position, destination)| aligned.iter().for_each(|part| absorb(destination, &part.metrics[position], update_cores)));
        rest.iter().for_each(|part| db.merge(&mut index, part, update_cores));
        db
    }

    fn merge(&mut self, index: &mut FxHashMap<(u64, ArcStr), usize>, source: &Self, update_cores: bool) {
        add!(self, source, total_time_ps, total_op_time_ps, normalized_total_op_time_ps);
        for (position, metrics) in source.metrics.iter().enumerate() {
            let aligned = self.metrics.get(position).is_some_and(|known| known.module == metrics.module && known.name == metrics.name);
            absorb(if aligned { &mut self.metrics[position] } else { self.entry(index, metrics.module, &metrics.name) }, metrics, update_cores);
        }
    }

    pub fn entry(&mut self, index: &mut FxHashMap<(u64, ArcStr), usize>, module: u64, name: &ArcStr) -> &mut Metrics {
        let position = *index.entry((module, name.clone())).or_insert_with(|| {
            self.metrics.push(Metrics { module, name: name.clone(), ..Default::default() });
            self.metrics.len() - 1
        });
        &mut self.metrics[position]
    }

    pub fn with_idle(mut self, total_time_ps: u64) -> Self {
        self.total_time_ps = total_time_ps.max(self.total_op_time_ps);
        let idle = self.total_time_ps.wrapping_sub(self.total_op_time_ps);
        self.metrics.push(Metrics { name: IDLE.into(), category: IDLE.into(), time_ps: idle, self_time_ps: idle, ..Default::default() });
        self
    }

    pub fn sorted(&self) -> Vec<&Metrics> {
        let mut result: Vec<&Metrics> = self.metrics.iter().collect();
        result.sort_by(|a, b| b.self_time_ps.cmp(&a.self_time_ps).then_with(|| a.name.cmp(&b.name)));
        result
    }
}

/// `span` is the first start and the last end of the derived operations, if the plane does not have them.
pub fn convert_tensor_core(plane: &Plane, map: &[u8], templates: &Templates, span: Option<(u64, u64)>) -> Db {
    let ((mut first, mut last), mut builder, reader) = (span.unwrap_or((u64::MAX, 0)), Builder::new(templates), EventReader::new(plane));
    for line in &plane.lines {
        let is_step = line.name == "Steps" || line.name == "Sparse Core Steps";
        let is_op = matches!(line.name.as_str(), "XLA Ops" | "Framework Ops" | "Sparse Core Ops");
        if !is_step && !is_op {
            continue;
        }
        let mut stack: Vec<(&Ev, EventStats, (u64, u64), u64)> = Vec::new();
        let mut finish = |(event, stats, span, children): (&Ev, EventStats, (u64, u64), u64)| builder.add(event, &stats, (span.1, span.1.saturating_sub(children)), true);
        for event in &line.events {
            let stats = reader.read(map, event);
            let span = stats.span(event);
            first = first.min(span.0);
            last = last.max(span.0.saturating_add(span.1));
            if is_op && line.labels.is_empty() {
                while let Some(top) = stack.last() {
                    if top.2.0 <= span.0 && top.2.0.saturating_add(top.2.1) >= span.0.saturating_add(span.1) {
                        break;
                    }
                    finish(stack.pop().unwrap());
                }
                if let Some(top) = stack.last_mut() {
                    top.3 = top.3.saturating_add(span.1);
                }
                stack.push((event, stats, span, 0));
            }
        }
        stack.into_iter().rev().for_each(finish);
    }
    builder.finish().with_idle(last.wrapping_sub(first))
}

fn perf_env(plane: &Plane) -> Perf {
    let bandwidths = [
        "peak_hbm_bw_gigabytes_per_second",
        "peak_sram_rd_bw_gigabytes_per_second",
        "peak_sram_wr_bw_gigabytes_per_second",
        "peak_cmem_rd_bw_gigabytes_per_second",
        "peak_cmem_wr_bw_gigabytes_per_second",
        "peak_vmem_rd_bw_gigabytes_per_second",
        "peak_vmem_wr_bw_gigabytes_per_second",
    ]
    .map(|name| plane.own_double(name))
    .to_vec();
    let peak_tera_flops = plane.own_double("peak_teraflops_per_second");
    Perf { peak_tera_flops, ridge_point: peak_tera_flops * 1e3 / bandwidths[0], cmem: bandwidths[3] > 0.0 || bandwidths[4] > 0.0, bandwidths }
}

/// The parts of the op statistics that a tool reads. Each part has the parts before it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Part {
    /// The operations of the devices, the module names, and the header of the steps.
    Device,
    /// Also the programs of the steps.
    Programs,
    /// Also the other fields of the steps and the operations of the host.
    All,
}

/// Prepared planes with the modules that borrow their file. Fields drop in order, so the modules drop before the file.
pub struct Kept {
    modules: Vec<(u64, crate::hlo::Module<'static>)>,
    pub fused: bool,
    pub part: Part,
    finished: bool,
    map: Vec<u8>,
    pub planes: Vec<Plane>,
}

impl Kept {
    pub fn map(&self) -> &[u8] {
        &self.map
    }

    /// Adds the regions, the groups, and the derived lines, if the planes do not have them.
    pub fn finish(&mut self) {
        if !std::mem::replace(&mut self.finished, true) {
            crate::finish(&mut self.planes, &self.map, false);
        }
    }

    /// Adds the fused children to the operations of a trace that does not have them.
    pub fn fuse(&mut self, stats: &mut OpStats) {
        if !self.fused && stats.tpu {
            let modules = crate::hlo::parse_modules(crate::hlo::protos(&self.planes, &self.map));
            crate::hlo::attach_fused(&modules, &mut stats.db);
            // SAFETY: the modules borrow `map`, and `Kept` drops them before `map`. The heap buffer of `map` does not move.
            self.modules = unsafe { std::mem::transmute::<Vec<(u64, crate::hlo::Module<'_>)>, Vec<(u64, crate::hlo::Module<'static>)>>(modules) };
        }
        self.fused = true;
    }

    /// Adds the parts of the op statistics that `part` has and the statistics do not have. `part` comes after the part of the statistics.
    pub fn upgrade(&mut self, stats: &mut OpStats, part: Part) {
        self.finish();
        let (planes, map) = (&self.planes, &self.map[..]);
        let (mut extra, host) = rayon::join(|| steps(planes, map, &all_templates(planes, map), part), || (part == Part::All).then(|| crate::tools::framework_op_stats::host_db(planes, map)));
        if stats.tpu {
            crate::xplane::steps::fix(&mut extra, &stats.db);
        }
        stats.extra = Arc::new(extra);
        if let Some((host, infeed_enqueue)) = host {
            (stats.host, stats.infeed_enqueue) = (host, infeed_enqueue);
        }
        self.part = part;
    }
}

fn all_templates(planes: &[Plane], map: &[u8]) -> Vec<Templates> {
    planes.par_iter().map(|plane| if is_tensor_core(&plane.name) { templates(plane, map) } else { Templates::default() }).collect()
}

/// The steps that `part` has. The programs of the steps on a GPU come with the other fields.
fn steps(planes: &[Plane], map: &[u8], templates: &[Templates], part: Part) -> crate::xplane::steps::Extra {
    match part {
        Part::Device => crate::xplane::steps::header(planes, map),
        Part::Programs if crate::xplane::gpu::devices(planes).is_empty() => crate::xplane::steps::programs(planes, map, templates),
        _ => crate::xplane::steps::extra(planes, map, templates),
    }
}

/// Without `fused`, the operations have no fused children. Only the op profile and the HLO statistics use them.
pub fn load_kept(map: Vec<u8>, fused: bool, part: Part) -> anyhow::Result<Option<(Arc<OpStats>, Kept)>> {
    let Some(mut planes) = crate::parse_checked(&map)? else { return Ok(None) };
    // The modules come from the metadata plane only, so they parse at the same time as the other planes change.
    let protos = if fused && planes.iter().any(|plane| plane.name.starts_with("/device:TPU:")) { crate::hlo::protos(&planes, &map) } else { Vec::new() };
    // The operations of the TPUs do not use the groups, and of the derived lines, they use only the span that `op_span` gives.
    let finished = part > Part::Device || planes.iter().any(|plane| plane.name.starts_with(crate::xplane::gpu::PREFIX));
    let spans = !finished && !crate::xplane::derive::is_grouped(&planes);
    let modules = std::thread::scope(|scope| {
        let modules = scope.spawn(|| crate::hlo::parse_modules(protos));
        if finished {
            crate::finish(&mut planes, &map, false);
        }
        modules.join().unwrap()
    });
    let stats = op_stats(&planes, &map, &modules, part, spans);
    // SAFETY: the modules borrow `map`, and `Kept` drops them before `map`. The heap buffer of `map` does not move.
    let modules = unsafe { std::mem::transmute::<Vec<(u64, crate::hlo::Module<'_>)>, Vec<(u64, crate::hlo::Module<'static>)>>(modules) };
    Ok(Some((Arc::new(stats), Kept { modules, fused, part, finished, map, planes })))
}

pub fn load(path: &std::path::Path) -> anyhow::Result<Option<Arc<OpStats>>> {
    let Some((stats, kept)) = load_kept(crate::read_file(path)?, true, Part::All)? else { return Ok(None) };
    crate::release(kept);
    Ok(Some(stats))
}

impl OpStats {
    pub fn combine(all: &[Option<Arc<Self>>]) -> Option<Arc<Self>> {
        let all: Vec<&Arc<Self>> = all.iter().map(Option::as_ref).collect::<Option<_>>()?;
        let first = *all.first()?;
        if all.len() == 1 {
            return Some(first.clone());
        }
        let (db, host) = (Db::combined(all.iter().map(|stats| &stats.db), true), Db::combined(all.iter().map(|stats| &stats.host), false));
        let (mut programs, mut perf) = (HashMap::new(), Perf::default());
        for stats in &all {
            stats.programs.iter().for_each(|(&id, name)| _ = programs.entry(id).or_insert_with(|| name.clone()));
            if stats.perf.peak_tera_flops > 0.0 {
                perf.peak_tera_flops = stats.perf.peak_tera_flops;
            }
            if perf.bandwidths.is_empty() {
                perf.bandwidths.clone_from(&stats.perf.bandwidths);
            }
            if stats.perf.ridge_point > 0.0 {
                perf.ridge_point = stats.perf.ridge_point;
            }
        }
        let mut extra = crate::xplane::steps::combine(&all.iter().map(|stats| &*stats.extra).collect::<Vec<_>>());
        (extra.megacore, extra.merged_vmem) = (false, false);
        let extra = Arc::new(extra);
        let tpu = all.iter().any(|stats| stats.tpu);
        let infeed_enqueue = all.iter().fold((0, 0), |sum, stats| (sum.0 + stats.infeed_enqueue.0, sum.1 + stats.infeed_enqueue.1));
        Some(Arc::new(Self { db, perf, tpu, host, infeed_enqueue, extra, programs, kernels: crate::xplane::gpu::sorted_kernels(all.iter().flat_map(|stats| stats.kernels.iter().cloned()).collect()) }))
    }
}

/// Without `Part::All`, the host operations are empty.
fn op_stats(planes: &[Plane], map: &[u8], modules: &[(u64, crate::hlo::Module)], part: Part, spans: bool) -> OpStats {
    let first = planes.iter().find(|plane| plane.name.starts_with("/device:TPU:"));
    let tpu = first.is_some();
    let gpus = crate::xplane::gpu::devices(planes);
    let templates = all_templates(planes, map);
    let device = || {
        if !gpus.is_empty() {
            return crate::xplane::gpu::device(planes, map, &gpus);
        }
        let parts: Vec<Db> = planes
            .par_iter()
            .zip(&templates)
            .filter(|(plane, _)| is_tensor_core(&plane.name))
            .map(|(plane, templates)| convert_tensor_core(plane, map, templates, spans.then(|| crate::xplane::derive::op_span(plane, map)).flatten()))
            .collect();
        let mut db = Db::combined(&parts, true);
        crate::release(parts);
        if !modules.is_empty() {
            crate::hlo::attach_fused(modules, &mut db);
        }
        (db, Vec::new())
    };
    // Each part runs on a thread outside the pool, so no part waits for the work of another that the pool stole.
    let ((db, kernels), mut extra, ((host, infeed_enqueue), programs)) = std::thread::scope(|scope| {
        let side = scope.spawn(|| {
            rayon::join(
                || if part == Part::All { crate::tools::framework_op_stats::host_db(planes, map) } else { Default::default() },
                || crate::hlo::protos(planes, map).into_par_iter().map(|(id, proto)| (id, crate::hlo::module_name(proto))).collect::<HashMap<u64, String>>(),
            )
        });
        let extra = scope.spawn(|| steps(planes, map, &templates, part));
        let device = scope.spawn(device);
        (device.join().unwrap(), extra.join().unwrap(), side.join().unwrap())
    });
    crate::release(templates);
    if tpu {
        crate::xplane::steps::fix(&mut extra, &db);
    }
    let perf = first.map_or_else(|| gpus.first().map_or_else(Perf::default, |plane| crate::xplane::gpu::perf_env(plane)), perf_env);
    OpStats { db, perf, tpu, host, infeed_enqueue, extra: Arc::new(extra), programs, kernels }
}
