pub mod derive;
pub mod gpu;
pub mod group;
pub mod steps;

use anyhow::Context;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

static NAMES: LazyLock<HashMap<&str, (bool, [u8; ROLES])>> = LazyLock::new(|| {
    let mut names: HashMap<&str, (bool, [u8; ROLES])> = HashMap::new();
    INTERNAL_EVENTS.split(' ').for_each(|name| names.entry(name).or_default().0 = true);
    for (index, connection) in CONNECTIONS.iter().enumerate() {
        connection.producers.split(' ').for_each(|name| names.entry(name).or_default().1[index] = 1);
        connection.consumers.split(' ').for_each(|name| names.entry(name).or_default().1[index] = 2);
    }
    names
});

pub const NONE_GROUP: i64 = i64::MIN;
const EPOCH_NS: i64 = 1_000_000_000_000_000;
const INTERNAL_EVENTS: &str = "MemoryAllocation MemoryDeallocation PrefetchProduce PrefetchConsume ParallelInterleaveProduce ParallelInterleaveConsume ParallelInterleaveInitializeInput ParallelMapProduce ParallelMapConsume MapAndBatchProduce MapAndBatchConsume ParseExampleProduce ParseExampleConsume";
const QUEUE_CONSUMERS: &str = "RunProgramRequest HostCallbackRequest TransferH2DRequest TransferPreprocessedH2DRequest TransferD2HRequest OnDeviceSendRequest OnDeviceRecvRequest OnDeviceSendRecvLocalRequest CustomWait OnDeviceSendRequestMulti OnDeviceRecvRequestMulti PjrtAsyncWait";
const INTERNAL_STATS: [&str; 8] = ["_pt", "_p", "_ct", "_c", "_r", "flops", "program_id", "symbol_id"];
macro_rules! kinds {
    ($($kind:ident: $text:literal),*) => {
        const STATS: [&str; [$($text),*].len()] = [$($text),*];
        #[allow(non_camel_case_types, dead_code, clippy::upper_case_acronyms)]
        #[repr(usize)]
        enum Kinds { $($kind),* }
        $(const $kind: usize = Kinds::$kind as usize;)*
    };
}

kinds!(FLOW: "flow", GROUP: "group_id", ASYNC: "_a", STEP: "step_name", PT: "_pt", P: "_p", CT: "_ct", C: "_c", CORRELATION: "correlation_id", REQUEST_ID: "request_id", QUEUE_ADDR: "queue_addr", DEVICE_ORDINAL: "device_ordinal", QUEUE_ID: "queue_id", RUN_ID: "run_id", CORE_TYPE: "core_type", PID: "_pid", ROOT: "_r", STEP_ID: "id", ITER_NUM: "iter_num");
const NO_KIND: u8 = u8::MAX;
const MAX_TAG: u64 = (1 << 29) - 1;
const ID_SLACK: u64 = 1024;
const ID_FACTOR: u64 = 4;
const RUN_ID_MASK: u64 = (1 << 27) - 1;
const LAUNCH: u64 = 12;
const LEGACY: u64 = 1;
const ROOTS: [(&str, i64, bool); 4] = [("TraceContext", 1, false), ("ProcessBatch", 2, true), ("OrbaxServing::ProcessBatch", 2, true), ("BatchingSessionRun", 1, true)];
const REGION: [&str; 3] = ["ThreadpoolListener::StartRegion", "ThreadpoolListener::StopRegion", "ThreadpoolListener::Region"];

struct Connection {
    producers: &'static str,
    consumers: &'static str,
    context: u64,
    stats: &'static [usize],
}

const ROLES: usize = 3;
const CONNECTIONS: [Connection; ROLES] = [
    Connection { producers: "EnqueueRequestLocked", consumers: QUEUE_CONSUMERS, context: 11, stats: &[REQUEST_ID, QUEUE_ADDR] },
    Connection { producers: "DoEnqueueProgram", consumers: "CompleteCallbacks", context: LAUNCH, stats: &[DEVICE_ORDINAL, QUEUE_ID, RUN_ID, CORE_TYPE] },
    Connection { producers: "ExecutorState::Process", consumers: "TpuExecuteOp", context: LEGACY, stats: &[STEP_ID, ITER_NUM] },
];

pub enum Field<'a> {
    Num(u64),
    Bytes(usize, &'a [u8]),
}

#[inline(always)]
pub fn varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let first = *buf.get(*pos)?;
    if first < 0x80 {
        *pos += 1;
        return Some(u64::from(first));
    }
    let (mut value, mut shift) = (0u64, 0);
    while shift < 64 {
        let byte = *buf.get(*pos)?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Some(value);
        }
        shift += 7;
    }
    None
}

pub struct Fields<'a> {
    buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Iterator for Fields<'a> {
    type Item = (u32, Field<'a>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let (buf, mut pos) = (self.buf, self.pos);
        if pos >= buf.len() {
            return None;
        }
        self.pos = usize::MAX;
        let key = varint(buf, &mut pos)?;
        if !(1..=MAX_TAG).contains(&(key >> 3)) {
            return None;
        }
        let field = match key & 7 {
            0 => Field::Num(varint(buf, &mut pos)?),
            1 => {
                let value = u64::from_le_bytes(*buf.get(pos..)?.first_chunk()?);
                pos += 8;
                Field::Num(value)
            }
            2 => {
                let length = usize::try_from(varint(buf, &mut pos)?).ok()?;
                let body = buf.get(pos..pos.checked_add(length)?)?;
                pos += length;
                Field::Bytes(pos - length, body)
            }
            5 => {
                let value = u32::from_le_bytes(*buf.get(pos..)?.first_chunk()?);
                pos += 4;
                Field::Num(u64::from(value))
            }
            _ => return None,
        };
        self.pos = pos;
        Some(((key >> 3) as u32, field))
    }
}

impl Fields<'_> {
    pub fn complete(&self) -> bool {
        self.pos == self.buf.len()
    }
}

pub fn fields(buf: &[u8]) -> Fields<'_> {
    Fields { buf, pos: 0 }
}

pub struct Ev {
    pub ts: u64,
    pub dur: u64,
    pub group: i64,
    pub raw: (u32, u32),
    pub meta: u32,
    pub eager: Option<bool>,
    /// The event has a field of stats.
    pub has_stats: bool,
    /// A stat of the event can have a kind of `STATS`. Without one, the links of the event come only from its metadata.
    pub linked: bool,
}

impl Ev {
    /// The bytes of the event for a reader of its stats. An event without stats gives no bytes.
    pub fn stats_raw<'a>(&self, map: &'a [u8]) -> &'a [u8] {
        if self.has_stats { slice(map, self.raw) } else { &[] }
    }
}

#[derive(Default)]
pub struct Line {
    pub id: i64,
    pub display_id: i64,
    pub name: String,
    pub display_name: String,
    pub events: Vec<Ev>,
    pub labels: Vec<Box<str>>,
    pub longs: Vec<Box<str>>,
    pub steps: FxHashMap<usize, Step>,
    pub args: FxHashMap<usize, Vec<String>>,
    pub timestamp_ns: i64,
}

#[derive(Clone, Default)]
pub struct Step {
    pub name: String,
    pub stats: Vec<(&'static str, i64)>,
}

type Link = Option<(u64, u64)>;

#[derive(Default)]
pub struct Links {
    pub flow: Option<u64>,
    pub group: Option<i64>,
    pub asynchronous: Option<u64>,
    pub producer: Link,
    pub consumer: Link,
    pub pid: Option<i32>,
    pub root: Option<i64>,
    pub step: Option<String>,
    /// The first integer value of the stat `correlation_id`, as `event_stat` finds it.
    pub correlation: Option<u64>,
}

type Stats = [Option<u64>; STATS.len()];

#[derive(Default)]
pub struct Meta {
    pub name: Box<str>,
    pub display: Box<str>,
    pub raw: (u32, u32),
    long: (u32, u32),
    pub internal: bool,
    root: Option<(i64, bool)>,
    step: Option<String>,
    base: Option<Box<Stats>>,
    role: [u8; ROLES],
}

impl Meta {
    pub fn long_name<'a>(&self, map: &'a [u8]) -> Cow<'a, str> {
        String::from_utf8_lossy(slice(map, self.long))
    }

    pub fn full_name<'a>(&'a self, map: &'a [u8]) -> Cow<'a, str> {
        if self.display.is_empty() { Cow::Borrowed(&self.name) } else { self.long_name(map) }
    }
}

#[derive(Default)]
pub struct Plane {
    pub id: i64,
    pub name: String,
    pub meta: Vec<Meta>,
    pub stat_names: Vec<Box<str>>,
    pub lines: Vec<Line>,
    pub pid: Option<i32>,
    pub origin_ns: i64,
    pub own: Vec<(Box<str>, Own)>,
    host: u64,
    kind: Vec<u8>,
    present: [bool; STATS.len()],
    correlation: Option<usize>,
}

pub enum Own {
    Int(i64),
    Double(f64),
    Text(String),
}

#[derive(Clone, Copy)]
pub enum Value<'a> {
    Int(i64),
    Uint(u64),
    Double(f64),
    Str(&'a [u8]),
    Ref(u64),
    Bytes(&'a [u8]),
}

impl Value<'_> {
    pub fn int(&self) -> Option<i64> {
        match self {
            Value::Int(value) => Some(*value),
            Value::Uint(value) => Some(*value as i64),
            _ => None,
        }
    }

    pub fn signed(&self) -> i64 {
        if let Value::Int(value) = self { *value } else { 0 }
    }
}

pub struct Stat<'a> {
    pub id: usize,
    pub value: Value<'a>,
}

pub fn slice(map: &[u8], (offset, length): (u32, u32)) -> &[u8] {
    &map[offset as usize..(offset + length) as usize]
}

pub fn stat(body: &[u8], keep: impl Fn(usize) -> bool) -> Option<Stat<'_>> {
    let (mut id, mut value) = (0, Value::Bytes(&[]));
    for (tag, field) in fields(body) {
        match (tag, field) {
            (1, Field::Num(stat_id)) => {
                id = stat_id as usize;
                if !keep(id) {
                    return None;
                }
            }
            (2, Field::Num(bits)) => value = Value::Double(f64::from_bits(bits)),
            (3, Field::Num(number)) => value = Value::Uint(number),
            (4, Field::Num(number)) => value = Value::Int(number as i64),
            (5, Field::Bytes(_, text)) => value = Value::Str(text),
            (6, Field::Bytes(_, data)) => value = Value::Bytes(data),
            (7, Field::Num(reference)) => value = Value::Ref(reference),
            _ => {}
        }
    }
    keep(id).then_some(Stat { id, value })
}

pub fn stats(raw: &[u8], field: u32, keep: impl Fn(usize) -> bool) -> impl Iterator<Item = Stat<'_>> {
    nested(raw, field).filter_map(move |body| stat(body, &keep))
}

pub fn nested(data: &[u8], wanted: u32) -> impl Iterator<Item = &[u8]> {
    fields(data).filter_map(move |(tag, field)| if let (true, Field::Bytes(_, body)) = (tag == wanted, field) { Some(body) } else { None })
}

pub fn event_stat(raw: &[u8], id: Option<usize>) -> Option<Value<'_>> {
    let id = id?;
    stats(raw, 4, |stat| stat == id).next().map(|stat| stat.value)
}

/// For each id, the same value as `event_stat`, from one pass over the stats.
pub fn event_stats<const N: usize>(raw: &[u8], ids: [Option<usize>; N]) -> [Option<Value<'_>>; N] {
    let mut out = [None; N];
    if ids.iter().all(Option::is_none) {
        return out;
    }
    for body in nested(raw, 4) {
        let first = std::cell::Cell::new(None);
        let Some(stat) = stat(body, |id| ids.contains(&Some(id)) && first.replace(Some(id)).is_none_or(|first| first == id)) else { continue };
        for (slot, id) in out.iter_mut().zip(ids) {
            if slot.is_none() && id == Some(stat.id) {
                *slot = Some(stat.value);
            }
        }
    }
    out
}

pub fn event_group(raw: &[u8], group: i64, id: Option<usize>) -> Option<i64> {
    if group == NONE_GROUP { event_stat(raw, id).map(|value| value.signed()) } else { Some(group) }
}

fn table<T: Default>(entries: Vec<(u64, T)>) -> Option<Vec<T>> {
    let bound = (entries.len() as u64 * ID_FACTOR + ID_SLACK).min(u64::from(u32::MAX));
    if entries.iter().any(|(id, _)| *id >= bound) {
        return None;
    }
    let size = entries.iter().map(|(id, _)| id + 1).max().unwrap_or(0);
    let mut table = Vec::new();
    table.resize_with(size as usize, T::default);
    for (id, value) in entries {
        table[id as usize] = value;
    }
    Some(table)
}

fn named(bytes: &[u8]) -> Option<(u64, &[u8], &[u8])> {
    let mut entry = (0, &[][..], &[][..]);
    let mut entries = fields(bytes);
    for (tag, field) in &mut entries {
        match (tag, field) {
            (1, Field::Num(id)) => entry.0 = id,
            (2, Field::Bytes(_, name)) => entry.1 = name,
            (4, Field::Bytes(_, display)) => entry.2 = display,
            _ => {}
        }
    }
    entries.complete().then_some(entry)
}

fn line((offset, bytes): (usize, &[u8]), limit: u64, kind: &[u8], checked: Option<&AtomicBool>) -> Option<Line> {
    let (mut line, mut bodies) = (Line::default(), Vec::with_capacity(bytes.len() / 32));
    let mut entries = fields(bytes);
    for (tag, field) in &mut entries {
        match (tag, field) {
            (1, Field::Num(id)) => line.id = id as i64,
            (10, Field::Num(id)) => line.display_id = id as i64,
            (2, Field::Bytes(_, name)) => line.name = String::from_utf8_lossy(name).into(),
            (11, Field::Bytes(_, name)) => line.display_name = String::from_utf8_lossy(name).into(),
            (3, Field::Num(nanos)) => line.timestamp_ns = nanos as i64,
            (4, Field::Bytes(start, body)) => bodies.push((start as u32, body.len() as u32)),
            _ => {}
        }
    }
    if !entries.complete() {
        return None;
    }
    let complete = AtomicBool::new(true);
    bodies
        .par_iter()
        .with_min_len(4096)
        .map(|&(start, len)| {
            let mut event = Ev { ts: 0, dur: 0, group: NONE_GROUP, raw: ((offset + start as usize) as u32, len), meta: 0, eager: None, has_stats: false, linked: false };
            let mut parts = fields(&bytes[start as usize..][..len as usize]);
            for (tag, field) in &mut parts {
                match (tag, field) {
                    (1, Field::Num(id)) => event.meta = id.min(limit) as u32,
                    (2, Field::Num(ts)) => event.ts = ts,
                    (3, Field::Num(dur)) => event.dur = dur,
                    (4, Field::Bytes(_, body)) => {
                        event.has_stats = true;
                        if let Some(valid) = checked
                            && !crate::tools::counters::valid_stat(body)
                        {
                            valid.store(false, Ordering::Relaxed);
                        }
                        event.linked = event.linked || !matches!(fields(body).next(), Some((1, Field::Num(id))) if kind.get(id as usize).is_none_or(|&kind| kind == NO_KIND));
                    }
                    _ => {}
                }
            }
            if !parts.complete() {
                complete.store(false, Ordering::Relaxed);
            }
            event
        })
        .collect_into_vec(&mut line.events);
    complete.into_inner().then_some(line)
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn hash(kind: u64, parts: &[u64]) -> u64 {
    parts.iter().fold(mix(kind), |acc, part| mix(acc ^ part))
}

pub fn parse(buf: &[u8]) -> anyhow::Result<Vec<Plane>> {
    parse_checking(buf, None)
}

/// Clears `checked` if a stat of an event does not pass the check of `valid_space`.
pub fn parse_checking(buf: &[u8], checked: Option<&AtomicBool>) -> anyhow::Result<Vec<Plane>> {
    anyhow::ensure!(u32::try_from(buf.len()).is_ok(), "The profile is larger than 4 GiB");
    let mut entries = fields(buf);
    let (mut spans, mut host) = (Vec::new(), None);
    for (tag, field) in &mut entries {
        match (tag, field) {
            (1, Field::Bytes(offset, bytes)) => spans.push((offset, bytes)),
            (4, Field::Bytes(_, name)) => host = host.or(Some(name)),
            _ => {}
        }
    }
    anyhow::ensure!(entries.complete(), "The XSpace is not valid");
    let mut planes: Vec<Plane> = spans.par_iter().map(|&span| Plane::parse(span, buf, checked)).collect::<anyhow::Result<_>>()?;
    let origin = origin_ns(&planes);
    let host = host.map_or(0, |name| hash(name.len() as u64, &name.iter().map(|&byte| byte as u64).collect::<Vec<_>>()));
    planes.iter_mut().for_each(|plane| (plane.origin_ns, plane.host) = (origin, host));
    planes.par_iter_mut().flat_map(|plane| plane.lines.par_iter_mut()).for_each(|line| {
        let base = line.base_ps(origin);
        for event in &mut line.events {
            event.ts = event.ts.saturating_add(base);
            event.dur = event.dur.min(u64::MAX - event.ts);
        }
    });
    Ok(planes)
}

impl Plane {
    fn parse((offset, bytes): (usize, &[u8]), buf: &[u8], checked: Option<&AtomicBool>) -> anyhow::Result<Self> {
        let (mut spans, mut metas, mut names) = (Vec::new(), Vec::new(), Vec::new());
        let mut plane = Self::default();
        let mut entries = fields(bytes);
        for (tag, field) in &mut entries {
            match (tag, field) {
                (1, Field::Num(id)) => plane.id = id as i64,
                (2, Field::Bytes(_, name)) => plane.name = String::from_utf8_lossy(name).into(),
                (3, Field::Bytes(start, body)) => spans.push((offset + start, body)),
                (4 | 5, Field::Bytes(entry_start, entry)) => {
                    let (mut parts, mut key) = (fields(entry), None);
                    for (entry_tag, value) in &mut parts {
                        let (value_start, value) = match (entry_tag, value) {
                            (1, Field::Num(number)) => {
                                key = Some(number);
                                continue;
                            }
                            (2, Field::Bytes(value_start, value)) => (value_start, value),
                            _ => continue,
                        };
                        let (id, long, display) = named(value).context("The metadata is not valid")?;
                        let id = key.unwrap_or(id);
                        let name: Box<str> = if display.is_empty() { String::from_utf8_lossy(long).into() } else { Box::default() };
                        if tag == 4 {
                            let raw = ((offset + entry_start + value_start) as u32, value.len() as u32);
                            let long = if long.is_empty() { (0, 0) } else { ((long.as_ptr() as usize - buf.as_ptr() as usize) as u32, long.len() as u32) };
                            let (internal, root) = (NAMES.get(&*name).is_some_and(|known| known.0), ROOTS.iter().find(|root| root.0 == &*name).map(|root| (root.1, root.2)));
                            metas.push((id, Meta { internal, root, name, display: String::from_utf8_lossy(display).into(), raw, long, ..Default::default() }));
                        } else {
                            names.push((id, name));
                        }
                    }
                    anyhow::ensure!(parts.complete(), "The metadata is not valid");
                }
                _ => {}
            }
        }
        anyhow::ensure!(entries.complete(), "The XPlane is not valid");
        plane.meta = table(metas).context("An event metadata ID is out of range")?;
        plane.stat_names = table(names).context("A stat metadata ID is out of range")?;
        let limit = plane.meta.len() as u64;
        plane.kind = plane.stat_names.iter().map(|name| STATS.iter().position(|stat| stat == &&**name).map_or(NO_KIND, |kind| kind as u8)).collect();
        plane.lines = spans.par_iter().map(|&span| line(span, limit, &plane.kind, checked)).collect::<Option<Vec<Line>>>().context("An XLine is not valid")?;
        for line in plane.lines.iter_mut().filter(|line| !line.events.is_sorted_by_key(|event| (event.ts, Reverse(event.dur)))) {
            line.events.sort_by_key(|event| (event.ts, Reverse(event.dur)));
        }
        if plane.lines.iter().flat_map(|line| &line.events).any(|event| u64::from(event.meta) == limit) {
            plane.meta.push(Meta::default());
        }
        plane.correlation = plane.kind.iter().position(|&kind| kind == CORRELATION as u8);
        for &kind in plane.kind.iter().filter(|&&kind| kind != NO_KIND) {
            plane.present[kind as usize] = true;
        }
        plane.own = stats(bytes, 6, |_| true)
            .map(|stat| {
                let value = match stat.value {
                    Value::Int(value) => Own::Int(value),
                    Value::Uint(value) => Own::Int(value as i64),
                    Value::Double(value) => Own::Double(value),
                    value => Own::Text(plane.text(&value)),
                };
                (plane.stat_names.get(stat.id).cloned().unwrap_or_default(), value)
            })
            .collect();
        plane.pid = match plane.own.iter().find(|(name, _)| &**name == "process_id") {
            Some((_, Own::Int(value))) => Some(*value as i32),
            _ => None,
        };
        let ready: Vec<bool> = CONNECTIONS.iter().map(|connection| connection.stats.iter().all(|&kind| plane.present[kind] || kind == CORE_TYPE)).collect();
        let bases: Vec<(Option<Box<Stats>>, Option<String>)> = plane
            .meta
            .par_iter()
            .map(|meta| {
                let mut base = Stats::default();
                plane.apply(&mut base, slice(buf, meta.raw), 5, |_| true);
                (base.iter().any(Option::is_some).then(|| Box::new(base)), plane.step(slice(buf, meta.raw), 5))
            })
            .collect();
        for (meta, (base, step)) in plane.meta.iter_mut().zip(bases) {
            let roles = NAMES.get(&*meta.name).map_or([0; ROLES], |known| known.1);
            (meta.base, meta.step, meta.role) = (base, step, std::array::from_fn(|index| if ready[index] { roles[index] } else { 0 }));
        }
        Ok(plane)
    }

    fn apply(&self, out: &mut Stats, raw: &[u8], field: u32, keep: impl Fn(usize) -> bool) {
        for stat in stats(raw, field, |id| self.kind.get(id).is_some_and(|&kind| kind != NO_KIND && keep(kind as usize))) {
            let kind = self.kind[stat.id] as usize;
            if let Some(value) = stat.value.int() {
                out[kind] = Some(value as u64);
            }
        }
    }

    fn step(&self, raw: &[u8], field: u32) -> Option<String> {
        let found = stats(raw, field, |id| self.present[STEP] && self.kind.get(id) == Some(&(STEP as u8)));
        found.filter(|stat| matches!(stat.value, Value::Str(_) | Value::Ref(_))).last().map(|stat| self.text(&stat.value))
    }

    pub fn group_of(&self, meta: u32, raw: &[u8]) -> Option<i64> {
        let mut out = self.meta[meta as usize].base.as_deref().copied().unwrap_or_default();
        self.apply(&mut out, raw, 4, |kind| kind == GROUP);
        out[GROUP].map(|group| group as i64)
    }

    pub fn links(&self, meta: u32, raw: &[u8], ordinal: Option<u64>) -> Links {
        let mut out = self.meta[meta as usize].base.as_deref().copied().unwrap_or_default();
        let (mut correlation, mut step, mut first_correlation) = (None, None, None);
        for body in nested(raw, 4) {
            let only_correlation = std::cell::Cell::new(true);
            let Some(stat) = stat(body, |id| {
                only_correlation.set(only_correlation.get() && Some(id) == self.correlation);
                self.kind.get(id).is_some_and(|&kind| kind != NO_KIND)
            }) else {
                continue;
            };
            if first_correlation.is_none() && only_correlation.get() {
                first_correlation = stat.value.int().map(|value| value as u64);
            }
            let kind = self.kind[stat.id] as usize;
            if let Some(value) = stat.value.int() {
                out[kind] = Some(value as u64);
            }
            match (kind, &stat.value) {
                (CORRELATION, Value::Uint(id)) => correlation = Some(*id),
                (CORRELATION, _) => correlation = Some(0),
                (STEP, Value::Str(_) | Value::Ref(_)) => step = Some(stat.value),
                _ => {}
            }
        }
        let get = |kind: usize| out[kind];
        let (mut producer, mut consumer) = (get(P).zip(get(PT)), get(C).zip(get(CT)));
        let meta = &self.meta[meta as usize];
        for (connection, &role) in CONNECTIONS.iter().zip(&meta.role) {
            let parts: Option<Vec<u64>> = (role != 0)
                .then(|| {
                    connection
                        .stats
                        .iter()
                        .map(|&kind| if self.present[kind] { get(kind).map(|value| if kind == RUN_ID && connection.context == LAUNCH { value & RUN_ID_MASK } else { value }) } else { Some(0) })
                        .collect()
                })
                .flatten();
            if let Some(parts) = parts {
                let link = Some((hash(connection.context, &parts), connection.context));
                if role == 1 { producer = link } else { consumer = link }
            }
        }
        if let (Some(ordinal), Some(queue), Some(run)) = (ordinal, get(QUEUE_ID), get(RUN_ID)) {
            consumer = Some((hash(LAUNCH, &[ordinal, queue, run, get(CORE_TYPE).unwrap_or(0)]), LAUNCH));
        }
        let mut pid = get(PID).map(|pid| pid as i32);
        if producer.is_some() || (consumer.is_some() && pid.is_none()) {
            pid = self.pid.or(pid);
        }
        let synthesize = |id: u64, kind: u64, direction: u64, category: u64| Some(category << 58 | ((hash(kind, &[id]) ^ self.host) & ((1 << 56) - 1)) << 2 | direction);
        let correlated = correlation.map_or_else(|| get(FLOW), |id| synthesize(id, u64::MAX, if self.name.starts_with("/device:") { 1 } else { 2 }, 9));
        let stated = get(ROOT).map(|level| level as i64);
        Links {
            flow: consumer.map_or_else(|| producer.map_or(correlated, |(id, kind)| synthesize(id, kind, 2, kind.min(63))), |(id, kind)| synthesize(id, kind, 1, kind.min(63))),
            group: get(GROUP).map(|group| group as i64),
            asynchronous: get(ASYNC),
            producer,
            consumer,
            pid: pid.filter(|_| ![producer, consumer].iter().flatten().any(|&(_, kind)| kind == LAUNCH)),
            root: match meta.root {
                Some((level, true)) => Some(level),
                Some((level, false)) => stated.or(Some(level)),
                None => stated,
            },
            step: step.map(|value| self.text(&value)).or_else(|| meta.step.clone()),
            correlation: first_correlation,
        }
    }

    pub fn own_double(&self, name: &str) -> f64 {
        self.own.iter().find_map(|(key, value)| (&**key == name).then_some(value)).map_or(0.0, |value| if let Own::Double(value) = value { *value } else { 0.0 })
    }

    pub fn own_text(&self, name: &str) -> Option<&str> {
        self.own.iter().find_map(|(key, value)| match value {
            Own::Text(text) if &**key == name => Some(text.as_str()),
            _ => None,
        })
    }

    pub fn has_roots(&self) -> bool {
        self.present[ROOT] || self.meta.iter().any(|meta| meta.root.is_some())
    }

    pub fn has_contexts(&self) -> bool {
        [PT, P, CT, C].iter().any(|&kind| self.present[kind])
    }

    pub fn id(&self, name: &str) -> Option<usize> {
        self.stat_names.iter().position(|stat| &**stat == name)
    }

    pub fn find<'a>(&self, map: &'a [u8], meta: u32, raw: (u32, u32), id: Option<usize>) -> Option<Value<'a>> {
        let id = id?;
        let find = |raw: (u32, u32), field: u32| stats(slice(map, raw), field, |stat| stat == id).next().map(|stat| stat.value);
        find(raw, 4).or_else(|| find(self.meta[meta as usize].raw, 5))
    }

    pub fn stat<'a>(&self, map: &'a [u8], meta: u32, raw: (u32, u32), name: &str) -> Option<Value<'a>> {
        self.find(map, meta, raw, self.id(name))
    }

    pub fn named_stats<'a>(&'a self, map: &'a [u8], meta: u32, raw: (u32, u32)) -> impl Iterator<Item = (u32, &'a Box<str>, Stat<'a>)> {
        [(self.meta[meta as usize].raw, 5), (raw, 4)].into_iter().flat_map(move |(raw, field)| {
            stats(slice(map, raw), field, |_| true).filter_map(move |stat| Some((field, self.stat_names.get(stat.id).filter(|name| !INTERNAL_STATS.contains(&&***name))?, stat)))
        })
    }

    pub fn text(&self, value: &Value) -> String {
        self.text_cow(value).into_owned()
    }

    pub fn text_cow<'a>(&'a self, value: &Value<'a>) -> Cow<'a, str> {
        match value {
            Value::Str(bytes) => String::from_utf8_lossy(bytes),
            Value::Ref(id) => Cow::Borrowed(self.stat_names.get(*id as usize).map_or("", |name| name)),
            _ => Cow::Borrowed(""),
        }
    }

    pub fn consumes_number(&self, id: usize) -> bool {
        self.kind.get(id).is_some_and(|&kind| kind <= ASYNC as u8)
    }

    pub fn add_threadpool_regions(&mut self, map: &[u8]) {
        let find = |name: &str| self.meta.iter().position(|meta| &*meta.name == name);
        let (Some(start), Some(stop)) = (find(REGION[0]), find(REGION[1])) else { return };
        let consumer = self.kind.iter().position(|&kind| kind as usize == C);
        let region = find(REGION[2]).unwrap_or_else(|| {
            self.meta.push(Meta { name: REGION[2].into(), ..Default::default() });
            self.meta.len() - 1
        });
        let origin = self.origin_ns;
        for line in &mut self.lines {
            let (mut started, mut regions) = (None, Vec::new());
            for event in &line.events {
                if event.meta as usize == start {
                    if stats(slice(map, event.raw), 4, |id| Some(id) == consumer).any(|stat| matches!(stat.value, Value::Int(_) | Value::Uint(_))) {
                        started = Some(event.ts).filter(|&ts| line.absolute_ps(ts, origin) != 0);
                    }
                } else if let Some(started) = started.filter(|_| event.meta as usize == stop) {
                    regions.push(Ev { ts: started, dur: event.ts.saturating_sub(started), group: NONE_GROUP, raw: (0, 0), meta: region as u32, eager: None, has_stats: false, linked: false });
                }
            }
            if !regions.is_empty() {
                line.events.extend(regions);
                line.events.sort_by_key(|event| (event.ts, Reverse(event.dur)));
            }
        }
    }
}

impl Line {
    pub fn base_ps(&self, origin_ns: i64) -> u64 {
        u64::try_from(self.timestamp_ns.saturating_sub(origin_ns)).unwrap_or(0).saturating_mul(1000)
    }

    pub fn absolute_ps(&self, ts: u64, origin_ns: i64) -> u64 {
        ts.wrapping_add((self.timestamp_ns as u64).wrapping_mul(1000).wrapping_sub(self.base_ps(origin_ns)))
    }

    pub fn resource_id(&self) -> u32 {
        (if self.display_id != 0 { self.display_id } else { self.id }) as u32
    }
}

pub fn origin_ns(planes: &[Plane]) -> i64 {
    let earliest = planes.iter().flat_map(|plane| plane.lines.iter().map(|line| line.timestamp_ns)).filter(|&t| t > 0).min().unwrap_or(0);
    if earliest > EPOCH_NS { earliest } else { 0 }
}
