use crate::trace::json::{CONTEXT_TYPES, HOST_PID_STRIDE, View, devices, ordered};
use crate::trace::{NONE_FLOW, NONE_RESOURCE};
use crate::xplane::{NONE_GROUP, Value};
use prost::Message;
use rayon::prelude::*;

#[derive(Clone, PartialEq, Message)]
pub(crate) struct SeriesMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) process_id: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) thread_id: u32,
    #[prost(uint32, tag = "3")]
    pub(crate) name_ref: u32,
}

#[derive(Clone, PartialEq, Message)]
pub(crate) struct EventMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) flow_id: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) flow_category: u32,
    #[prost(uint32, tag = "3")]
    pub(crate) group_id: u32,
    #[prost(uint32, tag = "4")]
    pub(crate) serial: u32,
    #[prost(oneof = "CounterValue", tags = "5, 6")]
    pub(crate) counter_value: Option<CounterValue>,
}

#[derive(Clone, PartialEq, prost::Oneof)]
pub(crate) enum CounterValue {
    #[prost(double, tag = "5")]
    Double(f64),
    #[prost(uint64, tag = "6")]
    Uint(u64),
}

#[derive(Clone, PartialEq, Message)]
pub(crate) struct Series {
    #[prost(message, optional, tag = "1")]
    pub(crate) metadata: Option<SeriesMetadata>,
    #[prost(uint64, repeated, tag = "2")]
    pub(crate) deltas: Vec<u64>,
    #[prost(uint64, repeated, tag = "3")]
    pub(crate) durations: Vec<u64>,
    #[prost(uint32, repeated, tag = "4")]
    pub(crate) name_refs: Vec<u32>,
    #[prost(message, repeated, tag = "5")]
    pub(crate) event_metadata: Vec<EventMetadata>,
}

#[derive(Clone, PartialEq, Message)]
pub(crate) struct Thread {
    #[prost(string, tag = "1")]
    pub(crate) name: String,
    #[prost(uint32, tag = "2")]
    pub(crate) id: u32,
    #[prost(uint32, optional, tag = "3")]
    pub(crate) sort_index: Option<u32>,
}

#[derive(Clone, PartialEq, Message)]
pub(crate) struct Process {
    #[prost(string, tag = "1")]
    pub(crate) name: String,
    #[prost(uint32, tag = "2")]
    pub(crate) id: u32,
    #[prost(uint32, optional, tag = "3")]
    pub(crate) sort_index: Option<u32>,
    #[prost(message, repeated, tag = "4")]
    pub(crate) threads: Vec<Thread>,
}

#[derive(Clone, PartialEq, Message)]
pub(crate) struct Metadata {
    #[prost(message, repeated, tag = "1")]
    pub(crate) processes: Vec<Process>,
}

#[derive(Clone, PartialEq, Message)]
pub(crate) struct Detail {
    #[prost(string, tag = "1")]
    pub(crate) name: String,
    #[prost(bool, tag = "2")]
    pub(crate) value: bool,
}

#[derive(Clone, PartialEq, Message)]
pub(crate) struct Response {
    #[prost(message, repeated, tag = "1")]
    pub(crate) complete_events: Vec<Series>,
    #[prost(message, repeated, tag = "2")]
    pub(crate) counter_events: Vec<Series>,
    #[prost(message, repeated, tag = "3")]
    pub(crate) async_events: Vec<Series>,
    #[prost(message, optional, tag = "4")]
    pub(crate) metadata: Option<Metadata>,
    #[prost(string, repeated, tag = "5")]
    pub(crate) interned_strings: Vec<String>,
    #[prost(message, repeated, tag = "6")]
    pub(crate) details: Vec<Detail>,
    #[prost(uint64, optional, tag = "7")]
    pub(crate) full_timespan_start_ps: Option<u64>,
    #[prost(uint64, optional, tag = "8")]
    pub(crate) full_timespan_end_ps: Option<u64>,
}

pub fn render(views: &[View], full_dma: Option<bool>) -> Vec<u8> {
    let mut response = Response::default();
    let offset = if full_dma.is_some() { 0 } else { HOST_PID_STRIDE };
    let processes = devices(views).into_iter().map(|(pid, device)| {
        let threads = device.resources.iter().map(|(&id, name)| Thread { name: name.clone(), id, sort_index: Some(id) }).collect();
        Process { name: device.name.clone(), id: pid - offset, sort_index: Some(pid - offset), threads }
    });
    response.metadata = Some(Metadata { processes: processes.collect() });
    let tpu = views.iter().any(|view| !view.trace.tpu_devices.is_empty());
    if let Some(full_dma) = full_dma {
        response.details.push(Detail { name: "mpmd_pipeline_view".into(), value: false });
        if tpu {
            response.details.push(Detail { name: "full_dma".into(), value: full_dma });
        }
    }
    response.full_timespan_start_ps = views.iter().filter(|view| !view.trace.events.is_empty()).map(|view| view.trace.min_ps).min();
    response.full_timespan_end_ps = views.iter().filter(|view| !view.trace.events.is_empty()).map(|view| view.trace.max_ps).max();
    let ordered = ordered(views);
    let track = |&(host, index): &(u32, u32)| (host, views[host as usize].trace.events[index as usize].track);
    // The series hold the indices of the trace names and of the context types. The interned IDs come after, in the order of the series.
    let mut built: Vec<(u32, u32, Series)> = ordered
        .par_chunk_by(|left, right| track(left) == track(right))
        .map(|run| {
            let (host, view) = (run[0].0, &views[run[0].0 as usize]);
            let first = &view.trace.events[run[0].1 as usize];
            let unbound = first.resource == NONE_RESOURCE;
            let tag = if unbound { 2 + u32::from(first.flow != NONE_FLOW) } else { 1 };
            let (thread_id, name_ref) = if unbound { (0, first.name) } else { (first.resource, 0) };
            let process_id = first.device + (host + 1) * HOST_PID_STRIDE - offset;
            let mut series = Series { metadata: Some(SeriesMetadata { process_id, thread_id, name_ref }), ..Default::default() };
            let mut last = 0;
            for &(_, index) in run {
                let event = &view.trace.events[index as usize];
                series.deltas.push(event.ts - last);
                last = event.ts;
                let mut metadata = EventMetadata::default();
                if tag == 2 {
                    metadata.counter_value = first_value(view, event);
                } else {
                    series.durations.push(event.dur);
                    if !unbound {
                        series.name_refs.push(event.name);
                    }
                    if event.flow != NONE_FLOW {
                        metadata.flow_id = view.trace.flow_ids[event.flow as usize] as u32;
                        metadata.flow_category = if matches!(event.flow_cat, 0 | 1) { 0 } else { u32::from(event.flow_cat) };
                    }
                    if event.group != NONE_GROUP {
                        metadata.group_id = event.group as u32;
                    }
                    metadata.serial = if full_dma.is_some() { event.serial } else { 0 };
                }
                series.event_metadata.push(metadata);
            }
            (host, tag, series)
        })
        .collect();
    let mut interner = crate::trace::Interner::default();
    interner.intern("");
    let mut ids: Vec<Vec<u32>> = views.iter().map(|view| vec![u32::MAX; view.trace.names.len()]).collect();
    for (host, tag, series) in &mut built {
        let mut name = |interner: &mut crate::trace::Interner, name: &mut u32| {
            let id = &mut ids[*host as usize][*name as usize];
            if *id == u32::MAX {
                *id = interner.intern(&views[*host as usize].trace.names[*name as usize]);
            }
            *name = *id;
        };
        if *tag != 1 {
            name(&mut interner, &mut series.metadata.as_mut().unwrap().name_ref);
        }
        for (index, metadata) in series.event_metadata.iter_mut().enumerate() {
            if let Some(slot) = series.name_refs.get_mut(index) {
                name(&mut interner, slot);
            }
            if metadata.flow_category != 0 {
                metadata.flow_category = interner.intern(CONTEXT_TYPES.split('|').nth(metadata.flow_category as usize).unwrap_or(""));
            }
        }
    }
    response.interned_strings = interner.0.into_iter().collect();
    // Protobuf writes the fields in the order of their tags, so the series come before the other fields of the response.
    built.sort_by_key(|(_, tag, _)| *tag);
    let parts: Vec<Vec<u8>> = built
        .into_par_iter()
        .map(|(_, tag, series)| {
            let mut out = Vec::with_capacity(prost::encoding::message::encoded_len(tag, &series));
            prost::encoding::message::encode(tag, &series, &mut out);
            out
        })
        .collect();
    let mut raw = parts.concat();
    response.encode(&mut raw).unwrap();
    zstd::bulk::compress(&raw, 1).unwrap()
}

fn first_value(view: &View, event: &crate::trace::Event) -> Option<CounterValue> {
    match view.planes[event.plane as usize].named_stats(view.map, event.meta, event.raw).next()?.2.value {
        Value::Double(value) => Some(CounterValue::Double(value)),
        Value::Uint(value) => Some(CounterValue::Uint(value)),
        _ => None,
    }
}
