use crate::trace::json::{CONTEXT_TYPES, HOST_PID_STRIDE, View, devices, ordered};
use crate::trace::{NONE_FLOW, NONE_RESOURCE};
use crate::xplane::{NONE_GROUP, Value};
use prost::Message;

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
    let mut interner = crate::trace::Interner::default();
    interner.intern("");
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
    let mut start = 0;
    while start < ordered.len() {
        let track = |&(host, index): &(u32, u32)| (host, views[host as usize].trace.events[index as usize].track);
        let end = start + ordered[start..].iter().take_while(|entry| track(entry) == track(&ordered[start])).count();
        let (host, first) = (ordered[start].0, &views[ordered[start].0 as usize].trace.events[ordered[start].1 as usize]);
        let (view, pid) = (&views[host as usize], first.device + (host + 1) * HOST_PID_STRIDE - offset);
        let unbound = first.resource == NONE_RESOURCE;
        let counter = unbound && first.flow == NONE_FLOW;
        let (thread_id, name_ref) = if unbound { (0, interner.intern(&view.trace.names[first.name as usize])) } else { (first.resource, 0) };
        let mut series = Series { metadata: Some(SeriesMetadata { process_id: pid, thread_id, name_ref }), ..Default::default() };
        let mut last = 0;
        for &(_, index) in &ordered[start..end] {
            let event = &view.trace.events[index as usize];
            series.deltas.push(event.ts - last);
            last = event.ts;
            let mut metadata = EventMetadata::default();
            if counter {
                metadata.counter_value = first_value(view, event);
            } else {
                series.durations.push(event.dur);
                if !unbound {
                    series.name_refs.push(interner.intern(&view.trace.names[event.name as usize]));
                }
                if event.flow != NONE_FLOW {
                    metadata.flow_id = view.trace.flow_ids[event.flow as usize] as u32;
                    if !matches!(event.flow_cat, 0 | 1) {
                        metadata.flow_category = interner.intern(CONTEXT_TYPES.split('|').nth(event.flow_cat as usize).unwrap_or(""));
                    }
                }
                if event.group != NONE_GROUP {
                    metadata.group_id = event.group as u32;
                }
                metadata.serial = if full_dma.is_some() { event.serial } else { 0 };
            }
            series.event_metadata.push(metadata);
        }
        match (counter, unbound) {
            (true, _) => response.counter_events.push(series),
            (false, true) => response.async_events.push(series),
            (false, false) => response.complete_events.push(series),
        }
        start = end;
    }
    response.interned_strings = interner.into_strings();
    let mut compressed = Vec::new();
    ruzstd::encoding::compress(&response.encode_to_vec()[..], &mut compressed, ruzstd::encoding::CompressionLevel::Fastest);
    compressed
}

fn first_value(view: &View, event: &crate::trace::Event) -> Option<CounterValue> {
    match view.planes[event.plane as usize].named_stats(view.map, event.meta, event.raw).next()?.2.value {
        Value::Double(value) => Some(CounterValue::Double(value)),
        Value::Uint(value) => Some(CounterValue::Uint(value)),
        _ => None,
    }
}
