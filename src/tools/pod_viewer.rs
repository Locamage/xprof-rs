use crate::hlo::proto_text::quoted;
use crate::tools::input_pipeline_analyzer::{EMPTY_INTERSECT, NO_STEP_MARKER};
use crate::tools::op_profile::proto_double;
use crate::tools::opstats::{OpStats, pico_to_micro};
use crate::xplane::steps::{
    Breakdown, DEVICE_COLLECTIVES, DEVICE_COMPUTE_16, DEVICE_COMPUTE_32, DEVICE_TO_DEVICE, DEVICE_TO_HOST, DEVICE_WAIT_DEVICE, DEVICE_WAIT_HOST, Extra, HOST_COMPILE, HOST_COMPUTE, HOST_PREPARE,
    HOST_TO_DEVICE, HOST_WAIT_INPUT, StepInfo, UNKNOWN_TIME,
};
use std::fmt::Write;

const GENERIC_EVENTS: [(&str, &[u32]); 9] = [
    ("Device compute", &[DEVICE_COMPUTE_32, DEVICE_COMPUTE_16]),
    ("Device to device", &[DEVICE_TO_DEVICE, DEVICE_WAIT_DEVICE]),
    ("Device collective communication", &[DEVICE_COLLECTIVES]),
    ("Host compute", &[HOST_COMPUTE]),
    ("Kernel launch", &[HOST_PREPARE]),
    ("Input", &[HOST_WAIT_INPUT, HOST_TO_DEVICE, DEVICE_WAIT_HOST]),
    ("Output", &[DEVICE_TO_HOST]),
    ("Compilation", &[HOST_COMPILE]),
    ("All others", &[UNKNOWN_TIME]),
];

const EMPTY_RECORD: &str = "{\"hostName\":\"\",\"chipId\":0,\"nodeId\":0,\"stepNum\":0,\"totalDurationUs\":0,\"bottleneck\":\"\",\"stepBreakdownUs\":{}}";

fn record(host_name: &str, step_num: u32, info: &StepInfo) -> String {
    let breakdown_ps = GENERIC_EVENTS.map(|(_, kinds)| match &info.breakdown {
        Breakdown::Types(types) => kinds.iter().map(|kind| types.get(kind).copied().unwrap_or(0)).sum(),
        Breakdown::Categories(_) => 0,
    });
    let bottleneck = breakdown_ps.iter().zip(GENERIC_EVENTS).map(|(&ps, (name, _))| (ps, name)).max().map_or("", |metric| metric.1);
    let mut out = format!("{{\"hostName\":{},\"chipId\":0,\"nodeId\":0,\"stepNum\":{step_num},\"totalDurationUs\":", quoted(host_name));
    proto_double(&mut out, pico_to_micro(info.duration));
    write!(out, ",\"bottleneck\":{},\"stepBreakdownUs\":{{", quoted(bottleneck)).unwrap();
    for (index, ps) in breakdown_ps.iter().enumerate() {
        write!(out, "{}\"{}\":", if index > 0 { "," } else { "" }, index + 1).unwrap();
        proto_double(&mut out, pico_to_micro(*ps));
    }
    out.push_str("}}");
    out
}

pub fn render(extra: &Extra) -> String {
    let records: Vec<String> =
        extra.steps.iter().flat_map(|step| step.cores.iter().filter_map(move |(core, info)| extra.cores.get(core).map(|details| record(&details.hostname, step.num, info)))).collect();
    let mut out = String::from("{\"podStatsSequence\":{\"podStatsMap\":[");
    let mut next = 0;
    for (position, step) in extra.steps.iter().enumerate() {
        write!(out, "{}{{\"stepNum\":{},\"podStatsPerCore\":{{", if position > 0 { "," } else { "" }, step.num).unwrap();
        for (index, core) in step.cores.keys().enumerate() {
            write!(out, "{}\"{core}\":", if index > 0 { "," } else { "" }).unwrap();
            out.push_str(records.get(next).map_or(EMPTY_RECORD, String::as_str));
            next += 1;
        }
        out.push_str("},\"channelDb\":[],\"coreIdToReplicaIdMap\":{},\"allReduceOpDb\":[]}");
    }
    let warning = match (extra.steps.is_empty(), extra.empty_intersect) {
        (false, _) => String::new(),
        (true, true) => quoted(EMPTY_INTERSECT),
        (true, false) => quoted(NO_STEP_MARKER),
    };
    write!(out, "]}},\"diagnostics\":{{\"info\":[],\"warnings\":[{warning}],\"errors\":[]}},\"stepBreakdownEvents\":[").unwrap();
    for (index, (name, _)) in GENERIC_EVENTS.iter().enumerate() {
        write!(out, "{}{{\"id\":{},\"name\":{}}}", if index > 0 { "," } else { "" }, index + 1, quoted(name)).unwrap();
    }
    write!(out, "],\"deviceType\":{}}}", quoted(&extra.device_type)).unwrap();
    out
}

pub fn json(stats: &OpStats) -> String {
    render(&stats.extra)
}

#[cfg(test)]
#[path = "../tests/inline/tools/pod_viewer.rs"]
mod tests;
