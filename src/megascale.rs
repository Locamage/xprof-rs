use crate::hlo::{Module, Shape, module_name, protos};
use crate::opstats::safe_divide;
use crate::table::{number, string};
use crate::xplane::{Ev, Field, Line, Plane, Value, fields, slice, stats};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

const COLUMNS: &str = r#"{"cols":[{"id":"rendezvous_name","label":"Rendezvous Name","type":"string"},{"id":"recv_op_name","label":"Recv Op Name","type":"string"},{"id":"send_op_name","label":"Send Op Name","type":"string"},{"id":"transfer_type","label":"Transfer Type","type":"string"},{"id":"slack_time","label":"Slack Time (ms)","type":"number"},{"id":"host_stall","label":"Host Stall (ms)","type":"string"},{"id":"observed_duration","label":"Observed Duration (ms)","type":"number"},{"id":"send_stall","label":"Send Stall (ms)","type":"number"},{"id":"send_done_stall","label":"SendDone Stall (ms)","type":"number"},{"id":"recv_stall","label":"Recv Stall (ms)","type":"number"},{"id":"recv_done_stall","label":"RecvDone Stall (ms)","type":"number"},{"id":"stall_duration","label":"Stall Duration (ms)","type":"number"},{"id":"total_stall","label":"Aggregated Total Stall (ms)","type":"number"},{"id":"occurrences","label":"Occurrences","type":"number"},{"id":"net_tx_bytes","label":"Data Transmitted Size","type":"number"},{"id":"required_bandwidth","label":"Required Bandwidth (Gbps)","type":"number"}],"rows":["#;
const ALL_HOSTS: &str = "ALL_HOSTS";
const TPU_PREFIX: &str = "/device:TPU:";
const FIRST_TPU: &str = "/device:TPU:0";
const RENDEZVOUS_ATTRIBUTE: &str = "_xla_host_transfer_rendezvous";
const TRANSFER_ATTRIBUTE: &str = "_xla_megascale_transfer_type";
const BANDWIDTH_FACTOR: f64 = 8E-9 / 1E-6;
const UNITS: &[u8] = b"KMGTPE";
const SLACK: usize = 0;
const OBSERVED: usize = 1;
const STALL: usize = 2;
const SEND: usize = 3;
const SEND_DONE: usize = 4;
const RECV: usize = 5;
const RECV_DONE: usize = 6;

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Summary {
    pub rendezvous: String,
    pub transfer_type: String,
    pub recv_op_name: String,
    pub send_op_name: String,
    pub occurrences: u64,
    pub bytes_transmitted_over_network: u64,
    pub times_us: [u64; 7],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    Send,
    Recv,
    SendDone,
    RecvDone,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: Opcode,
    pub channel_id: u64,
    pub rendezvous: Option<String>,
    pub transfer_type: Option<String>,
    pub size: i64,
}

#[derive(Default)]
struct OpState {
    start_time: u64,
    overlap_base: u64,
    transfer_type: String,
    stall_duration_ns: u64,
    send_op_name: String,
    replica_group_size: i64,
    send_ps: u64,
    send_done_ps: u64,
    recv_ps: u64,
}

pub struct Visit<'a> {
    pub timestamp_ns: i64,
    pub duration_ps: i64,
    pub display_name: &'a str,
    pub collective_info: &'a [u8],
}

#[derive(Default)]
pub struct Tracker {
    channels: HashMap<u64, String>,
    states: HashMap<String, OpState>,
    group_sizes: HashMap<String, i64>,
    summaries: Vec<Summary>,
    positions: HashMap<String, usize>,
    overlap: u64,
}

fn replica_group_size(info: &[u8]) -> i64 {
    let mut groups = fields(info).filter_map(|(tag, field)| if let Field::Bytes(_, body) = field { Some((tag, body)) } else { None });
    let (one_to_one, endpoints): (Vec<_>, Vec<_>) = groups.by_ref().filter(|(tag, _)| *tag == 2 || *tag == 3).partition(|(tag, _)| *tag == 3);
    if one_to_one.is_empty() { endpoints.first().map_or(0, |(_, group)| fields(group).filter(|(tag, field)| *tag == 1 && matches!(field, Field::Bytes(..))).count() as i64) } else { 1 }
}

pub fn transmitted_bytes(size: i64, group: i64, transfer_type: &str) -> u64 {
    if group == 0 {
        return 0;
    }
    let ratio = safe_divide((group - 1) as f64, group as f64);
    match transfer_type {
        "ONE_TO_ONE" => group.wrapping_mul(size) as u64,
        "ALL_GATHER" => safe_divide((group - 1).wrapping_mul(size) as f64, group as f64) as u64,
        "ALL_REDUCE" => (2.0 * ratio * size as f64) as u64,
        "ALL_TO_ALL" => (ratio * size as f64) as u64,
        "REDUCE_SCATTER" => size.wrapping_mul(group - 1) as u64,
        _ => 0,
    }
}

fn averaged(mut summaries: Vec<Summary>) -> Vec<Summary> {
    for summary in &mut summaries {
        let occurrences = summary.occurrences as f64;
        summary.times_us.iter_mut().for_each(|time| *time = safe_divide(*time as f64, occurrences) as u64);
    }
    summaries
}

impl Tracker {
    pub fn visit(&mut self, instruction: &Instruction, visit: &Visit) {
        let rendezvous = match &instruction.rendezvous {
            Some(name) => {
                self.channels.insert(instruction.channel_id, name.clone());
                name.clone()
            }
            None => match self.channels.get(&instruction.channel_id) {
                Some(name) => name.clone(),
                None => return,
            },
        };
        let duration_ns = (visit.duration_ps as u64) as f64 / 1e3;
        let duration_ps = visit.duration_ps as u64;
        let group = match instruction.opcode {
            Opcode::Send => *self.group_sizes.entry(rendezvous.clone()).or_insert_with(|| replica_group_size(visit.collective_info)),
            _ => 0,
        };
        let overlap = self.overlap;
        let state = self.states.entry(rendezvous.clone()).or_insert_with(|| OpState { overlap_base: overlap, ..Default::default() });
        state.stall_duration_ns = (state.stall_duration_ns as f64 + duration_ns) as u64;
        match instruction.opcode {
            Opcode::Send => {
                state.start_time = visit.timestamp_ns as u64;
                state.transfer_type = instruction.transfer_type.clone().unwrap_or_default();
                state.overlap_base = overlap;
                state.stall_duration_ns = duration_ns as u64;
                state.send_op_name = visit.display_name.to_string();
                state.send_ps = duration_ps;
                state.replica_group_size = group;
            }
            Opcode::Recv => state.recv_ps = duration_ps,
            Opcode::SendDone => state.send_done_ps = duration_ps,
            Opcode::RecvDone if state.start_time != 0 => {
                let index = *self.positions.entry(rendezvous.clone()).or_insert_with(|| {
                    self.summaries.push(Summary { rendezvous: rendezvous.clone(), ..Default::default() });
                    self.summaries.len() - 1
                });
                let summary = &mut self.summaries[index];
                let end_ns = (visit.timestamp_ns as f64 + duration_ns) as i64;
                let slack_us = ((visit.timestamp_ns as u64).wrapping_sub(state.start_time).wrapping_sub(overlap.wrapping_sub(state.overlap_base)) as f64 / 1e3) as u64;
                let observed_us = (((end_ns as u64) as f64 / 1e3) as u64).wrapping_sub((state.start_time as f64 / 1e3) as u64);
                let times = &mut summary.times_us;
                times[SLACK] = times[SLACK].wrapping_add(slack_us);
                times[STALL] = times[STALL].wrapping_add((state.stall_duration_ns as f64 / 1e3) as u64);
                times[OBSERVED] = times[OBSERVED].wrapping_add(observed_us);
                times[SEND] = (times[SEND] as f64 + state.send_ps as f64 / 1e6) as u64;
                times[RECV] = (times[RECV] as f64 + state.recv_ps as f64 / 1e6 / 1e6) as u64;
                times[SEND_DONE] = (times[SEND_DONE] as f64 + state.send_done_ps as f64 / 1e6) as u64;
                times[RECV_DONE] = (times[RECV_DONE] as f64 + duration_ps as f64 / 1e6) as u64;
                summary.occurrences += 1;
                summary.bytes_transmitted_over_network = transmitted_bytes(instruction.size, state.replica_group_size, &state.transfer_type);
                summary.transfer_type.clone_from(&state.transfer_type);
                summary.recv_op_name = visit.display_name.to_string();
                summary.send_op_name.clone_from(&state.send_op_name);
            }
            _ => {}
        }
        self.overlap = overlap.wrapping_add(duration_ns as u64);
    }
}

pub fn combine(analyses: &[Vec<Summary>]) -> Vec<Summary> {
    let mut combined: Vec<Summary> = Vec::new();
    for slack in analyses.iter().flatten() {
        let index = combined.iter().position(|summary| summary.rendezvous == slack.rendezvous).unwrap_or_else(|| {
            combined.push(Summary { rendezvous: slack.rendezvous.clone(), ..Default::default() });
            combined.len() - 1
        });
        let summary = &mut combined[index];
        for (to, from) in summary.times_us.iter_mut().zip(slack.times_us) {
            *to = to.wrapping_add(from.wrapping_mul(slack.occurrences));
        }
        summary.occurrences += slack.occurrences;
        summary.bytes_transmitted_over_network = slack.bytes_transmitted_over_network;
        summary.recv_op_name.clone_from(&slack.recv_op_name);
        summary.send_op_name.clone_from(&slack.send_op_name);
        summary.transfer_type.clone_from(&slack.transfer_type);
    }
    averaged(combined)
}

pub fn human_bytes(bytes: i64) -> String {
    if bytes == i64::MIN {
        return "-8E".into();
    }
    let (sign, mut value) = if bytes < 0 { ("-", -bytes) } else { ("", bytes) };
    if value < 1024 {
        return format!("{sign}{value}B");
    }
    let mut unit = 0;
    while value >= 1024 * 1024 {
        value /= 1024;
        unit += 1;
    }
    format!("{sign}{:.2$}{}", value as f64 / 1024.0, UNITS[unit] as char, if unit == 0 { 1 } else { 2 })
}

pub fn table(summaries: &[Summary]) -> String {
    let mut out = format!("[{COLUMNS}");
    for (index, summary) in summaries.iter().enumerate() {
        out.push_str(if index > 0 { ",{\"c\":[" } else { "{\"c\":[" });
        for (position, text) in [&summary.rendezvous, &summary.recv_op_name, &summary.send_op_name, &summary.transfer_type].into_iter().enumerate() {
            out.push_str(if position > 0 { ",{\"v\":" } else { "{\"v\":" });
            string(&mut out, text);
            out.push('}');
        }
        let cell = |out: &mut String, value: f64| {
            out.push_str(",{\"v\":");
            number(out, value);
            out.push('}');
        };
        let times = summary.times_us;
        cell(&mut out, times[SLACK] as f64 / 1e3);
        out.push_str(",{\"v\":\"-\"}");
        for value in [times[OBSERVED], times[SEND], times[SEND_DONE], times[RECV], times[RECV_DONE], times[STALL], times[STALL].wrapping_mul(summary.occurrences)] {
            cell(&mut out, value as f64 / 1e3);
        }
        cell(&mut out, summary.occurrences as f64);
        out.push_str(",{\"f\":");
        string(&mut out, &human_bytes(summary.bytes_transmitted_over_network as i64));
        out.push_str(",\"v\":");
        number(&mut out, summary.bytes_transmitted_over_network as f64);
        out.push('}');
        cell(&mut out, if times[SLACK] == 0 { f64::from(i32::MAX) } else { safe_divide(summary.bytes_transmitted_over_network as f64, times[SLACK] as f64) * BANDWIDTH_FACTOR });
        out.push_str("]}");
    }
    out.push_str("]}]");
    out
}

fn instruction(module: &Module, name: &str) -> Option<Instruction> {
    let wanted = crate::hlo::sanitize(name);
    let node = (0..module.nodes.len()).filter(|&node| module.nodes[node].name == wanted).find(|&node| module.inst(node).name == name)?;
    let inst = module.inst(node);
    let attributes = inst.frontend_attributes.as_deref().map(|attributes| &attributes.map);
    let attribute = |key: &str| attributes.and_then(|map| map.get(key)).cloned();
    let opcode = match inst.opcode.as_str() {
        "send" => Opcode::Send,
        "recv" => Opcode::Recv,
        "send-done" => Opcode::SendDone,
        "recv-done" => Opcode::RecvDone,
        _ => Opcode::Other,
    };
    let shape = inst.shape.as_deref().cloned().unwrap_or_default();
    let size = match (shape.is_array(), shape.is_tuple()) {
        (true, _) => shape.unpadded_bytes(),
        (false, true) => shape.tuple_shapes.iter().filter(|element| element.is_array()).map(Shape::unpadded_bytes).sum(),
        _ => 0,
    };
    Some(Instruction { opcode, channel_id: inst.channel_id as u64, rendezvous: attribute(RENDEZVOUS_ATTRIBUTE), transfer_type: attribute(TRANSFER_ATTRIBUTE), size })
}

fn absolute(line: &Line, event: &Ev, origin: i64) -> (i64, i64) {
    let offset = event.ts.wrapping_sub((line.timestamp_ns.wrapping_sub(origin) as u64).wrapping_mul(1000)) as i64;
    let timestamp_ps = line.timestamp_ns.wrapping_mul(1000).wrapping_add(offset);
    (timestamp_ps, (line.timestamp_ns as f64 + (offset as u64) as f64 / 1e3) as i64)
}

fn metadata_text<'a>(plane: &'a Plane, map: &'a [u8], meta: u32, id: Option<usize>) -> Option<Value<'a>> {
    let id = id?;
    stats(slice(map, plane.meta.get(meta as usize)?.raw), 5, |stat| stat == id).next().map(|stat| stat.value)
}

pub fn analyze(planes: &[Plane], map: &[u8]) -> Vec<Summary> {
    let cores = planes.iter().filter(|plane| plane.name.strip_prefix(TPU_PREFIX).is_some_and(|id| id.bytes().all(|byte| byte.is_ascii_digit()))).count();
    let Some(plane) = planes.iter().find(|plane| plane.name == FIRST_TPU).filter(|_| cores > 0) else { return Vec::new() };
    let modules: HashMap<String, &[u8]> = protos(planes, map).into_iter().rev().map(|(program, proto)| (format!("{}({program})", module_name(proto)), proto)).collect();
    let mut parsed: HashMap<String, Option<Module>> = HashMap::new();
    let mut instructions: HashMap<String, Instruction> = HashMap::new();
    let mut tracker = Tracker::default();
    let origin = crate::xplane::origin_ns(planes);
    let Some(module_line) = plane.lines.iter().find(|line| line.name == "XLA Modules") else { return averaged(tracker.summaries) };
    let (mut current, category_id, info_id) = (None, plane.id("hlo_category"), plane.id("dcn_collective_info"));
    for line in plane.lines.iter().filter(|line| line.name == "XLA Ops") {
        for event in &line.events {
            let category = metadata_text(plane, map, event.meta, category_id).map(|value| plane.text(&value)).unwrap_or_default();
            let (timestamp_ps, timestamp_ns) = absolute(line, event, origin);
            let span = (timestamp_ps, timestamp_ps.wrapping_add(event.dur as i64));
            let module_span = |index: usize| {
                let module = &module_line.events[index];
                let begin = absolute(module_line, module, origin).0;
                (begin, begin.wrapping_add(module.dur as i64))
            };
            let includes = |outer: (i64, i64)| outer.0 <= span.0 && span.1 <= outer.1;
            let mut containing = current.filter(|&index| includes(module_span(index)));
            if containing.is_none() {
                for index in current.map_or(0, |index| index + 1)..module_line.events.len() {
                    let candidate = module_span(index);
                    if candidate.0 > span.1 {
                        break;
                    }
                    if candidate.1 < span.0 {
                        continue;
                    }
                    current = Some(index);
                    containing = includes(candidate).then_some(index);
                    break;
                }
            }
            let Some(index) = containing else { continue };
            if !category.contains("host send") && !category.contains("host recv") {
                continue;
            }
            let module_meta = module_line.events[index].meta as usize;
            let module = if module_meta < plane.meta.len() { plane.meta[module_meta].full_name(map) } else { Cow::Borrowed("") };
            let display = plane.meta.get(event.meta as usize).map_or("", |meta| &meta.display);
            let key = format!("{module}_{display}");
            if !instructions.contains_key(&key) {
                let loaded = parsed.entry(module.to_string()).or_insert_with(|| modules.get(module.as_ref()).map(|proto| Module::parse(Cow::Borrowed(proto))).filter(|module| module.valid));
                let Some(found) = loaded.as_ref().and_then(|module| instruction(module, display)) else { continue };
                instructions.insert(key.clone(), found);
            }
            let info = match metadata_text(plane, map, event.meta, info_id) {
                Some(Value::Bytes(bytes)) => bytes,
                _ => &[],
            };
            tracker.visit(&instructions[&key], &Visit { timestamp_ns, duration_ps: event.dur as i64, display_name: display, collective_info: info });
        }
    }
    averaged(tracker.summaries)
}

pub fn json(paths: &[PathBuf], host: &str) -> Option<String> {
    let mut analyses: Vec<(String, Vec<Summary>)> = Vec::new();
    for path in paths {
        let map = crate::read_file(path).ok()?;
        if !crate::run_tools::tools(&map)?.contains(&"megascale_stats") {
            return Some(table(&[]));
        }
        let planes = crate::xplane::parse(&map).ok()?;
        analyses.push((crate::host_name(path), analyze(&planes, &map)));
    }
    if host == ALL_HOSTS {
        return Some(table(&combine(&analyses.into_iter().map(|(_, analysis)| analysis).collect::<Vec<_>>())));
    }
    analyses.into_iter().find(|(name, _)| name == host).map(|(_, analysis)| table(&analysis))
}

#[cfg(test)]
#[path = "tests/inline/megascale.rs"]
mod tests;
