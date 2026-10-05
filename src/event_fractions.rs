use crate::derive::{STEP_LINE, is_derived, is_tensor_core};
use crate::group::is_sparse_core;
use crate::opstats::EventReader;
use crate::xplane::{NONE_GROUP, Plane, Value, event_group, event_stat, slice};
use std::collections::{BTreeMap, HashMap, HashSet};

const OP_LINES: [&str; 3] = ["XLA Ops", "Framework Ops", "Sparse Core Ops"];
pub const BARRIER_CORES: &str = "barrier-cores";
const DUMMY_BARRIER_PS: u64 = 1250;
const STEP_DURATION_RATIO: f64 = 0.01;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Fractions {
    pub chips: BTreeMap<String, Vec<f32>>,
    pub hosts: BTreeMap<String, Vec<f32>>,
}

fn plane_steps(plane: &Plane, map: &[u8]) -> HashMap<i64, u64> {
    let (tensor, sparse) = (is_tensor_core(&plane.name), is_sparse_core(&plane.name));
    let (group_id, correlation_id) = (plane.id("group_id"), plane.id("correlation_id"));
    let mut markers: HashMap<i64, u64> = HashMap::new();
    let mut ops: HashSet<i64> = HashSet::new();
    let reader = EventReader::new(plane);
    for line in &plane.lines {
        if line.id as i32 == STEP_LINE as i32 || (tensor && line.name == "Steps") || (sparse && line.name == "Sparse Core Steps") {
            markers = HashMap::new();
            for event in &line.events {
                if let Some(step) = event_group(slice(map, event.raw), event.group, group_id) {
                    let longest = markers.entry(step).or_default();
                    *longest = (*longest).max(reader.span(map, event).1);
                }
            }
        } else if is_derived(line.id) {
        } else if tensor || sparse {
            if (tensor && OP_LINES.contains(&line.name.as_str())) || (sparse && line.name == "Sparse Core Ops") {
                ops = line
                    .events
                    .iter()
                    .filter_map(|event| if event.group == NONE_GROUP { event_stat(slice(map, event.raw), group_id).map(|value| value.int().unwrap_or(0)) } else { Some(event.group) })
                    .collect();
            }
        } else {
            for event in &line.events {
                let correlation = event_stat(slice(map, event.raw), correlation_id).map_or(-1, |value| value.signed());
                let step = event_stat(slice(map, event.raw), group_id).map_or(-1, |value| value.signed());
                if correlation >= 0 && step >= 0 {
                    ops.insert(step);
                }
            }
        }
    }
    if ops.is_empty() {
        return HashMap::new();
    }
    markers.retain(|step, _| ops.contains(step));
    markers
}

pub fn analyze(planes: &[Plane], map: &[u8], hostname: &str, target: &str) -> Fractions {
    let mut fractions: BTreeMap<i64, BTreeMap<String, f64>> = BTreeMap::new();
    let mut durations: HashMap<i64, u64> = HashMap::new();
    let steps: HashMap<&str, HashMap<i64, u64>> = planes.iter().map(|plane| (plane.name.as_str(), plane_steps(plane, map))).collect();
    for plane in planes {
        let plane_steps = &steps[plane.name.as_str()];
        let wanted: HashSet<u32> = (0..plane.meta.len()).filter(|&meta| plane.meta[meta].full_name(map).contains(target)).map(|meta| meta as u32).collect();
        let (group_id, duration_id) = (plane.id("group_id"), plane.id("device_duration_ps"));
        for event in plane.lines.iter().flat_map(|line| &line.events).filter(|event| wanted.contains(&event.meta)) {
            let Some(step) = event_group(slice(map, event.raw), event.group, group_id) else { continue };
            let Some(&step_ps) = plane_steps.get(&step).filter(|&&step_ps| step_ps != 0) else { continue };
            let Some(duration) = event_stat(slice(map, event.raw), duration_id) else { continue };
            let event_ps = if let Value::Uint(value) = duration { value } else { 0 };
            if target == BARRIER_CORES && (event_ps == 0 || event_ps == DUMMY_BARRIER_PS) {
                continue;
            }
            let portion = (event_ps as f64 / step_ps as f64) as f32;
            *fractions.entry(step).or_default().entry(plane.name.clone()).or_default() += f64::from(portion);
            durations.insert(step, step_ps);
        }
    }
    if fractions.len() >= 3 {
        fractions.pop_first();
        fractions.pop_last();
    } else if fractions.len() == 2 {
        let (first, second) = (*fractions.keys().next().unwrap(), *fractions.keys().last().unwrap());
        let (first_ps, second_ps) = (durations[&first] as f64, durations[&second] as f64);
        if second_ps < STEP_DURATION_RATIO * first_ps {
            fractions.remove(&second);
        } else if first_ps < STEP_DURATION_RATIO * second_ps {
            fractions.remove(&first);
        }
    }
    let mut result = Fractions::default();
    for planes in fractions.values() {
        for (plane, fraction) in planes {
            result.chips.entry(plane.clone()).or_default().push(*fraction as f32);
        }
    }
    let host: Vec<f32> = fractions.values().filter(|planes| !planes.is_empty()).map(|planes| (planes.values().sum::<f64>() / planes.len() as f64) as f32).collect();
    result.hosts.insert(hostname.to_string(), host);
    result
}

pub fn accumulate(result: Fractions, combined: &mut Fractions) {
    for (chip, fractions) in result.chips {
        combined.chips.entry(chip).or_default().extend(fractions);
    }
    for (host, fractions) in result.hosts {
        combined.hosts.entry(host).or_default().extend(fractions);
    }
}

#[cfg(test)]
#[path = "tests/inline/event_fractions.rs"]
mod tests;
