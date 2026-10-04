use crate::derive::is_tensor_core;
use crate::group::is_sparse_core;
use crate::opstats::EventReader;
use crate::xplane::{Ev, NONE_GROUP, Plane, Value, slice, stats};
use std::collections::{BTreeMap, HashMap, HashSet};

const DERIVED_MIN: i32 = 0xdead_beef_u32 as i32;
const DERIVED_MAX: i32 = DERIVED_MIN + 389;
const OP_LINES: [&str; 3] = ["XLA Ops", "Framework Ops", "Sparse Core Ops"];
pub const BARRIER_CORES: &str = "barrier-cores";
const DUMMY_BARRIER_PS: u64 = 1250;
const STEP_DURATION_RATIO: f64 = 0.01;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Fractions {
    pub chips: BTreeMap<String, Vec<f32>>,
    pub hosts: BTreeMap<String, Vec<f32>>,
}

fn own<'a>(map: &'a [u8], event: &Ev, id: Option<usize>) -> Option<Value<'a>> {
    let id = id?;
    stats(slice(map, event.raw), 4, |stat| stat == id).next().map(|stat| stat.value)
}

fn int_value(value: &Value) -> i64 {
    if let Value::Int(value) = value { *value } else { 0 }
}

fn group(map: &[u8], event: &Ev, id: Option<usize>, widen: bool) -> Option<i64> {
    if event.group != NONE_GROUP {
        return Some(event.group);
    }
    own(map, event, id).map(|value| if widen { value.int().unwrap_or(0) } else { int_value(&value) })
}

fn plane_steps(plane: &Plane, map: &[u8]) -> HashMap<i64, u64> {
    let (tensor, sparse) = (is_tensor_core(&plane.name), is_sparse_core(&plane.name));
    let (group_id, correlation_id) = (plane.id("group_id"), plane.id("correlation_id"));
    let mut markers: HashMap<i64, u64> = HashMap::new();
    let mut ops: HashSet<i64> = HashSet::new();
    let reader = EventReader::new(plane);
    for line in &plane.lines {
        let id = line.id as i32;
        if id == DERIVED_MIN || (tensor && line.name == "Steps") || (sparse && line.name == "Sparse Core Steps") {
            markers = HashMap::new();
            for event in &line.events {
                if let Some(step) = group(map, event, group_id, false) {
                    let longest = markers.entry(step).or_default();
                    *longest = (*longest).max(reader.span(map, event).1);
                }
            }
        } else if (DERIVED_MIN..=DERIVED_MAX).contains(&id) {
        } else if tensor || sparse {
            if (tensor && OP_LINES.contains(&line.name.as_str())) || (sparse && line.name == "Sparse Core Ops") {
                ops = line.events.iter().filter_map(|event| group(map, event, group_id, true)).collect();
            }
        } else {
            for event in &line.events {
                let correlation = own(map, event, correlation_id).map_or(-1, |value| int_value(&value));
                let step = own(map, event, group_id).map_or(-1, |value| int_value(&value));
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

pub fn raw_name<'a>(plane: &'a Plane, map: &'a [u8], meta: usize) -> std::borrow::Cow<'a, str> {
    let meta = &plane.meta[meta];
    if meta.display.is_empty() { std::borrow::Cow::Borrowed(&*meta.name) } else { meta.long_name(map) }
}

pub fn analyze(planes: &[Plane], map: &[u8], hostname: &str, target: &str) -> Fractions {
    let mut fractions: BTreeMap<i64, BTreeMap<String, f64>> = BTreeMap::new();
    let mut durations: HashMap<i64, u64> = HashMap::new();
    let steps: HashMap<&str, HashMap<i64, u64>> = planes.iter().map(|plane| (plane.name.as_str(), plane_steps(plane, map))).collect();
    for plane in planes {
        let plane_steps = &steps[plane.name.as_str()];
        let wanted: HashSet<u32> = (0..plane.meta.len()).filter(|&meta| raw_name(plane, map, meta).contains(target)).map(|meta| meta as u32).collect();
        let (group_id, duration_id) = (plane.id("group_id"), plane.id("device_duration_ps"));
        for event in plane.lines.iter().flat_map(|line| &line.events).filter(|event| wanted.contains(&event.meta)) {
            let Some(step) = group(map, event, group_id, false) else { continue };
            let Some(&step_ps) = plane_steps.get(&step).filter(|&&step_ps| step_ps != 0) else { continue };
            let Some(duration) = own(map, event, duration_id) else { continue };
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
