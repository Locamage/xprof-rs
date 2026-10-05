use super::xspace::{V, XSpace};
use crate::counters::{events, planes};
use crate::xplane::{Field, fields};
use prost::Message;
use std::path::{Path, PathBuf};

pub const DEMO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/demo.xplane.pb");
pub const TRAINING_STEP_PS: i64 = 20_000_000_000;
pub const MEDIUM_STEP_PS: i64 = 35_000_000_000;
const V6E_PEAK_TFLOPS_BF16: f64 = 918.0;
const V6E_PEAK_HBM_GB_S: f64 = 1638.0;
const RIDGE_DERATING: f64 = 0.9313;
const STEPS: i64 = 4;
const SUB_STEP_NS: f64 = 10_000_000.0;
const IGNORED_LINES: [&str; 4] = ["_counters_", "power throttle", "steps", "xla modules"];
const CORE_DETAILS: &[u8] = b"\x0a\x04host";
const OP_FLOPS: i64 = 1_000_000_000_000;

type Line = (String, String, Vec<(f64, f64)>);

pub fn demo_query(name: &str, command: &str, flags: &[&str]) -> crate::cli::json::J {
    let path = demo(&super::cli_support::scratch(name));
    let argv: Vec<&str> = [command, path.to_str().unwrap()].into_iter().chain(flags.iter().copied()).collect();
    super::cli_support::ok(&argv)
}

pub fn training_trace(step_ps: i64) -> Vec<u8> {
    let mut space = XSpace::default();
    let plane = space.tpu(0, "TPU v6 lite", V6E_PEAK_TFLOPS_BF16, V6E_PEAK_HBM_GB_S, None);
    plane.id = 0;
    plane.add_stat("core_details", V::Bytes(CORE_DETAILS.to_vec()));
    plane.named_line(0, "Steps");
    plane.named_line(1, "XLA Modules");
    plane.named_line(2, "XLA Ops");
    for (symbol, name, category) in [(1, "fusion.1", "convolution fusion"), (2, "convolution.2", "convolution")] {
        plane.metadata_stats(name, &[("program_id", 1.into()), ("symbol_id", symbol.into()), ("hlo_category", category.into()), ("flops", OP_FLOPS.into())]);
    }
    for step in 0..STEPS {
        let (start, group) = (step * step_ps, [("group_id", V::from(step + 1))]);
        plane.event(0, &(step + 1).to_string(), start, step_ps, &group);
        plane.event(1, "jit_train_step(1)", start, step_ps, &[("group_id", (step + 1).into()), ("program_id", 1.into())]);
        plane.event(2, "fusion.1", start + step_ps / 200, step_ps * 3 / 4, &group);
        plane.event(2, "convolution.2", start + step_ps * 76 / 100, step_ps * 23 / 100, &group);
    }
    space.encode_to_vec()
}

pub fn session(dir: &Path, name: &str, data: &[u8]) -> PathBuf {
    let run = dir.join("plugins/profile/run");
    std::fs::create_dir_all(&run).unwrap();
    let path = run.join(name);
    std::fs::write(&path, data).unwrap();
    path
}

pub fn demo(dir: &Path) -> PathBuf {
    session(dir, "demo.xplane.pb", &std::fs::read(DEMO).unwrap())
}

fn device_lines(path: &Path) -> Vec<Line> {
    let map = std::fs::read(path).unwrap();
    let mut found = Vec::new();
    for plane in planes(&map, |_| true) {
        for bytes in &plane.lines {
            let (mut name, mut timestamp) = (String::new(), 0i64);
            for (tag, field) in fields(bytes) {
                match (tag, field) {
                    (2, Field::Bytes(_, text)) => name = String::from_utf8_lossy(text).into_owned(),
                    (3, Field::Num(value)) => timestamp = value as i64,
                    _ => {}
                }
            }
            let spans = events(bytes).map(|event| (timestamp as f64 + event.offset_ps as i64 as f64 / 1000.0, event.duration_ps as i64 as f64 / 1000.0)).collect();
            found.push((plane.name.clone(), name, spans));
        }
    }
    found
}

pub fn oracle_step_time_ms(path: &Path) -> f64 {
    let lines = device_lines(path);
    let average = |keep: &dyn Fn(&str, &str) -> bool| {
        let durations: Vec<f64> = lines
            .iter()
            .filter(|(plane, line, _)| keep(plane, line))
            .flat_map(|(_, _, spans)| spans.iter().filter(|(_, duration)| *duration > SUB_STEP_NS).map(|(_, duration)| duration / 1_000_000.0))
            .collect();
        (!durations.is_empty()).then(|| durations.iter().sum::<f64>() / durations.len() as f64)
    };
    average(&|plane, line| plane.starts_with("/device:") && !plane.contains("SparseCore") && line.trim() == "Steps")
        .or_else(|| average(&|_, line| line.trim() == "Steps"))
        .or_else(|| average(&|_, line| line.to_uppercase().contains("XLA MODULES")))
        .unwrap_or(0.0)
}

pub fn oracle_duty_cycle(path: &Path) -> f64 {
    let mut intervals: Vec<(i64, i64)> = device_lines(path)
        .into_iter()
        .filter(|(plane, line, _)| plane.starts_with("/device:") && !IGNORED_LINES.iter().any(|ignored| line.to_lowercase().contains(ignored)))
        .flat_map(|(_, _, spans)| spans.into_iter().map(|(start, duration)| (start as i64, start as i64 + duration as i64)))
        .filter(|(start, end)| end > start)
        .collect();
    intervals.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (start, end) in intervals {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    let (Some(first), Some(last)) = (merged.first(), merged.last()) else { return 0.0 };
    let span = last.1 - first.0;
    if span <= 0 { 0.0 } else { (merged.iter().map(|(start, end)| end - start).sum::<i64>() as f64 / span as f64).min(1.0) }
}

pub fn oracle_ridge_point(peak_flops: f64, peak_bandwidth: f64) -> f64 {
    if peak_bandwidth > 0.0 { RIDGE_DERATING * peak_flops / peak_bandwidth } else { 0.0 }
}

pub fn datasheet_violations(step_time_ms: f64, duty_cycle: f64) -> Vec<String> {
    let mut violations = Vec::new();
    if step_time_ms <= 0.0 {
        violations.push(format!("Step time must be positive (got {step_time_ms} ms)"));
    }
    if !(0.0..=1.0).contains(&duty_cycle) {
        violations.push(format!("Duty cycle must be in [0, 1] (got {duty_cycle})"));
    }
    violations
}
