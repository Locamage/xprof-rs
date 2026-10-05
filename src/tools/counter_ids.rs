use std::collections::HashMap;
use std::sync::LazyLock;

const V6E: &str = include_str!("../data/tpu_counters_v6e.txt");
const V7X: &str = include_str!("../data/tpu_counters_v7x.txt");

pub static V6E_IDS: LazyLock<HashMap<&str, u64>> = LazyLock::new(|| entries(V6E).collect());
pub static V7X_IDS: LazyLock<HashMap<&str, u64>> = LazyLock::new(|| entries(V7X).collect());
static V6E_NAMES: LazyLock<String> = LazyLock::new(|| names_json(V6E));
static V7X_NAMES: LazyLock<String> = LazyLock::new(|| names_json(V7X));

pub fn names(device_type: &str) -> Option<&'static str> {
    match device_type {
        "v6e" => Some(V6E_NAMES.as_str()),
        "v7x" => Some(V7X_NAMES.as_str()),
        _ => None,
    }
}

/// Each line of a counter table is `NAME VALUE`, in the order of the `XProf` header.
fn entries(table: &str) -> impl Iterator<Item = (&str, u64)> {
    table.lines().filter_map(|line| line.split_once(' ').and_then(|(name, value)| Some((name, value.parse().ok()?))))
}

pub fn names_json(table: &str) -> String {
    let entries: Vec<String> = entries(table).map(|(name, value)| format!("{{\"name\": \"{}\", \"val\": {value}}}", name.to_lowercase())).collect();
    format!("[{}]", entries.join(", "))
}

#[cfg(test)]
#[path = "../tests/inline/tools/counter_ids.rs"]
mod tests;
