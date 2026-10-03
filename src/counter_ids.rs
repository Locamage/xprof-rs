use std::collections::HashMap;
use std::sync::LazyLock;

const V6E: &str = include_str!("tpu_counter_ids_v6e.h");
const V7X: &str = include_str!("tpu_counter_ids_v7x.h");

pub static V6E_IDS: LazyLock<HashMap<String, u64>> = LazyLock::new(|| enumerators(V6E).into_iter().collect());
pub static V7X_IDS: LazyLock<HashMap<String, u64>> = LazyLock::new(|| enumerators(V7X).into_iter().collect());
static V6E_NAMES: LazyLock<String> = LazyLock::new(|| names_json(V6E));
static V7X_NAMES: LazyLock<String> = LazyLock::new(|| names_json(V7X));

pub fn names(device_type: &str) -> Option<&'static str> {
    match device_type {
        "v6e" => Some(V6E_NAMES.as_str()),
        "v7x" => Some(V7X_NAMES.as_str()),
        _ => None,
    }
}

fn integer(text: &str) -> Option<u64> {
    let digits = text.replace('_', "");
    let lower = digits.to_ascii_lowercase();
    match lower.get(..2) {
        Some("0x") => u64::from_str_radix(&lower[2..], 16).ok(),
        Some("0o") => u64::from_str_radix(&lower[2..], 8).ok(),
        Some("0b") => u64::from_str_radix(&lower[2..], 2).ok(),
        _ if lower.len() > 1 && lower.starts_with('0') && lower.bytes().any(|byte| byte != b'0') => None,
        _ => lower.parse().ok(),
    }
}

fn enumerators(header: &str) -> Vec<(String, u64)> {
    let code: Vec<&str> = header.lines().map(|line| line.split_once("//").map_or(line, |(code, _)| code)).collect();
    let code = code.join("\n");
    let body = code.split("uint64_t").skip(1).find_map(|rest| rest.trim_start().strip_prefix('{')?.split_once('}').map(|(body, _)| body).filter(|body| !body.is_empty())).unwrap_or("");
    body.split(',').filter_map(|entry| entry.trim().split_once('=').and_then(|(name, value)| Some((name.trim().to_string(), integer(value.trim())?)))).collect()
}

pub fn names_json(header: &str) -> String {
    let entries: Vec<String> = enumerators(header).into_iter().map(|(name, value)| format!("{{\"name\": \"{}\", \"val\": {value}}}", name.to_lowercase())).collect();
    format!("[{}]", entries.join(", "))
}

#[cfg(test)]
#[path = "tests/inline/counter_ids.rs"]
mod tests;
