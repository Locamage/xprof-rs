pub mod client;
pub mod hlo;
pub mod json;
pub mod ops;
pub mod overview;
pub mod steps;
pub mod xplane;

use crate::obj;
use client::{Client, Local};
use json::J;
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use std::io::Write;
use std::path::Path;

const SPILL_BYTES: usize = 10 * 1024 * 1024;
const REPORT: &str = "https://github.com/openxla/xprof/issues";
const ALIASES: [&str; 3] = ["--session_dir", "--session_path", "--source"];
const SESSION_LIKE: [&str; 11] =
    ["session_id", "source", "baseline_session_id", "optimized_session_id", "run_name", "module_name", "instruction_name", "func_name", "kernel_name", "host_name", "host"];
const KEYWORDS: [&str; 32] = [
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
    "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
];

type Handler = fn(&dyn Client, &Args) -> Result<Out, Error>;

pub const COMMANDS: [(&str, &str, Handler); 30] = [
    ("aggregate_xplane_events", "session_id! plane_regex event_regex bypass_cache | logdir", xplane::aggregate_xplane_events),
    ("check_host_boundness", "session_id! func_name bypass_cache | logdir", steps::check_host_boundness),
    ("compute_utilization", "session_id! | kernel_name duration_us force_duration host device output_format raw_bytes bypass_cache logdir", xplane::get_kernel_utilization),
    ("get_avg_step_time", "source! | func_name output_format bypass_cache logdir", xplane::get_avg_step_time),
    ("get_device_information", "session_id! bypass_cache | logdir", overview::get_device_information),
    (
        "get_graph_viewer",
        "session_id | symbol_id symbol_type graph_type module_name output_type show_metadata node_name graph_width merge_fusion tag tool op_profile_limit use_xplane bypass_cache logdir",
        hlo::get_graph_viewer,
    ),
    ("get_hlo_module_content", "session_id! fmt module_name max_lines | print_metadata bypass_cache logdir", hlo::get_hlo_module_content),
    ("get_hlo_neighborhood", "session_id! instruction_name radius module_name | op_name print_metadata bypass_cache logdir", hlo::get_hlo_neighborhood),
    ("get_hlo_op_profile", "session_id! top_n view category path depth sort_by bypass_cache | logdir", ops::get_hlo_op_profile),
    ("get_hlo_stats", "session_id! | limit sort_by category_filter bypass_cache logdir", ops::get_hlo_stats),
    ("get_hlo_text", "session_id! path module_name op_name bypass_cache | logdir", hlo::get_hlo_text),
    ("get_hosts", "session_id! bypass_cache | logdir", overview::get_hosts),
    ("get_kernel_stats", "source session_id | kernel_name limit output_format include_summary device_to_use trace_matchers bypass_cache logdir", xplane::get_kernel_stats),
    ("get_kernel_utilization", "session_id! | kernel_name duration_us force_duration host device output_format raw_bytes bypass_cache logdir", xplane::get_kernel_utilization),
    ("get_kpi_metrics", "session_id! | bypass_cache logdir", overview::get_kpi_metrics),
    ("get_llo_analysis", "session_id! host kernel bypass_cache | logdir", xplane::get_llo_analysis),
    ("get_llo_debug_string", "session_id! host bypass_cache | logdir", xplane::get_llo_debug_string),
    ("get_memory_profile", "session_id! | bypass_cache logdir", overview::get_memory_profile),
    ("get_overview", "session_id! include_command bypass_cache | logdir", overview::get_overview),
    ("get_peak_allocations", "session_id! | limit min_size_mib output_format include_summary aggregate_instructions bypass_cache logdir", hlo::get_peak_allocations),
    ("get_profile_summary", "session_id! bypass_cache | logdir", ops::get_profile_summary),
    ("get_roofline_model", "session_id! | top_n group_by bypass_cache logdir", overview::get_roofline_model),
    ("get_step_trace", "session_id! | step_num limit device_core include_summary bypass_cache logdir", steps::get_step_trace),
    ("get_top_hlo_ops", "session_id! | limit category_filter bypass_cache logdir", ops::get_top_hlo_ops),
    ("get_utilization_viewer", "session_id! | host device node bypass_cache logdir", overview::get_utilization_viewer),
    ("get_xspace_proto", "session_id! as_text output_path bypass_cache | logdir **", xplane::get_xspace_proto),
    ("list_hlo_modules", "session_id! | bypass_cache logdir", hlo::list_hlo_modules),
    ("list_xplane_events", "session_id! | plane_regex event_regex start_time_ps end_time_ps max_events offset bypass_cache logdir", xplane::list_xplane_events),
    ("upload_trace", "file_path! ttl tag run_name | logdir bypass_cache **", xplane::upload_trace),
    (
        "verify_numerical_parity",
        "kernel_ref! kernel_candidate! shapes! dtype_str tier max_allowed_ulp p99_9_allowed_ulp seed regimes kernel_oracle device_kind | logdir bypass_cache",
        xplane::verify_numerical_parity,
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    FileNotFound,
    Os,
    Value,
    Runtime,
    Assertion,
    Type,
    NotImplemented,
    Import,
    Fire,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: Kind,
    pub message: String,
}

impl Error {
    pub fn new(kind: Kind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }

    pub fn name(&self) -> &'static str {
        match self.kind {
            Kind::FileNotFound => "FileNotFoundError",
            Kind::Os => "OSError",
            Kind::Value => "ValueError",
            Kind::Runtime => "RuntimeError",
            Kind::Assertion => "AssertionError",
            Kind::Type => "TypeError",
            Kind::NotImplemented => "NotImplementedError",
            Kind::Import => "ImportError",
            Kind::Fire => "FireError",
        }
    }

    pub fn repr(&self) -> String {
        if self.message.is_empty() { format!("{}()", self.name()) } else { format!("{}({})", self.name(), json::py_repr(&self.message)) }
    }

    fn reason(&self) -> (&'static str, i32) {
        match self.kind {
            Kind::Type | Kind::Fire => ("USAGE_ERROR", 2),
            Kind::FileNotFound | Kind::Os => ("PATH_ERROR", 3),
            Kind::Value => ("INVALID_VALUE", 4),
            _ => ("INTERNAL_ERROR", 1),
        }
    }
}

pub fn bypass(on: bool) -> (&'static str, String) {
    ("bypass_cache", if on { "True" } else { "False" }.to_string())
}

pub fn rethrow(error: Error, message: impl FnOnce(&Error) -> String) -> Error {
    if matches!(error.kind, Kind::FileNotFound | Kind::Value | Kind::Fire) { error } else { Error::new(Kind::Runtime, message(&error)) }
}

pub fn round(value: f64, digits: usize) -> f64 {
    if value.is_finite() { format!("{value:.digits$}").parse().unwrap_or(value) } else { value }
}

pub fn fsum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values = values.into_iter();
    let Some(mut total) = values.next().map(|first| 0.0 + first) else { return 0.0 };
    let mut compensation = 0.0;
    for value in values {
        let next = total + value;
        compensation += if total.abs() >= value.abs() { (total - next) + value } else { (value - next) + total };
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() { total + compensation } else { total }
}

pub fn stdev(values: &[f64]) -> f64 {
    let parts: Vec<(i64, i32)> = values
        .iter()
        .map(|value| {
            let bits = value.to_bits();
            let (exponent, fraction) = (((bits >> 52) & 0x7ff) as i32, bits & ((1 << 52) - 1));
            let (mantissa, exponent) = if exponent == 0 { (fraction, -1074) } else { (fraction | 1 << 52, exponent - 1075) };
            let zeros = mantissa.trailing_zeros().min(52);
            let (mantissa, exponent) = (mantissa >> zeros, exponent + zeros as i32);
            (if bits >> 63 == 1 { -(mantissa as i64) } else { mantissa as i64 }, exponent)
        })
        .filter(|&(mantissa, _)| mantissa != 0)
        .collect();
    // The result does not change with the scale, so the scale comes from the values that are not zero.
    let low = parts.iter().map(|(_, exponent)| *exponent).min().unwrap_or(0);
    let count = BigInt::from(values.len());
    let (sum, squares) = if parts.iter().all(|&(mantissa, exponent)| 64 - mantissa.unsigned_abs().leading_zeros() as i32 + exponent - low <= 62) {
        let (mut sum, mut squares, mut partial) = (0i128, BigInt::default(), 0u128);
        for &(mantissa, exponent) in &parts {
            let value = i128::from(mantissa << (exponent - low));
            sum += value;
            let square = (value * value) as u128;
            partial = partial.checked_add(square).unwrap_or_else(|| {
                squares += partial;
                square
            });
        }
        (BigInt::from(sum), squares + partial)
    } else {
        let scaled: Vec<BigInt> = parts.into_iter().map(|(mantissa, exponent)| BigInt::from(mantissa) << (exponent - low) as usize).collect();
        (scaled.iter().sum(), scaled.iter().map(|value| value * value).sum())
    };
    let (numerator, denominator) = ((&count * squares - &sum * &sum).magnitude().clone(), (&count * (&count - 1u32)).magnitude().clone());
    let shift = (numerator.bits() as i64 - denominator.bits() as i64 - 109).div_euclid(2);
    let (numerator, denominator) = if shift >= 0 { (numerator, denominator << (2 * shift) as usize) } else { (numerator << (-2 * shift) as usize, denominator) };
    let root = (&numerator / &denominator).sqrt();
    let odd = if &root * &root * &denominator == numerator { root } else { root | BigUint::from(1u32) };
    odd.to_u64().unwrap_or(u64::MAX) as f64 * 2f64.powi(shift as i32 + low)
}

pub fn fail<T>(kind: Kind, message: impl Into<String>) -> Result<T, Error> {
    Err(Error::new(kind, message))
}

#[derive(Debug, PartialEq)]
pub enum Out {
    Text(String),
    Bytes(Vec<u8>),
    Value(J),
}

impl From<J> for Out {
    fn from(value: J) -> Self {
        Self::Text(value.dumps())
    }
}

#[derive(Default, Debug)]
pub struct Args {
    pub values: Vec<(String, J)>,
}

impl Args {
    pub fn get(&self, name: &str) -> Option<&J> {
        self.values.iter().find(|(key, _)| key == name).map(|(_, value)| value).filter(|value| **value != J::Null)
    }

    pub fn text(&self, name: &str) -> Option<String> {
        self.get(name).map(J::text)
    }

    pub fn string(&self, name: &str, default: &str) -> String {
        self.text(name).unwrap_or_else(|| default.to_string())
    }

    pub fn session(&self) -> String {
        self.string("session_id", "")
    }

    pub fn int(&self, name: &str, default: i64) -> Result<i64, Error> {
        self.get(name)
            .map_or(Ok(default), |value| value.int().map(|number| number as i64).ok_or_else(|| Error::new(Kind::Type, format!("'>' not supported between instances of '{}' and 'int'", kind(value)))))
    }

    pub fn flag(&self, name: &str, default: bool) -> bool {
        self.values.iter().find(|(key, _)| key == name).map_or(default, |(_, value)| value.truthy())
    }
}

fn kind(value: &J) -> &'static str {
    match value {
        J::Null => "NoneType",
        J::Bool(_) => "bool",
        J::Int(_) => "int",
        J::Float(_) => "float",
        J::Str(_) => "str",
        J::List(_) => "tuple",
        J::Map(_) => "dict",
    }
}

struct Spec {
    positional: Vec<(String, bool)>,
    keyword: Vec<String>,
    varkw: bool,
}

fn spec(text: &str) -> Spec {
    let (positional, keyword) = text.split_once('|').unwrap_or((text, ""));
    Spec {
        positional: positional.split_whitespace().map(|name| (name.trim_end_matches('!').to_string(), name.ends_with('!'))).collect(),
        keyword: keyword.split_whitespace().filter(|name| *name != "**").map(str::to_string).collect(),
        varkw: keyword.contains("**"),
    }
}

fn is_flag(argument: &str) -> bool {
    let bytes = argument.as_bytes();
    argument.starts_with("--") || (bytes.len() >= 2 && bytes[0] == b'-' && bytes[1].is_ascii_alphabetic())
}

pub fn literal(text: &str) -> J {
    /// Python refuses more nested brackets than this.
    const MAX_DEPTH: usize = 200;
    struct Parser<'a> {
        bytes: &'a [u8],
        at: usize,
        depth: usize,
    }
    impl Parser<'_> {
        fn skip(&mut self) {
            while self.bytes.get(self.at).is_some_and(|byte| *byte == b' ' || *byte == b'\t') {
                self.at += 1;
            }
        }
        fn eat(&mut self, byte: u8) -> bool {
            self.skip();
            let found = self.bytes.get(self.at) == Some(&byte);
            self.at += usize::from(found);
            found
        }
        fn sequence(&mut self, close: u8) -> Option<(Vec<J>, bool)> {
            let (mut items, mut comma) = (Vec::new(), false);
            loop {
                if self.eat(close) {
                    return Some((items, comma));
                }
                items.push(self.value()?);
                comma = self.eat(b',');
                if !comma {
                    return self.eat(close).then_some((items, comma));
                }
            }
        }
        fn value(&mut self) -> Option<J> {
            self.skip();
            let start = self.at;
            let first = *self.bytes.get(self.at)?;
            match first {
                b'(' | b'[' => {
                    self.at += 1;
                    self.depth += 1;
                    if self.depth > MAX_DEPTH {
                        return None;
                    }
                    let (items, comma) = self.sequence(if first == b'(' { b')' } else { b']' })?;
                    self.depth -= 1;
                    Some(if first == b'(' && items.len() == 1 && !comma { items.into_iter().next()? } else { J::List(items) })
                }
                b'\'' | b'"' => {
                    self.at += 1;
                    let mut out = String::new();
                    let rest = std::str::from_utf8(&self.bytes[self.at..]).ok()?;
                    let mut chars = rest.char_indices();
                    while let Some((index, character)) = chars.next() {
                        match character {
                            '\\' => match chars.next()?.1 {
                                'n' => out.push('\n'),
                                't' => out.push('\t'),
                                'r' => out.push('\r'),
                                '0' => out.push('\0'),
                                other @ ('\\' | '\'' | '"') => out.push(other),
                                other => out.extend(['\\', other]),
                            },
                            character if character as u32 == u32::from(first) => {
                                self.at += index + 1;
                                return Some(J::Str(out));
                            }
                            character => out.push(character),
                        }
                    }
                    None
                }
                b'-' | b'+' => {
                    self.at += 1;
                    if matches!(self.bytes[self.at..].iter().find(|byte| !matches!(byte, b' ' | b'\t' | b'(')), Some(b'-' | b'+')) {
                        return None;
                    }
                    match self.value()? {
                        J::Int(number) => Some(J::Int(if first == b'-' { -number } else { number })),
                        J::Float(number) => Some(J::Float(if first == b'-' { -number } else { number })),
                        _ => None,
                    }
                }
                _ => {
                    while self.bytes.get(self.at).is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_' || *byte == b'.' || *byte >= 0x80) {
                        self.at += 1;
                    }
                    let token = std::str::from_utf8(&self.bytes[start..self.at]).ok()?;
                    if (first.is_ascii_digit() || first == b'.') && token.contains(['e', 'E']) && matches!(self.bytes.get(self.at), Some(b'-' | b'+')) && token.ends_with(['e', 'E']) {
                        self.at += 1;
                        while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                            self.at += 1;
                        }
                    }
                    let token = std::str::from_utf8(&self.bytes[start..self.at]).ok()?;
                    match token {
                        "True" => Some(J::Bool(true)),
                        "False" => Some(J::Bool(false)),
                        "None" => Some(J::Null),
                        _ if first.is_ascii_digit() || first == b'.' => {
                            if token.contains('_') {
                                return None;
                            }
                            if token.bytes().all(|byte| byte.is_ascii_digit()) {
                                return (token == "0" || !token.starts_with('0') || token.bytes().all(|byte| byte == b'0')).then(|| token.parse().ok().map(J::Int))?;
                            }
                            let radix = |prefix: &str, base| token.strip_prefix(prefix).and_then(|digits| i128::from_str_radix(digits, base).ok()).map(J::Int);
                            radix("0x", 16).or_else(|| radix("0o", 8)).or_else(|| radix("0b", 2)).or_else(|| token.parse::<f64>().ok().filter(|_| !token.contains(['i', 'n', 'I', 'N'])).map(J::Float))
                        }
                        _ if token.contains('.') || token.is_empty() || KEYWORDS.contains(&token) => None,
                        _ => Some(J::Str(token.to_string())),
                    }
                }
            }
        }
    }
    let mut parser = Parser { bytes: text.as_bytes(), at: 0, depth: 0 };
    let parsed = (|| {
        let first = parser.value()?;
        if !parser.eat(b',') {
            parser.skip();
            return (parser.at == parser.bytes.len()).then_some(first);
        }
        let mut items = vec![first];
        loop {
            parser.skip();
            if parser.at == parser.bytes.len() {
                return Some(J::List(items));
            }
            items.push(parser.value()?);
            if !parser.eat(b',') {
                parser.skip();
                return (parser.at == parser.bytes.len()).then_some(J::List(items));
            }
        }
    })();
    parsed.unwrap_or_else(|| J::Str(text.to_string()))
}

fn bind(name: &str, spec: &Spec, tokens: &[String]) -> Result<(Args, Vec<String>), Error> {
    let names: Vec<&str> = spec.positional.iter().map(|(name, _)| name.as_str()).chain(spec.keyword.iter().map(String::as_str)).collect();
    let (mut keywords, mut unknown, mut positional) = (Vec::<(String, String)>::new(), Vec::new(), Vec::new());
    let mut index = 0;
    while index < tokens.len() {
        let argument = &tokens[index];
        index += 1;
        if !is_flag(argument) {
            positional.push(argument.clone());
            continue;
        }
        let stripped = argument.trim_start_matches('-');
        let (key, value) = match stripped.split_once('=') {
            Some((key, value)) => (key.replace('-', "_"), Some(value.to_string())),
            None => (stripped.replace('-', "_"), None),
        };
        let boolean = value.is_none() && tokens.get(index).is_none_or(|next| is_flag(next));
        let shortcut: Vec<&&str> = names.iter().filter(|candidate| key.len() == 1 && candidate.starts_with(&key)).collect();
        let keyword = if names.contains(&key.as_str()) || (boolean && key.starts_with("no") && names.contains(&&key[2..])) || spec.varkw {
            key.clone()
        } else if shortcut.len() == 1 {
            shortcut[0].to_string()
        } else if shortcut.len() > 1 {
            return fail(
                Kind::Fire,
                format!(
                    "The argument '{argument}' is ambiguous as it could refer to any of the following arguments: [{}]",
                    shortcut.iter().map(|name| json::py_repr(name)).collect::<Vec<_>>().join(", ")
                ),
            );
        } else {
            unknown.push(argument.clone());
            if value.is_none() && !boolean {
                unknown.push(tokens[index].clone());
                index += 1;
            }
            continue;
        };
        let (keyword, value) = match value {
            Some(value) => (keyword, value),
            None if boolean && !names.contains(&keyword.as_str()) && keyword.starts_with("no") && !spec.varkw => (keyword[2..].to_string(), "False".into()),
            None if boolean => (keyword, "True".into()),
            None => {
                index += 1;
                (keyword, tokens[index - 1].clone())
            }
        };
        keywords.retain(|(existing, _)| *existing != keyword);
        keywords.push((keyword, value));
    }
    let mut args = Args::default();
    let mut remaining = positional.into_iter();
    for (parameter, required) in &spec.positional {
        if let Some(position) = keywords.iter().position(|(key, _)| key == parameter) {
            let (key, value) = keywords.remove(position);
            args.values.push((key, literal(&value)));
        } else if let Some(value) = remaining.next() {
            args.values.push((parameter.clone(), literal(&value)));
        } else if *required {
            return fail(Kind::Fire, format!("The function received no value for the required argument: {parameter}\nUsage: xprof {name} {}", usage(spec)));
        }
    }
    args.values.extend(keywords.into_iter().map(|(key, value)| (key, literal(&value))));
    Ok((args, unknown.into_iter().chain(remaining).collect()))
}

fn usage(spec: &Spec) -> String {
    let required: Vec<String> = spec.positional.iter().filter(|(_, required)| *required).map(|(name, _)| name.to_uppercase()).collect();
    format!("{} <flags>", required.join(" "))
}

fn wrap(spec: &Spec, args: &mut Args) -> Result<Option<std::path::PathBuf>, Error> {
    let has = |name: &str| spec.positional.iter().any(|(parameter, _)| parameter == name) || spec.keyword.iter().any(|parameter| parameter == name);
    let logdir = args.values.iter().find(|(key, _)| key == "logdir").map(|(_, value)| value.clone()).filter(|value| *value != J::Null);
    if matches!(logdir, Some(J::Bool(_))) {
        return fail(Kind::Fire, "The --logdir flag requires a value.");
    }
    let first = spec.positional.first().map(|(name, _)| args.values.iter().find(|(key, _)| key == name).map_or(J::Null, |(_, value)| value.clone()));
    if first.as_ref().is_some_and(|value| !value.truthy()) {
        return fail(Kind::Value, "session_id cannot be an empty string.");
    }
    let target = match (&logdir, &first) {
        (Some(logdir), _) => Some(logdir.text()),
        (None, Some(J::Str(text))) if has("session_id") && text.contains(['/', '\\', '.']) => Some(text.clone()),
        _ => None,
    };
    if let Some(target) = target.filter(|target| !has("destination") && !has("run_name") && !target.starts_with("gs://")) {
        let path = expand(&target);
        if !path.exists() {
            return fail(Kind::FileNotFound, format!("Trace path '{target}' does not exist."));
        }
        if path.is_dir() && client::traces(&path).is_empty() {
            return fail(Kind::FileNotFound, format!("No .xplane.pb or .xspace.pb files found in directory '{target}' (DATA_ABSENT)."));
        }
    }
    for (key, value) in &mut args.values {
        if SESSION_LIKE.contains(&key.as_str()) && matches!(value, J::Int(_) | J::Float(_)) {
            *value = J::Str(value.text());
        }
    }
    Ok(logdir.map(|logdir| expand(&logdir.text())))
}

pub fn expand(path: &str) -> std::path::PathBuf {
    match path.strip_prefix('~').filter(|rest| rest.is_empty() || rest.starts_with('/')) {
        Some(rest) => std::path::PathBuf::from(format!("{}{rest}", std::env::var("HOME").unwrap_or_default())),
        None => std::path::PathBuf::from(path),
    }
}

pub fn preprocess(argv: &[String]) -> Vec<String> {
    let (mut tokens, mut alias, mut index) = (Vec::new(), None, 0);
    while index < argv.len() {
        let argument = &argv[index];
        index += 1;
        let (flag, value) = match argument.split_once('=').filter(|_| argument.starts_with('-')) {
            Some((flag, value)) => (flag, Some(value.to_string())),
            None => (argument.as_str(), None),
        };
        if ALIASES.contains(&flag) {
            alias = Some(value.unwrap_or_else(|| match argv.get(index).filter(|next| !next.starts_with('-')) {
                Some(next) => {
                    index += 1;
                    next.clone()
                }
                None => String::new(),
            }));
            continue;
        }
        tokens.push(argument.clone());
    }
    if let Some(alias) = alias {
        tokens.insert(tokens.len().min(1), alias);
    }
    tokens
}

fn spill(command: &str, bytes: &[u8]) -> Option<String> {
    if bytes.len() <= SPILL_BYTES {
        return None;
    }
    let name = COMMANDS.iter().find(|(name, _, _)| *name == command).map_or(command, |(name, _, _)| if *name == "compute_utilization" { "get_kernel_utilization" } else { name });
    let unique = format!("{:08x}", (std::process::id() as u64 ^ std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |time| time.as_nanos() as u64)) & 0xffff_ffff);
    let path = client::temp_dir().join(format!("xprof_spill_{name}_{unique}.json"));
    std::fs::write(&path, bytes).ok()?;
    let mib = (bytes.len() as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0;
    let report = obj! {
        "status" => "SAVED_TO_FILE",
        "size_bytes" => bytes.len(),
        "size_mib" => mib,
        "file_path" => path.to_string_lossy().into_owned(),
        "message" => format!("Output payload ({} MB) exceeded 10 MB threshold. Saved to file to prevent terminal buffer overflow.", json::J::Float(mib).text()),
    };
    Some(report.dumps())
}

fn render(out: Out) -> Vec<u8> {
    let mut bytes = match out {
        Out::Text(text) => text.into_bytes(),
        Out::Bytes(data) => {
            let quote = if data.contains(&b'\'') && !data.contains(&b'"') { b'"' } else { b'\'' };
            let mut text = format!("b{}", quote as char);
            for byte in data {
                match byte {
                    b'\\' => text.push_str("\\\\"),
                    b'\n' => text.push_str("\\n"),
                    b'\r' => text.push_str("\\r"),
                    b'\t' => text.push_str("\\t"),
                    byte if byte == quote => text.extend(['\\', byte as char]),
                    0x20..=0x7e => text.push(byte as char),
                    other => text.push_str(&format!("\\x{other:02x}")),
                }
            }
            text.push(quote as char);
            text.into_bytes()
        }
        Out::Value(J::List(items)) => items.iter().map(|item| if let J::Str(text) = item { text.replace('\n', " ") } else { item.compact() }).collect::<Vec<_>>().join("\n").into_bytes(),
        Out::Value(J::Map(entries)) => {
            let visible: Vec<&(String, J)> = entries.iter().filter(|(key, _)| !key.starts_with('_')).collect();
            let width = visible.iter().map(|(key, _)| key.chars().count()).max().unwrap_or(0) + 1;
            let lines: Vec<String> =
                visible.iter().map(|(key, value)| format!("{:width$} {}", format!("{key}:"), if let J::Str(text) = value { text.replace('\n', " ") } else { value.compact() })).collect();
            if lines.is_empty() { "{}".into() } else { lines.join("\n").into_bytes() }
        }
        Out::Value(value) => value.compact().into_bytes(),
    };
    bytes.push(b'\n');
    bytes
}

fn emit(error: &Error) -> (i32, Vec<u8>, String) {
    if error.kind == Kind::Fire {
        return (2, Vec::new(), format!("ERROR: {}\n", error.message));
    }
    let (reason, code) = error.reason();
    let message = if code == 1 { format!("{}\nPlease report to {REPORT}", error.message) } else { error.message.clone() };
    let mut payload = obj! {"status" => "ERROR", "reason" => reason, "error" => message.clone()};
    let traceback = format!("Traceback (most recent call last):\n{}: {}", error.name(), error.message);
    if code == 1 {
        payload.set("traceback", traceback.clone());
    }
    let detail = if code == 1 { format!("\n{traceback}\n") } else { String::new() };
    (code, format!("{}\n", payload.dumps()).into_bytes(), format!("{reason}: {message}\n{detail}"))
}

/// Runs both at the same time when the client can do this.
pub fn both<A: Send, B: Send>(client: &dyn Client, first: impl FnOnce(&dyn Client) -> A, second: impl FnOnce(&dyn Client) -> B + Send) -> (A, B) {
    match client.shared() {
        Some(shared) => std::thread::scope(|scope| {
            let second = scope.spawn(move || second(shared));
            (first(shared), second.join().unwrap())
        }),
        None => (first(client), second(client)),
    }
}

pub fn invoke(client: &mut Local, command: &str, argv: &[String]) -> Result<(Out, Vec<String>), Error> {
    let Some((_, text, handler)) = COMMANDS.iter().find(|(name, _, _)| *name == command) else {
        return fail(
            Kind::Fire,
            format!(
                "Could not consume arg: {command}\nUsage: xprof <command> | <flags> [ARGS]...\n  available commands:    {} | server",
                COMMANDS.iter().map(|(name, _, _)| *name).collect::<Vec<_>>().join(" | ")
            ),
        );
    };
    let spec = spec(text);
    let (mut args, leftover) = bind(command, &spec, argv)?;
    client.logdir = wrap(&spec, &mut args)?.or(client.logdir.take());
    Ok((handler(client, &args)?, leftover))
}

pub fn execute(argv: &[String]) -> Option<(i32, Vec<u8>, String)> {
    let argv = &preprocess(argv);
    let command = argv.first()?;
    if command.starts_with('-') || command == "server" {
        return None;
    }
    if let Some((_, text, _)) = COMMANDS.iter().find(|(name, _, _)| name == command).filter(|_| argv.iter().skip(1).any(|argument| argument == "--help" || argument == "-h")) {
        return Some((0, format!("NAME\n    xprof {command}\n\nSYNOPSIS\n    xprof {command} {}\n", usage(&spec(text))).into_bytes(), String::new()));
    }
    // The process exits soon after. To drop the loaded planes only costs time.
    Some(match invoke(Box::leak(Box::new(Local { fused: command == "check_host_boundness", ..Local::default() })), command, &argv[1..]) {
        Ok((_, leftover)) if !leftover.is_empty() => emit(&Error::new(Kind::Fire, format!("Could not consume arg: {}", leftover[0]))),
        Ok((out, _)) => {
            let spilled = match &out {
                Out::Text(text) => spill(command, text.as_bytes()),
                Out::Bytes(data) => spill(command, data),
                Out::Value(_) => None,
            };
            (0, render(spilled.map_or(out, Out::Text)), String::new())
        }
        Err(error) => emit(&error),
    })
}

pub fn run(argv: &[String]) -> Option<i32> {
    let (code, out, err) = execute(argv)?;
    _ = std::io::stdout().write_all(&out);
    _ = std::io::stderr().write_all(err.as_bytes());
    Some(code)
}

pub fn read(path: &Path) -> Result<Vec<u8>, Error> {
    let big = std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() > (16 << 20));
    (if big { crate::read_file(path) } else { std::fs::read(path) }).map_err(|error| {
        Error::new(
            if error.kind() == std::io::ErrorKind::NotFound { Kind::FileNotFound } else { Kind::Os },
            format!("[Errno {}] {}: {}", error.raw_os_error().unwrap_or(0), error.to_string().split(" (os error").next().unwrap_or_default(), json::py_repr(&path.to_string_lossy())),
        )
    })
}
