use crate::server::run_tools::python_string_into;
use rayon::prelude::*;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::cell::Cell;
use std::fmt::{self, Write};

/// A list with more items writes parts of this many items in parallel.
const PARALLEL_ITEMS: usize = 4096;

#[derive(Clone, Debug, PartialEq, Default)]
pub enum J {
    #[default]
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    List(Vec<Self>),
    Map(Vec<(String, Self)>),
    /// The tuples, sets and dicts of the command line, which Python keeps apart from lists and objects.
    Tuple(Vec<Self>),
    Set(Vec<Self>),
    Dict(Vec<(Self, Self)>),
}

#[macro_export]
macro_rules! obj {
    ($($key:expr => $value:expr),* $(,)?) => { $crate::cli::json::J::Map(vec![$(($key.to_string(), $crate::cli::json::J::from($value))),*]) };
}

macro_rules! from {
    ($($kind:ty => $variant:ident as $target:ty),*) => { $(impl From<$kind> for J { fn from(value: $kind) -> J { J::$variant(value as $target) } })* };
}

from!(i32 => Int as i128, i64 => Int as i128, u64 => Int as i128, usize => Int as i128, i128 => Int as i128, f64 => Float as f64, bool => Bool as bool);

impl From<&str> for J {
    fn from(value: &str) -> Self {
        Self::Str(value.to_string())
    }
}

impl From<String> for J {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<&String> for J {
    fn from(value: &String) -> Self {
        Self::Str(value.clone())
    }
}

impl From<&Self> for J {
    fn from(value: &Self) -> Self {
        value.clone()
    }
}

impl<T: Into<Self>> From<Vec<T>> for J {
    fn from(items: Vec<T>) -> Self {
        Self::List(items.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Self>> From<Option<T>> for J {
    fn from(value: Option<T>) -> Self {
        value.map_or(Self::Null, Into::into)
    }
}

impl J {
    pub fn parse(text: &str) -> Option<Self> {
        if text.len() >= SPLIT
            && let Some(value) = split_parse(text, 0)
        {
            return Some(value);
        }
        whole(text)
    }

    /// The deepest nesting of objects, which is the nesting of messages that protobuf counts.
    pub fn object_depth(&self) -> usize {
        match self {
            Self::List(items) => items.iter().map(Self::object_depth).max().unwrap_or(0),
            Self::Map(entries) => {
                1 + entries
                    .iter()
                    .map(|(_, value)| value.object_depth())
                    .max()
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Map(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn at(&self, key: &str) -> &Self {
        self.get(key).unwrap_or(&Self::Null)
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn set(&mut self, key: &str, value: impl Into<Self>) {
        if let Self::Map(entries) = self {
            insert(entries, key.to_string(), value.into());
        }
    }

    pub fn entries(&self) -> &[(String, Self)] {
        match self {
            Self::Map(entries) => entries,
            _ => &[],
        }
    }

    pub fn items(&self) -> &[Self] {
        match self {
            Self::List(items) => items,
            _ => &[],
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    pub fn truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(flag) => *flag,
            Self::Int(number) => *number != 0,
            Self::Float(number) => *number != 0.0,
            Self::Str(text) => !text.is_empty(),
            Self::List(items) | Self::Tuple(items) | Self::Set(items) => !items.is_empty(),
            Self::Map(entries) => !entries.is_empty(),
            Self::Dict(entries) => !entries.is_empty(),
        }
    }

    pub fn float(&self) -> Option<f64> {
        match self {
            Self::Bool(flag) => Some(f64::from(u8::from(*flag))),
            Self::Int(number) => Some(*number as f64),
            Self::Float(number) => Some(*number),
            Self::Str(text) => py_float(text),
            _ => None,
        }
    }

    pub fn int(&self) -> Option<i128> {
        match self {
            Self::Bool(flag) => Some(i128::from(*flag)),
            Self::Int(number) => Some(*number),
            Self::Float(number) if number.is_finite() => Some(number.trunc() as i128),
            Self::Str(text) => py_int(text),
            _ => None,
        }
    }

    pub fn text(&self) -> String {
        match self {
            Self::Null => "None".into(),
            Self::Bool(flag) => if *flag { "True" } else { "False" }.into(),
            Self::Int(number) => number.to_string(),
            Self::Float(number) => crate::tools::table::repr(*number),
            Self::Str(text) => text.clone(),
            Self::List(items) => format!(
                "[{}]",
                items.iter().map(Self::repr).collect::<Vec<_>>().join(", ")
            ),
            Self::Map(entries) => format!(
                "{{{}}}",
                entries
                    .iter()
                    .map(|(key, value)| format!("{}: {}", py_repr(key), value.repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Tuple(items) if items.len() == 1 => format!("({},)", items[0].repr()),
            Self::Tuple(items) => format!(
                "({})",
                items.iter().map(Self::repr).collect::<Vec<_>>().join(", ")
            ),
            Self::Set(items) => format!(
                "{{{}}}",
                items.iter().map(Self::repr).collect::<Vec<_>>().join(", ")
            ),
            Self::Dict(entries) => format!(
                "{{{}}}",
                entries
                    .iter()
                    .map(|(key, value)| format!("{}: {}", key.repr(), value.repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    /// Python `==`, where `True == 1 == 1.0`.
    pub fn equals(&self, other: &Self) -> bool {
        let whole = |value: &Self| match value {
            Self::Bool(flag) => Some(i128::from(*flag)),
            Self::Int(number) => Some(*number),
            _ => None,
        };
        match (self, other) {
            (Self::Float(left), Self::Float(right)) => left == right,
            (Self::Float(float), number) | (number, Self::Float(float))
                if whole(number).is_some() =>
            {
                float.fract() == 0.0
                    && float.abs() < 1e38
                    && *float as i128 == whole(number).unwrap_or_default()
            }
            _ if whole(self).is_some() && whole(other).is_some() => whole(self) == whole(other),
            (Self::List(left), Self::List(right)) | (Self::Tuple(left), Self::Tuple(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(left, right)| left.equals(right))
            }
            _ => self == other,
        }
    }

    pub fn repr(&self) -> String {
        match self {
            Self::Str(text) => py_repr(text),
            other => other.text(),
        }
    }

    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, Some(0), true);
        out
    }

    pub fn compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, None, false);
        out
    }

    fn write(&self, out: &mut String, depth: Option<usize>, ascii: bool) {
        let open = |out: &mut String, depth: Option<usize>| {
            if let Some(depth) = depth {
                out.push('\n');
                out.extend(std::iter::repeat_n(' ', 2 * depth));
            }
        };
        let inner = depth.map(|depth| depth + 1);
        let separator = |index: usize| {
            if index == 0 {
                ""
            } else if depth.is_some() {
                ","
            } else {
                ", "
            }
        };
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
            Self::Int(number) => _ = write!(out, "{number}"),
            Self::Float(number) if number.is_nan() => out.push_str("NaN"),
            Self::Float(number) if number.is_infinite() => out.push_str(if *number > 0.0 {
                "Infinity"
            } else {
                "-Infinity"
            }),
            Self::Float(number) => out.push_str(&crate::tools::table::repr(*number)),
            Self::Str(text) => string(out, text, ascii),
            Self::List(items) | Self::Tuple(items) | Self::Set(items) if items.is_empty() => {
                out.push_str("[]");
            }
            Self::Map(entries) if entries.is_empty() => out.push_str("{}"),
            Self::Dict(entries) if entries.is_empty() => out.push_str("{}"),
            Self::List(items) | Self::Tuple(items) | Self::Set(items) => {
                out.push('[');
                let part = |out: &mut String, (start, items): (usize, &[Self])| {
                    for (index, item) in (start..).zip(items) {
                        out.push_str(separator(index));
                        open(out, inner);
                        item.write(out, inner, ascii);
                    }
                };
                if items.len() > PARALLEL_ITEMS {
                    let parts: Vec<String> = items
                        .par_chunks(PARALLEL_ITEMS)
                        .enumerate()
                        .map(|(chunk, items)| {
                            let mut text = String::new();
                            part(&mut text, (chunk * PARALLEL_ITEMS, items));
                            text
                        })
                        .collect();
                    parts.iter().for_each(|text| out.push_str(text));
                } else {
                    part(out, (0, items));
                }
                open(out, depth);
                out.push(']');
            }
            Self::Map(entries) => {
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    out.push_str(separator(index));
                    open(out, inner);
                    string(out, key, ascii);
                    out.push_str(": ");
                    value.write(out, inner, ascii);
                }
                open(out, depth);
                out.push('}');
            }
            Self::Dict(entries) => Self::Map(
                entries
                    .iter()
                    .map(|(key, value)| (key.text(), value.clone()))
                    .collect(),
            )
            .write(out, depth, ascii),
        }
    }
}

fn string(out: &mut String, text: &str, ascii: bool) {
    if ascii {
        python_string_into(out, text);
    } else {
        out.push_str(&serde_json::to_string(text).unwrap());
    }
}

pub fn py_float(text: &str) -> Option<f64> {
    py_number(text)
}

pub fn py_int(text: &str) -> Option<i128> {
    py_number(text)
}

fn py_number<T: std::str::FromStr>(text: &str) -> Option<T> {
    let trimmed = text.trim();
    trimmed
        .replace('_', "")
        .parse()
        .ok()
        .filter(|_| !trimmed.starts_with('_') && !trimmed.ends_with('_') && !trimmed.contains("__"))
}

/// The ranges of `str.isdigit` in Python 3.12 (Unicode 15.0).
#[rustfmt::skip]
const DIGITS: [(u32, u32); 83] = [
    (0x30, 0x39), (0xB2, 0xB3), (0xB9, 0xB9), (0x660, 0x669), (0x6F0, 0x6F9), (0x7C0, 0x7C9), (0x966, 0x96F), (0x9E6, 0x9EF), (0xA66, 0xA6F), (0xAE6, 0xAEF),
    (0xB66, 0xB6F), (0xBE6, 0xBEF), (0xC66, 0xC6F), (0xCE6, 0xCEF), (0xD66, 0xD6F), (0xDE6, 0xDEF), (0xE50, 0xE59), (0xED0, 0xED9), (0xF20, 0xF29),
    (0x1040, 0x1049), (0x1090, 0x1099), (0x1369, 0x1371), (0x17E0, 0x17E9), (0x1810, 0x1819), (0x1946, 0x194F), (0x19D0, 0x19DA), (0x1A80, 0x1A89),
    (0x1A90, 0x1A99), (0x1B50, 0x1B59), (0x1BB0, 0x1BB9), (0x1C40, 0x1C49), (0x1C50, 0x1C59), (0x2070, 0x2070), (0x2074, 0x2079), (0x2080, 0x2089),
    (0x2460, 0x2468), (0x2474, 0x247C), (0x2488, 0x2490), (0x24EA, 0x24EA), (0x24F5, 0x24FD), (0x24FF, 0x24FF), (0x2776, 0x277E), (0x2780, 0x2788),
    (0x278A, 0x2792), (0xA620, 0xA629), (0xA8D0, 0xA8D9), (0xA900, 0xA909), (0xA9D0, 0xA9D9), (0xA9F0, 0xA9F9), (0xAA50, 0xAA59), (0xABF0, 0xABF9),
    (0xFF10, 0xFF19), (0x104A0, 0x104A9), (0x10A40, 0x10A43), (0x10D30, 0x10D39), (0x10E60, 0x10E68), (0x11052, 0x1105A), (0x11066, 0x1106F),
    (0x110F0, 0x110F9), (0x11136, 0x1113F), (0x111D0, 0x111D9), (0x112F0, 0x112F9), (0x11450, 0x11459), (0x114D0, 0x114D9), (0x11650, 0x11659),
    (0x116C0, 0x116C9), (0x11730, 0x11739), (0x118E0, 0x118E9), (0x11950, 0x11959), (0x11C50, 0x11C59), (0x11D50, 0x11D59), (0x11DA0, 0x11DA9),
    (0x11F50, 0x11F59), (0x16A60, 0x16A69), (0x16AC0, 0x16AC9), (0x16B50, 0x16B59), (0x1D7CE, 0x1D7FF), (0x1E140, 0x1E149), (0x1E2F0, 0x1E2F9),
    (0x1E4F0, 0x1E4F9), (0x1E950, 0x1E959), (0x1F100, 0x1F10A), (0x1FBF0, 0x1FBF9)
];

/// `str.isdigit` of Python.
pub fn py_isdigit(text: &str) -> bool {
    !text.is_empty()
        && text.chars().all(|c| {
            DIGITS
                .get(DIGITS.partition_point(|&(_, high)| high < c as u32))
                .is_some_and(|&(low, _)| low <= c as u32)
        })
}

pub fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character == quote => out.extend(['\\', quote]),
            character if (character as u32) < 0x20 || character as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", character as u32));
            }
            character => out.push(character),
        }
    }
    out.push(quote);
    out
}

/// The code splits a container of this length or more at its top-level commas. It parses the parts in parallel.
const SPLIT: usize = 1 << 16;
/// The code splits containers only at this many levels or fewer. Each level scans its text again.
const SPLIT_LEVELS: usize = 8;

fn trim(text: &str) -> &str {
    text.trim_matches([' ', '\t', '\n', '\r'])
}

/// The index after the quote at the end of the string that starts at `start`.
fn string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut index = start + 1;
    loop {
        index += memchr::memchr2(b'"', b'\\', bytes.get(index..)?)?;
        if bytes[index] == b'"' {
            return Some(index + 1);
        }
        index += 2;
    }
}

fn backslashes_before(bytes: &[u8], index: usize) -> usize {
    bytes[..index]
        .iter()
        .rev()
        .take_while(|&&byte| byte == b'\\')
        .count()
}

/// The commas in `bytes[start..end]` at the lowest depth the range reaches, the depth change, and that lowest depth.
fn chunk_commas(
    bytes: &[u8],
    start: usize,
    end: usize,
    inside: bool,
) -> Option<(Vec<usize>, isize, isize)> {
    let mut index = if inside {
        string_end(bytes, start - backslashes_before(bytes, start) - 1)?
    } else {
        start
    };
    let (mut depth, mut lowest, mut commas) = (0isize, 0isize, Vec::new());
    while index < end {
        match bytes[index] {
            b'"' => index = string_end(bytes, index)? - 1,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth < lowest {
                    (lowest, commas) = (depth, Vec::new());
                }
            }
            b',' if depth == lowest => commas.push(index),
            _ => {}
        }
        index += 1;
    }
    Some((commas, depth, lowest))
}

/// Splits the inside of a container at its top-level commas. Threads scan the chunks in parallel. A chunk starts in a
/// string when the earlier chunks hold an odd number of quotes. A quote after an odd run of backslashes does not count.
fn parts(text: &str) -> Option<Vec<&str>> {
    let bytes = text.as_bytes();
    let size = (bytes.len() / rayon::current_num_threads()).max(SPLIT);
    let starts: Vec<usize> = (0..bytes.len()).step_by(size).collect();
    let quotes: Vec<usize> = starts
        .par_iter()
        .map(|&start| {
            memchr::memchr_iter(b'"', &bytes[start..(start + size).min(bytes.len())])
                .filter(|&at| backslashes_before(bytes, start + at).is_multiple_of(2))
                .count()
        })
        .collect();
    let mut inside = false;
    let starts: Vec<(usize, bool)> = starts
        .into_iter()
        .zip(quotes)
        .map(|(start, count)| {
            let was = inside;
            inside ^= count % 2 == 1;
            (start, was)
        })
        .collect();
    let scanned = starts
        .into_par_iter()
        .map(|(start, inside)| chunk_commas(bytes, start, (start + size).min(bytes.len()), inside))
        .collect::<Option<Vec<_>>>()?;
    let (mut depth, mut start, mut parts) = (0isize, 0, Vec::new());
    for (commas, change, lowest) in scanned {
        match depth + lowest {
            ..0 => return None,
            0 => {
                for comma in commas {
                    parts.push(&text[start..comma]);
                    start = comma + 1;
                }
            }
            _ => {}
        }
        depth += change;
    }
    parts.push(&text[start..]);
    (depth == 0).then_some(parts)
}

/// Python refuses JSON that nests deeper than about this.
const MAX_DEPTH: usize = 1000;

fn insert(entries: &mut Vec<(String, J)>, key: String, value: J) {
    match entries.iter_mut().find(|(name, _)| *name == key) {
        Some(slot) => slot.1 = value,
        None => entries.push((key, value)),
    }
}

/// Gives what `serde_json` gives. The depth limit is that of `J`, not that of `serde_json`.
fn whole(text: &str) -> Option<J> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    deserializer.disable_recursion_limit();
    let value = J::deserialize(&mut deserializer).ok()?;
    deserializer.end().ok().map(|()| value)
}

thread_local! {
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// One level of nesting while `J` deserializes. It fails at `MAX_DEPTH`.
struct Level;

impl Level {
    fn enter<E: serde::de::Error>() -> Result<Self, E> {
        DEPTH.with(|depth| {
            if depth.get() == MAX_DEPTH {
                return Err(E::custom("recursion limit exceeded"));
            }
            depth.set(depth.get() + 1);
            Ok(Self)
        })
    }
}

impl Drop for Level {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Gives what `serde_json` gives, or `None` when the text is not valid JSON.
fn split_parse(text: &str, level: usize) -> Option<J> {
    let text = trim(text);
    let (open, close) = (*text.as_bytes().first()?, *text.as_bytes().last()?);
    if text.len() < SPLIT
        || level == SPLIT_LEVELS
        || !matches!((open, close), (b'[', b']') | (b'{', b'}'))
    {
        return whole(text);
    }
    let parts = parts(&text[1..text.len() - 1])?;
    if let [only] = parts[..]
        && trim(only).is_empty()
    {
        return Some(if open == b'[' {
            J::List(Vec::new())
        } else {
            J::Map(Vec::new())
        });
    }
    if open == b'[' {
        return parts
            .into_par_iter()
            .map(|part| split_parse(part, level + 1))
            .collect::<Option<_>>()
            .map(J::List);
    }
    let members: Vec<(String, J)> = parts
        .into_par_iter()
        .map(|part| {
            let part = trim(part);
            let end = string_end(part.as_bytes(), 0).filter(|_| part.starts_with('"'))?;
            Some((
                serde_json::from_str(&part[..end]).ok()?,
                split_parse(trim(&part[end..]).strip_prefix(':')?, level + 1)?,
            ))
        })
        .collect::<Option<_>>()?;
    let mut entries = Vec::with_capacity(members.len());
    for (key, value) in members {
        insert(&mut entries, key, value);
    }
    Some(J::Map(entries))
}

impl<'de> Deserialize<'de> for J {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Walk;
        impl<'de> Visitor<'de> for Walk {
            type Value = J;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("JSON")
            }
            fn visit_unit<E>(self) -> Result<J, E> {
                Ok(J::Null)
            }
            fn visit_bool<E>(self, value: bool) -> Result<J, E> {
                Ok(J::Bool(value))
            }
            fn visit_i64<E>(self, value: i64) -> Result<J, E> {
                Ok(J::Int(value.into()))
            }
            fn visit_u64<E>(self, value: u64) -> Result<J, E> {
                Ok(J::Int(value.into()))
            }
            fn visit_f64<E>(self, value: f64) -> Result<J, E> {
                Ok(J::Float(value))
            }
            fn visit_str<E>(self, value: &str) -> Result<J, E> {
                Ok(J::Str(value.to_string()))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<J, A::Error> {
                let _level = Level::enter()?;
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(J::List(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<J, A::Error> {
                let _level = Level::enter()?;
                let mut entries = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, J>()? {
                    insert(&mut entries, key, value);
                }
                Ok(J::Map(entries))
            }
        }
        deserializer.deserialize_any(Walk)
    }
}
