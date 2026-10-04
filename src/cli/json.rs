use crate::run_tools::python_string_into;
use rayon::prelude::*;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::fmt::{self, Write};

#[derive(Clone, Debug, PartialEq, Default)]
pub enum J {
    #[default]
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    List(Vec<J>),
    Map(Vec<(String, J)>),
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
    fn from(value: &str) -> J {
        J::Str(value.to_string())
    }
}

impl From<String> for J {
    fn from(value: String) -> J {
        J::Str(value)
    }
}

impl From<&String> for J {
    fn from(value: &String) -> J {
        J::Str(value.clone())
    }
}

impl From<&J> for J {
    fn from(value: &J) -> J {
        value.clone()
    }
}

impl<T: Into<J>> From<Vec<T>> for J {
    fn from(items: Vec<T>) -> J {
        J::List(items.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<J>> From<Option<T>> for J {
    fn from(value: Option<T>) -> J {
        value.map_or(J::Null, Into::into)
    }
}

impl J {
    pub fn parse(text: &str) -> Option<J> {
        if text.len() >= SPLIT
            && let Some(value) = split_parse(text)
        {
            return Some(value);
        }
        serde_json::from_str(text).ok()
    }

    pub fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Map(entries) => entries.iter().find(|(name, _)| name == key).map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn at(&self, key: &str) -> &J {
        self.get(key).unwrap_or(&J::Null)
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn set(&mut self, key: &str, value: impl Into<J>) {
        if let J::Map(entries) = self {
            let value = value.into();
            match entries.iter_mut().find(|(name, _)| name == key) {
                Some(slot) => slot.1 = value,
                None => entries.push((key.to_string(), value)),
            }
        }
    }

    pub fn entries(&self) -> &[(String, J)] {
        match self {
            J::Map(entries) => entries,
            _ => &[],
        }
    }

    pub fn items(&self) -> &[J] {
        match self {
            J::List(items) => items,
            _ => &[],
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            J::Str(text) => Some(text),
            _ => None,
        }
    }

    pub fn truthy(&self) -> bool {
        match self {
            J::Null => false,
            J::Bool(flag) => *flag,
            J::Int(number) => *number != 0,
            J::Float(number) => *number != 0.0,
            J::Str(text) => !text.is_empty(),
            J::List(items) => !items.is_empty(),
            J::Map(entries) => !entries.is_empty(),
        }
    }

    pub fn float(&self) -> Option<f64> {
        match self {
            J::Bool(flag) => Some(f64::from(u8::from(*flag))),
            J::Int(number) => Some(*number as f64),
            J::Float(number) => Some(*number),
            J::Str(text) => py_float(text),
            _ => None,
        }
    }

    pub fn int(&self) -> Option<i128> {
        match self {
            J::Bool(flag) => Some(i128::from(*flag)),
            J::Int(number) => Some(*number),
            J::Float(number) if number.is_finite() => Some(number.trunc() as i128),
            J::Str(text) => py_int(text),
            _ => None,
        }
    }

    pub fn text(&self) -> String {
        match self {
            J::Null => "None".into(),
            J::Bool(flag) => if *flag { "True" } else { "False" }.into(),
            J::Int(number) => number.to_string(),
            J::Float(number) => crate::table::repr(*number),
            J::Str(text) => text.clone(),
            J::List(items) => format!("[{}]", items.iter().map(J::repr).collect::<Vec<_>>().join(", ")),
            J::Map(entries) => format!("{{{}}}", entries.iter().map(|(key, value)| format!("{}: {}", py_repr(key), value.repr())).collect::<Vec<_>>().join(", ")),
        }
    }

    pub fn repr(&self) -> String {
        match self {
            J::Str(text) => py_repr(text),
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
        match self {
            J::Null => out.push_str("null"),
            J::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
            J::Int(number) => _ = write!(out, "{number}"),
            J::Float(number) if number.is_nan() => out.push_str("NaN"),
            J::Float(number) if number.is_infinite() => out.push_str(if *number > 0.0 { "Infinity" } else { "-Infinity" }),
            J::Float(number) => out.push_str(&crate::table::repr(*number)),
            J::Str(text) => string(out, text, ascii),
            J::List(items) if items.is_empty() => out.push_str("[]"),
            J::Map(entries) if entries.is_empty() => out.push_str("{}"),
            J::List(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    out.push_str(if index == 0 {
                        ""
                    } else if depth.is_some() {
                        ","
                    } else {
                        ", "
                    });
                    open(out, inner);
                    item.write(out, inner, ascii);
                }
                open(out, depth);
                out.push(']');
            }
            J::Map(entries) => {
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    out.push_str(if index == 0 {
                        ""
                    } else if depth.is_some() {
                        ","
                    } else {
                        ", "
                    });
                    open(out, inner);
                    string(out, key, ascii);
                    out.push_str(": ");
                    value.write(out, inner, ascii);
                }
                open(out, depth);
                out.push('}');
            }
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
    let trimmed = text.trim();
    let digits = trimmed.replace('_', "");
    digits.parse().ok().filter(|_| !trimmed.starts_with('_') && !trimmed.ends_with('_') && !trimmed.contains("__"))
}

pub fn py_int(text: &str) -> Option<i128> {
    let trimmed = text.trim();
    trimmed.replace('_', "").parse().ok().filter(|_| !trimmed.starts_with('_') && !trimmed.ends_with('_') && !trimmed.contains("__"))
}

pub fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character == quote => out.extend(['\\', quote]),
            character if (character as u32) < 0x20 || character as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", character as u32)),
            character => out.push(character),
        }
    }
    out.push(quote);
    out
}

/// The code splits a container of this length or more at its top-level commas. It parses the parts in parallel.
const SPLIT: usize = 1 << 16;

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
    bytes[..index].iter().rev().take_while(|&&byte| byte == b'\\').count()
}

/// The commas in `bytes[start..end]` at the lowest depth the range reaches, the depth change, and that lowest depth.
fn chunk_commas(bytes: &[u8], start: usize, end: usize, inside: bool) -> Option<(Vec<usize>, isize, isize)> {
    let mut index = start;
    if inside {
        index = string_end(bytes, start - backslashes_before(bytes, start) - 1)?;
    }
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
        .map(|&start| memchr::memchr_iter(b'"', &bytes[start..(start + size).min(bytes.len())]).filter(|&at| backslashes_before(bytes, start + at).is_multiple_of(2)).count())
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
    let scanned = starts.into_par_iter().map(|(start, inside)| chunk_commas(bytes, start, (start + size).min(bytes.len()), inside)).collect::<Option<Vec<_>>>()?;
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

/// Gives what `serde_json` gives, or `None` when the text is not valid JSON.
fn split_parse(text: &str) -> Option<J> {
    let text = trim(text);
    let (open, close) = (*text.as_bytes().first()?, *text.as_bytes().last()?);
    if text.len() < SPLIT || !matches!((open, close), (b'[', b']') | (b'{', b'}')) {
        return serde_json::from_str(text).ok();
    }
    let parts = parts(&text[1..text.len() - 1])?;
    if let [only] = parts[..]
        && trim(only).is_empty()
    {
        return Some(if open == b'[' { J::List(Vec::new()) } else { J::Map(Vec::new()) });
    }
    if open == b'[' {
        return parts.into_par_iter().map(split_parse).collect::<Option<_>>().map(J::List);
    }
    let members: Vec<(String, J)> = parts
        .into_par_iter()
        .map(|part| {
            let part = trim(part);
            let end = string_end(part.as_bytes(), 0).filter(|_| part.starts_with('"'))?;
            Some((serde_json::from_str(&part[..end]).ok()?, split_parse(trim(&part[end..]).strip_prefix(':')?)?))
        })
        .collect::<Option<_>>()?;
    let mut entries: Vec<(String, J)> = Vec::with_capacity(members.len());
    for (key, value) in members {
        match entries.iter_mut().find(|(name, _)| *name == key) {
            Some(slot) => slot.1 = value,
            None => entries.push((key, value)),
        }
    }
    Some(J::Map(entries))
}

impl<'de> Deserialize<'de> for J {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<J, D::Error> {
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
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(J::List(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<J, A::Error> {
                let mut entries: Vec<(String, J)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, J>()? {
                    match entries.iter_mut().find(|(name, _)| *name == key) {
                        Some(slot) => slot.1 = value,
                        None => entries.push((key, value)),
                    }
                }
                Ok(J::Map(entries))
            }
        }
        deserializer.deserialize_any(Walk)
    }
}
