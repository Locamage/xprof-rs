use crate::hlo::general;
use crate::op_profile::proto_double;
use crate::xplane::{Field, fields, varint};
use prost::Message;
use prost_reflect::{DescriptorPool, Kind, MessageDescriptor};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::LazyLock;

const DESCRIPTORS: &[u8] = include_bytes!("hlo_descriptors.pb");

static POOL: LazyLock<DescriptorPool> = LazyLock::new(|| {
    let mut pool = DescriptorPool::global();
    pool.decode_file_descriptor_set(DESCRIPTORS).unwrap();
    pool
});

struct Spec {
    number: u32,
    name: String,
    json_name: String,
    kind: Kind,
    map: bool,
    list: bool,
    presence: bool,
    oneof: Option<String>,
}

static MESSAGES: LazyLock<FxHashMap<String, (bool, Vec<Spec>)>> = LazyLock::new(|| {
    let specs = |message: &MessageDescriptor| {
        let mut specs: Vec<Spec> = message
            .fields()
            .map(|field| Spec {
                number: field.number(),
                name: field.name().to_string(),
                json_name: field.json_name().to_string(),
                kind: field.kind(),
                map: field.is_map(),
                list: field.is_list(),
                presence: field.supports_presence(),
                oneof: field.containing_oneof().map(|oneof| oneof.full_name().to_string()),
            })
            .collect();
        specs.sort_by_key(|spec| spec.number);
        specs
    };
    POOL.all_messages().map(|message| (message.full_name().to_string(), (message.is_map_entry(), specs(&message)))).collect()
});

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Text,
    Json,
    JsonAlways,
}

pub fn enum_name(name: &str, number: i32) -> String {
    POOL.get_enum_by_name(name).and_then(|kind| kind.get_value(number)).map_or_else(String::new, |value| value.name().to_string())
}

pub fn c_escape(out: &mut String, bytes: &[u8]) {
    for &byte in bytes {
        match byte {
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            b'"' => out.push_str("\\\""),
            b'\'' => out.push_str("\\'"),
            b'\\' => out.push_str("\\\\"),
            0x20..=0x7e => out.push(byte as char),
            byte => write!(out, "\\{:03o}", byte).unwrap(),
        }
    }
}

pub fn json_string(out: &mut String, text: &str) {
    out.push('"');
    let mut rest = text;
    loop {
        let plain = rest.bytes().position(|byte| !(0x20..0x7f).contains(&byte) || matches!(byte, b'"' | b'\\' | b'<' | b'>')).unwrap_or(rest.len());
        out.push_str(&rest[..plain]);
        rest = &rest[plain..];
        let Some(character) = rest.chars().next() else { break };
        rest = &rest[character.len_utf8()..];
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0'..='\u{1f}'
            | '<'
            | '>'
            | '\u{7f}'..='\u{9f}'
            | '\u{ad}'
            | '\u{600}'..='\u{603}'
            | '\u{6dd}'
            | '\u{70f}'
            | '\u{17b4}'
            | '\u{17b5}'
            | '\u{200b}'..='\u{200f}'
            | '\u{2028}'..='\u{202e}'
            | '\u{2060}'..='\u{2064}'
            | '\u{206a}'..='\u{206f}'
            | '\u{feff}'
            | '\u{fff9}'..='\u{fffb}'
            | '\u{1d173}'..='\u{1d17a}'
            | '\u{e0001}'
            | '\u{e0020}'..='\u{e007f}' => {
                let mut units = [0u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    write!(out, "\\u{:04x}", unit).unwrap();
                }
            }
            character => out.push(character),
        }
    }
    out.push('"');
}

fn decode(kind: &Kind, raw: u64) -> i128 {
    match kind {
        Kind::Int32 | Kind::Sfixed32 | Kind::Enum(_) => i128::from(raw as i32),
        Kind::Sint32 => i128::from(((raw as u32 >> 1) as i32) ^ -((raw as u32 & 1) as i32)),
        Kind::Sint64 => i128::from(((raw >> 1) as i64) ^ -((raw & 1) as i64)),
        Kind::Int64 | Kind::Sfixed64 => i128::from(raw as i64),
        Kind::Uint32 | Kind::Fixed32 => i128::from(raw as u32),
        Kind::Bool => i128::from(raw != 0),
        _ => i128::from(raw),
    }
}

fn floating(value: f64, digits: [usize; 2], round_trips: impl Fn(&str) -> bool) -> String {
    match general(value, digits[0]) {
        _ if value.is_nan() => "nan".into(),
        short if round_trips(&short) => short,
        _ => general(value, digits[1]),
    }
}

fn message_name(kind: &Kind) -> &str {
    kind.as_message().map_or("", MessageDescriptor::full_name)
}

fn last(entry: &[u8], number: u32) -> Option<Field<'_>> {
    fields(entry).filter(|(found, _)| *found == number).last().map(|(_, value)| value)
}

fn emit(out: &mut String, depth: usize, spec: &Spec, value: &Field, format: Format) {
    let kind = &spec.kind;
    let json = format != Format::Text;
    if !json {
        out.extend(std::iter::repeat_n("  ", depth));
        out.push_str(&spec.name);
    }
    match (value, kind) {
        (Field::Bytes(_, bytes), Kind::Message(_)) if json => print(out, depth, message_name(kind), bytes, format),
        (Field::Bytes(_, bytes), Kind::Message(_)) => {
            out.push_str(" {\n");
            print(out, depth + 1, message_name(kind), bytes, format);
            out.extend(std::iter::repeat_n("  ", depth));
            out.push('}');
        }
        (Field::Bytes(_, bytes), _) if json => json_string(out, &crate::xplane::lossy(bytes)),
        (Field::Bytes(_, bytes), _) => {
            out.push_str(": \"");
            c_escape(out, bytes);
            out.push('"');
        }
        (&Field::Num(raw), kind) => {
            if !json {
                out.push_str(": ");
            }
            match kind {
                Kind::Double if json => proto_double(out, f64::from_bits(raw)),
                Kind::Float if json => proto_double(out, f64::from(f32::from_bits(raw as u32))),
                Kind::Double => out.push_str(&floating(f64::from_bits(raw), [15, 17], |text| text.parse() == Ok(f64::from_bits(raw)))),
                Kind::Float => out.push_str(&floating(f64::from(f32::from_bits(raw as u32)), [6, 9], |text| text.parse() == Ok(f32::from_bits(raw as u32)))),
                Kind::Bool => write!(out, "{}", raw != 0).unwrap(),
                Kind::Enum(values) => match values.get_value(raw as i32) {
                    Some(value) if json => json_string(out, value.name()),
                    Some(value) => out.push_str(value.name()),
                    None => write!(out, "{}", raw as i32).unwrap(),
                },
                Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 | Kind::Uint64 | Kind::Fixed64 if json => write!(out, "\"{}\"", decode(kind, raw)).unwrap(),
                kind => write!(out, "{}", decode(kind, raw)).unwrap(),
            }
        }
    }
    if !json {
        out.push('\n');
    }
}

pub fn print_hlo(out: &mut String, buf: &[u8]) {
    print(out, 0, "xla.HloProto", buf, Format::Text);
}

pub fn json(message: &str, value: &impl Message, always_print_defaults: bool) -> String {
    let mut out = String::new();
    print(&mut out, 0, message, &value.encode_to_vec(), if always_print_defaults { Format::JsonAlways } else { Format::Json });
    out
}

fn print(out: &mut String, depth: usize, message: &str, buf: &[u8], format: Format) {
    let (map_entry, specs) = &MESSAGES[message];
    let json = format != Format::Text;
    let always = |spec: &Spec| *map_entry || (format == Format::JsonAlways && !spec.presence);
    let mut found: Vec<Option<Vec<Field>>> = specs.iter().map(|spec| always(spec).then(Vec::new)).collect();
    for (number, value) in fields(buf) {
        let Ok(index) = specs.binary_search_by_key(&number, |spec| spec.number) else { continue };
        let spec = &specs[index];
        let kind = &spec.kind;
        if let Some(oneof) = &spec.oneof {
            for (_, slot) in specs.iter().zip(&mut found).filter(|(other, _)| other.number != number && other.oneof.as_ref() == Some(oneof)) {
                *slot = None;
            }
        }
        let items = found[index].get_or_insert_with(Vec::new);
        match value {
            Field::Bytes(_, bytes) if !matches!(kind, Kind::String | Kind::Bytes | Kind::Message(_)) => {
                let width = match kind {
                    Kind::Double | Kind::Fixed64 | Kind::Sfixed64 => 8,
                    Kind::Float | Kind::Fixed32 | Kind::Sfixed32 => 4,
                    _ => 0,
                };
                let mut pos = 0;
                while pos < bytes.len() {
                    let value = match width {
                        0 => varint(bytes, &mut pos),
                        _ => bytes.get(pos..pos + width).map(|chunk| {
                            pos += width;
                            chunk.iter().rev().fold(0, |value, &byte| value << 8 | u64::from(byte))
                        }),
                    };
                    let Some(value) = value else { break };
                    items.push(Field::Num(value));
                }
            }
            value => items.push(value),
        }
    }
    if json {
        out.push('{');
    }
    let start = out.len();
    for (spec, items) in specs.iter().zip(&found).filter_map(|(spec, items)| Some((spec, items.as_ref()?))) {
        let kind = &spec.kind;
        let key = |out: &mut String| {
            if json {
                write!(out, "{}\"{}\":", if out.len() > start { "," } else { "" }, spec.json_name).unwrap();
            }
        };
        if spec.map {
            let [Spec { kind: key_kind, .. }, value_spec @ Spec { kind: value_kind, .. }] = &MESSAGES[message_name(kind)].1[..] else { continue };
            let mut entries: BTreeMap<(i128, &[u8]), &[u8]> = BTreeMap::new();
            for item in items {
                let Field::Bytes(_, bytes) = item else { continue };
                let entry_key = match last(bytes, 1) {
                    Some(Field::Bytes(_, text)) => (0, text),
                    Some(Field::Num(raw)) => (decode(key_kind, raw), &[][..]),
                    None => (0, &[][..]),
                };
                entries.insert(entry_key, bytes);
            }
            if !json {
                entries.values().for_each(|bytes| emit(out, depth, spec, &Field::Bytes(0, bytes), format));
            } else if !entries.is_empty() || always(spec) {
                key(out);
                out.push('{');
                for (index, ((number, text), bytes)) in entries.iter().enumerate() {
                    let name = match key_kind {
                        Kind::String => crate::xplane::lossy(text).into_owned(),
                        Kind::Bool => (*number != 0).to_string(),
                        _ => number.to_string(),
                    };
                    out.push_str(if index > 0 { "," } else { "" });
                    json_string(out, &name);
                    out.push(':');
                    let empty = if matches!(value_kind, Kind::Message(_) | Kind::String | Kind::Bytes) { Field::Bytes(0, &[]) } else { Field::Num(0) };
                    emit(out, depth, value_spec, &last(bytes, 2).unwrap_or(empty), format);
                }
                out.push('}');
            }
        } else if spec.list && !(json && items.is_empty() && !always(spec)) {
            key(out);
            out.push_str(if json { "[" } else { "" });
            let texts: Vec<String> = match kind.as_message() {
                None => Vec::new(),
                Some(_) => items
                    .par_iter()
                    .map(|item| {
                        let mut text = String::new();
                        emit(&mut text, depth, spec, item, format);
                        text
                    })
                    .collect(),
            };
            for (index, item) in items.iter().enumerate() {
                out.push_str(if json && index > 0 { "," } else { "" });
                match texts.get(index) {
                    Some(text) => out.push_str(text),
                    None => emit(out, depth, spec, item, format),
                }
            }
            out.push_str(if json { "]" } else { "" });
        } else if kind.as_message().is_some() {
            let merged = match items.as_slice() {
                [Field::Bytes(_, bytes)] => Cow::Borrowed(*bytes),
                _ => Cow::Owned(items.iter().flat_map(|item| if let Field::Bytes(_, bytes) = item { bytes.to_vec() } else { Vec::new() }).collect()),
            };
            key(out);
            emit(out, depth, spec, &Field::Bytes(0, merged.as_ref()), format);
        } else {
            let value = match items.last() {
                Some(Field::Bytes(_, bytes)) => Field::Bytes(0, bytes),
                Some(&Field::Num(raw)) => Field::Num(raw),
                None if matches!(kind, Kind::String | Kind::Bytes) => Field::Bytes(0, &[]),
                None => Field::Num(0),
            };
            let default = match value {
                Field::Bytes(_, bytes) => bytes.is_empty(),
                Field::Num(raw) if json => raw == 0,
                Field::Num(raw) => decode(kind, raw) == 0,
            };
            if always(spec) || spec.presence || !default {
                key(out);
                emit(out, depth, spec, &value, format);
            }
        }
    }
    if json {
        out.push('}');
    }
}

pub fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    json_string(&mut out, text);
    out
}
