use crate::run_tools::python_string;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::fmt;

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
            J::Int(number) => out.push_str(&number.to_string()),
            J::Float(number) if number.is_nan() => out.push_str("NaN"),
            J::Float(number) if number.is_infinite() => out.push_str(if *number > 0.0 { "Infinity" } else { "-Infinity" }),
            J::Float(number) => out.push_str(&crate::table::repr(*number)),
            J::Str(text) if ascii => out.push_str(&python_string(text)),
            J::Str(text) => out.push_str(&serde_json::to_string(text).unwrap()),
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
                    J::Str(key.clone()).write(out, inner, ascii);
                    out.push_str(": ");
                    value.write(out, inner, ascii);
                }
                open(out, depth);
                out.push('}');
            }
        }
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
                let mut object = J::Map(Vec::new());
                while let Some((key, value)) = map.next_entry::<String, J>()? {
                    object.set(&key, value);
                }
                Ok(object)
            }
        }
        deserializer.deserialize_any(Walk)
    }
}
