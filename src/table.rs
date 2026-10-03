use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::LazyLock;

const MIN_EXPONENT: i32 = -4;
const MAX_EXPONENT: i32 = 15;
const CACHED_POWERS: i32 = 79;
const ROWS_PER_CHUNK: usize = 256;

pub enum Cell {
    Number(f64),
    Text(String),
    Boolean(bool),
}

#[derive(Default)]
pub struct Table {
    pub columns: Vec<(String, &'static str, String, Option<&'static str>)>,
    pub props: BTreeMap<String, String>,
    pub rows: Vec<Vec<Cell>>,
}

static POWERS: LazyLock<Vec<(u64, i32)>> = LazyLock::new(|| {
    (0..CACHED_POWERS)
        .map(|index| {
            let (mut mantissa, mut exponent, k) = (1u128 << 123, -123, -300 + 8 * index);
            for _ in 0..k.unsigned_abs() {
                if k > 0 {
                    mantissa *= 10;
                } else {
                    let shift = mantissa.leading_zeros();
                    mantissa = (mantissa << shift) / 10;
                    exponent -= shift as i32;
                }
                let excess = 4 - mantissa.leading_zeros() as i32;
                (mantissa, exponent) = if excess > 0 { (mantissa >> excess, exponent + excess) } else { (mantissa << -excess, exponent + excess) };
            }
            (((mantissa >> 60) + (mantissa >> 59 & 1)) as u64, exponent + 60)
        })
        .collect()
});

fn multiply(x: (u64, i32), y: (u64, i32)) -> (u64, i32) {
    let product = u128::from(x.0) * u128::from(y.0) + (1u128 << 63);
    ((product >> 64) as u64, x.1 + y.1 + 64)
}

fn normalize(mut x: (u64, i32)) -> (u64, i32) {
    let shift = x.0.leading_zeros();
    x.0 <<= shift;
    x.1 -= shift as i32;
    x
}

fn round_digits(digits: &mut [u8], dist: u64, delta: u64, mut rest: u64, ten_k: u64) {
    let last = digits.len() - 1;
    while rest < dist && delta - rest >= ten_k && (rest + ten_k < dist || dist - rest > rest + ten_k - dist) {
        digits[last] -= 1;
        rest += ten_k;
    }
}

fn grisu2(value: f64) -> (Vec<u8>, i32) {
    let bits = value.to_bits();
    let (exponent, fraction) = (bits >> 52, bits & ((1 << 52) - 1));
    let v = if exponent == 0 { (fraction, -1074) } else { (fraction + (1 << 52), exponent as i32 - 1075) };
    let plus = normalize((2 * v.0 + 1, v.1 - 1));
    let minus_raw = if fraction == 0 && exponent > 1 { (4 * v.0 - 1, v.1 - 2) } else { (2 * v.0 - 1, v.1 - 1) };
    let minus = (minus_raw.0 << (minus_raw.1 - plus.1), plus.1);
    let v = normalize(v);
    let f = -60 - plus.1 - 1;
    let k = (f * 78913) / (1 << 18) + i32::from(f > 0);
    let index = (300 + k + 7) / 8;
    let ((cached_f, cached_e), cached_k) = (POWERS[index as usize], -300 + 8 * index);
    let w = multiply(v, (cached_f, cached_e));
    let upper = (multiply(plus, (cached_f, cached_e)).0 - 1, plus.1 + cached_e + 64);
    let lower = (multiply(minus, (cached_f, cached_e)).0 + 1, upper.1);
    let mut decimal_exponent = -cached_k;
    let (mut delta, mut dist) = (upper.0 - lower.0, upper.0 - w.0);
    let shift = -upper.1;
    let one = 1u64 << shift;
    let mut high = (upper.0 >> shift) as u32;
    let mut low = upper.0 & (one - 1);
    let mut digits = Vec::with_capacity(20);
    let mut power = 10u32.pow(high.ilog10());
    let mut remaining = high.ilog10() as i32 + 1;
    while remaining > 0 {
        digits.push(b'0' + (high / power) as u8);
        high %= power;
        remaining -= 1;
        let rest = (u64::from(high) << shift) + low;
        if rest <= delta {
            decimal_exponent += remaining;
            round_digits(&mut digits, dist, delta, rest, u64::from(power) << shift);
            return (digits, decimal_exponent);
        }
        power /= 10;
    }
    let mut count = 0;
    loop {
        low *= 10;
        digits.push(b'0' + (low >> shift) as u8);
        low &= one - 1;
        count += 1;
        delta *= 10;
        dist *= 10;
        if low <= delta {
            break;
        }
    }
    decimal_exponent -= count;
    round_digits(&mut digits, dist, delta, low, one);
    (digits, decimal_exponent)
}

fn decimal(out: &mut String, digits: &str, point: i32, max_exponent: i32) {
    let length = digits.len() as i32;
    if length <= point && point <= max_exponent {
        out.push_str(digits);
        out.extend(std::iter::repeat_n('0', (point - length) as usize));
        out.push_str(".0");
    } else if 0 < point && point <= max_exponent {
        out.push_str(&digits[..point as usize]);
        out.push('.');
        out.push_str(&digits[point as usize..]);
    } else if MIN_EXPONENT < point && point <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-point) as usize));
        out.push_str(digits);
    } else {
        out.push_str(&digits[..1]);
        if length > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        write!(out, "e{}{:02}", if point - 1 < 0 { '-' } else { '+' }, (point - 1).abs()).unwrap();
    }
}

pub fn number(out: &mut String, value: f64) {
    if !value.is_finite() {
        out.push_str("null");
        return;
    }
    if value.is_sign_negative() {
        out.push('-');
    }
    if value == 0.0 {
        out.push_str("0.0");
        return;
    }
    let (digits, decimal_exponent) = grisu2(value.abs());
    decimal(out, std::str::from_utf8(&digits).unwrap(), digits.len() as i32 + decimal_exponent, MAX_EXPONENT);
}

pub fn string(out: &mut String, text: &str) {
    if text.bytes().any(|byte| byte < 0x20 || byte == b'"' || byte == b'\\') {
        out.push_str(&serde_json::to_string(text).unwrap());
    } else {
        out.push('"');
        out.push_str(text);
        out.push('"');
    }
}

impl Table {
    pub fn new(columns: &[(&str, &'static str, &str)]) -> Table {
        Table { columns: columns.iter().map(|&(id, kind, label)| (id.into(), kind, label.into(), None)).collect(), ..Default::default() }
    }

    pub fn column(&mut self, id: &str, kind: &'static str, label: &str) {
        self.columns.push((id.into(), kind, label.into(), None));
    }

    pub fn prop(&mut self, key: &str, value: impl Into<String>) {
        self.props.insert(key.into(), value.into());
    }

    pub fn row(&mut self) -> &mut Vec<Cell> {
        self.rows.push(Vec::new());
        self.rows.last_mut().unwrap()
    }

    pub fn json(&self) -> String {
        let mut out = String::from("{\"cols\":[");
        for (index, (id, kind, label, role)) in self.columns.iter().enumerate() {
            out.push_str(if index > 0 { ",{\"id\":" } else { "{\"id\":" });
            string(&mut out, id);
            out.push_str(",\"label\":");
            string(&mut out, label);
            if let Some(role) = role {
                out.push_str(",\"p\":{\"role\":");
                string(&mut out, role);
                out.push('}');
            }
            out.push_str(",\"type\":");
            string(&mut out, kind);
            out.push('}');
        }
        out.push(']');
        if !self.props.is_empty() {
            out.push_str(",\"p\":{");
            for (index, (key, value)) in self.props.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                string(&mut out, key);
                out.push(':');
                string(&mut out, value);
            }
            out.push('}');
        }
        let rows: Vec<String> = self
            .rows
            .par_chunks(ROWS_PER_CHUNK)
            .map(|rows| {
                let mut out = String::with_capacity(rows.len() * 512);
                for row in rows {
                    out.push_str(if out.is_empty() { "{\"c\":[" } else { ",{\"c\":[" });
                    for (position, cell) in row.iter().enumerate() {
                        out.push_str(if position > 0 { ",{\"v\":" } else { "{\"v\":" });
                        match cell {
                            Cell::Number(value) => number(&mut out, *value),
                            Cell::Text(text) => string(&mut out, text),
                            Cell::Boolean(value) => out.push_str(if *value { "true" } else { "false" }),
                        }
                        out.push('}');
                    }
                    out.push_str("]}");
                }
                out
            })
            .collect();
        out.push_str(",\"rows\":[");
        out.push_str(&rows.join(","));
        out.push_str("]}");
        out
    }
}

pub fn repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    let mut out = String::from(if value.is_sign_negative() { "-" } else { "" });
    if value.is_infinite() || value == 0.0 {
        out.push_str(if value == 0.0 { "0.0" } else { "inf" });
        return out;
    }
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    decimal(&mut out, &mantissa.replace('.', ""), exponent.parse::<i32>().unwrap() + 1, MAX_EXPONENT + 1);
    out
}
