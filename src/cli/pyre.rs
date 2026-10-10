//! Python `re` patterns. The parse follows `re._parser` of Python 3.12, so a bad pattern gives the message of Python. The search uses `fancy-regex`.

use super::json::py_repr;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::LazyLock;

const MAXREPEAT: u128 = u32::MAX as u128;
const MAXGROUPS: u128 = 1073741823;
const MAXCODE: u128 = u32::MAX as u128;
const MAXWIDTH: u128 = 1 << 64;

const TEMPLATE: u32 = 1;
const IGNORECASE: u32 = 2;
const LOCALE: u32 = 4;
const MULTILINE: u32 = 8;
const DOTALL: u32 = 16;
const UNICODE: u32 = 32;
const VERBOSE: u32 = 64;
const DEBUG: u32 = 128;
const ASCII: u32 = 256;
const TYPE_FLAGS: u32 = ASCII | LOCALE | UNICODE;
const GLOBAL_FLAGS: u32 = DEBUG | TEMPLATE;
/// Not a Python flag. It tells `emit` to count the repeats in the `fancy-regex` machine, because the `regex` machine of a large count is too large.
const COUNTED: u32 = 1 << 16;

/// The groups of lower case characters with the same upper case, from `re._casefix` of Python 3.12.
const CASE_GROUPS: [&[u32]; 24] = [
    &[0x69, 0x131],
    &[0x73, 0x17f],
    &[0xb5, 0x3bc],
    &[0x345, 0x3b9, 0x1fbe],
    &[0x390, 0x1fd3],
    &[0x3b0, 0x1fe3],
    &[0x3b2, 0x3d0],
    &[0x3b5, 0x3f5],
    &[0x3b8, 0x3d1],
    &[0x3ba, 0x3f0],
    &[0x3c0, 0x3d6],
    &[0x3c1, 0x3f1],
    &[0x3c2, 0x3c3],
    &[0x3c6, 0x3d5],
    &[0x432, 0x1c80],
    &[0x434, 0x1c81],
    &[0x43e, 0x1c82],
    &[0x441, 0x1c83],
    &[0x442, 0x1c84, 0x1c85],
    &[0x44a, 0x1c86],
    &[0x463, 0x1c87],
    &[0x1c88, 0xa64b],
    &[0x1e61, 0x1e9b],
    &[0xfb05, 0xfb06],
];

/// The characters that `lower` changes, and their lower case.
static LOWERED: LazyLock<[Vec<(u32, u32)>; 2]> = LazyLock::new(|| [false, true].map(|ascii| (0..=0x10ffff).filter_map(|c| Some((c, lower(c, ascii))).filter(|&(c, low)| c != low)).collect()));

/// `_sre.unicode_tolower`, or `_sre.ascii_tolower` for the `a` flag. It is the first character of the full lower case.
fn lower(c: u32, ascii: bool) -> u32 {
    match char::from_u32(c) {
        Some(c) if ascii => c.to_ascii_lowercase() as u32,
        Some(c) => c.to_lowercase().next().map_or(c as u32, u32::from),
        None => c,
    }
}

/// `_sre.unicode_iscased`, or `_sre.ascii_iscased` for the `a` flag.
fn cased(c: u32, ascii: bool) -> bool {
    match char::from_u32(c) {
        Some(c) if ascii => c.is_ascii_alphabetic(),
        Some(c) => lower(c as u32, false) != c as u32 || c.to_uppercase().next() != Some(c),
        None => false,
    }
}

/// The items of a set that Python matches without case. Python puts the lower case of each item, and the other characters of its `CASE_GROUPS`, in the set. Then a character matches if its lower case is in the set.
fn fold(items: &[Item], ascii: bool) -> Vec<Item> {
    let inside = |c: u32| items.iter().any(|item| matches!(*item, Item::Literal(value) if value == c) || matches!(*item, Item::Range(low, high) if low <= c && c <= high));
    let lowered = &LOWERED[usize::from(ascii)];
    let mut extra: Vec<u32> = lowered.iter().filter(|&&(c, _)| inside(c)).map(|&(_, low)| low).collect();
    if !ascii {
        for group in CASE_GROUPS {
            if group.iter().any(|&c| inside(c) || extra.contains(&c)) {
                extra.extend(group.iter());
            }
        }
    }
    extra.sort_unstable();
    extra.dedup();
    let mut out = items.to_vec();
    out.extend(lowered.iter().filter(|&&(_, low)| inside(low) || extra.binary_search(&low).is_ok()).map(|&(c, _)| Item::Literal(c)));
    out.extend(extra.into_iter().map(Item::Literal));
    out
}

fn combine(flags: u32, add: u32, delete: u32) -> u32 {
    (if add & TYPE_FLAGS != 0 { flags & !TYPE_FLAGS } else { flags } | add) & !delete
}

/// The set that Python checks the first character of a match against, if a group changes the `a` or `u` flag of the set. Python makes the check with the global flags.
fn first_set(mut seq: &[Node], global: u32) -> Option<(bool, Vec<Item>)> {
    let mut flags = global;
    loop {
        let node = match seq.first()? {
            Node::Branch(branches) if branches.iter().all(|branch| branch.first().is_some_and(|first| Some(first) == branches[0].first())) => &branches[0][0],
            Node::Branch(branches) => {
                let mut items = Vec::new();
                for branch in branches {
                    match branch.as_slice() {
                        [Node::Literal(value)] => items.push(Item::Literal(*value)),
                        [Node::Set(false, set)] => items.extend(set),
                        _ => return None,
                    }
                }
                return ((flags ^ global) & ASCII != 0 && plain(&items, flags)).then_some((false, items));
            }
            node => node,
        };
        match node {
            Node::Group(_, add, delete, body) => {
                flags = combine(flags, *add, *delete);
                seq = body;
            }
            Node::Set(negate, items) => return ((flags ^ global) & ASCII != 0 && plain(items, flags)).then(|| (*negate, items.clone())),
            _ => return None,
        }
    }
}

/// A set that Python can check as the first character: with a category, and with no cased character for the `i` flag.
fn plain(items: &[Item], flags: u32) -> bool {
    let ascii = flags & ASCII != 0;
    items.iter().any(|item| matches!(item, Item::Category(_)))
        && (flags & IGNORECASE == 0
            || items.iter().all(|item| match *item {
                Item::Literal(c) => !cased(c, ascii),
                Item::Range(low, high) => high <= 0xffff && !(low..=high).any(|c| cased(c, ascii)),
                Item::Category(_) => true,
            }))
}

/// A compiled Python pattern and its smallest width. `search` is `re.search(...) is not None`.
pub struct Pattern(fancy_regex::Regex, usize);

impl Pattern {
    pub fn search(&self, text: &str) -> bool {
        let least = self.1;
        (least == 0 || text.len() >= least && (text.len() >= 4 * least || text.chars().count() >= least)) && self.0.is_match(text).unwrap_or(false)
    }
}

/// The text of the Python `RecursionError`.
pub const RECURSION: &str = "maximum recursion depth exceeded";

/// `re.compile(pattern)`. The error is the text of the Python exception. Python stops with `RECURSION` when the parse goes `frames` calls deep.
pub fn compile(pattern: &str, frames: usize) -> Result<Pattern, String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut state = State { flags: 0, names: HashMap::new(), widths: vec![None], behind: None, refs: Vec::new(), frames };
    let mut source = Source::new(&chars)?;
    let parsed = parse_sub(&mut source, &mut state, false, 0)?;
    if state.flags & ASCII == 0 {
        state.flags |= UNICODE;
    } else if state.flags & UNICODE != 0 {
        return Err("ASCII and UNICODE flags are incompatible".into());
    }
    if source.next.is_some() {
        return Err(source.error("unbalanced parenthesis", 0));
    }
    if let Some((group, position)) = state.refs.iter().find(|(group, _)| *group >= state.widths.len() as u128) {
        return Err(located(&chars, &format!("invalid group reference {group}"), *position as isize));
    }
    check(&parsed, &state)?;
    let build = |flags| {
        let mut out = String::new();
        if let Some((negate, items)) = first_set(&parsed, state.flags) {
            out.push_str("(?=");
            emit_set(negate, &items, state.flags & ASCII != 0, &mut out);
            out.push(')');
        }
        emit(&parsed, flags, &state, true, &mut out);
        fancy_regex::RegexBuilder::new(&out).backtrack_limit(usize::MAX).build()
    };
    let least = width(&parsed, &state).0.min(MAXCODE) as usize;
    build(state.flags).or_else(|_| build(state.flags | COUNTED)).map(|regex| Pattern(regex, least)).map_err(|error| error.to_string())
}

#[derive(Clone, Copy, PartialEq)]
enum Tok {
    Char(char),
    Escape(char),
}

impl Tok {
    fn len(self) -> usize {
        if let Tok::Escape(_) = self { 2 } else { 1 }
    }

    fn text(self) -> String {
        match self {
            Tok::Char(c) => c.to_string(),
            Tok::Escape(c) => format!("\\{c}"),
        }
    }

    fn char(self) -> Option<char> {
        if let Tok::Char(c) = self { Some(c) } else { None }
    }

    fn alpha(self) -> bool {
        self.char().is_some_and(char::is_alphabetic)
    }
}

fn len(tok: Option<Tok>) -> usize {
    tok.map_or(0, Tok::len)
}

fn located(pattern: &[char], message: &str, position: isize) -> String {
    let mut text = format!("{message} at position {position}");
    if pattern.contains(&'\n') {
        let before = &pattern[..(position.max(0) as usize).min(pattern.len())];
        let line = before.iter().filter(|&&c| c == '\n').count() + 1;
        let column = position - before.iter().rposition(|&c| c == '\n').map_or(-1, |index| index as isize);
        text += &format!(" (line {line}, column {column})");
    }
    text
}

struct Source<'a> {
    pattern: &'a [char],
    index: usize,
    next: Option<Tok>,
}

type Parsed<T> = Result<T, String>;

impl<'a> Source<'a> {
    fn new(pattern: &'a [char]) -> Parsed<Self> {
        let mut source = Source { pattern, index: 0, next: None };
        source.advance()?;
        Ok(source)
    }

    fn advance(&mut self) -> Parsed<()> {
        let Some(&c) = self.pattern.get(self.index) else {
            self.next = None;
            return Ok(());
        };
        if c == '\\' {
            let Some(&escaped) = self.pattern.get(self.index + 1) else {
                return Err(located(self.pattern, "bad escape (end of pattern)", self.pattern.len() as isize - 1));
            };
            self.index += 2;
            self.next = Some(Tok::Escape(escaped));
        } else {
            self.index += 1;
            self.next = Some(Tok::Char(c));
        }
        Ok(())
    }

    fn matches(&mut self, c: char) -> Parsed<bool> {
        if self.next == Some(Tok::Char(c)) {
            self.advance()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn get(&mut self) -> Parsed<Option<Tok>> {
        let this = self.next;
        self.advance()?;
        Ok(this)
    }

    fn get_while(&mut self, count: usize, keep: fn(char) -> bool) -> Parsed<String> {
        let mut result = String::new();
        for _ in 0..count {
            match self.next {
                Some(Tok::Char(c)) if keep(c) => result.push(c),
                _ => break,
            }
            self.advance()?;
        }
        Ok(result)
    }

    fn get_until(&mut self, terminator: char, name: &str) -> Parsed<String> {
        let mut result = String::new();
        loop {
            let c = self.get()?;
            match c {
                None if result.is_empty() => return Err(self.error(&format!("missing {name}"), 0)),
                None => return Err(self.error(&format!("missing {terminator}, unterminated name"), result.chars().count())),
                Some(Tok::Char(c)) if c == terminator => {
                    if result.is_empty() {
                        return Err(self.error(&format!("missing {name}"), 1));
                    }
                    return Ok(result);
                }
                Some(tok) => result += &tok.text(),
            }
        }
    }

    fn tell(&self) -> usize {
        self.index - len(self.next)
    }

    fn seek(&mut self, index: usize) -> Parsed<()> {
        self.index = index;
        self.advance()
    }

    fn error(&self, message: &str, offset: usize) -> String {
        located(self.pattern, message, self.tell() as isize - offset as isize)
    }

    fn check_name(&self, name: &str, offset: usize) -> Parsed<()> {
        let mut chars = name.chars();
        let identifier = chars.next().is_some_and(|first| first == '_' || unicode_ident::is_xid_start(first)) && chars.all(unicode_ident::is_xid_continue);
        if identifier { Ok(()) } else { Err(self.error(&format!("bad character in group name {}", py_repr(name)), name.chars().count() + offset)) }
    }
}

struct State {
    flags: u32,
    names: HashMap<String, usize>,
    widths: Vec<Option<(u128, u128)>>,
    behind: Option<usize>,
    refs: Vec<(u128, usize)>,
    frames: usize,
}

impl State {
    fn closed(&self, group: usize) -> bool {
        self.widths.get(group).is_some_and(Option::is_some)
    }

    fn check_behind(&self, group: usize, source: &Source) -> Parsed<()> {
        match self.behind {
            Some(_) if !self.closed(group) => Err(source.error("cannot refer to an open group", 0)),
            Some(first) if group >= first => Err(source.error("cannot refer to group defined in the same lookbehind subpattern", 0)),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Category {
    Digit,
    NotDigit,
    Space,
    NotSpace,
    Word,
    NotWord,
}

#[derive(Clone, Copy, PartialEq)]
enum Item {
    Literal(u32),
    Range(u32, u32),
    Category(Category),
}

#[derive(Clone, Copy, PartialEq)]
enum At {
    Beginning,
    End,
    BeginningString,
    EndString,
    Boundary,
    NonBoundary,
}

#[derive(Clone, Copy, PartialEq)]
enum Repeat {
    Greedy,
    Lazy,
    Possessive,
}

#[derive(PartialEq)]
enum Node {
    Literal(u32),
    Set(bool, Vec<Item>),
    Any,
    At(At),
    Repeat(u128, u128, Repeat, Box<Node>),
    Group(Option<usize>, u32, u32, Vec<Node>),
    Atomic(Vec<Node>),
    Branch(Vec<Vec<Node>>),
    Assert(bool, bool, Vec<Node>),
    GroupRef(usize),
    Exists(usize, Vec<Node>, Option<Vec<Node>>),
}

enum Escaped {
    Item(Item),
    At(At),
    GroupRef(usize),
}

fn category(c: char) -> Option<Escaped> {
    Some(match c {
        'A' => Escaped::At(At::BeginningString),
        'b' => Escaped::At(At::Boundary),
        'B' => Escaped::At(At::NonBoundary),
        'Z' => Escaped::At(At::EndString),
        'd' => Escaped::Item(Item::Category(Category::Digit)),
        'D' => Escaped::Item(Item::Category(Category::NotDigit)),
        's' => Escaped::Item(Item::Category(Category::Space)),
        'S' => Escaped::Item(Item::Category(Category::NotSpace)),
        'w' => Escaped::Item(Item::Category(Category::Word)),
        'W' => Escaped::Item(Item::Category(Category::NotWord)),
        _ => return None,
    })
}

fn control(c: char) -> Option<u32> {
    Some(match c {
        'a' => 7,
        'b' => 8,
        'f' => 12,
        'n' => 10,
        'r' => 13,
        't' => 9,
        'v' => 11,
        '\\' => 92,
        _ => return None,
    })
}

fn hex(c: char) -> bool {
    c.is_ascii_hexdigit()
}

fn octal(c: char) -> bool {
    ('0'..='7').contains(&c)
}

/// The `\x`, `\u`, `\U`, and `\N` escapes, and the escapes that are the same in a set and outside it.
fn unicode_escape(source: &mut Source, c: char) -> Parsed<Option<u32>> {
    let (digits, full) = match c {
        'x' => (2, 4),
        'u' => (4, 6),
        'U' => (8, 10),
        'N' => {
            if !source.matches('{')? {
                return Err(source.error("missing {", 0));
            }
            let name = source.get_until('}', "character name")?;
            return Err(source.error(&format!("undefined character name {}", py_repr(&name)), name.chars().count() + 4));
        }
        _ => return Ok(None),
    };
    let escape = format!("\\{c}{}", source.get_while(digits, hex)?);
    if escape.len() != full {
        return Err(source.error(&format!("incomplete escape {escape}"), escape.len()));
    }
    let value = u32::from_str_radix(&escape[2..], 16).unwrap();
    if value > 0x10FFFF {
        return Err(source.error(&format!("bad escape {escape}"), escape.len()));
    }
    Ok(Some(value))
}

fn bad_escape(source: &Source, c: char) -> Parsed<u32> {
    if c.is_ascii_alphabetic() || c.is_ascii_digit() { Err(source.error(&format!("bad escape \\{c}"), 2)) } else { Ok(c as u32) }
}

fn class_escape(source: &mut Source, c: char) -> Parsed<Item> {
    if let Some(value) = control(c) {
        return Ok(Item::Literal(value));
    }
    if let Some(Escaped::Item(item)) = category(c) {
        return Ok(item);
    }
    if let Some(value) = unicode_escape(source, c)? {
        return Ok(Item::Literal(value));
    }
    if octal(c) {
        let escape = format!("\\{c}{}", source.get_while(2, octal)?);
        let value = u32::from_str_radix(&escape[1..], 8).unwrap();
        if value > 0o377 {
            return Err(source.error(&format!("octal escape value {escape} outside of range 0-0o377"), escape.len()));
        }
        return Ok(Item::Literal(value));
    }
    bad_escape(source, c).map(Item::Literal)
}

fn escape(source: &mut Source, c: char, state: &State) -> Parsed<Escaped> {
    if let Some(code) = category(c) {
        return Ok(code);
    }
    if let Some(value) = control(c) {
        return Ok(Escaped::Item(Item::Literal(value)));
    }
    if let Some(value) = unicode_escape(source, c)? {
        return Ok(Escaped::Item(Item::Literal(value)));
    }
    if c == '0' {
        let digits = source.get_while(2, octal)?;
        return Ok(Escaped::Item(Item::Literal(u32::from_str_radix(&format!("0{digits}"), 8).unwrap())));
    }
    if c.is_ascii_digit() {
        let mut escape = format!("\\{c}");
        if let Some(Tok::Char(next)) = source.next.filter(|next| next.char().is_some_and(|c| c.is_ascii_digit())) {
            source.advance()?;
            escape.push(next);
            if octal(c) && octal(next) && source.next.and_then(Tok::char).is_some_and(octal) {
                escape.push(source.get()?.and_then(Tok::char).unwrap());
                let value = u32::from_str_radix(&escape[1..], 8).unwrap();
                if value > 0o377 {
                    return Err(source.error(&format!("octal escape value {escape} outside of range 0-0o377"), escape.len()));
                }
                return Ok(Escaped::Item(Item::Literal(value)));
            }
        }
        let group: usize = escape[1..].parse().unwrap();
        if group < state.widths.len() {
            if !state.closed(group) {
                return Err(source.error("cannot refer to an open group", escape.len()));
            }
            state.check_behind(group, source)?;
            return Ok(Escaped::GroupRef(group));
        }
        return Err(source.error(&format!("invalid group reference {group}"), escape.len() - 1));
    }
    bad_escape(source, c).map(|value| Escaped::Item(Item::Literal(value)))
}

fn parse_sub(source: &mut Source, state: &mut State, mut verbose: bool, nested: usize) -> Parsed<Vec<Node>> {
    if nested >= state.frames {
        return Err(RECURSION.into());
    }
    let mut items = Vec::new();
    loop {
        let first = nested == 0 && items.is_empty();
        items.push(parse(source, state, verbose, nested + 1, first)?);
        if !source.matches('|')? {
            break;
        }
        if nested == 0 {
            verbose = state.flags & VERBOSE != 0;
        }
    }
    Ok(if items.len() == 1 { items.pop().unwrap() } else { vec![Node::Branch(items)] })
}

fn parse_set(source: &mut Source) -> Parsed<Node> {
    let here = source.tell() - 1;
    let mut set = Vec::new();
    let negate = source.matches('^')?;
    let unterminated = |source: &Source| source.error("unterminated character set", source.tell() - here);
    loop {
        let Some(this) = source.get()? else { return Err(unterminated(source)) };
        if this == Tok::Char(']') && !set.is_empty() {
            break;
        }
        let first = match this {
            Tok::Escape(c) => class_escape(source, c)?,
            Tok::Char(c) => Item::Literal(c as u32),
        };
        if !source.matches('-')? {
            set.push(first);
            continue;
        }
        let Some(that) = source.get()? else { return Err(unterminated(source)) };
        if that == Tok::Char(']') {
            set.extend([first, Item::Literal('-' as u32)]);
            break;
        }
        let second = match that {
            Tok::Escape(c) => class_escape(source, c)?,
            Tok::Char(c) => Item::Literal(c as u32),
        };
        match (first, second) {
            (Item::Literal(low), Item::Literal(high)) if low <= high => set.push(Item::Range(low, high)),
            _ => return Err(source.error(&format!("bad character range {}-{}", this.text(), that.text()), this.len() + 1 + that.len())),
        }
    }
    Ok(Node::Set(negate, set))
}

fn too_large() -> String {
    "the repetition number is too large".into()
}

/// Parses the bounds of `{m,n}` after the `{`. `None` is a `{` that is a literal.
fn bounds(source: &mut Source) -> Parsed<Option<(u128, u128)>> {
    let here = source.tell();
    let digits = |source: &mut Source| -> Parsed<String> {
        let mut digits = String::new();
        while let Some(c) = source.next.and_then(Tok::char).filter(char::is_ascii_digit) {
            digits.push(c);
            source.advance()?;
        }
        Ok(digits)
    };
    let low = digits(source)?;
    let high = if source.matches(',')? { digits(source)? } else { low.clone() };
    if !source.matches('}')? {
        source.seek(here)?;
        return Ok(None);
    }
    let number = |digits: &str| digits.parse::<u128>().unwrap_or(u128::MAX);
    let (mut min, mut max) = (0, MAXREPEAT);
    if !low.is_empty() {
        min = number(&low);
        if min >= MAXREPEAT {
            return Err(too_large());
        }
    }
    if !high.is_empty() {
        max = number(&high);
        if max >= MAXREPEAT {
            return Err(too_large());
        }
        if max < min {
            return Err(source.error("min repeat greater than max repeat", source.tell() - here));
        }
    }
    Ok(Some((min, max)))
}

fn parse_flags(source: &mut Source, state: &mut State, first: char) -> Parsed<Option<(u32, u32)>> {
    let flag = |c: char| match c {
        'i' => Some(IGNORECASE),
        'L' => Some(LOCALE),
        'm' => Some(MULTILINE),
        's' => Some(DOTALL),
        'x' => Some(VERBOSE),
        'a' => Some(ASCII),
        't' => Some(TEMPLATE),
        'u' => Some(UNICODE),
        _ => None,
    };
    let (mut add, mut delete, mut c) = (0, 0, first);
    if c != '-' {
        loop {
            let value = flag(c).unwrap();
            if c == 'L' {
                return Err(source.error("bad inline flags: cannot use 'L' flag with a str pattern", 0));
            }
            add |= value;
            if value & TYPE_FLAGS != 0 && add & TYPE_FLAGS != value {
                return Err(source.error("bad inline flags: flags 'a', 'u' and 'L' are incompatible", 0));
            }
            let Some(next) = source.get()? else { return Err(source.error("missing -, : or )", 0)) };
            match next.char() {
                Some(end @ (')' | '-' | ':')) => {
                    c = end;
                    break;
                }
                Some(next) if flag(next).is_some() => c = next,
                _ => return Err(source.error(if next.alpha() { "unknown flag" } else { "missing -, : or )" }, next.len())),
            }
        }
    }
    if c == ')' {
        state.flags |= add;
        return Ok(None);
    }
    if add & GLOBAL_FLAGS != 0 {
        return Err(source.error("bad inline flags: cannot turn on global flag", 1));
    }
    if c == '-' {
        let Some(mut next) = source.get()? else { return Err(source.error("missing flag", 0)) };
        if next.char().and_then(flag).is_none() {
            return Err(source.error(if next.alpha() { "unknown flag" } else { "missing flag" }, next.len()));
        }
        loop {
            let value = next.char().and_then(flag).unwrap();
            if value & TYPE_FLAGS != 0 {
                return Err(source.error("bad inline flags: cannot turn off flags 'a', 'u' and 'L'", 0));
            }
            delete |= value;
            let Some(after) = source.get()? else { return Err(source.error("missing :", 0)) };
            if after == Tok::Char(':') {
                break;
            }
            if after.char().and_then(flag).is_none() {
                return Err(source.error(if after.alpha() { "unknown flag" } else { "missing :" }, after.len()));
            }
            next = after;
        }
    }
    if delete & GLOBAL_FLAGS != 0 {
        return Err(source.error("bad inline flags: cannot turn off global flag", 1));
    }
    if add & delete != 0 {
        return Err(source.error("bad inline flags: flag turned on and off", 1));
    }
    Ok(Some((add, delete)))
}

fn unterminated(source: &Source, start: usize) -> String {
    source.error("missing ), unterminated subpattern", source.tell() - start)
}

fn unexpected_end(source: &Source) -> String {
    source.error("unexpected end of pattern", 0)
}

fn parse(source: &mut Source, state: &mut State, mut verbose: bool, nested: usize, first: bool) -> Parsed<Vec<Node>> {
    if nested >= state.frames {
        return Err(RECURSION.into());
    }
    let mut sub: Vec<Node> = Vec::new();
    while let Some(this) = source.next {
        if matches!(this, Tok::Char('|' | ')')) {
            break;
        }
        source.advance()?;
        if verbose {
            if matches!(this, Tok::Char(' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')) {
                continue;
            }
            if this == Tok::Char('#') {
                while !matches!(source.get()?, None | Some(Tok::Char('\n'))) {}
                continue;
            }
        }
        let c = match this {
            Tok::Escape(c) => {
                sub.push(match escape(source, c, state)? {
                    Escaped::Item(Item::Literal(value)) => Node::Literal(value),
                    Escaped::Item(item) => Node::Set(false, vec![item]),
                    Escaped::At(at) => Node::At(at),
                    Escaped::GroupRef(group) => Node::GroupRef(group),
                });
                continue;
            }
            Tok::Char(c) => c,
        };
        match c {
            '[' => sub.push(parse_set(source)?),
            '*' | '+' | '?' | '{' => {
                let here = source.tell();
                let (min, max) = match c {
                    '?' => (0, 1),
                    '*' => (0, MAXREPEAT),
                    '+' => (1, MAXREPEAT),
                    _ => {
                        let Some(bounds) = (source.next != Some(Tok::Char('}'))).then(|| bounds(source)).transpose()?.flatten() else {
                            sub.push(Node::Literal('{' as u32));
                            continue;
                        };
                        bounds
                    }
                };
                match sub.last() {
                    None | Some(Node::At(_)) => return Err(source.error("nothing to repeat", source.tell() - here + 1)),
                    Some(Node::Repeat(..)) => return Err(source.error("multiple repeat", source.tell() - here + 1)),
                    _ => {}
                }
                let kind = if source.matches('?')? {
                    Repeat::Lazy
                } else if source.matches('+')? {
                    Repeat::Possessive
                } else {
                    Repeat::Greedy
                };
                let item = sub.pop().unwrap();
                sub.push(Node::Repeat(min, max, kind, Box::new(item)));
            }
            '.' => sub.push(Node::Any),
            '^' => sub.push(Node::At(At::Beginning)),
            '$' => sub.push(Node::At(At::End)),
            '(' => {
                let start = source.tell() - 1;
                let (mut capture, mut atomic, mut name, mut add, mut delete) = (true, false, None, 0, 0);
                if source.matches('?')? {
                    let Some(kind) = source.get()? else { return Err(unexpected_end(source)) };
                    match kind.char() {
                        Some('P') => {
                            if source.matches('<')? {
                                let group = source.get_until('>', "group name")?;
                                source.check_name(&group, 1)?;
                                name = Some(group);
                            } else if source.matches('=')? {
                                let group = source.get_until(')', "group name")?;
                                source.check_name(&group, 1)?;
                                let offset = group.chars().count() + 1;
                                let Some(&id) = state.names.get(&group) else { return Err(source.error(&format!("unknown group name {}", py_repr(&group)), offset)) };
                                if !state.closed(id) {
                                    return Err(source.error("cannot refer to an open group", offset));
                                }
                                state.check_behind(id, source)?;
                                sub.push(Node::GroupRef(id));
                                continue;
                            } else {
                                let Some(other) = source.get()? else { return Err(unexpected_end(source)) };
                                return Err(source.error(&format!("unknown extension ?P{}", other.text()), other.len() + 2));
                            }
                        }
                        Some(':') => capture = false,
                        Some('#') => {
                            loop {
                                if source.next.is_none() {
                                    return Err(source.error("missing ), unterminated comment", source.tell() - start));
                                }
                                if source.get()? == Some(Tok::Char(')')) {
                                    break;
                                }
                            }
                            continue;
                        }
                        Some(mut look @ ('=' | '!' | '<')) => {
                            let (behind, outer) = (look == '<', state.behind);
                            if behind {
                                let Some(next) = source.get()? else { return Err(unexpected_end(source)) };
                                match next.char() {
                                    Some(next @ ('=' | '!')) => look = next,
                                    _ => return Err(source.error(&format!("unknown extension ?<{}", next.text()), next.len() + 2)),
                                }
                                if outer.is_none() {
                                    state.behind = Some(state.widths.len());
                                }
                            }
                            let body = parse_sub(source, state, verbose, nested + 1)?;
                            if behind && outer.is_none() {
                                state.behind = None;
                            }
                            if !source.matches(')')? {
                                return Err(unterminated(source, start));
                            }
                            sub.push(Node::Assert(behind, look == '!', body));
                            continue;
                        }
                        Some('(') => {
                            let condition = source.get_until(')', "group name")?;
                            let offset = condition.chars().count() + 1;
                            let group = if condition.bytes().all(|byte| byte.is_ascii_digit()) {
                                let group = condition.parse::<u128>().unwrap_or(u128::MAX);
                                if group == 0 {
                                    return Err(source.error("bad group number", offset));
                                }
                                if group >= MAXGROUPS {
                                    return Err(source.error(&format!("invalid group reference {group}"), offset));
                                }
                                if !state.refs.iter().any(|(known, _)| *known == group) {
                                    state.refs.push((group, source.tell() - offset));
                                }
                                group as usize
                            } else {
                                source.check_name(&condition, 1)?;
                                *state.names.get(&condition).ok_or_else(|| source.error(&format!("unknown group name {}", py_repr(&condition)), offset))?
                            };
                            state.check_behind(group, source)?;
                            let yes = parse(source, state, verbose, nested + 1, false)?;
                            let no = if source.matches('|')? {
                                let no = parse(source, state, verbose, nested + 1, false)?;
                                if source.next == Some(Tok::Char('|')) {
                                    return Err(source.error("conditional backref with more than two branches", 0));
                                }
                                Some(no)
                            } else {
                                None
                            };
                            if !source.matches(')')? {
                                return Err(unterminated(source, start));
                            }
                            sub.push(Node::Exists(group, yes, no));
                            continue;
                        }
                        Some('>') => (capture, atomic) = (false, true),
                        Some(c) if "iLmsxatu-".contains(c) => match parse_flags(source, state, c)? {
                            None => {
                                if !first || !sub.is_empty() {
                                    return Err(source.error("global flags not at the start of the expression", source.tell() - start));
                                }
                                verbose = state.flags & VERBOSE != 0;
                                continue;
                            }
                            Some(flags) => (add, delete, capture) = (flags.0, flags.1, false),
                        },
                        _ => return Err(source.error(&format!("unknown extension ?{}", kind.text()), kind.len() + 1)),
                    }
                }
                let group = if capture {
                    let id = state.widths.len();
                    state.widths.push(None);
                    if id as u128 >= MAXGROUPS {
                        return Err(source.error("too many groups", name.as_ref().map_or(0, |name: &String| name.chars().count()) + 1));
                    }
                    if let Some(name) = &name {
                        if let Some(&old) = state.names.get(name) {
                            return Err(source.error(&format!("redefinition of group name {} as group {id}; was group {old}", py_repr(name)), name.chars().count() + 1));
                        }
                        state.names.insert(name.clone(), id);
                    }
                    Some(id)
                } else {
                    None
                };
                let verbose = (verbose || add & VERBOSE != 0) && delete & VERBOSE == 0;
                let body = parse_sub(source, state, verbose, nested + 1)?;
                if !source.matches(')')? {
                    return Err(unterminated(source, start));
                }
                if let Some(id) = group {
                    state.widths[id] = Some(width(&body, state));
                }
                sub.push(if atomic { Node::Atomic(body) } else { Node::Group(group, add, delete, body) });
            }
            c => sub.push(Node::Literal(c as u32)),
        }
    }
    Ok(sub)
}

fn width(seq: &[Node], state: &State) -> (u128, u128) {
    let (mut low, mut high) = (0u128, 0u128);
    for node in seq {
        let (min, max) = match node {
            Node::Branch(branches) => branches.iter().map(|branch| width(branch, state)).fold((MAXWIDTH, 0), |(low, high), (min, max)| (low.min(min), high.max(max))),
            Node::Atomic(body) | Node::Group(.., body) => width(body, state),
            Node::Repeat(min, max, _, item) => {
                let (item_low, item_high) = width(std::slice::from_ref(item), state);
                low = low.saturating_add(item_low * min);
                if *max == MAXREPEAT && item_high != 0 {
                    high = MAXWIDTH;
                } else {
                    high = high.saturating_add(item_high * max);
                }
                continue;
            }
            Node::Literal(_) | Node::Set(..) | Node::Any => (1, 1),
            Node::GroupRef(group) => state.widths[*group].unwrap_or((0, 0)),
            Node::Exists(_, yes, no) => {
                let (min, max) = width(yes, state);
                match no.as_ref().map(|no| width(no, state)) {
                    Some((no_min, no_max)) => (min.min(no_min), max.max(no_max)),
                    None => (0, max),
                }
            }
            Node::At(_) | Node::Assert(..) => (0, 0),
        };
        low = low.saturating_add(min);
        high = high.saturating_add(max);
    }
    (low.min(MAXWIDTH), high.min(MAXWIDTH))
}

/// The checks of `re._compiler`, in the order of the compile.
fn check(seq: &[Node], state: &State) -> Parsed<()> {
    for node in seq {
        match node {
            Node::Assert(behind, _, body) => {
                if *behind {
                    let (low, high) = width(body, state);
                    if low > MAXCODE {
                        return Err("looks too much behind".into());
                    }
                    if low != high {
                        return Err("look-behind requires fixed-width pattern".into());
                    }
                }
                check(body, state)?;
            }
            Node::Branch(branches) => branches.iter().try_for_each(|branch| check(branch, state))?,
            Node::Atomic(body) | Node::Group(.., body) => check(body, state)?,
            Node::Repeat(.., item) => check(std::slice::from_ref(item), state)?,
            Node::Exists(_, yes, no) => {
                check(yes, state)?;
                check(no.as_deref().unwrap_or_default(), state)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn literal(value: u32, out: &mut String) -> bool {
    match char::from_u32(value) {
        Some(c) if c.is_ascii_alphanumeric() || c == '_' => out.push(c),
        Some(_) => _ = write!(out, "\\x{{{value:x}}}"),
        None => return false,
    }
    true
}

fn class(category: Category, ascii: bool) -> &'static str {
    match (category, ascii) {
        (Category::Digit, false) => r"\d",
        (Category::NotDigit, false) => r"\D",
        (Category::Space, false) => r"\s\x1c-\x1f",
        (Category::NotSpace, false) => r"[^\s\x1c-\x1f]",
        (Category::Word, false) => r"\p{L}\p{N}_",
        (Category::NotWord, false) => r"[^\p{L}\p{N}_]",
        (Category::Digit, true) => "0-9",
        (Category::NotDigit, true) => "[^0-9]",
        (Category::Space, true) => r" \t\n\r\x0b\x0c",
        (Category::NotSpace, true) => r"[^ \t\n\r\x0b\x0c]",
        (Category::Word, true) => "0-9A-Za-z_",
        (Category::NotWord, true) => "[^0-9A-Za-z_]",
    }
}

fn emit_set(negate: bool, items: &[Item], ascii: bool, out: &mut String) {
    let mut inner = String::new();
    for item in items {
        match *item {
            Item::Literal(value) => _ = literal(value, &mut inner),
            Item::Range(low, high) => {
                for (low, high) in [(low, high.min(0xD7FF)), (low.max(0xE000), high)] {
                    if low <= high {
                        literal(low, &mut inner);
                        inner.push('-');
                        literal(high, &mut inner);
                    }
                }
            }
            Item::Category(category) => inner += class(category, ascii),
        }
    }
    match (inner.is_empty(), negate) {
        (true, false) => out.push_str(r"[^\s\S]"),
        (true, true) => out.push_str(r"[\s\S]"),
        (false, _) => _ = write!(out, "[{}{inner}]", if negate { "^" } else { "" }),
    }
}

fn emit(seq: &[Node], flags: u32, state: &State, end: bool, out: &mut String) {
    let ascii = flags & ASCII != 0;
    let fold = flags & IGNORECASE != 0;
    let word = if ascii { "[0-9A-Za-z_]" } else { r"[\p{L}\p{N}_]" };
    for (index, node) in seq.iter().enumerate() {
        let last = end && index + 1 == seq.len();
        match node {
            Node::Literal(value) if fold && cased(*value, ascii) => emit_set(false, &self::fold(&[Item::Literal(*value)], ascii), ascii, out),
            Node::Literal(value) => {
                if !literal(*value, out) {
                    out.push_str(r"[^\s\S]");
                }
            }
            Node::Set(negate, items) if fold => emit_set(*negate, &self::fold(items, ascii), ascii, out),
            Node::Set(negate, items) => emit_set(*negate, items, ascii, out),
            Node::Any => out.push_str(if flags & DOTALL != 0 { r"[\s\S]" } else { r"[^\n]" }),
            Node::At(at) => out.push_str(&match at {
                At::Beginning if flags & MULTILINE != 0 => "(?m:^)".into(),
                At::Beginning | At::BeginningString => r"\A".into(),
                At::End if flags & MULTILINE != 0 => "(?m:$)".into(),
                At::End if last => r"\n?\z".into(),
                At::End => r"(?=\n?\z)".into(),
                At::EndString => r"\z".into(),
                At::Boundary => format!("(?:(?<={word})(?!{word})|(?<!{word})(?={word}))"),
                At::NonBoundary => format!(r"(?:(?<={word})(?={word})|(?<!{word})(?!{word})(?:(?<=[\s\S])|(?=[\s\S])))"),
            }),
            // `fancy-regex` does not repeat a match with no width. Such a repeat is one try of the match.
            Node::Repeat(min, max, kind, item) if width(std::slice::from_ref(&**item), state).1 == 0 => {
                let one = std::slice::from_ref(&**item);
                match (*min, *max, kind) {
                    (0, 0, _) => {
                        out.push_str(r"(?:[^\s\S]");
                        emit(one, flags, state, false, out);
                        out.push_str("|)");
                    }
                    (0, _, Repeat::Lazy) => {
                        out.push_str("(?:|");
                        emit(one, flags, state, false, out);
                        out.push(')');
                    }
                    (0, _, _) => {
                        out.push_str(if *kind == Repeat::Possessive { "(?>" } else { "(?:" });
                        emit(one, flags, state, false, out);
                        out.push_str("|)");
                    }
                    _ => emit(one, flags, state, false, out),
                }
            }
            Node::Repeat(min, max, kind, item) => {
                if *kind == Repeat::Possessive {
                    out.push_str("(?>");
                }
                let counted = flags & COUNTED != 0 && !matches!((*min, *max), (0 | 1, MAXREPEAT) | (0, 1));
                let wrap = counted || matches!(**item, Node::Repeat(..) | Node::At(_) | Node::Exists(..));
                out.push_str(if wrap { "(?:" } else { "" });
                emit(std::slice::from_ref(item), flags, state, false, out);
                out.push_str(if counted {
                    "(?=))"
                } else if wrap {
                    ")"
                } else {
                    ""
                });
                match (*min, *max) {
                    (0, 1) => out.push('?'),
                    (0, MAXREPEAT) => out.push('*'),
                    (1, MAXREPEAT) => out.push('+'),
                    (min, MAXREPEAT) => _ = write!(out, "{{{min},}}"),
                    (min, max) if min == max => _ = write!(out, "{{{min}}}"),
                    (min, max) => _ = write!(out, "{{{min},{max}}}"),
                }
                match kind {
                    Repeat::Lazy => out.push('?'),
                    Repeat::Possessive => out.push(')'),
                    Repeat::Greedy => {}
                }
            }
            Node::Group(group, add, delete, body) => {
                let flags = combine(flags, *add, *delete);
                out.push_str(if group.is_some() { "(" } else { "(?:" });
                emit(body, flags, state, last, out);
                out.push(')');
            }
            Node::Atomic(body) => {
                out.push_str("(?>");
                emit(body, flags, state, last, out);
                out.push(')');
            }
            Node::Branch(branches) => {
                out.push_str("(?:");
                for (index, branch) in branches.iter().enumerate() {
                    if index > 0 {
                        out.push('|');
                    }
                    emit(branch, flags, state, last, out);
                }
                out.push(')');
            }
            // A Python look-around is atomic. A look-behind goes back by its fixed width, then looks ahead, because `fancy-regex` cannot look behind a reference.
            Node::Assert(behind, negate, body) => {
                let back = if *behind { width(body, state).0 } else { 0 };
                out.push_str(match (back > 0, negate) {
                    (false, false) => "(?=(?>",
                    (false, true) => "(?!(?>",
                    (true, false) => "(?<=(?=(?>",
                    (true, true) => "(?<!(?=(?>",
                });
                emit(body, flags, state, false, out);
                out.push_str("))");
                if back > 0 {
                    _ = write!(out, r"[\s\S]{{{back}}})");
                }
            }
            Node::GroupRef(group) => _ = write!(out, "{}\\k<{group}>)", if fold { "(?i:" } else { "(?:" }),
            // `fancy-regex` reads `(?(1)|)` as a test that group 1 has a match.
            Node::Exists(group, yes, no) => {
                let (mut when, mut other) = (String::new(), String::new());
                emit(yes, flags, state, last, &mut when);
                emit(no.as_deref().unwrap_or_default(), flags, state, last, &mut other);
                if !when.is_empty() || !other.is_empty() {
                    _ = write!(out, "(?({group}){when}|{other})");
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/inline/cli/pyre.rs"]
mod tests;
