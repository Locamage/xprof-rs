use super::client::{Client, TRACE_SUFFIXES, traces};
use super::json::{J, py_repr};
use super::pyre::{self, Pattern};
use super::{Args, Error, Kind, Out, bypass, fail, fsum, rethrow, round, stdev};
use crate::obj;
use crate::tools::counters::{Event, Plane, checked, events, planes};
use crate::xplane::{Field, Value, fields, stats};
use indexmap::IndexMap;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::borrow::Cow;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

const MAX_SCANNED: usize = 5_000_000;
const NAME_STATS: [&str; 4] = ["msg", "message", "annotation", "label"];
const DEVICE_LINES: [&str; 3] = ["XLA OPS", "PALLAS", "LLO OPS"];
const EXCLUDED_LINES: [&str; 5] = ["COUNTER", "MODULES", "OVERLAY", "SYNC FLAG", "SENSOR"];
const XSPACE_SPILL_BYTES: usize = 10 * 1024 * 1024;
const LLO_REMEDIATION: &str = "To enable LLO tracing, ensure the workload is executed with LIBTPU_INIT_ARGS=\"--xla_xprof_enable_custom_call_tracing=true --xla_xprof_register_llo_debug_info=true\" exported strictly BEFORE 'import jax'. Prerequisites: Python 3.11+ (Python 3.12 recommended via uv), JAX >= 0.11.0 (default Cloud TPU VM images running Python 3.10 cap JAX at 0.6.2 and lack LLO flag support), and xprof-nightly.";
const NUMERICAL_DEPENDENCIES: &str = "Required numerical dependencies are not installed in current environment: No module named 'ml_dtypes'. Please install numpy and ml_dtypes (e.g. 'pip install numpy ml_dtypes') to use verify_numerical_parity.";

struct Visit<'a> {
    plane: &'a Plane<'a>,
    line: &'a str,
    timestamp: i64,
    event: Event<'a>,
}

impl Visit<'_> {
    fn start_ns(&self) -> f64 {
        self.timestamp as f64 + self.event.offset_ps as i64 as f64 / 1000.0
    }

    fn duration_ns(&self) -> f64 {
        self.event.duration_ps as i64 as f64 / 1000.0
    }

    fn stats(&self) -> Vec<(String, String)> {
        stats(self.event.raw, 4, |_| true)
            .map(|stat| (self.plane.stat_names.get(&(stat.id as u64)).map_or_else(|| stat.id.to_string(), |name| String::from_utf8_lossy(name).into_owned()), value_text(stat.value)))
            .collect()
    }

    /// The first value with this name that is not empty.
    fn stat(&self, name: &str) -> Option<String> {
        stats(self.event.raw, 4, |_| true)
            .filter(|stat| self.plane.stat_names.get(&(stat.id as u64)).is_some_and(|known| *known == name.as_bytes()))
            .map(|stat| value_text(stat.value))
            .find(|value| !value.is_empty())
    }

    fn raw_name(&self) -> String {
        self.plane.metadata.get(&self.event.meta).map_or_else(|| self.event.meta.to_string(), |(name, _)| String::from_utf8_lossy(name).into_owned())
    }

    fn name(&self) -> String {
        let name = self.raw_name();
        if is_number(&name) {
            return self.stats().into_iter().find(|(key, value)| NAME_STATS.contains(&key.as_str()) && !value.is_empty()).map_or(name, |(_, value)| value);
        }
        name
    }
}

/// A name that is a number gets its text from the stats of each event.
fn is_number(name: &str) -> bool {
    !name.is_empty() && name.chars().all(char::is_numeric)
}

fn value_text(value: Value) -> String {
    match value {
        Value::Double(number) => format!("{number:.6}"),
        Value::Uint(number) => number.to_string(),
        Value::Int(number) => number.to_string(),
        Value::Str(bytes) => String::from_utf8_lossy(bytes).into_owned(),
        Value::Ref(id) => id.to_string(),
        Value::Bytes(_) => "<bytes>".into(),
    }
}

/// Runs `scan` on every line of every trace file, with its plane and its index. The lines run in parallel. The results keep the order of the planes and of the lines.
fn scan_lines<R: Send>(paths: &[PathBuf], scan: impl Fn(&Plane, usize) -> R + Sync) -> Result<Vec<R>, Error> {
    scan_until(paths, |plane, index, _| scan(plane, index), |_| 0, usize::MAX)
}

/// The results of `scan` for each line, in the order of the files. The scan stops after `limit` events. `count` gives the number of events in a result.
fn scan_until<R: Send>(paths: &[PathBuf], scan: impl Fn(&Plane, usize, usize) -> R + Sync, count: impl Fn(&R) -> usize + Sync, mut limit: usize) -> Result<Vec<R>, Error> {
    let mut results = Vec::new();
    for path in paths {
        let scanned = checked(&super::read(path)?, |map| {
            let planes = planes(map, |_| true);
            let lines: Vec<(&Plane, usize)> = planes.iter().flat_map(|plane| (0..plane.lines.len()).map(move |index| (plane, index))).collect();
            // A few lines can hold most of the events. The largest lines start first, so that the scan does not wait for a large line at the end.
            let mut order: Vec<usize> = (0..lines.len()).collect();
            order.sort_unstable_by_key(|&at| std::cmp::Reverse(lines[at].0.lines[lines[at].1].len()));
            let next = AtomicUsize::new(0);
            let mut done: Vec<(usize, R)> = (0..rayon::current_num_threads())
                .into_par_iter()
                .with_max_len(1)
                .flat_map_iter(|_| std::iter::from_fn(|| order.get(next.fetch_add(1, Relaxed)).map(|&at| (at, scan(lines[at].0, lines[at].1, usize::MAX)))))
                .collect();
            done.sort_unstable_by_key(|item| item.0);
            let mut found: Vec<R> = done.into_iter().map(|item| item.1).collect();
            let mut at = 0;
            while at < found.len() && count(&found[at]) <= limit {
                limit -= count(&found[at]);
                at += 1;
            }
            if at < found.len() {
                found.truncate(at);
                if limit > 0 {
                    found.push(scan(lines[at].0, lines[at].1, limit));
                }
                limit = 0;
            }
            found
        });
        results.extend(scanned.ok_or_else(|| Error::new(Kind::Value, "Failed to parse XSpace protobuf data"))?);
    }
    Ok(results)
}

fn visit_line(plane: &Plane, index: usize, mut each: impl FnMut(&Visit) -> bool) -> bool {
    let (bytes, mut line, mut timestamp) = (plane.lines[index], String::new(), 0);
    for (tag, field) in fields(bytes) {
        match (tag, field) {
            (2, Field::Bytes(_, name)) => line = String::from_utf8_lossy(name).into_owned(),
            (3, Field::Num(value)) => timestamp = value as i64,
            _ => {}
        }
    }
    events(bytes).all(|event| each(&Visit { plane, line: &line, timestamp, event }))
}

/// The name of an event and if it matches. Only the events with a number as name need work for each event.
struct Names<'a> {
    pattern: &'a Pattern,
    known: FxHashMap<u64, (String, bool, bool)>,
}

impl<'a> Names<'a> {
    fn new(pattern: &'a Pattern) -> Self {
        Names { pattern, known: FxHashMap::default() }
    }

    fn resolve(&mut self, visit: &Visit) -> (Cow<'_, str>, bool) {
        let pattern = self.pattern;
        let (name, numeric, matched) = self.known.entry(visit.event.meta).or_insert_with(|| {
            let name = visit.raw_name();
            let matched = pattern.search(&name);
            (name.clone(), is_number(&name), matched)
        });
        if *numeric {
            let name = visit.name();
            let matched = pattern.search(&name);
            return (Cow::Owned(name), matched);
        }
        (Cow::Borrowed(name.as_str()), *matched)
    }
}

fn sources(client: &dyn Client, source: &str) -> Result<Vec<PathBuf>, Error> {
    let path = Path::new(source);
    if path.exists() || source.starts_with('/') || TRACE_SUFFIXES.iter().any(|suffix| source.ends_with(suffix)) {
        if !path.exists() {
            return fail(Kind::FileNotFound, format!("Path does not exist: {}", py_repr(source)));
        }
        if path.is_dir() {
            let found = traces(path);
            if found.is_empty() {
                return fail(Kind::FileNotFound, format!("No .xplane.pb or .xspace.pb files found in directory: {}", py_repr(source)));
            }
            return Ok(found);
        }
        return Ok(vec![path.to_path_buf()]);
    }
    client.xspace_paths(&client.run_dir(source)?)
}

fn regex(pattern: &str) -> Result<Pattern, Error> {
    pyre::compile(pattern, 986).map_err(|message| Error::new(Kind::Runtime, message))
}

fn plane_filter(pattern: &Pattern) -> impl Fn(&Plane) -> bool {
    move |plane| pattern.search(&plane.name)
}

pub fn list_xplane_events(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let compute = || -> Result<J, Error> {
        let number = |name: &str, operator: &str, number_first: bool| {
            args.get(name)
                .map(|value| {
                    value.float().ok_or_else(|| {
                        let (left, right) = if number_first { ("int", super::kind(value)) } else { (super::kind(value), "int") };
                        Error::new(Kind::Type, format!("'{operator}' not supported between instances of '{left}' and '{right}'"))
                    })
                })
                .transpose()
        };
        let (start, end) = (number("start_time_ps", "<", true)?, number("end_time_ps", ">", true)?);
        let (max_events, offset) = (number("max_events", "<=", false)?.unwrap_or(100.0), number("offset", "<", true)?.unwrap_or(0.0));
        let (planes_re, events_re) = (regex(&args.string("plane_regex", ".*"))?, regex(&args.string("event_regex", ".*"))?);
        let keep = plane_filter(&planes_re);
        let wanted = if max_events <= 0.0 { f64::INFINITY } else { offset.max(0.0).ceil() + max_events.ceil() };
        // Each plane keeps the first matches that can be in the answer. The planes join in order.
        let parts = scan_lines(&sources(client, &session)?, |plane, index| {
            let (mut listed, mut matched, mut names) = (Vec::new(), 0usize, Names::new(&events_re));
            if keep(plane) {
                visit_line(plane, index, |visit| {
                    let offset_ps = (visit.start_ns() * 1000.0).trunc();
                    let duration_ps = (visit.duration_ns() * 1000.0).trunc();
                    if start.is_some_and(|start| offset_ps < start) || end.is_some_and(|end| offset_ps + duration_ps > end) {
                        return true;
                    }
                    let (name, found) = names.resolve(visit);
                    if found {
                        matched += 1;
                        if (listed.len() as f64) < wanted {
                            listed.push((visit.line.to_string(), name.into_owned(), offset_ps as i128, duration_ps as i128));
                        }
                    }
                    true
                });
            }
            (plane.name.clone(), listed, matched)
        })?;
        let (mut listed, mut matched, mut skipped) = (Vec::new(), 0usize, 0usize);
        for (plane, found, count) in parts {
            matched += count;
            for (line, name, offset_ps, duration_ps) in found {
                if (skipped as f64) < offset {
                    skipped += 1;
                } else if max_events <= 0.0 || (listed.len() as f64) < max_events {
                    listed.push(obj! {"plane" => plane.clone(), "line_id" => line, "event" => name, "offset_ps" => offset_ps, "duration_ps" => duration_ps});
                }
            }
        }
        let returned = listed.len();
        Ok(obj! {"events" => listed, "returned" => returned, "total_matched" => matched, "truncated" => matched as f64 > returned as f64 + offset})
    };
    Ok(compute().map_err(|error| rethrow(error, |error| format!("Error listing XPlane events for session {session}: {}", error.message)))?.into())
}

pub fn aggregate_xplane_events(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let compute = || -> Result<J, Error> {
        let (planes_re, events_re) = (regex(&args.string("plane_regex", ".*"))?, regex(&args.string("event_regex", ".*"))?);
        let keep = plane_filter(&planes_re);
        let (mut durations, mut scanned): (IndexMap<String, Vec<i128>, FxBuildHasher>, usize) = (IndexMap::default(), 0);
        let paths = sources(client, &session)?;
        // The scan stops after the first event above the limit, also in the middle of a line.
        let parts = scan_until(
            &paths,
            |plane, index, cap| {
                let (mut found, mut count, mut names) = (IndexMap::<String, Vec<i128>, FxBuildHasher>::default(), 0usize, Names::new(&events_re));
                // The slot in `found` of each metadata with a name that is not a number. `None` is a name that does not match.
                let mut slots: FxHashMap<u64, Option<usize>> = FxHashMap::default();
                if keep(plane) {
                    visit_line(plane, index, |visit| {
                        count += 1;
                        let duration = || (visit.duration_ns() * 1000.0).trunc() as i128;
                        if let Some(slot) = slots.get(&visit.event.meta) {
                            if let Some(slot) = *slot {
                                found[slot].push(duration());
                            }
                            return count < cap;
                        }
                        let (name, matched) = names.resolve(visit);
                        let slot = matched.then(|| found.get_index_of(&*name).unwrap_or_else(|| found.insert_full(name.to_string(), Vec::new()).0));
                        if let Some(slot) = slot {
                            found[slot].push(duration());
                        }
                        if let Cow::Borrowed(_) = name {
                            slots.insert(visit.event.meta, slot);
                        }
                        count < cap
                    });
                }
                (found, count)
            },
            |part| part.1,
            MAX_SCANNED + 1,
        )?;
        for (found, count) in parts {
            scanned += count;
            for (name, values) in found {
                durations.entry(name).or_default().extend(values);
            }
        }
        let mut results: Vec<J> = durations
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(name, values)| {
                let total: i128 = values.iter().sum();
                let floats: Vec<f64> = values.iter().map(|value| *value as f64).collect();
                obj! {
                    "event" => name,
                    "count" => values.len(),
                    "total_duration_ps" => total,
                    "avg_duration_ps" => total as f64 / values.len() as f64,
                    "min_duration_ps" => values.iter().min().copied(),
                    "max_duration_ps" => values.iter().max().copied(),
                    "std_dev_ps" => if values.len() > 1 { stdev(&floats) } else { 0.0 },
                }
            })
            .collect();
        results.sort_by_key(|result| std::cmp::Reverse(result.at("total_duration_ps").int().unwrap_or(0)));
        let count = results.len();
        Ok(obj! {"aggregates" => results, "returned" => count, "total_matched" => count, "events_scanned" => scanned, "truncated" => scanned > MAX_SCANNED})
    };
    Ok(compute().map_err(|error| rethrow(error, |error| format!("Error aggregating XPlane events for session {session}: {}", error.message)))?.into())
}

fn serialized(client: &dyn Client, session: &str, host: &str) -> Result<Vec<u8>, Error> {
    let mut paths = client.xspace_paths(&client.run_dir(session)?)?;
    if !host.is_empty() {
        paths.retain(|path| super::client::host(path) == host);
        if paths.is_empty() {
            return fail(Kind::FileNotFound, format!("No traces found for host {} in session {}", py_repr(host), py_repr(session)));
        }
    }
    match paths.as_slice() {
        [path] => super::read(path),
        _ => fail(Kind::NotImplemented, "Multi-host XSpace serialization is not supported in OSS because xplane_pb2 is not exposed."),
    }
}

pub fn get_xspace_proto(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    if args.flag("as_text", false) {
        return fail(Kind::NotImplemented, "as_text=True is not supported in OSS because xplane_pb2 is not exposed.");
    }
    let session = args.session();
    let data = serialized(client, &session, "")?;
    if let Some(output) = args.text("output_path").filter(|path| !path.is_empty()) {
        let path = PathBuf::from(&output);
        let io = |error: std::io::Error| Error::new(Kind::Os, error.to_string());
        path.parent().filter(|parent| !parent.as_os_str().is_empty()).map(std::fs::create_dir_all).transpose().map_err(io)?;
        std::fs::write(&path, &data).map_err(io)?;
        return Ok(Out::Text(output));
    }
    if data.len() > XSPACE_SPILL_BYTES {
        let safe: String = session.chars().map(|character| if character.is_ascii_alphanumeric() || character == '_' || character == '-' { character } else { '_' }).collect();
        let path = PathBuf::from(format!("/tmp/xspace_{safe}.pb"));
        crate::replace_file(&path, &data).map_err(|error| Error::new(Kind::Os, error.to_string()))?;
        return Ok(obj! {
            "status" => "SAVED_TO_FILE",
            "size_bytes" => data.len(),
            "size_mib" => round(data.len() as f64 / 1048576.0, 2),
            "file_path" => path.to_string_lossy().into_owned(),
            "message" => "XSpace payload exceeded 10 MB. Saved to file to prevent terminal buffer overflow. Specify --output_path to customize.",
        }
        .into());
    }
    Ok(Out::Bytes(data))
}

pub fn union_ns(mut intervals: Vec<(i128, i128)>) -> i128 {
    intervals.sort_by_key(|interval| interval.0);
    let mut merged: Vec<(i128, i128)> = Vec::new();
    for (start, end) in intervals {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged.iter().map(|(start, end)| end - start).sum()
}

fn kernel_markdown(records: &[J], kernel: Option<&str>) -> String {
    let mut lines = vec![kernel.map_or_else(|| "# Top Kernels by Duration".to_string(), |kernel| format!("# Kernel Stats for `{kernel}`")), String::new()];
    lines.push("| Kernel | Total Duration (us) | Execution Count | Avg Duration (us) |".into());
    lines.push("| :--- | ---: | ---: | ---: |".into());
    for row in records {
        lines.push(format!(
            "| `{}` | {:.2} | {} | {:.2} |",
            row.at("kernel_name").text(),
            row.at("total_duration_us").float().unwrap_or(0.0),
            row.at("execution_count").text(),
            row.at("avg_duration_us").float().unwrap_or(0.0)
        ));
    }
    lines.join("\n")
}

pub fn get_kernel_stats(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let source =
        args.get("source").filter(|value| value.truthy()).or_else(|| args.get("session_id")).map(J::text).ok_or_else(|| Error::new(Kind::Value, "Must provide either 'source' or 'session_id'."))?;
    let (kernel, format, summary) = (args.text("kernel_name").filter(|name| !name.is_empty()), args.string("output_format", "json"), args.flag("include_summary", false));
    let matchers: Vec<String> = match args.get("trace_matchers") {
        Some(J::List(items)) => items.iter().map(J::text).collect(),
        Some(value) if value.truthy() => value.text().chars().map(String::from).collect(),
        _ => Vec::new(),
    };
    let compute = || -> Result<Out, Error> {
        let patterns: Vec<_> = matchers.iter().map(|pattern| (pyre::compile(pattern, 982), pattern)).collect();
        let deep = AtomicBool::new(false);
        let parts = scan_lines(&sources(client, &source)?, |plane, index| {
            let (mut durations, mut intervals, mut steps) = (IndexMap::<String, Vec<f64>, FxBuildHasher>::default(), Vec::new(), Vec::new());
            if !plane.name.starts_with("/device:") {
                return (durations, intervals, steps);
            }
            let (tpu, mut kind) = (plane.name.to_uppercase().contains("TPU"), None);
            let wanted = |name: &str| {
                kernel.as_ref().is_none_or(|kernel| kernel == name)
                    && (patterns.is_empty()
                        || patterns.iter().any(|(compiled, pattern)| match compiled {
                            Ok(compiled) => compiled.search(name) || name.contains(pattern.as_str()),
                            Err(message) if message == pyre::RECURSION => {
                                deep.store(true, Relaxed);
                                true
                            }
                            Err(_) => name.contains(pattern.as_str()),
                        }))
            };
            // The slot in `durations` of each metadata with a name that is not a number, for the events without `tf_op_name`. `None` is a name that the filters skip.
            let mut slots: FxHashMap<u64, Option<usize>> = FxHashMap::default();
            visit_line(plane, index, |visit| {
                let &mut (skipped, modules) = kind.get_or_insert_with(|| {
                    let line = visit.line.to_uppercase();
                    let skipped = if tpu { !DEVICE_LINES.iter().any(|word| line.contains(word)) } else { EXCLUDED_LINES.iter().any(|word| line.contains(word)) };
                    (skipped, tpu && summary && line.contains("XLA MODULES"))
                });
                if skipped {
                    if modules {
                        steps.push(visit.duration_ns() / 1000.0);
                    }
                    return true;
                }
                let tf_op = visit.stat("tf_op_name");
                let cached = tf_op.is_none().then(|| slots.get(&visit.event.meta).copied()).flatten();
                let slot = cached.unwrap_or_else(|| {
                    let by_meta = tf_op.is_none() && !is_number(&visit.raw_name());
                    let name = tf_op.unwrap_or_else(|| visit.name());
                    let slot = wanted(&name).then(|| durations.get_index_of(&name).unwrap_or_else(|| durations.insert_full(name, Vec::new()).0));
                    if by_meta {
                        slots.insert(visit.event.meta, slot);
                    }
                    slot
                });
                let Some(slot) = slot else { return true };
                durations[slot].push(visit.duration_ns() / 1000.0);
                if summary {
                    let start = visit.start_ns().trunc() as i128;
                    intervals.push((start, start + visit.duration_ns().trunc() as i128));
                }
                true
            });
            (durations, intervals, steps)
        })?;
        if deep.into_inner() {
            return Err(Error::new(Kind::Runtime, pyre::RECURSION));
        }
        let (mut durations, mut intervals, mut steps) = (IndexMap::<String, Vec<f64>, FxBuildHasher>::default(), Vec::new(), Vec::new());
        for (found, spans, times) in parts {
            for (name, values) in found {
                durations.entry(name).or_default().extend(values);
            }
            intervals.extend(spans);
            steps.extend(times);
        }
        if durations.is_empty() {
            let message = format!("No kernel stats found for session {source}{}", kernel.as_ref().map_or(String::new(), |kernel| format!(" and kernel {kernel}")));
            return Ok(match format.as_str() {
                "dict" if summary => Out::Value(obj! {
                    "total_device_duration_ns" => 0,
                    "total_device_duration_us" => 0.0,
                    "total_device_duration_ms" => 0.0,
                    "kernel_records" => J::List(Vec::new()),
                    "step_durations_us" => J::List(Vec::new()),
                    "stats" => obj! {"mean_us" => 0.0, "std_us" => 0.0},
                }),
                "dict" => Out::Value(J::List(Vec::new())),
                "markdown" => Out::Text(format!("# Info\n{message}\n")),
                _ => obj! {"info" => message}.into(),
            });
        }
        let mut records = Vec::new();
        for (name, values) in durations {
            let total = fsum(values.iter().copied());
            let canonical = name.trim().split(':').next().unwrap_or_default().split_whitespace().next();
            let canonical = if name.is_empty() { String::new() } else { canonical.ok_or_else(|| Error::new(Kind::Runtime, "list index out of range"))?.to_string() };
            records.push(obj! {
                "kernel_name" => name.clone(),
                "canonical_name" => canonical.clone(),
                "short_name" => canonical.clone(),
                "hlo_op_name" => canonical,
                "raw_name" => name,
                "total_duration_us" => round(total, 4),
                "execution_count" => values.len(),
                "avg_duration_us" => round(total / values.len() as f64, 4),
            });
        }
        records.sort_by(|left, right| right.at("total_duration_us").float().partial_cmp(&left.at("total_duration_us").float()).unwrap_or(std::cmp::Ordering::Equal));
        if kernel.is_none() {
            let limit = args.int("limit", 10)?;
            if limit > 0 {
                records.truncate(limit as usize);
            }
        }
        if summary {
            let total_ns = union_ns(intervals);
            let total_us = total_ns as f64 / 1000.0;
            let mean = if steps.is_empty() { total_us } else { fsum(steps.iter().copied()) / steps.len() as f64 };
            let report = obj! {
                "total_device_duration_ns" => total_ns,
                "total_device_duration_us" => total_us,
                "total_device_duration_ms" => total_ns as f64 / 1_000_000.0,
                "kernel_records" => records,
                "step_durations_us" => steps.clone(),
                "stats" => obj! {"mean_us" => round(mean, 4), "std_us" => round(if steps.len() > 1 { stdev(&steps) } else { 0.0 }, 4)},
            };
            return Ok(if format == "dict" { Out::Value(report) } else { report.into() });
        }
        Ok(match format.as_str() {
            "dict" => Out::Value(J::List(records)),
            "markdown" => Out::Text(kernel_markdown(&records, kernel.as_deref())),
            _ => J::List(records).into(),
        })
    };
    compute().map_err(|error| rethrow(error, |error| format!("Failed to get kernel stats: {}", error.message)))
}

pub fn get_avg_step_time(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let source = args.string("source", "");
    let function = args.text("func_name").filter(|name| !name.is_empty());
    let compute = || -> Result<Out, Error> {
        let parts = scan_lines(&sources(client, &source)?, |plane, index| {
            let (mut durations, mut line) = (Vec::new(), None);
            if plane.name.starts_with("/device:") {
                visit_line(plane, index, |visit| {
                    if *line.get_or_insert_with(|| visit.line.to_uppercase().contains("XLA MODULES")) && function.as_ref().is_none_or(|function| visit.raw_name().contains(function.as_str())) {
                        durations.push(visit.duration_ns() / 1_000_000.0);
                    }
                    true
                });
            }
            durations
        })?;
        let durations: Vec<f64> = parts.into_iter().flatten().collect();
        if durations.is_empty() {
            return fail(Kind::Value, format!("No steps matching func_name '{}' found in {source}.", args.get("func_name").map_or("None".into(), J::text)));
        }
        let report = obj! {"avg_step_time_ms" => round(fsum(durations.iter().copied()) / durations.len() as f64, 4), "step_count" => durations.len()};
        Ok(if args.string("output_format", "json") == "dict" { Out::Value(report) } else { report.into() })
    };
    compute().map_err(|error| rethrow(error, |error| format!("Failed to get average step time: {}", error.message)))
}

pub fn get_kernel_utilization(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    if session.is_empty() {
        return fail(Kind::Value, "session_id or raw_bytes must be provided.");
    }
    let mut params: Vec<(&str, String)> = Vec::new();
    params.extend(args.text("kernel_name").filter(|name| !name.is_empty()).map(|name| ("kernel", name)));
    params.extend(args.get("duration_us").map(|value| ("duration_us", value.text())));
    params.extend(args.flag("force_duration", false).then(|| ("force_duration", "True".to_string())));
    params.extend(args.get("device").map(|value| ("device_id", value.text())));
    params.push(bypass(args.flag("bypass_cache", false)));
    let text = if session.starts_with('/') || session.starts_with("./") {
        let path = Path::new(&session);
        if !path.exists() {
            return fail(Kind::FileNotFound, format!("Path does not exist: {}", py_repr(&session)));
        }
        let files = if path.is_dir() { traces(path) } else { vec![path.to_path_buf()] };
        if files.is_empty() {
            return fail(Kind::FileNotFound, format!("No .xplane.pb or .xspace.pb files found in directory: {}", py_repr(&session)));
        }
        let option = |key: &str| params.iter().find(|(name, _)| *name == key).map(|(_, value)| value.as_str());
        let rendered = <[PathBuf; 1]>::try_from(files).ok().and_then(|[file]| super::client::kernel_utilization(&file, option));
        rendered.ok_or_else(|| Error::new(Kind::Runtime, format!("Failed to compute utilization from file {}: no data returned.", py_repr(&session))))?
    } else {
        let fetched = client
            .fetch_text("kernel_utilization.json", &session, &params)
            .map_err(|error| Error::new(Kind::Runtime, format!("Error fetching kernel_utilization.json for session {}: {}", py_repr(&session), error.repr())))?;
        fetched.ok_or_else(|| Error::new(Kind::FileNotFound, format!("No utilization data returned for session {}.", py_repr(&session))))?
    };
    if args.string("output_format", "json") == "dict" {
        return Ok(Out::Value(J::parse(&text).ok_or_else(|| Error::new(Kind::Value, "Expecting value: line 1 column 1 (char 0)"))?));
    }
    Ok(Out::Text(text))
}

fn llo(client: &dyn Client, args: &Args, failure: &str) -> Result<Out, Error> {
    let session = args.session();
    let path_like = Path::new(&session).exists() || session.starts_with(['/', '.', '\\']) || TRACE_SUFFIXES.iter().any(|suffix| session.ends_with(suffix));
    let compute = || -> Result<(), Error> {
        if path_like {
            return Ok(());
        }
        let hosts = client.hosts(&session)?;
        let host = match args.string("host", "") {
            host if host.is_empty() => hosts.first().cloned().unwrap_or_default(),
            host if !hosts.contains(&host) => {
                return fail(Kind::Value, format!("Invalid host: '{host}'. Available hosts: [{}]", hosts.iter().map(|host| py_repr(host)).collect::<Vec<_>>().join(", ")));
            }
            host => host,
        };
        serialized(client, &session, &host).map(drop)
    };
    compute().map_err(|error| rethrow(error, |error| format!("Error analyzing LLO data: {}", error.message)))?;
    Ok(obj! {"status" => "UNAVAILABLE", "reason" => "LLO_DATA_ABSENT", "error" => failure, "remediation" => LLO_REMEDIATION}.into())
}

pub fn get_llo_analysis(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    llo(client, args, "Failed to analyze LLO from xspace (LLO trace data is not available in this session).")
}

pub fn get_llo_debug_string(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    llo(client, args, "Failed to extract LLO debug string (LLO trace data is not available in this session).")
}

fn resolve(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut existing = absolute.as_path();
    let mut rest = Vec::new();
    while !existing.exists() {
        let (Some(parent), Some(name)) = (existing.parent(), existing.file_name()) else { break };
        rest.push(name.to_os_string());
        existing = parent;
    }
    let mut resolved = existing.canonicalize().unwrap_or_else(|_| existing.to_path_buf());
    for name in rest.into_iter().rev() {
        if name == ".." {
            resolved.pop();
        } else if name != "." {
            resolved.push(name);
        }
    }
    resolved
}

pub fn upload_trace(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let Some(logdir) = client.logdir() else { return fail(Kind::Value, "Logdir not set in client. Provide logdir.") };
    let file = args.string("file_path", "");
    let source = Path::new(&file);
    if !source.exists() {
        return fail(Kind::FileNotFound, format!("Source trace file '{file}' does not exist."));
    }
    let name = source.file_name().unwrap_or_default().to_string_lossy().into_owned();
    if !TRACE_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)) {
        return fail(Kind::Value, format!("Unsupported file format '{name}'. Only XSpace formats (.xplane.pb, .xspace.pb) are supported by XProf."));
    }
    let run = args.text("run_name").filter(|run| !run.is_empty()).unwrap_or_else(|| "imported_trace".into());
    if run.contains(['/', '\\']) || run.contains("..") || run.starts_with('/') {
        return fail(Kind::Value, format!("Invalid run_name '{run}': must be a single directory name without path separators or traversal components."));
    }
    let base = resolve(&logdir.join("plugins/profile"));
    let destination = resolve(&base.join(&run));
    if !destination.starts_with(&base) || destination == base {
        return fail(Kind::Value, format!("Invalid run_name '{run}': resolves outside the logdir profile hierarchy."));
    }
    let io = |error: std::io::Error| Error::new(if error.kind() == std::io::ErrorKind::NotFound { Kind::FileNotFound } else { Kind::Os }, error.to_string());
    std::fs::create_dir_all(&destination).map_err(io)?;
    let target = destination.join(&name);
    let identity = |path: &Path| std::fs::metadata(path).ok().map(|meta| (meta.dev(), meta.ino()));
    if identity(source).is_some_and(|source| identity(&target) == Some(source)) {
        return fail(Kind::Os, format!("PosixPath({}) and PosixPath({}) are the same file", py_repr(&file), py_repr(&target.to_string_lossy())));
    }
    std::fs::copy(source, &target).map_err(io)?;
    Ok(obj! {
        "status" => "success",
        "message" => format!("Successfully imported trace to run '{run}'"),
        "run_name" => run,
        "run_path" => destination.to_string_lossy().into_owned(),
        "imported_file" => target.to_string_lossy().into_owned(),
    }
    .into())
}

pub fn verify_numerical_parity(_: &dyn Client, _: &Args) -> Result<Out, Error> {
    fail(Kind::Import, NUMERICAL_DEPENDENCIES)
}
