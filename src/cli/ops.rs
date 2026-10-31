use super::client::Client;
use super::json::J;
use super::{Args, Error, Kind, Out, bypass, fail, fsum, rethrow, round};
use crate::obj;
use rayon::prelude::*;
use regex::Regex;
use std::cmp::Ordering;
use std::sync::LazyLock;

const VIEWS: [&str; 5] = ["grouped", "category", "summary", "flat", "tree"];
const SORTS: [&str; 3] = ["time", "flops", "bytes"];
const NAME_CATEGORIES: [&str; 10] = [
    "custom-call",
    "fusion",
    "convolution",
    "dot",
    "reduce",
    "copy",
    "reshape",
    "broadcast",
    "while",
    "tuple",
];
const COLLECTIVES: [&str; 4] = ["all-gather", "all-reduce", "reduce-scatter", "collective"];
const CUSTOM_CALLS: [&str; 2] = ["custom-call", "custom_call"];
const NO_LABEL: [&str; 4] = ["IDLE", "idle", "unknown", ""];
const SUMMARY_LIMIT: usize = 10;
const NO_PROFILE: &str = "No HLO op_profile found in trace. For JAX traces, ensure compilation is captured in the trace or pass XLA_FLAGS='--xla_dump_to=<path> --xla_dump_hlo_as_proto'.";
/// The default `max_recursion_depth` of `json_format.Parse` in protobuf.
const MAX_MESSAGE_DEPTH: usize = 100;
const TOO_DEEP: &str = "Message too deep. Max recursion depth is 100";
static TARGET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"custom_call_target="([^"]+)""#).expect("regular expression is valid")
});
static OP_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"%([^%=]+) =").expect("regular expression is valid"));

pub fn field<'a>(node: &'a J, name: &str) -> &'a J {
    let camel = |key: &str| {
        let mut after = false;
        key.chars().eq(name.chars().filter_map(|letter| {
            if letter == '_' {
                after = true;
                None
            } else {
                Some(if std::mem::take(&mut after) {
                    letter.to_ascii_uppercase()
                } else {
                    letter
                })
            }
        }))
    };
    node.entries()
        .iter()
        .find(|(key, _)| camel(key))
        .or_else(|| node.entries().iter().find(|(key, _)| key == name))
        .map_or(&J::Null, |(_, value)| value)
}

fn number(node: &J, name: &str) -> f64 {
    field(field(node, "metrics"), name).float().unwrap_or(0.0)
}

fn integer(node: &J, name: &str) -> i128 {
    field(field(node, "metrics"), name).int().unwrap_or(0)
}

fn bytes(node: &J) -> J {
    total(
        field(field(node, "metrics"), "raw_bytes_accessed_array")
            .items()
            .iter()
            .map(|value| J::Float(value.float().unwrap_or(0.0))),
    )
}

fn total(values: impl Iterator<Item = J>) -> J {
    let values: Vec<J> = values.collect();
    if values.iter().all(|value| matches!(value, J::Int(_))) {
        J::Int(values.iter().filter_map(J::int).sum())
    } else {
        J::Float(fsum(
            values.iter().map(|value| value.float().unwrap_or(0.0)),
        ))
    }
}

fn label(node: &J, default: &str) -> String {
    Some(field(node, "name"))
        .filter(|name| **name != J::Null)
        .map(J::text)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| default.into())
}

fn children(node: &J) -> &[J] {
    field(node, "children").items()
}

fn present(node: &J, name: &str) -> bool {
    *field(node, name) != J::Null
}

fn descending<T>(items: &mut [T], key: impl Fn(&T) -> f64) {
    items.sort_by(|left, right| {
        key(right)
            .partial_cmp(&key(left))
            .unwrap_or(Ordering::Equal)
    });
}

/// Frees the profile on another thread, so the caller does not wait for it.
struct Parsed(J);

impl std::ops::Deref for Parsed {
    type Target = J;
    fn deref(&self) -> &J {
        &self.0
    }
}

impl Drop for Parsed {
    fn drop(&mut self) {
        crate::release(std::mem::take(&mut self.0));
    }
}

fn too_deep() -> Error {
    Error::new(
        Kind::Value,
        format!("Failed to parse op_profile proto: ParseError('{TOO_DEEP}')"),
    )
}

fn profile(
    client: &dyn Client,
    session: &str,
    params: &[(&str, String)],
    missing: String,
    wrap: &str,
) -> Result<Parsed, Error> {
    let fetched = (|| {
        let parsed = client
            .op_profile(session, params)?
            .ok_or_else(|| Error::new(Kind::FileNotFound, missing))?;
        if parsed.object_depth() > MAX_MESSAGE_DEPTH {
            Err(too_deep())
        } else {
            Ok(parsed)
        }
    })();
    fetched
        .map(Parsed)
        .map_err(|error| rethrow(error, |error| format!("{wrap}{}", error.repr())))
}

pub fn get_profile_summary(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let params: Vec<(&str, String)> = vec![
        ("format", "json".to_string()),
        bypass(args.flag("bypass_cache", false)),
    ];
    let profile = profile(
        client,
        &session,
        &params,
        format!("No HLO op_profile data found in profile for session {session}."),
        &format!("Error analyzing profile for session {session}: "),
    )?;
    let by_category = field(&profile, "by_category");
    let root = if number(by_category, "raw_time") > 0.0 {
        by_category
    } else {
        field(&profile, "by_program")
    };
    let total = number(root, "raw_time");
    let mut lines = vec![format!("Profile Summary for {session}")];
    if total > 0.0 {
        lines.push(format!("Total Time: {:.4} s", total / 1e12));
    }
    lines.extend([
        "\nTop Operations (by self time):".to_string(),
        "| Name | Self Time (s) | Fraction |".into(),
        "|---|---|---|".into(),
    ]);
    let mut nodes = Vec::new();
    let mut pending = vec![root];
    while let Some(node) = pending.pop().filter(|_| nodes.len() < SUMMARY_LIMIT) {
        if number(node, "raw_time") > 0.0 {
            nodes.push(node);
        }
        pending.extend(children(node).iter().rev());
    }
    descending(&mut nodes, |node| number(node, "raw_time"));
    for node in nodes {
        let name = label(node, "Unknown");
        let time = number(node, "raw_time");
        let fraction = if total == 0.0 { 0.0 } else { time / total };
        lines.push(format!(
            "| {} | {:.4} | {:.1}% |",
            name.replace('|', "\\|"),
            time / 1e12,
            fraction * 100.0
        ));
    }
    Ok(Out::Text(lines.join("\n")))
}

fn leaves(node: &J, prefix: &str, out: &mut Vec<J>) {
    let name = field(node, "name").str().unwrap_or_default();
    let full = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    };
    if !children(node).is_empty()
        && fsum(children(node).iter().map(|child| number(child, "raw_time"))) > 0.0
    {
        for child in children(node) {
            leaves(child, &full, out);
        }
        return;
    }
    if number(node, "raw_time") <= 0.0 {
        return;
    }
    let xla = field(node, "xla");
    let category = if present(node, "xla") && field(xla, "category").truthy() {
        field(xla, "category").text()
    } else if present(node, "category") {
        format!("Category: {name}")
    } else {
        let lower = name.to_lowercase();
        match NAME_CATEGORIES
            .iter()
            .find(|category| lower.contains(*category))
        {
            Some(category) => category.to_string(),
            None if name.starts_with('%') => name
                .trim_start_matches('%')
                .split('.')
                .next()
                .unwrap_or_default()
                .split('_')
                .next()
                .unwrap_or_default()
                .to_string(),
            None => "unknown".into(),
        }
    };
    let occurrences = integer(node, "occurrences");
    let mut item = obj! {
        "name" => full,
        "category" => category,
        "total_self_time_ms" => round(number(node, "raw_time") / 1e9, 4),
        "occurrences" => if occurrences > 0 { occurrences } else { 1 },
        "flops" => number(node, "raw_flops"),
        "bytes_accessed" => bytes(node),
    };
    let source = field(xla, "source_info");
    if present(node, "xla") && present(xla, "source_info") {
        if field(source, "file_name").truthy() {
            item.set("source_file", field(source, "file_name"));
        }
        if field(source, "line_number").int().unwrap_or(0) > 0 {
            item.set("source_line", field(source, "line_number").int());
        }
        if field(source, "stack_frame").truthy() {
            item.set("stack_frame", field(source, "stack_frame"));
        }
    }
    out.push(item);
}

fn tree(node: &J, depth: i64, limit: f64, prefix: &str) -> J {
    let name = label(node, "root");
    let path = if prefix.is_empty() {
        name.clone()
    } else {
        format!("{prefix}/{name}")
    };
    let mut sorted: Vec<&J> = children(node).iter().collect();
    descending(&mut sorted, |child| number(child, "raw_time"));
    let built: Vec<J> = if (depth as f64) < limit {
        sorted
            .into_iter()
            .filter(|child| number(child, "raw_time") > 0.0)
            .map(|child| tree(child, depth + 1, limit, &path))
            .collect()
    } else {
        Vec::new()
    };
    let mut result = obj! {"name" => name, "path" => path, "total_self_time_ms" => round(number(node, "raw_time") / 1e9, 4)};
    if present(node, "xla") {
        if field(field(node, "xla"), "category").truthy() {
            result.set("category", field(field(node, "xla"), "category"));
        }
        if number(node, "raw_flops") > 0.0 {
            result.set("flops", number(node, "raw_flops"));
        }
    }
    if !built.is_empty() {
        result.set("children", built);
        result.set("child_count", children(node).len());
        result.set("has_children", true);
    } else if !children(node).is_empty() {
        result.set("has_children", true);
        result.set("child_count", children(node).len());
    }
    result
}

pub fn get_hlo_op_profile(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let lower = |name: &str, default: &str| match args.raw(name) {
        None => Ok(default.to_string()),
        Some(J::Str(text)) => Ok(text.clone()),
        Some(value) => Err(super::no_attribute(value, "lower")),
    };
    let view = lower("view", "grouped")?;
    let view_key = view.trim().to_lowercase();
    if !VIEWS.contains(&view_key.as_str()) {
        return fail(
            Kind::Value,
            format!(
                "Invalid view mode '{view}'. Must be one of: 'grouped', 'category', 'summary', 'flat', 'tree'."
            ),
        );
    }
    let sort = lower("sort_by", "time")?;
    let sort_key = sort.trim().to_lowercase();
    if !SORTS.contains(&sort_key.as_str()) {
        return fail(
            Kind::Value,
            format!("Invalid sort_by '{sort}'. Must be one of: 'time', 'flops', 'bytes'."),
        );
    }
    let params: Vec<(&str, String)> = vec![
        ("format", "json".to_string()),
        bypass(args.flag("bypass_cache", false)),
    ];
    let wrap = format!("Error fetching HLO op profile for session {session}: ");
    let profile = profile(
        client,
        &session,
        &params,
        format!("No HLO op_profile found in trace for session {session}."),
        &wrap,
    )?;
    let wrapped = |error: Error| Error::new(Kind::Runtime, format!("{wrap}{}", error.repr()));
    // Python reads `top_n` only when the view uses it. A float is good for a comparison but not for a slice.
    let top_n = |sliced: fn(f64) -> bool| {
        let top_n = args.compared("top_n", 15.0, ">").map_err(wrapped)?;
        if matches!(args.raw("top_n"), Some(J::Float(_))) && sliced(top_n) {
            return Err(wrapped(Error::new(Kind::Type, super::SLICE)));
        }
        Ok(top_n)
    };
    let no_ops = || {
        fail(
            Kind::FileNotFound,
            format!("No HLO op_profile operations found in profile for session {session}."),
        )
    };
    let root = ["by_category", "by_program"]
        .iter()
        .map(|key| field(&profile, key))
        .find(|node| **node != J::Null && number(node, "raw_time") > 0.0);
    let Some(root) = root else { return no_ops() };
    let total_ms = number(root, "raw_time") / 1e9;
    let mut flat = Vec::new();
    leaves(root, "", &mut flat);
    if flat.is_empty() {
        return no_ops();
    }
    let key = |op: &J| match sort_key.as_str() {
        "flops" => op.at("flops").float().unwrap_or(0.0),
        "bytes" => op.at("bytes_accessed").float().unwrap_or(0.0),
        _ => op.at("total_self_time_ms").float().unwrap_or(0.0),
    };
    let mut groups: indexmap::IndexMap<String, Vec<J>> = indexmap::IndexMap::new();
    for op in &flat {
        groups
            .entry(op.at("category").text())
            .or_default()
            .push(op.clone());
    }
    let time_of = |ops: &[J]| {
        fsum(
            ops.iter()
                .map(|op| op.at("total_self_time_ms").float().unwrap_or(0.0)),
        )
    };
    let fraction = |part: f64, whole: f64| {
        if whole > 0.0 {
            round(part / whole, 4)
        } else {
            0.0
        }
    };
    let sorted = |ops: &[J]| {
        let mut ops = ops.to_vec();
        descending(&mut ops, key);
        ops
    };
    let top_name = |ops: &[J]| {
        ops.first().map_or(String::new(), |op| {
            op.at("name")
                .text()
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_string()
        })
    };
    let mut summary: Vec<J> = groups
        .iter()
        .map(|(category, ops)| {
            obj! {
                "category" => category,
                "total_self_time_ms" => round(time_of(ops), 4),
                "fraction_of_total_time" => fraction(time_of(ops), total_ms),
                "total_flops" => total(ops.iter().map(|op| op.at("flops").clone())),
                "bytes_accessed" => total(ops.iter().map(|op| op.at("bytes_accessed").clone())),
                "op_count" => ops.len(),
                "top_op" => top_name(&sorted(ops)),
            }
        })
        .collect();
    let summary_key = match sort_key.as_str() {
        "flops" => "total_flops",
        "bytes" => "bytes_accessed",
        _ => "total_self_time_ms",
    };
    descending(&mut summary, |entry| {
        entry.at(summary_key).float().unwrap_or(0.0)
    });
    let available: Vec<String> = summary
        .iter()
        .map(|entry| entry.at("category").text())
        .collect();
    let ops_of = |category: &str| groups.get(category).map_or(&[][..], Vec::as_slice);
    let with_fraction = |ops: Vec<J>, whole: f64| -> Vec<J> {
        ops.into_iter()
            .map(|mut op| {
                let share = fraction(op.at("total_self_time_ms").float().unwrap_or(0.0), whole);
                op.set("category_fraction", share);
                op
            })
            .collect()
    };
    if let Some(category) = args.get("category") {
        let J::Str(category) = category else {
            return Err(wrapped(super::no_attribute(category, "strip")));
        };
        let target = category.trim().to_lowercase();
        let matched = available
            .iter()
            .find(|name| name.to_lowercase() == target)
            .or_else(|| {
                available
                    .iter()
                    .find(|name| name.to_lowercase().contains(&target))
            });
        let Some(matched) = matched else {
            return fail(
                Kind::FileNotFound,
                format!(
                    "Category '{category}' not found in HLO op profile. Available categories: [{}]",
                    available
                        .iter()
                        .map(|name| super::json::py_repr(name))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        };
        let ops = ops_of(matched);
        let matched_time = time_of(ops);
        let top_n = top_n(|top_n| top_n > 0.0)?;
        let limit = if top_n > 0.0 {
            top_n as usize
        } else {
            ops.len()
        };
        let selected: Vec<J> = sorted(ops).into_iter().take(limit).collect();
        let operations = with_fraction(selected, matched_time);
        let top = top_name(&operations);
        return Ok(obj! {
            "category" => matched,
            "total_self_time_ms" => round(matched_time, 4),
            "fraction_of_total_time" => fraction(matched_time, total_ms),
            "operations" => operations,
            "navigation_hints" => obj! {
                "inspect_top_op_ast" => format!("xprof get_hlo_neighborhood <trace> --instruction_name='{top}'"),
                "inspect_graph" => format!("xprof get_graph_viewer <trace> --node_name='{top}'"),
                "back_to_categories" => "xprof get_hlo_op_profile <trace> --view=category",
            },
        }
        .into());
    }
    match view_key.as_str() {
        "flat" => {
            let top_n = top_n(|top_n| top_n > 0.0)?;
            let limit = if top_n > 0.0 {
                top_n as usize
            } else {
                flat.len()
            };
            Ok(J::List(sorted(&flat).into_iter().take(limit).collect()).into())
        }
        "category" | "summary" => {
            let top = available.first().cloned().unwrap_or_default();
            let mut hints = obj! {
                "drill_down_into_top_category" => format!("xprof get_hlo_op_profile <trace> --category='{top}'"),
                "view_grouped_ops" => "xprof get_hlo_op_profile <trace> --view=grouped",
                "available_categories" => available.clone(),
            };
            if let Some(communication) = available.iter().find(|name| {
                COLLECTIVES
                    .iter()
                    .any(|kind| name.to_lowercase().contains(kind))
            }) {
                hints.set(
                    "drill_down_into_communication",
                    format!("xprof get_hlo_op_profile <trace> --category='{communication}'"),
                );
            }
            Ok(obj! {"total_profile_time_ms" => round(total_ms, 4), "category_summary" => summary, "navigation_hints" => hints}.into())
        }
        "tree" => tree_view(root, args, wrapped),
        _ => {
            let top_n = top_n(|top_n| top_n > 0.0 && top_n <= 5.0)?;
            let per_category = if top_n <= 0.0 {
                usize::MAX
            } else {
                top_n.min(5.0) as usize
            };
            let grouped = summary.iter().map(|entry| {
                let category = entry.at("category").text();
                let ops: Vec<J> = sorted(ops_of(&category))
                    .into_iter()
                    .take(per_category)
                    .collect();
                (
                    category,
                    J::from(with_fraction(
                        ops,
                        entry.at("total_self_time_ms").float().unwrap_or(0.0),
                    )),
                )
            });
            Ok(obj! {
                "total_profile_time_ms" => round(total_ms, 4),
                "category_summary" => summary.clone(),
                "grouped_operations" => J::Map(grouped.collect()),
                "navigation_hints" => obj! {
                    "drill_down_category" => "xprof get_hlo_op_profile <trace> --category='<category_name>'",
                    "inspect_op_neighborhood" => "xprof get_hlo_neighborhood <trace> --instruction_name='<op_name>'",
                    "inspect_roofline" => "xprof get_roofline_model <trace>",
                    "explore_tree" => "xprof get_hlo_op_profile <trace> --view=tree --path='by_category' --depth=2",
                    "available_categories" => available,
                },
            }
            .into())
        }
    }
}

fn tree_view(root: &J, args: &Args, wrapped: impl Fn(Error) -> Error) -> Result<Out, Error> {
    let root_name = label(root, "root");
    let (mut target, mut target_path) = (root, root_name.clone());
    if let Some(path) = args.get("path").filter(|path| path.truthy()) {
        let J::Str(path) = path else {
            return Err(wrapped(super::no_attribute(path, "strip")));
        };
        let mut parts: Vec<&str> = path
            .trim_matches('/')
            .split('/')
            .filter(|part| !part.is_empty())
            .collect();
        if parts.first().is_some_and(|first| {
            [
                "root",
                "by_category",
                "by_program",
                root_name.to_lowercase().as_str(),
            ]
            .contains(&first.to_lowercase().as_str())
        }) {
            parts.remove(0);
        }
        let mut matched = vec![root_name];
        for part in parts {
            let lower = part.to_lowercase();
            let found = children(target).iter().find(|child| {
                let name = field(child, "name")
                    .str()
                    .unwrap_or_default()
                    .to_lowercase();
                name == lower || name.contains(&lower)
            });
            let Some(found) = found else {
                return fail(
                    Kind::FileNotFound,
                    format!("Path '{path}' not found in HLO op profile tree."),
                );
            };
            target = found;
            matched.push(label(found, "node"));
        }
        target_path = matched.join("/");
    }
    let depth = args.compared("depth", 2.0, ">").map_err(&wrapped)?;
    let limit = if depth > 0.0 {
        args.raw("depth").cloned().unwrap_or(J::Int(2))
    } else {
        J::Int(2)
    };
    let child_paths: Vec<String> = children(target)
        .iter()
        .filter(|child| number(child, "raw_time") > 0.0)
        .map(|child| {
            format!(
                "{target_path}/{}",
                field(child, "name").str().unwrap_or_default()
            )
        })
        .collect();
    let hint = child_paths.first().map_or_else(
        || "No further child paths.".to_string(),
        |first| {
            format!(
                "xprof get_hlo_op_profile <trace> --view=tree --path='{first}' --depth={}",
                limit.text()
            )
        },
    );
    Ok(obj! {
        "current_path" => target_path,
        "depth_limit" => &limit,
        "total_self_time_ms" => round(number(target, "raw_time") / 1e9, 4),
        "tree" => tree(target, 1, limit.float().unwrap_or_default(), ""),
        "navigation_hints" => obj! {
            "navigate_deeper_into_child" => hint,
            "switch_to_program_tree" => "xprof get_hlo_op_profile <trace> --view=tree --path='by_program' --depth=2",
            "available_child_paths" => child_paths,
        },
    }
    .into())
}

fn instructions(node: &J, prefix: &str, out: &mut Vec<J>) {
    let name = field(node, "name").str().unwrap_or_default();
    let xla = field(node, "xla");
    if present(node, "xla") && number(node, "raw_time") > 0.0 {
        let category = field(xla, "category").str().unwrap_or_default();
        let custom = CUSTOM_CALLS.contains(&category.to_lowercase().as_str());
        let provenance = field(xla, "provenance").str().unwrap_or_default();
        let expression = field(xla, "expression").str().unwrap_or_default();
        let mut label = name.to_string();
        if !provenance.is_empty() {
            if (NO_LABEL.contains(&name) || custom) && name != provenance {
                label = format!("{name} [{provenance}]");
            }
        } else if !expression.is_empty()
            && custom
            && let Some(target) = TARGET
                .captures(expression)
                .map(|found| found[1].to_string())
            && name != target
        {
            label = format!("{name} [{target}]");
        }
        let mut item = obj! {
            "name" => if prefix.is_empty() { label } else { format!("{prefix}/{label}") },
            "category" => category,
            "total_self_time_ms" => number(node, "raw_time") / 1e9,
            "occurrences" => integer(node, "occurrences"),
            "flops" => number(node, "raw_flops"),
            "bytes_accessed" => bytes(node),
        };
        let source = field(xla, "source_info");
        if present(xla, "source_info") {
            item.set(
                "source_file",
                field(source, "file_name").str().unwrap_or_default(),
            );
            item.set(
                "source_line",
                field(source, "line_number").int().unwrap_or(0),
            );
            if field(source, "stack_frame").truthy() {
                item.set("stack_frame", field(source, "stack_frame"));
            }
        }
        out.push(item);
    }
    let child_prefix = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    };
    for child in children(node) {
        instructions(child, &child_prefix, out);
    }
}

pub fn get_top_hlo_ops(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let mut params = vec![("format", "pb".to_string())];
    params.extend(args.flag("bypass_cache", false).then(|| bypass(true)));
    let profile = match profile(
        client,
        &session,
        &params,
        NO_PROFILE.into(),
        &format!("Error fetching top HLO ops for session {session}: "),
    ) {
        Err(error) if error == too_deep() => {
            return Ok(obj! {"error" => format!("Failed to parse JSON proto: {TOO_DEEP}")}.into());
        }
        other => other?,
    };
    let by_category = field(&profile, "by_category");
    let mut flat = Vec::new();
    if *by_category != J::Null && number(by_category, "raw_time") > 0.0 {
        instructions(by_category, "", &mut flat);
    } else if present(&profile, "by_program") {
        instructions(field(&profile, "by_program"), "", &mut flat);
    }
    if let Some(filter) = args.get("category_filter").filter(|filter| filter.truthy()) {
        let J::Str(filter) = filter else {
            return Err(super::no_attribute(filter, "strip"));
        };
        let target = filter.trim().to_lowercase();
        flat.retain(|op| op.at("category").text().trim().to_lowercase() == target);
    }
    if flat.is_empty() {
        return Ok(obj! {"top_by_time" => J::List(Vec::new()), "top_by_flops" => J::List(Vec::new()), "top_by_bytes_accessed" => J::List(Vec::new()), "total_matched" => 0, "has_by_program" => present(&profile, "by_program")}.into());
    }
    let limit = args.compared("limit", 10.0, ">")?;
    if let Some(message) = super::nlargest_float(args.raw("limit"), limit, flat.len()) {
        return fail(Kind::Type, message);
    }
    let top = |key: &str| {
        let mut ops: Vec<&J> = flat.iter().collect();
        descending(&mut ops, |op| op.at(key).float().unwrap_or(0.0));
        if limit > 0.0 {
            ops.truncate(limit as usize);
        }
        ops.into_iter().cloned().collect::<Vec<J>>()
    };
    let (by_time, by_flops, by_bytes) = (
        top("total_self_time_ms"),
        top("flops"),
        top("bytes_accessed"),
    );
    let custom = by_time.iter().chain(&by_flops).chain(&by_bytes).any(|op| {
        CUSTOM_CALLS.contains(&op.at("category").text().to_lowercase().as_str())
            || op.at("name").text().to_lowercase().contains("custom-call")
    });
    let mut result = obj! {"top_by_time" => by_time, "top_by_flops" => by_flops, "top_by_bytes_accessed" => by_bytes, "total_matched" => flat.len()};
    if custom {
        result.set("guidance", super::overview::CUSTOM_CALL_GUIDANCE);
    }
    Ok(result.into())
}

fn safe_int(value: Option<&J>) -> i128 {
    match value {
        Some(J::Str(text)) if !text.contains('.') => super::json::py_int(text).unwrap_or(0),
        Some(value) => value.float().map_or(0, |number| {
            if number.is_finite() {
                number.trunc() as i128
            } else {
                0
            }
        }),
        None => 0,
    }
}

pub fn stats_records(table: &J, category_filter: Option<&str>) -> Vec<J> {
    let columns: Vec<String> = table
        .at("cols")
        .items()
        .iter()
        .enumerate()
        .map(|(index, column)| {
            Some(column.at("id"))
                .filter(|id| id.truthy())
                .or(column.get("label"))
                .map_or(format!("col_{index}"), J::text)
                .to_lowercase()
        })
        .collect();
    let records = table.at("rows").items().par_iter().filter_map(|row| {
        let cells = row.at("c").items();
        let cell = |keys: &[&str]| -> Option<&J> {
            keys.iter().find_map(|key| {
                let value = cells.get(columns.iter().position(|column| column == key)?)?;
                let value = if let J::Map(_) = value { value.get("v")? } else { value };
                (*value != J::Null && !value.str().is_some_and(|text| text.trim().is_empty())).then_some(value)
            })
        };
        let text = |keys: &[&str], default: &str| cell(keys).map_or_else(|| default.to_string(), J::text);
        let float = |keys: &[&str]| cell(keys).and_then(J::float).unwrap_or(0.0);
        let category = text(&["hlo_category", "category"], "");
        if category_filter.is_some_and(|filter| !category.to_lowercase().contains(&filter.trim().to_lowercase())) {
            return None;
        }
        let expression = text(&["hlo_op_expression", "hlo_expression", "expression"], "");
        let op_name = match OP_NAME.captures(&expression) {
            Some(found) => found[1].to_string(),
            None if !expression.is_empty() => expression.chars().take(80).collect(),
            None => text(&["hlo_op_name", "op_name"], ""),
        };
        let self_time_percent = match cell(&["total_self_time_as_fraction"]) {
            Some(value) => value.float().unwrap_or(0.0) * 100.0,
            None => float(&["total_self_time_percent", "self_time_percent", "self_time_pct"]),
        };
        let (mut source_file, mut source_line) = (text(&["source_file", "file_name"], ""), safe_int(cell(&["source_line", "line_number"])));
        if source_file.is_empty() {
            let info = text(&["source_info"], "");
            match info.rsplit_once(':') {
                Some((file, line)) => {
                    source_file = file.to_string();
                    if source_line == 0 && !line.is_empty() && line.chars().all(char::is_numeric) {
                        source_line = line.parse().unwrap_or(0);
                    }
                }
                None => source_file = info,
            }
        }
        Some(obj! {
            "rank" => safe_int(cell(&["rank"])),
            "program_id" => safe_int(cell(&["program_id"])),
            "category" => category,
            "op_name" => op_name.trim(),
            "tf_op_name" => text(&["tf_op_name", "framework_op_name"], "").trim(),
            "occurrences" => safe_int(cell(&["occurrences"])),
            "total_time_us" => float(&["total_time_in_us", "total_time_us", "total_time"]),
            "total_self_time_us" => float(&["total_self_time_in_us", "total_self_time_us", "total_self_time", "self_time"]),
            "self_time_percent" => self_time_percent,
            "measured_flop_rate" => float(&["measured_flop_rate", "model_flop_rate", "normalized_flop_rate", "flop_rate"]),
            "flops" => float(&["flops_v2", "flops"]),
            "measured_memory_bw_gbs" => float(&["measured_memory_bw", "hbm_bw", "memory_bw", "bandwidth"]),
            "bound_by" => text(&["bound_by"], "Unknown"),
            "source_file" => source_file,
            "source_line" => source_line,
        })
    });
    records.collect()
}

pub fn get_hlo_stats(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let mut params = vec![
        ("format", "json".to_string()),
        ("tqx", "out:pb".to_string()),
    ];
    params.extend(args.flag("bypass_cache", false).then(|| bypass(true)));
    let data = client
        .fetch("hlo_stats.json", &session, &params)
        .map_err(|error| {
            Error::new(
                Kind::Runtime,
                format!(
                    "Error fetching HLO stats for session {session}: {}",
                    error.repr()
                ),
            )
        })?;
    let data = data.ok_or_else(|| {
        Error::new(
            Kind::Runtime,
            "Unexpected data type returned: <class 'NoneType'>",
        )
    })?;
    let text = String::from_utf8_lossy(&data);
    let table = J::parse(text.trim())
        .filter(|table| table.has("cols"))
        .ok_or_else(|| {
            Error::new(
                Kind::Value,
                "Failed to parse HloStatsDatabase proto: ParseError('Failed to load JSON')",
            )
        })?;
    let filter = args.get("category_filter").filter(|filter| filter.truthy());
    if let Some(filter) =
        filter.filter(|filter| filter.str().is_none() && !table.at("rows").items().is_empty())
    {
        return Err(super::no_attribute(filter, "strip"));
    }
    let mut records = stats_records(&table, filter.and_then(J::str));
    crate::release(table);
    if records.is_empty() {
        return fail(Kind::FileNotFound, "No HLO stats records found");
    }
    let key = match args
        .stripped("sort_by", "self_time")?
        .trim()
        .to_lowercase()
        .as_str()
    {
        "total_time" => "total_time_us",
        "occurrences" => "occurrences",
        "flops" => "flops",
        "bandwidth" => "measured_memory_bw_gbs",
        _ => "total_self_time_us",
    };
    descending(&mut records, |record| record.at(key).float().unwrap_or(0.0));
    let limit = args.compared("limit", 20.0, ">")?;
    if limit > 0.0 && matches!(args.raw("limit"), Some(J::Float(_))) {
        return fail(Kind::Type, super::SLICE);
    }
    if limit > 0.0 && records.len() as f64 > limit {
        crate::release(records.split_off(limit as usize));
    }
    Ok(J::List(records).into())
}
