use super::client::{Client, find};
use super::json::J;
use super::{Args, Error, Kind, Out, bypass, fail, rethrow};
use crate::obj;
use indexmap::IndexMap;
use regex::Regex;
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::LazyLock;

const SUFFIX: &str = ".hlo_proto.pb";
const NO_MODULE: &str = "NO_MODULE.hlo_proto.pb";
const NO_HLO: &str = "No compiled HLO module proto found in profile session. To capture HLO graphs, export XLA_FLAGS='--xla_dump_to=<logdir> --xla_dump_hlo_as_proto' before profiling.";
const NO_GRAPH: &str = "No graph_viewer data found in profile session. Ensure workload was profiled with XLA_FLAGS='--xla_dump_to=<logdir> --xla_dump_hlo_as_proto'.";
const SUGGESTIONS: usize = 10;
const NEIGHBORHOOD_RADIUS: i64 = 2;
static COMPUTATION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:ENTRY\s+)?%?([a-zA-Z0-9._-]+)\b.*\{\s*$").unwrap());
static INSTRUCTION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^%?([a-zA-Z0-9._-]+)\s*=(.*)$").unwrap());
static METADATA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)metadata=\{.*?\}").unwrap());
static NUMBERED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^([a-zA-Z_][a-zA-Z0-9_]*)\.\d+$").unwrap());

pub fn proto_files(client: &dyn Client, session: &str) -> Result<Vec<PathBuf>, Error> {
    let run = client.run_dir(session)?;
    if find(&run, &[SUFFIX], true).is_empty() {
        let paths = client.xspace_paths(&run)?;
        crate::hlo::extract(paths[0].parent().unwrap_or(&run), &paths);
    }
    let mut files = find(&run, &[SUFFIX], false);
    if files.is_empty() {
        files = find(&run, &[SUFFIX], true);
    }
    files.retain(|path| !path.file_name().is_some_and(|name| name == NO_MODULE));
    Ok(files)
}

fn module_names(files: &[PathBuf]) -> Vec<String> {
    files.iter().map(|path| path.file_name().unwrap_or_default().to_string_lossy().trim_end_matches(SUFFIX).to_string()).collect()
}

pub fn list_hlo_modules(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    Ok(Out::Text(match proto_files(client, &args.session()) {
        Err(error) => format!("Error listing HLO modules: {}", error.repr()),
        Ok(files) if files.is_empty() => "No HLO modules found. Ensure you have run a compilation or imported traces with HLO.".into(),
        Ok(files) => {
            let names = module_names(&files);
            std::iter::once(format!("Found {} HLO modules:", names.len())).chain(names.iter().enumerate().map(|(index, name)| format!("{index}. {name}"))).collect::<Vec<_>>().join("\n")
        }
    }))
}

fn target_module(client: &dyn Client, session: &str, module: Option<String>) -> Result<Result<String, String>, Error> {
    let files = proto_files(client, session)?;
    if files.is_empty() {
        return Ok(Err("No HLO proto found.".into()));
    }
    let available = module_names(&files);
    Ok(match module.filter(|module| !module.is_empty()) {
        Some(module) if !available.contains(&module) => Err(format!("Module '{module}' not found. Available: {}", available.join(", "))),
        Some(module) => Ok(module),
        None => Ok(available[0].clone()),
    })
}

fn hlo_text(client: &dyn Client, session: &str, module: &str, metadata: bool) -> Result<String, Error> {
    let params = [("type", if metadata { "long_txt" } else { "short_txt" }.to_string()), ("module_name", module.to_string())];
    client.fetch_text("graph_viewer.json", session, &params)?.ok_or_else(|| Error::new(Kind::Runtime, "'NoneType' object has no attribute 'splitlines'"))
}

pub fn module_content(client: &dyn Client, session: &str, format: &str, module: Option<String>, max_lines: i64, metadata: bool) -> String {
    let compute = || -> Result<String, Error> {
        let module = match target_module(client, session, module)? {
            Ok(module) => module,
            Err(message) => return Ok(message),
        };
        if format != "text" {
            return Ok(format!("Unsupported format: {format}"));
        }
        let text = hlo_text(client, session, &module, metadata)?;
        let lines: Vec<&str> = text.lines().collect();
        Ok(if max_lines > 0 && lines.len() > max_lines as usize {
            format!("{}\n... (truncated after {max_lines} lines, total {}). Use 'max_lines=-1' to see all)", lines[..max_lines as usize].join("\n"), lines.len())
        } else {
            text
        })
    };
    compute().unwrap_or_else(|error| format!("Error fetching HLO module content: {}", error.repr()))
}

pub fn get_hlo_module_content(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    Ok(Out::Text(module_content(client, &args.session(), &args.string("fmt", "text"), args.text("module_name"), args.int("max_lines", 2000)?, args.flag("print_metadata", false))))
}

fn operands(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let named = |byte: u8| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-');
    let attempt = |mut at: usize| {
        at += usize::from(bytes.get(at) == Some(&b'%'));
        let start = at;
        while bytes.get(at).is_some_and(|byte| named(*byte)) {
            at += 1;
        }
        (at > start && bytes.get(at).is_none_or(|byte| byte.is_ascii_whitespace() || matches!(byte, b',' | b')' | 0x0b))).then_some((start, at))
    };
    let (mut found, mut position) = (Vec::new(), 0);
    while position <= bytes.len() {
        let delimited = bytes.get(position).filter(|byte| byte.is_ascii_whitespace() || matches!(byte, b',' | b'(' | 0x0b)).and_then(|_| attempt(position + 1));
        match (position == 0).then(|| attempt(0)).flatten().or(delimited) {
            Some((start, end)) => {
                found.push(text[start..end].to_string());
                position = end;
            }
            None => position += 1,
        }
    }
    found
}

pub fn neighborhood(client: &dyn Client, session: &str, instruction: Option<String>, radius: impl Into<J>, module: Option<String>, op_name: Option<String>, metadata: bool) -> Result<String, Error> {
    if let (Some(instruction), Some(op_name)) = (&instruction, &op_name)
        && instruction != op_name
    {
        return fail(Kind::Value, format!("Conflicting arguments: instruction_name='{instruction}' and op_name='{op_name}' cannot both be specified with different values."));
    }
    let Some(target) = instruction.filter(|name| !name.is_empty()).or(op_name).filter(|name| !name.is_empty()) else {
        return fail(Kind::Value, "Either instruction_name or op_name must be provided to get_hlo_neighborhood.");
    };
    let target = target.strip_prefix('%').unwrap_or(&target).to_string();
    let radius = radius.into();
    let compute = || -> Result<String, Error> {
        let module = match target_module(client, session, module)? {
            Ok(module) => module,
            Err(message) => return Ok(message),
        };
        let text = hlo_text(client, session, &module, metadata)?;
        let (mut lines, mut operand_lists, mut users, mut computations) = (IndexMap::new(), IndexMap::<String, Vec<String>>::new(), IndexMap::<String, Vec<String>>::new(), IndexMap::new());
        let mut current = "unknown".to_string();
        for line in text.lines() {
            let stripped = line.trim();
            if let Some(found) = COMPUTATION.captures(stripped).filter(|_| !stripped.contains('=') && !stripped.starts_with("ROOT ")) {
                current = found[1].to_string();
                continue;
            }
            let clean = stripped.strip_prefix("ROOT ").unwrap_or(stripped);
            if let Some(found) = INSTRUCTION.captures(clean) {
                let name = found[1].to_string();
                lines.insert(name.clone(), stripped.to_string());
                computations.insert(name.clone(), current.clone());
                let listed = operands(&METADATA.replace_all(&found[2], ""));
                for operand in &listed {
                    users.entry(operand.clone()).or_default().push(name.clone());
                }
                operand_lists.insert(name, listed);
            }
        }
        if !lines.contains_key(&target) {
            let suggestions: Vec<&str> = lines.keys().take(SUGGESTIONS).map(String::as_str).collect();
            let hint = if suggestions.is_empty() { String::new() } else { format!(" Suggestions: {}", suggestions.join(", ")) };
            return Ok(format!("Instruction '{target}' not found in HLO module.{hint}"));
        }
        let (mut visited, mut queue, mut found) = (HashSet::from([target.clone()]), VecDeque::from([(target.clone(), 0)]), Vec::new());
        while let Some((name, distance)) = queue.pop_front() {
            let within = match &radius {
                J::Int(limit) => i128::from(distance) < *limit,
                J::Float(limit) => f64::from(distance) < *limit,
                other => return fail(Kind::Type, format!("'<' not supported between instances of 'int' and '{}'", super::kind(other))),
            };
            if within {
                for next in operand_lists.get(&name).into_iter().flatten().chain(users.get(&name).into_iter().flatten()) {
                    if lines.contains_key(next) && visited.insert(next.clone()) {
                        queue.push_back((next.clone(), distance + 1));
                    }
                }
            }
            found.push((distance, name));
        }
        found.sort();
        let header = format!("Neighborhood of '{target}' (radius={}):", radius.text());
        let body =
            found.iter().map(|(distance, name)| format!("{}[dist={distance}] [{}] {}", "  ".repeat(*distance as usize + 1), computations.get(name).map_or("unknown", String::as_str), lines[name]));
        Ok(std::iter::once(header).chain(body).collect::<Vec<_>>().join("\n"))
    };
    Ok(compute().unwrap_or_else(|error| format!("Error analyzing neighborhood: {}", error.repr())))
}

pub fn get_hlo_neighborhood(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let text = neighborhood(
        client,
        &args.session(),
        args.text("instruction_name"),
        args.get("radius").unwrap_or(&J::Int(NEIGHBORHOOD_RADIUS.into())),
        args.text("module_name"),
        args.text("op_name"),
        args.flag("print_metadata", false),
    )?;
    Ok(Out::Text(text))
}

pub fn get_hlo_text(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let compute = || -> Result<String, Error> {
        let text = match args.text("op_name").filter(|name| !name.is_empty()) {
            Some(op_name) => neighborhood(client, &session, Some(op_name), NEIGHBORHOOD_RADIUS, args.text("module_name"), None, false)?,
            None => module_content(client, &session, "text", args.text("module_name"), 2000, false),
        };
        if let Some(path) = args.text("path").filter(|path| !path.is_empty()) {
            let path = PathBuf::from(path);
            path.parent().filter(|parent| !parent.as_os_str().is_empty()).map(std::fs::create_dir_all).transpose().map_err(|error| Error::new(Kind::Os, error.to_string()))?;
            std::fs::write(&path, &text).map_err(|error| Error::new(Kind::Os, error.to_string()))?;
        }
        Ok(text)
    };
    Ok(Out::Text(compute().map_err(|_| Error::new(Kind::Runtime, "Error retrieving HLO text"))?))
}

pub fn get_graph_viewer(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let (session, symbol) = (args.session(), args.string("symbol_id", ""));
    if !session.is_empty() && !symbol.is_empty() {
        return fail(Kind::Value, "Cannot set both session_id and symbol_id");
    }
    if session.is_empty() && symbol.is_empty() {
        return fail(Kind::Value, "Either session_id or symbol_id must be provided");
    }
    let session = if symbol.is_empty() { session } else { "xsymbol".into() };
    let mut module = args.string("module_name", "");
    if symbol.is_empty() && module.is_empty() {
        module = proto_files(client, &session).ok().and_then(|files| module_names(&files).into_iter().next()).unwrap_or_default();
    }
    let mut options = vec![
        ("graph_type", args.string("graph_type", "xla")),
        ("type", args.string("output_type", "short_txt")),
        ("show_metadata", args.get("show_metadata").map_or("true".into(), |value| value.text().to_lowercase())),
    ];
    let int = |name: &str, default: i64| args.int(name, default);
    let (width, limit, xplane) = (int("graph_width", 1)?, int("op_profile_limit", 0)?, int("use_xplane", 0)?);
    let present = |name: &'static str, value: String| (!value.is_empty()).then_some((name, value));
    options.extend(args.flag("bypass_cache", false).then(|| bypass(true)));
    options.extend(present("symbol_id", symbol));
    options.extend(present("symbol_type", args.string("symbol_type", "")));
    options.extend(present("module_name", module));
    options.extend(present("node_name", args.string("node_name", "")));
    options.extend((width != 1).then(|| ("graph_width", width.to_string())));
    options.extend(args.get("merge_fusion").filter(|value| value.truthy()).map(|value| ("merge_fusion", value.text().to_lowercase())));
    options.extend(present("tag", args.string("tag", "")));
    options.extend(present("tool", args.string("tool", "")));
    options.extend((limit > 0).then(|| ("op_profile_limit", limit.to_string())));
    options.extend((xplane > 0).then(|| ("use_xplane", xplane.to_string())));
    let data = client.fetch_text("graph_viewer", &session, &options).map_err(|error| {
        if error.message.contains("Can not load hlo proto") || error.message.contains("No HLO") {
            Error::new(Kind::FileNotFound, NO_HLO)
        } else if error.kind == Kind::Value {
            error
        } else {
            Error::new(Kind::Runtime, format!("Error fetching data for graph_viewer: {}", error.repr()))
        }
    })?;
    Ok(Out::Text(data.ok_or_else(|| Error::new(Kind::FileNotFound, NO_GRAPH))?))
}

fn buffers(module: &J, min_size: f64, aggregate: bool, threshold: &J) -> Vec<(String, f64)> {
    let mut kept = Vec::new();
    let mut others = 0.0;
    let logical = module.at("bufferAssignment").at("logicalBuffers").items();
    let entries: Vec<(String, f64)> = if logical.is_empty() {
        module
            .at("maxHeap")
            .items()
            .iter()
            .map(|buffer| (buffer.get("instructionName").map_or("unknown".into(), J::text), buffer.get("logicalBufferSizeMib").map_or(Some(0.0), J::float).unwrap_or(0.0)))
            .collect()
    } else {
        logical
            .iter()
            .map(|buffer| (buffer.at("definedAt").get("instructionName").map_or("unknown".into(), J::text), buffer.get("size").map_or(Some(0), J::int).unwrap_or(0) as f64 / 1048576.0))
            .collect()
    };
    for (instruction, size) in entries {
        if aggregate && size < min_size {
            others += size;
        } else {
            kept.push((instruction, size));
        }
    }
    let descending = |items: &mut Vec<(String, f64)>| items.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(std::cmp::Ordering::Equal));
    if !aggregate {
        descending(&mut kept);
        return kept;
    }
    let mut groups: IndexMap<(String, u64), Vec<String>> = IndexMap::new();
    for (instruction, size) in kept {
        let base = NUMBERED.captures(&instruction).map_or_else(|| instruction.clone(), |found| format!("{}.*", &found[1]));
        groups.entry((base, size.to_bits())).or_default().push(instruction);
    }
    let mut aggregated: Vec<(String, f64)> = groups
        .into_iter()
        .map(|((base, bits), names)| {
            let size = f64::from_bits(bits);
            if names.len() > 1 { (format!("{base} ({} occurrences of size {} MiB)", names.len(), crate::hlo::general(size, 6)), size * names.len() as f64) } else { (names[0].clone(), size) }
        })
        .collect();
    if others > 0.0 {
        aggregated.push((format!("Others (< {} MiB)", threshold.text()), others));
    }
    descending(&mut aggregated);
    aggregated
}

pub fn get_peak_allocations(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let (limit, threshold) = (args.int("limit", 10)?, args.get("min_size_mib").cloned().unwrap_or(J::Float(1.0)));
    let min_size = threshold.float().ok_or_else(|| Error::new(Kind::Type, "'<' not supported between instances of 'float' and 'str'"))?;
    let (aggregate, summary) = (args.flag("aggregate_instructions", true), args.flag("include_summary", true));
    let params = [("format", "json".to_string())];
    let names: Vec<String> = match client.fetch_text("memory_viewer.json", &session, &params)? {
        Some(data) => {
            let names: Vec<String> = data.split(',').map(str::trim).filter(|name| !name.is_empty()).map(str::to_string).collect();
            if names.is_empty() {
                return fail(Kind::Value, "No HLO modules found in memory viewer data for the session");
            }
            names
        }
        None => match proto_files(client, &session) {
            Ok(files) if !files.is_empty() => module_names(&files),
            _ => return fail(Kind::Value, "No memory viewer data returned for the session"),
        },
    };
    let mut modules = Vec::new();
    for name in &names {
        let mut params = vec![("format", "json".to_string()), ("module_name", name.clone())];
        params.extend(args.flag("bypass_cache", false).then(|| bypass(true)));
        let data =
            client.fetch_text("memory_viewer.json", &session, &params).map_err(|error| rethrow(error, |error| format!("Failed to get peak allocations for session {session}: {}", error.message)))?;
        let Some(parsed) = data.and_then(|data| J::parse(&data)) else { continue };
        modules.push((name.clone(), parsed.get("totalBufferAllocationMib").cloned().unwrap_or(J::Int(0)), buffers(&parsed, min_size, aggregate, &threshold)));
    }
    let total = modules.len();
    modules.sort_by(|left, right| right.1.float().partial_cmp(&left.1.float()).unwrap_or(std::cmp::Ordering::Equal));
    if limit > 0 {
        modules.truncate(limit as usize);
    }
    if args.string("output_format", "json") == "markdown" {
        let mut lines = vec!["# Peak Memory Allocations by Module".to_string(), String::new()];
        if aggregate {
            lines.extend([
                "> [!NOTE]".to_string(),
                "> **Aggregation Logic:**".into(),
                "> - Buffers with similar names (e.g., `name.1`, `name.2`) and identical sizes are aggregated into `name.*`.".into(),
                "> - Buffers smaller than the threshold are aggregated into 'Others'.".into(),
                String::new(),
            ]);
        }
        if summary {
            lines.extend(["## Session Summary".to_string(), format!("- Total Modules: {total}")]);
            if !modules.is_empty() {
                lines.extend([String::new(), "| Module | Total HBM (MiB) |".into(), "| :--- | ---: |".into()]);
                lines.extend(modules.iter().map(|(name, hbm, _)| format!("| `{name}` | {:.2} |", hbm.float().unwrap_or(0.0))));
            }
            lines.push(String::new());
        }
        for (name, hbm, kept) in &modules {
            lines.extend([format!("## Module: `{name}`"), format!("Total HBM: {:.2} MiB", hbm.float().unwrap_or(0.0)), String::new(), "| Instruction | Size (MiB) |".into(), "| :--- | ---: |".into()]);
            lines.extend(kept.iter().map(|(instruction, size)| format!("| `{instruction}` | {size:.2} |")));
            lines.push(String::new());
        }
        return Ok(Out::Text(lines.join("\n")));
    }
    let listed: Vec<J> = modules
        .iter()
        .map(|(name, hbm, kept)| obj! {"module_name" => name, "total_hbm_mib" => hbm, "top_buffers" => kept.iter().map(|(instruction, size)| obj! {"instruction" => instruction, "size_mib" => *size}).collect::<Vec<_>>()})
        .collect();
    if summary {
        let top: Vec<J> = modules.iter().map(|(name, hbm, _)| obj! {"module_name" => name, "total_hbm_mib" => hbm}).collect();
        return Ok(obj! {"summary" => obj! {"total_modules" => total, "top_modules" => top}, "modules" => listed}.into());
    }
    Ok(J::List(listed).into())
}
