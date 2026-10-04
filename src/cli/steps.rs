use super::client::Client;
use super::json::J;
use super::ops::field;
use super::overview::utilization_viewer;
use super::{Args, Error, Kind, Out, bypass, fsum, rethrow, round, stdev};
use crate::obj;

const DEFAULT_STEP_LIMIT: i64 = 20;
const COMPUTE_COLUMNS: [&str; 5] = ["tccomputetimems", "scv0computetimems", "bccomputetimems", "devicecomputetimems", "noninfeedtimems"];
const COMMUNICATION_COLUMNS: [&str; 2] = ["devicetodevicetimems", "devicecollectivestimems"];
const INFEED_COLUMNS: [&str; 4] = ["tcinfeedtimems", "scv0infeedtimems", "bcinfeedtimems", "infeedtimems"];
const OUTFEED_COLUMNS: [&str; 2] = ["tcoutfeedtimems", "outfeedtimems"];
const OTHER_COLUMNS: [&str; 6] = ["tcidletimems", "hosttransfertimems", "hostcomputetimems", "kernellaunchtimems", "compiletimems", "othertimems"];
const STEP_COUNT_KEYS: [&str; 4] = ["total_steps", "num_steps", "step_count", "total_step_count"];
const ICI_PATTERNS: [&str; 6] = ["all-reduce", "all-gather", "all-to-all", "reduce-scatter", "collective-broadcast", "collective-permute"];
const HBM_PATTERNS: [&str; 3] = ["copy-start", "copy", "copy-done"];
const IDLE_RATIO_THRESHOLD: f64 = 10.0;
const MXU_IDLENESS_THRESHOLD: f64 = 70.0;
const HBM_THRESHOLD: f64 = 30.0;
const ICI_THRESHOLD: f64 = 30.0;
const DUTY_CYCLE_THRESHOLD: f64 = 50.0;
const MAX_UTILIZATION_HOSTS: i64 = 32;

fn finite(value: &J) -> f64 {
    value.float().filter(|number| number.is_finite()).unwrap_or(0.0)
}

fn percent(part: f64, whole: f64) -> f64 {
    if whole > 0.0 { round(part / whole * 100.0, 2) } else { 0.0 }
}

fn step(number: J, time: f64, [compute, communication, infeed, outfeed, idle]: [f64; 5], percents: [f64; 5], bottleneck: String, breakdown: Option<(f64, f64, f64)>) -> J {
    let mut out = obj! {
        "step_num" => number,
        "step_time_ms" => time,
        "compute_time_ms" => compute,
        "compute_percent" => percents[0],
        "communication_time_ms" => communication,
        "communication_percent" => percents[1],
        "infeed_time_ms" => infeed,
        "infeed_percent" => percents[2],
        "outfeed_time_ms" => outfeed,
        "outfeed_percent" => percents[3],
        "bottleneck" => bottleneck,
        "idle_time_ms" => idle,
        "idle_percent" => percents[4],
    };
    if let Some((all_reduce, send, recv)) = breakdown {
        out.set("communication_breakdown_ms", obj! {"all_reduce_ms" => all_reduce, "send_ms" => send, "recv_ms" => recv});
    }
    out
}

fn most_common(items: &[String]) -> Option<String> {
    let mut counts: Vec<(&String, usize)> = Vec::new();
    for item in items {
        match counts.iter_mut().find(|(name, _)| *name == item) {
            Some(entry) => entry.1 += 1,
            None => counts.push((item, 1)),
        }
    }
    counts.iter().fold(None, |best: Option<(&String, usize)>, &(name, count)| if best.is_none_or(|(_, top)| count > top) { Some((name, count)) } else { best }).map(|(name, _)| name.clone())
}

fn pod_viewer(raw: &str, step_num: Option<&J>, core: Option<&J>) -> Vec<J> {
    let Some(data) = J::parse(raw) else { return Vec::new() };
    let mut steps = Vec::new();
    let mut core_found = false;
    for entry in data.at("podStatsSequence").at("podStatsMap").items().iter().filter(|entry| matches!(entry, J::Map(_))) {
        let number = match entry.at("stepNum") {
            J::Str(text) => super::json::py_int(text).map_or_else(|| J::from(text), J::Int),
            J::Float(value) => J::Int(value.trunc() as i128),
            other => other.clone(),
        };
        if step_num.is_some_and(|wanted| number != *wanted) {
            continue;
        }
        let cores = entry.at("podStatsPerCore");
        let core_stats: Vec<&J> = match core {
            Some(core) => match cores.get(&core.text()) {
                Some(stats) => {
                    core_found = true;
                    vec![stats]
                }
                None => continue,
            },
            None => cores.entries().iter().map(|(_, stats)| stats).collect(),
        };
        if core_stats.is_empty() {
            continue;
        }
        let count = core_stats.len() as f64;
        let average = |key: &str| round(fsum(core_stats.iter().map(|stats| finite(stats.at(key)))) / count / 1000.0, 4);
        let breakdown = |key: &str| {
            let total: f64 = core_stats
                .iter()
                .map(|stats| Some(stats.at("stepBreakdownUs")).filter(|map| **map != J::Null).unwrap_or_else(|| stats.at("step_breakdown_us")))
                .filter(|map| matches!(map, J::Map(_)))
                .map(|map| finite(map.at(key)))
                .fold(0.0, |total, value| total + value);
            round(total / count / 1000.0, 4)
        };
        let total = average("totalDurationUs");
        let (mut compute, mut all_reduce, mut send, mut recv) = (breakdown("1"), breakdown("3"), 0.0, 0.0);
        let mut communication = round(all_reduce + breakdown("2"), 4);
        let (mut infeed, mut outfeed) = (breakdown("6"), breakdown("7"));
        if [compute, communication, infeed, outfeed].iter().all(|value| *value == 0.0) {
            (compute, all_reduce, send, recv) = (average("highFlopsComputeUs"), average("crsDurationUs"), average("sendDurationUs"), average("recvDurationUs"));
            communication = round(all_reduce + send + recv, 4);
            (infeed, outfeed) = (average("hostInfeedDurationUs"), average("hostOutfeedDurationUs"));
        }
        if [compute, communication, infeed, outfeed].iter().all(|value| *value == 0.0) {
            continue;
        }
        let idle = round((total - (compute + communication + infeed + outfeed)).max(0.0), 4);
        let shares = [compute, communication, infeed, outfeed, idle].map(|value| percent(value, total));
        let named: Vec<String> = core_stats.iter().map(|stats| stats.at("bottleneck")).filter(|value| value.truthy()).map(J::text).collect();
        let bottleneck = most_common(&named).unwrap_or_else(|| {
            let [compute, communication, infeed, outfeed, _] = shares;
            if communication > compute && communication > infeed && communication > outfeed {
                "Communication".into()
            } else if infeed > compute && infeed > outfeed {
                "Input / Infeed".into()
            } else if outfeed > compute {
                "Output / Outfeed".into()
            } else {
                "Compute".into()
            }
        });
        steps.push(step(number, total, [compute, communication, infeed, outfeed, idle], shares, bottleneck, Some((all_reduce, send, recv))));
    }
    if core.is_some() && !core_found { Vec::new() } else { steps }
}

fn input_pipeline(raw: &str, step_num: Option<&J>) -> (Vec<J>, Option<J>) {
    let Some(data) = J::parse(raw) else { return (Vec::new(), None) };
    let section = data.items().iter().find(|section| {
        let columns: Vec<String> = section.at("cols").items().iter().map(|column| column.at("id").str().unwrap_or_default().to_lowercase()).collect();
        section.has("cols") && section.has("rows") && columns.iter().any(|column| column.contains("step") || column.contains("infeed"))
    });
    let Some(section) = section else { return (Vec::new(), None) };
    let columns: Vec<Option<String>> = section.at("cols").items().iter().map(|column| column.get("id").filter(|_| matches!(column, J::Map(_))).map(|id| id.text().to_lowercase())).collect();
    let index = |name: &str| columns.iter().position(|column| column.as_deref() == Some(name));
    let mut steps = Vec::new();
    for row in section.at("rows").items() {
        let cells = row.at("c").items();
        if cells.is_empty() {
            continue;
        }
        let value = |name: &str| index(name).and_then(|at| cells.get(at)).map(|cell| cell.at("v").clone());
        let sum = |names: &[&str]| names.iter().filter_map(|name| value(name)).map(|cell| finite(&cell)).fold(0.0, |total, value| total + value);
        let number = match cells.get(index("stepnum").unwrap_or(0)).map(|cell| cell.at("v").clone()).unwrap_or_default() {
            J::Null => J::Null,
            J::Str(text) => super::json::py_int(&text).map_or(J::Str(text), J::Int),
            J::Float(number) if number.is_finite() => J::Int(number.trunc() as i128),
            other => other,
        };
        if step_num.is_some_and(|wanted| number != *wanted) {
            continue;
        }
        let (compute, communication, infeed, outfeed, other) = (sum(&COMPUTE_COLUMNS), sum(&COMMUNICATION_COLUMNS), sum(&INFEED_COLUMNS), sum(&OUTFEED_COLUMNS), sum(&OTHER_COLUMNS));
        let total = round(if index("steptimems").is_some() { sum(&["steptimems"]) } else { compute + communication + infeed + outfeed + other }, 4);
        let infeed_percent = match value("infeedpercentaverage").filter(|cell| *cell != J::Null) {
            Some(cell) => round(finite(&cell), 2),
            None => percent(infeed, total),
        };
        let idle = round(other.max(total - (compute + communication + infeed + outfeed)), 4);
        let candidates = [("Compute", compute), ("Communication", communication), ("Input / Infeed", infeed), ("Output / Outfeed", outfeed), ("Idle / Other", other)];
        let bottleneck = candidates.iter().fold(candidates[0], |best, candidate| if candidate.1 > best.1 { *candidate } else { best }).0.to_string();
        let percents = [percent(compute, total), percent(communication, total), infeed_percent, percent(outfeed, total), percent(idle, total)];
        steps.push(step(number, total, [round(compute, 4), round(communication, 4), round(infeed, 4), round(outfeed, 4), idle], percents, bottleneck, None));
    }
    (steps, Some(section.at("p").clone()).filter(J::truthy))
}

fn summarize(steps: &[J], props: Option<&J>) -> J {
    let column = |key: &str| steps.iter().map(|step| step.at(key).float().unwrap_or(0.0)).collect::<Vec<f64>>();
    let average = |key: &str| round(fsum(column(key)) / steps.len() as f64, 4);
    let times = column("step_time_ms");
    let (step, compute, communication, infeed, outfeed, idle) =
        (average("step_time_ms"), average("compute_time_ms"), average("communication_time_ms"), average("infeed_time_ms"), average("outfeed_time_ms"), average("idle_time_ms"));
    let named: Vec<String> = steps.iter().map(|step| step.at("bottleneck").text()).filter(|name| !name.is_empty()).collect();
    let primary = most_common(&named).unwrap_or_else(|| if communication > compute { "Communication" } else { "Compute" }.into());
    let mut summary = obj! {
        "total_steps" => steps.len(),
        "step_time_ms_average" => step,
        "step_time_ms_min" => round(times.iter().copied().fold(f64::INFINITY, f64::min), 4),
        "step_time_ms_max" => round(times.iter().copied().fold(f64::NEG_INFINITY, f64::max), 4),
        "step_time_ms_stddev" => if steps.len() > 1 { round(stdev(&times), 4) } else { 0.0 },
        "compute_time_ms_average" => compute,
        "compute_percent" => percent(compute, step),
        "communication_time_ms_average" => communication,
        "communication_percent" => percent(communication, step),
        "infeed_time_ms_average" => infeed,
        "infeed_percent" => percent(infeed, step),
        "outfeed_time_ms_average" => outfeed,
        "outfeed_percent" => percent(outfeed, step),
        "primary_bottleneck" => primary,
        "idle_time_ms_average" => idle,
        "idle_percent" => percent(idle, step),
        "is_aggregate" => false,
    };
    if let Some(conclusion) = props.and_then(|props| props.get("summary_conclusion")).filter(|value| **value != J::Null) {
        summary.set("conclusion", conclusion.clone());
    }
    summary
}

fn overview_steps(raw: &str) -> Option<J> {
    let sections = J::parse(raw)?;
    let mut all = J::Map(Vec::new());
    sections.items().iter().flat_map(|section| section.at("p").entries()).for_each(|(key, value)| all.set(key, value));
    let mut step = finite(all.at("steptime_ms_average"));
    if step <= 0.0 && all.has("stat_step_time") {
        step = super::json::py_float(all.at("stat_step_time").text().replace("ms", "").trim()).filter(|value| value.is_finite()).unwrap_or(0.0);
    }
    if step <= 0.0 {
        return None;
    }
    let pick = |first: &str, second: &str| finite(all.get(first).unwrap_or_else(|| all.at(second)));
    let (infeed, outfeed, idle) = (pick("tc_infeed_ms_average", "sc_infeed_ms_average"), pick("tc_outfeed_ms_average", "sc_outfeed_ms_average"), pick("tc_idle_ms_average", "sc_idle_ms_average"));
    let compute = (step - infeed - outfeed - idle).max(0.0);
    let infeed_percent = percent(infeed, step);
    let total_steps = STEP_COUNT_KEYS.iter().filter_map(|key| all.get(key)).find_map(|value| value.int().filter(|count| *count > 0)).or_else(|| {
        sections.items().iter().find_map(|section| {
            let stepped = section.has("rows") && section.has("cols") && section.at("cols").items().iter().any(|column| column.at("id").text().to_lowercase().contains("step"));
            (stepped && section.at("rows").truthy()).then(|| section.at("rows").items().len() as i128)
        })
    });
    let note = if total_steps.is_some() {
        "Aggregated overview statistics (individual step breakdown not available)."
    } else {
        "Aggregated overview statistics (individual step breakdown and step count not available)."
    };
    let bound = |key: &str| if all.has(key) { finite(all.at(key)) } else { step };
    Some(obj! {
        "total_steps" => total_steps,
        "step_time_ms_average" => round(step, 4),
        "step_time_ms_min" => round(bound("steptime_ms_min"), 4),
        "step_time_ms_max" => round(bound("steptime_ms_max"), 4),
        "step_time_ms_stddev" => round(finite(all.at("steptime_ms_standard_deviation")), 4),
        "compute_time_ms_average" => round(compute, 4),
        "compute_percent" => percent(compute, step),
        "communication_time_ms_average" => 0.0,
        "communication_percent" => 0.0,
        "infeed_time_ms_average" => round(infeed, 4),
        "infeed_percent" => infeed_percent,
        "outfeed_time_ms_average" => round(outfeed, 4),
        "outfeed_percent" => percent(outfeed, step),
        "primary_bottleneck" => if infeed_percent > 50.0 { "Input / Infeed" } else { "Compute" },
        "idle_time_ms_average" => round(idle, 4),
        "idle_percent" => percent(idle, step),
        "is_aggregate" => true,
        "note" => note,
    })
}

fn first_available(client: &dyn Client, session: &str, tools: &[&str], params: &[(&str, String)]) -> Result<Option<String>, Error> {
    for tool in tools {
        match client.fetch_text(tool, session, params) {
            Ok(Some(data)) => return Ok(Some(data)),
            Ok(None) => {}
            Err(error) if error.kind == Kind::Value => {}
            Err(error) if matches!(error.kind, Kind::Runtime | Kind::NotImplemented) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(None)
}

pub fn get_step_trace(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let (step_num, core, limit) = (args.get("step_num"), args.get("device_core"), args.int("limit", DEFAULT_STEP_LIMIT)?);
    let mut params = vec![("format", "json".to_string())];
    params.push(bypass(args.flag("bypass_cache", false)));
    let mut found: Option<(Vec<J>, Option<J>)> = None;
    if let Some(data) = first_available(client, &session, &["pod_viewer.json", "pod_viewer"], &params)? {
        let steps = pod_viewer(&data, step_num, core);
        if !steps.is_empty() {
            let summary = summarize(&steps, None);
            found = Some((steps, Some(summary)));
        }
    }
    if found.is_none()
        && let Some(data) = first_available(client, &session, &["input_pipeline_analyzer", "input_pipeline.json"], &params)?
    {
        let (steps, props) = input_pipeline(&data, step_num);
        if !steps.is_empty() {
            let summary = summarize(&steps, props.as_ref());
            found = Some((steps, Some(summary)));
        }
    }
    if found.is_none()
        && let Some(data) = first_available(client, &session, &["overview_page.json", "overview_page"], &params)?
    {
        found = overview_steps(&data).map(|summary| (Vec::new(), Some(summary)));
    }
    let Some((steps, summary)) = found else {
        return Ok(obj! {"status" => "NO_DATA", "message" => format!("No step trace data found for session '{session}' (the trace may lack step markers). Consider using 'list_xplane_events' to get more performance insights.")}.into());
    };
    let mut out = J::Map(Vec::new());
    if let Some(summary) = summary.filter(|_| args.flag("include_summary", true)) {
        out.set("summary", summary);
    }
    let shown = if limit > 0 { limit as usize } else { steps.len() };
    out.set("step_breakdown", steps.into_iter().take(shown).collect::<Vec<_>>());
    Ok(out.into())
}

fn hlo_times(node: &J) -> (f64, f64, f64) {
    let metrics = node.at("metrics");
    let children = node.at("children").items();
    let mut times = (0.0, 0.0, 0.0);
    if children.is_empty() && metrics.at("occurrences").float().unwrap_or(0.0) > 0.0 {
        let name = node.at("name").text().to_lowercase();
        let category = node.at("xla").get("category").map_or(String::new(), |category| category.text().to_lowercase());
        let time = Some(metrics.at("rawTime")).filter(|value| value.truthy()).and_then(J::float).unwrap_or(0.0);
        let matches = |patterns: &[&str]| patterns.iter().any(|pattern| name.contains(pattern) || category.contains(pattern));
        if matches(&ICI_PATTERNS) {
            times.2 += time;
        } else if matches(&HBM_PATTERNS) {
            times.1 += time;
        } else {
            times.0 += time;
        }
    }
    for child in children.iter().filter(|child| matches!(child, J::Map(_))) {
        let (compute, hbm, ici) = hlo_times(child);
        times = (times.0 + compute, times.1 + hbm, times.2 + ici);
    }
    times
}

fn barrier(client: &dyn Client, session: &str) -> (f64, usize) {
    let durations = client.barrier_durations(session).unwrap_or_default();
    if durations.is_empty() { (0.0, 0) } else { (durations.iter().sum::<f64>() / durations.len() as f64 / 1000.0, durations.len()) }
}

fn utilization_metrics(client: &dyn Client, session: &str, hosts: i64, bypass: bool) -> [f64; 4] {
    let mut metrics = [0.0; 4];
    for host in 0..hosts.clamp(1, MAX_UTILIZATION_HOSTS) {
        let Ok(data) = utilization_viewer(client, session, host, 0, 0, bypass) else { break };
        if !data.truthy() || data.has("error") {
            break;
        }
        if data.at("status").str() == Some("NO_DATA") {
            let message = data.at("message").text();
            if message.contains("No data returned for session") || message.contains("No hardware performance counter events found") {
                break;
            }
            continue;
        }
        if !data.has("message") {
            for (slot, key) in metrics.iter_mut().zip(["idleness_percent", "hbm_bandwidth_utilization_percent", "ici_read_utilization_percent", "ici_write_utilization_percent"]) {
                if let Some(value) = data.get(key).and_then(J::float) {
                    *slot = value;
                }
            }
            break;
        }
    }
    metrics
}

fn leading_number(value: &J, pattern: fn(char) -> bool) -> Option<String> {
    let text = value.text();
    let start = text.find(pattern)?;
    Some(text[start..].chars().take_while(|character| pattern(*character)).collect())
}

pub fn check_host_boundness(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let bypass = args.flag("bypass_cache", false);
    let compute = || -> Result<J, Error> {
        let mut params = vec![("format", "json".to_string())];
        params.push(super::bypass(bypass));
        let raw = client.fetch_text("overview_page.json", &session, &params)?.ok_or_else(|| Error::new(Kind::FileNotFound, format!("No overview data returned for session {session}")))?;
        let sections = J::parse(&raw).ok_or_else(|| Error::new(Kind::Value, "Expecting value: line 1 column 1 (char 0)"))?;
        let section = |index: usize, key: &str| {
            sections.items().get(index).filter(|section| matches!(section, J::Map(_))).map_or(J::Map(Vec::new()), |section| section.get(key).cloned().unwrap_or_else(|| J::Map(Vec::new())))
        };
        let pick = |key: &str| Some(sections.at(key)).filter(|value| matches!(value, J::Map(_))).unwrap_or(&sections).clone();
        let (overview, step_table, environment, steps) = match &sections {
            J::List(_) => (section(0, "p"), section(1, "p"), section(2, "p"), section(1, "rows").items().len()),
            J::Map(_) => (
                pick("performance_summary"),
                pick("step_time"),
                pick("run_environment"),
                sections.get("number_of_steps").map_or(sections.at("rows").items().len() as i128, |count| count.int().unwrap_or(0)) as usize,
            ),
            _ => return Err(Error::new(Kind::Value, format!("Unexpected overview data format for session {session}"))),
        };
        let duty = super::json::py_float(&overview.get("device_duty_cycle_percent").map_or_else(|| "0.0%".into(), J::text).replace('%', "")).unwrap_or(0.0);
        let digits = |value: Option<&J>| leading_number(value.unwrap_or(&J::from("1")), |character| character.is_ascii_digit()).and_then(|text| text.parse::<i64>().ok()).unwrap_or(1).max(1);
        let (cores, hosts) = (digits(environment.get("device_core_count")), digits(environment.get("host_count")));
        let step_value = Some(step_table.at("steptime_ms_average"))
            .filter(|value| value.truthy())
            .or_else(|| Some(step_table.at("sc_step_time_ms_average")).filter(|value| value.truthy()))
            .cloned()
            .unwrap_or_else(|| J::from("0.0"));
        let average_step = leading_number(&step_value, |character| character.is_ascii_digit() || character == '.').and_then(|text| super::json::py_float(&text)).unwrap_or(0.0);
        let total = average_step * steps as f64;
        if total == 0.0 || steps == 0 {
            return Ok(obj! {
                "status" => "INSUFFICIENT_DATA",
                "reasons" => vec![format!("Session {session} lacks valid step timing duration telemetry in overview_page.json.")],
                "recommendations" => vec!["Capture a new XProf trace with step profiling enabled to measure host boundness."],
            });
        }
        let mut hlo_params = vec![("group_by", "category".to_string())];
        hlo_params.push(super::bypass(bypass));
        let profile_times = |client: &dyn Client| {
            let profile = client.fetch_text("hlo_op_profile.json", &session, &hlo_params).ok().flatten().and_then(|raw| J::parse(&raw));
            let times = profile.as_ref().map(|profile| field(profile, "by_category")).filter(|node| matches!(node, J::Map(_)) && node.truthy()).map(hlo_times);
            crate::release(profile);
            times
        };
        let others = |client: &dyn Client| (barrier(client, &session), utilization_metrics(client, &session, hosts, bypass));
        let (times, ((barrier_average, barriers), [idleness, hbm, ici_read, ici_write])) = super::both(client, profile_times, others);
        let available = times.is_some();
        let (compute_ps, hbm_ps, ici_ps) = times.unwrap_or((0.0, 0.0, 0.0));
        let scale = |picoseconds: f64| picoseconds / 1e9 / cores as f64;
        let (compute_ms, hbm_ms, ici_ms) = (scale(compute_ps), scale(hbm_ps), scale(ici_ps));
        let barrier_ms = if barriers > 0 { barrier_average * steps as f64 } else { 0.0 };
        let idle = total - (compute_ms + hbm_ms + ici_ms);
        let pure_idle = (idle - barrier_ms).max(0.0);
        let active = total - pure_idle;
        let ratio = if active > 0.0 { pure_idle / active } else { 0.0 } * 100.0;
        let idle_chips = cores as f64 * if total > 0.0 { pure_idle / total } else { 0.0 };
        let (idle_high, mxu_high, hbm_low, ici_low) = (ratio > IDLE_RATIO_THRESHOLD, idleness > MXU_IDLENESS_THRESHOLD, hbm < HBM_THRESHOLD, ici_read < ICI_THRESHOLD && ici_write < ICI_THRESHOLD);
        let (mut reasons, mut recommendations) = (Vec::new(), Vec::new());
        let status = if idle_high && mxu_high && hbm_low && ici_low {
            reasons.push(format!("Workload is host-bound: Idle Time Ratio exceeds {IDLE_RATIO_THRESHOLD:.1}% of active compute."));
            recommendations.push(format!("Opportunity Size: Hardware waste = {idle_chips:.1} idle chips."));
            recommendations.push("Review data pipeline for inefficiencies (e.g., tf.data transformations or PyGrain).".into());
            recommendations.push(if args.get("func_name").is_some_and(J::truthy) {
                "Use Lumini xprof_check_host_boundness with func_name to run automated step dispersion analysis.".into()
            } else {
                "Provide `func_name` to enable automated Dispersion Analysis and check for transient stalls.".into()
            });
            "HOST_BOUND"
        } else if duty > DUTY_CYCLE_THRESHOLD && (!available || !idle_high) {
            if available {
                reasons.push(format!("TPU duty cycle ({duty:.1}%) is high and idle ratio is low."));
                "NOT_HOST_BOUND"
            } else {
                reasons.push(format!("TPU duty cycle is high ({duty:.1}%), but HLO telemetry is missing. Unable to determine if workload is host-bound."));
                "UNKNOWN"
            }
        } else {
            reasons.push("Workload is not host-bound. One or more host-bound condition thresholds were not met:".into());
            if !idle_high {
                reasons.push(format!("- Idle Time Ratio ({ratio:.1}%) is <= {IDLE_RATIO_THRESHOLD:.1}%."));
            }
            if !mxu_high {
                reasons.push(format!("- MXU Idleness ({idleness:.1}%) is <= {MXU_IDLENESS_THRESHOLD:.1}%."));
            }
            if !hbm_low {
                reasons.push(format!("- HBM Bandwidth Utilization ({hbm:.1}%) is >= {HBM_THRESHOLD:.1}%."));
            }
            if !ici_low {
                reasons.push(format!("- ICI Utilization (Read: {ici_read:.1}%, Write: {ici_write:.1}%) is >= {ICI_THRESHOLD:.1}%."));
            }
            "NOT_HOST_BOUND"
        };
        Ok(obj! {
            "status" => status,
            "metrics" => obj! {
                "tpu_duty_cycle_percent" => round(duty, 2),
                "idle_time_ratio_percent" => round(ratio, 2),
                "equivalent_idle_chips" => round(idle_chips, 2),
                "mxu_idleness_percent" => round(idleness, 2),
                "hbm_bandwidth_utilization_percent" => round(hbm, 2),
                "ici_read_utilization_percent" => round(ici_read, 2),
                "ici_write_utilization_percent" => round(ici_write, 2),
                "scaled_compute_time_ms" => round(compute_ms, 2),
                "scaled_hbm_time_ms" => round(hbm_ms, 2),
                "scaled_ici_time_ms" => round(ici_ms, 2),
                "scaled_barrier_time_ms" => round(barrier_ms, 2),
                "pure_idle_time_ms" => round(pure_idle, 2),
                "total_duration_ms" => round(total, 2),
                "number_of_steps" => steps,
                "core_count" => cores,
            },
            "reasons" => reasons,
            "recommendations" => recommendations,
        })
    };
    Ok(compute().map_err(|error| rethrow(error, |error| format!("Error during host boundness check: {}", error.message)))?.into())
}
