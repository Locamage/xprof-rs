use super::client::Client;
use super::json::{J, py_repr};
use super::{Args, Error, Kind, Out, bypass, fail, fsum, rethrow, round};
use crate::obj;
use regex::Regex;
use std::sync::LazyLock;

const PERFORMANCE_SUMMARY_KEYS: [&str; 22] = [
    "steptime_ms_average",
    "steptime_ms_standard_deviation",
    "tc_idle_ms_average",
    "tc_infeed_ms_average",
    "tc_outfeed_ms_average",
    "host_transfer_ms_average",
    "sc_step_time_ms_average",
    "sc_idle_ms_average",
    "sc_infeed_ms_average",
    "sc_outfeed_ms_average",
    "mxu_utilization_percent",
    "flop_rate_utilization_relative_to_roofline",
    "device_duty_cycle_percent",
    "memory_bw_utilization_relative_to_hw_limit",
    "program_goodput_percent",
    "host_tf_op_percent",
    "device_tf_op_percent",
    "host_op_time_eager_percent",
    "device_op_time_eager_percent",
    "device_idle_time_percent",
    "hbm_bw_utilization_percent",
    "host_idle_time_percent",
];
const RUN_ENVIRONMENT_KEYS: [&str; 10] =
    ["is_training", "profile_start_time", "profile_duration_ms", "host_count", "task_count", "device_type", "device_core_count", "change_list", "build_time", "build_target"];
const ZERO_PERCENTS: [&str; 3] = ["0.0%", "0%", "0.0"];
const BANDWIDTH_RENAMES: [(&str, &str); 2] = [("peak_hbm_bw", "peak_hbm_bw_gibs"), ("peak_vmem_bw", "peak_vmem_bw_gibs")];
const BYTES_PER_GIB: f64 = 1073741824.0;
const BYTES_PER_MIB: f64 = 1048576.0;
const CUSTOM_CALL_GUIDANCE: &str = "Op-level metrics unavailable for custom calls. Use get_llo_analysis, get_llo_debug_string, and aggregate_xplane_events for Pallas kernels.";
static XID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"--streamz_default_root_labels=\S*xmanager:int:(\d+)").unwrap());
static TITLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"title=['"]([^'"]+)['"]"#).unwrap());
static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static NUMBERED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:XLU|MXU)\d+$").unwrap());

pub fn row_dict(table: &J, row: &J) -> J {
    let ids = table.at("cols").items().iter().map(|column| column.get("id").map_or(String::new(), J::text));
    J::Map(ids.zip(row.at("c").items().iter().map(|cell| if let J::Map(_) = cell { cell.at("v").clone() } else { cell.clone() })).collect())
}

fn table_rows(data: &str) -> Option<(J, Vec<J>, J)> {
    let parsed = J::parse(data)?;
    let table = parsed.items().first()?.clone();
    let rows = table.at("rows").items().iter().map(|row| row_dict(&table, row)).collect();
    Some((table.at("p").clone(), rows, table))
}

fn zero(value: Option<&J>) -> bool {
    value.is_none_or(|value| !value.truthy() || value.str().is_some_and(|text| ZERO_PERCENTS.contains(&text)))
}

fn percent(value: f64) -> String {
    format!("{:.2}%", value * 100.0)
}

fn roofline_fallback(client: &dyn Client, session: &str, summary: &mut J) -> Option<()> {
    let needs_flop = zero(summary.get("flop_rate_utilization_relative_to_roofline"));
    let needs_memory = zero(summary.get("memory_bw_utilization_relative_to_hw_limit")) && zero(summary.get("hbm_bw_utilization_percent"));
    if !needs_flop && !needs_memory {
        return None;
    }
    let data = client.fetch_text("roofline_model.json", session, &[]).ok()?.or_else(|| client.fetch_text("roofline_model", session, &[]).ok()?)?;
    let (_, rows, _) = table_rows(&data)?;
    let program = rows.first()?;
    let value = |key: &str| program.at(key).float();
    if let Some(efficiency) = value("roofline_efficiency") {
        summary.set("roofline_efficiency_percent", percent(efficiency));
        if needs_flop {
            summary.set("flop_rate_utilization_relative_to_roofline", percent(efficiency));
        }
    }
    if let Some(efficiency) = value("compute_efficiency") {
        summary.set("compute_efficiency_percent", percent(efficiency));
    }
    if let Some(utilization) = value("max_mem_bw_utilization") {
        summary.set("max_mem_bw_utilization_percent", percent(utilization));
        if needs_memory {
            summary.set("memory_bw_utilization_relative_to_hw_limit", percent(utilization));
            summary.set("hbm_bw_utilization_percent", percent(utilization));
        }
    }
    if program.at("bound_by").truthy() {
        summary.set("bound_by", program.at("bound_by"));
    }
    if let Some(intensity) = value("operational_intensity") {
        summary.set("operational_intensity_flop_per_byte", round(intensity, 4));
    }
    Some(())
}

pub fn overview(client: &dyn Client, session: &str, include_command: bool, bypass: bool) -> Result<J, Error> {
    let compute = || -> Result<J, Error> {
        let params: Vec<(&str, String)> = vec![("format", "json".to_string()), super::bypass(bypass)];
        let data = match client.fetch_text("overview_page", session, &params)? {
            Some(data) => Some(data),
            None => client.fetch_text("overview_page.json", session, &params)?,
        };
        let data = data.ok_or_else(|| Error::new(Kind::FileNotFound, "No overview data returned for the session"))?;
        let sections = J::parse(&data).ok_or_else(|| Error::new(Kind::Value, "Expecting value: line 1 column 1 (char 0)"))?;
        if sections.items().is_empty() {
            return fail(Kind::Value, "Unexpected overview page data format");
        }
        let mut all = J::Map(Vec::new());
        sections.items().iter().flat_map(|section| section.at("p").entries()).for_each(|(key, value)| all.set(key, value));
        let environment = sections.items().iter().find_map(|section| {
            let columns: Vec<String> = section.get("rows").and(section.get("cols"))?.items().iter().map(|column| column.at("id").text()).collect();
            let row = section.at("rows").items().first().filter(|row| row.at("c").truthy())?;
            let cell = |id: &str| columns.iter().position(|column| column == id).map(|index| row.at("c").items().get(index).map_or(J::Null, |cell| cell.at("v").clone()));
            columns.iter().any(|column| column == "host_id").then(|| (cell("host_id"), cell("bns_address"), cell("command_line")))
        });
        let (hostname, bns, command) = environment.unwrap_or_default();
        let mut summary = J::Map(all.entries().iter().filter(|(key, _)| key.starts_with("stat_") || key.starts_with("sc_") || PERFORMANCE_SUMMARY_KEYS.contains(&key.as_str())).cloned().collect());
        roofline_fallback(client, session, &mut summary);
        let mut run =
            J::Map(all.entries().iter().filter(|(key, _)| (key.starts_with("run_") || RUN_ENVIRONMENT_KEYS.contains(&key.as_str())) && (key != "run_command" || include_command)).cloned().collect());
        if let Some(hostname) = hostname.filter(J::truthy) {
            run.set("hostname", hostname);
        }
        if let Some(bns) = bns.filter(J::truthy) {
            run.set("bns", bns);
        }
        let command = command.filter(J::truthy);
        if let Some(command) = command.as_ref().filter(|_| include_command && !run.has("run_command")) {
            run.set("run_command", command);
        }
        if let Some(device) = all.get("device_type") {
            run.set("device_type", device.clone());
        }
        if let Some(xid) = command.as_ref().and_then(|command| XID.captures(&command.text()).map(|found| found[1].to_string())) {
            run.set("xid", xid);
        }
        Ok(obj! {"performance_summary" => summary, "run_environment" => run})
    };
    compute().map_err(|error| rethrow(error, |error| format!("Error fetching overview data: {}", error.message)))
}

pub fn get_overview(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    Ok(overview(client, &args.session(), args.flag("include_command", false), args.flag("bypass_cache", false))?.into())
}

struct Memory {
    capacity: i128,
    peak: i128,
    stack: i128,
    heap: i128,
    free: i128,
    fragmentation: f64,
}

fn integer(value: &J, default: i128) -> Result<i128, Error> {
    match value {
        J::Null => Ok(default),
        value => {
            value.int().filter(|_| !matches!(value, J::Str(text) if text.contains('.'))).ok_or_else(|| Error::new(Kind::Value, format!("invalid literal for int() with base 10: {}", value.repr())))
        }
    }
}

fn parse_memory(data: &str) -> Result<Memory, Error> {
    let parsed = J::parse(data).ok_or_else(|| Error::new(Kind::Value, "Failed to parse JSON for memory profile: JSONDecodeError('Expecting value: line 1 column 1 (char 0)')"))?;
    let devices = if let J::Map(_) = parsed { vec![parsed] } else { parsed.items().to_vec() };
    let Some(first) = devices.first() else { return fail(Kind::Value, "Unexpected memory profile data format: not a list or empty") };
    let mut memory = Memory { capacity: -1, peak: -1, stack: -1, heap: -1, free: -1, fragmentation: -1.0 };
    let take = |summary: &J, peak_key: &str, memory: &mut Memory| -> Result<(), Error> {
        memory.capacity = memory.capacity.max(integer(summary.at("memoryCapacity"), -1)?);
        let stats = summary.at("peakStats");
        let peak = integer(stats.at(peak_key), -1)?;
        if peak > memory.peak {
            memory.peak = peak;
            memory.stack = integer(stats.at("stackReservedBytes"), -1)?;
            memory.heap = integer(stats.at("heapAllocatedBytes"), -1)?;
            memory.free = integer(stats.at("freeMemoryBytes"), -1)?;
            memory.fragmentation = stats.get("fragmentation").map_or(Some(-1.0), J::float).unwrap_or(-1.0);
        }
        Ok(())
    };
    if first.has("cols") && first.has("rows") {
        return fail(Kind::Value, "Table format memory profile not fully supported");
    } else if first.has("memoryProfileSummary") {
        for device in &devices {
            let summary = device.at("memoryProfileSummary");
            if !device.truthy() || !device.has("memoryProfileSummary") {
                continue;
            }
            memory.capacity = memory.capacity.max(integer(summary.at("memoryCapacity"), -1)?);
            if summary.at("peakStats").has("peakBytesUsageHbm") {
                _ = take(summary, "peakBytesUsageHbm", &mut memory);
            }
        }
    } else if first.has("peakMemoryUsageMiB") {
        for item in devices.iter().filter(|item| item.has("peakMemoryUsageMiB")) {
            if let Some(peak) = item.at("peakMemoryUsageMiB").float().map(|value| value * BYTES_PER_MIB).filter(|peak| *peak > memory.peak as f64) {
                memory.peak = peak as i128;
            }
        }
    } else if first.has("memoryProfilePerAllocator") {
        for (_, allocator) in first.at("memoryProfilePerAllocator").entries() {
            take(allocator.at("profileSummary"), "peakBytesInUse", &mut memory)?;
        }
    } else {
        return fail(Kind::Value, "Unknown memory profile JSON format");
    }
    Ok(memory)
}

pub fn memory_profile(client: &dyn Client, session: &str, bypass: bool) -> Result<J, Error> {
    let attempt = |host: Option<&str>| -> (Option<Memory>, Option<String>) {
        let mut params = vec![("format", "json".to_string())];
        params.extend(bypass.then(|| super::bypass(true)));
        params.extend(host.map(|host| ("host", host.to_string())));
        match client.fetch_text("memory_profile.json", session, &params) {
            Ok(None) => (None, None),
            Ok(Some(data)) => match parse_memory(&data) {
                Ok(memory) => (Some(memory), None),
                Err(error) => (None, Some(error.message)),
            },
            Err(error) if error.kind == Kind::Value => (None, Some(error.message)),
            Err(error) => (None, Some(format!("Error fetching memory profile: {}", error.repr()))),
        }
    };
    let valid = |memory: &Memory| memory.capacity > 0 && memory.peak > 0;
    let (mut best, mut error) = attempt(None);
    if !best.as_ref().is_some_and(valid) {
        match client.hosts(session) {
            Err(failure) => error = Some(format!("Failed to get hosts: {}", failure.repr())),
            Ok(hosts) => {
                for host in hosts.iter().filter(|host| !host.is_empty()) {
                    match attempt(Some(host)) {
                        (Some(memory), _) if valid(&memory) => {
                            (best, error) = (Some(memory), None);
                            break;
                        }
                        (Some(memory), failure) if best.is_none() => (best, error) = (Some(memory), failure),
                        (None, Some(failure)) => error = Some(failure),
                        _ => {}
                    }
                }
            }
        }
    }
    let gib = |bytes: i128, positive: bool| if bytes > 0 || (!positive && bytes >= 0) { round(bytes as f64 / BYTES_PER_GIB, 2) } else { -1.0 };
    let Some(memory) = best else {
        return match error {
            Some(error) if error.contains("Failed to parse JSON") || error.contains("Value error") => fail(Kind::Value, error),
            Some(error) => fail(Kind::Runtime, format!("Error fetching memory profile: {error}")),
            None => Ok(obj! {
                "memory_capacity_gib" => -1.0,
                "peak_memory_usage_gib" => -1.0,
                "peak_usage_details" => obj! {"stack_reservation_gib" => -1.0, "heap_allocation_gib" => -1.0, "free_memory_gib" => -1.0, "fragmentation_percent" => -1.0, "utilization_percent" => -1.0},
            }),
        };
    };
    let (capacity, peak) = (gib(memory.capacity, true), gib(memory.peak, true));
    let utilization = if capacity > 0.0 && peak > 0.0 { round(peak / capacity * 100.0, 2) } else { -1.0 };
    Ok(obj! {
        "memory_capacity_gib" => capacity,
        "peak_memory_usage_gib" => peak,
        "peak_usage_details" => obj! {
            "stack_reservation_gib" => gib(memory.stack, false),
            "heap_allocation_gib" => gib(memory.heap, false),
            "free_memory_gib" => gib(memory.free, false),
            "fragmentation_percent" => if memory.fragmentation >= 0.0 { round(memory.fragmentation * 100.0, 2) } else { -1.0 },
            "utilization_percent" => utilization,
        },
    })
}

pub fn get_memory_profile(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    Ok(memory_profile(client, &args.session(), args.flag("bypass_cache", false))?.into())
}

pub fn get_kpi_metrics(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let overview = overview(client, &session, false, args.flag("bypass_cache", false)).map_err(|error| rethrow(error, |error| format!("Error in get_overview: {}", error.message)))?;
    let memory = memory_profile(client, &session, args.flag("bypass_cache", false)).unwrap_or_else(|_| obj! {"peak_memory_usage_gib" => "N/A"});
    let pick = |section: &J, key: &str| section.get(key).cloned().unwrap_or_else(|| J::from("N/A"));
    let (summary, run) = (overview.at("performance_summary"), overview.at("run_environment"));
    Ok(obj! {
        "step_time_ms" => pick(summary, "steptime_ms_average"),
        "duty_cycle_percent" => pick(summary, "device_duty_cycle_percent"),
        "mxu_utilization_percent" => pick(summary, "mxu_utilization_percent"),
        "roofline_utilization" => pick(summary, "flop_rate_utilization_relative_to_roofline"),
        "peak_hbm_gib" => pick(&memory, "peak_memory_usage_gib"),
        "accelerator_info" => obj! {"device_type" => pick(run, "device_type"), "device_core_count" => pick(run, "device_core_count")},
    }
    .into())
}

fn device_info(props: &J) -> J {
    J::Map(props.entries().iter().map(|(key, value)| (key.clone(), value.float().filter(|_| !matches!(value, J::Bool(_) | J::Null)).map_or_else(|| value.clone(), J::Float))).collect())
}

pub fn get_device_information(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let params = [bypass(args.flag("bypass_cache", false))];
    let compute = || -> Result<J, Error> {
        let data = client.fetch_text("roofline_model.json", &session, &params)?.ok_or_else(|| Error::new(Kind::FileNotFound, format!("No roofline model data returned for session {session}.")))?;
        let parsed = J::parse(&data).ok_or_else(|| Error::new(Kind::Value, "Failed to parse roofline model data: JSONDecodeError('Expecting value: line 1 column 1 (char 0)')"))?;
        let table = parsed.items().first().ok_or_else(|| Error::new(Kind::Value, "Unexpected roofline model data format"))?;
        let info = device_info(table.at("p"));
        let mut renamed =
            J::Map(info.entries().iter().map(|(key, value)| (BANDWIDTH_RENAMES.iter().find(|(from, _)| from == key).map_or(key.as_str(), |(_, to)| to).to_string(), value.clone())).collect());
        let units: Vec<(String, J)> = BANDWIDTH_RENAMES.iter().filter(|(_, to)| renamed.has(to)).map(|(_, to)| (to.to_string(), J::from("GiB/s"))).collect();
        if !units.is_empty() {
            renamed.set("units", J::Map(units));
        }
        Ok(renamed)
    };
    Ok(compute().map_err(|error| rethrow(error, |error| format!("Error fetching device information for session {session}: {}", error.repr())))?.into())
}

pub fn get_hosts(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let session = args.session();
    let hosts = client.hosts(&session).map_err(|error| rethrow(error, |error| format!("Error fetching hosts for session {session}: {}", error.repr())))?;
    if hosts.is_empty() {
        return fail(Kind::FileNotFound, format!("No hosts found for session {session}."));
    }
    Ok(obj! {"hosts" => hosts.into_iter().map(|host| obj! {"hostname" => host}).collect::<Vec<_>>()}.into())
}

fn strip_html(text: &J) -> String {
    let Some(text) = text.str().filter(|text| !text.is_empty()) else { return String::new() };
    match TITLE.captures(text) {
        Some(found) => found[1].replace('\n', " -> "),
        None => TAG.replace_all(text, "").trim().to_string(),
    }
}

pub fn get_roofline_model(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let (session, top_n) = (args.session(), args.int("top_n", 15)?);
    let params = [bypass(args.flag("bypass_cache", false))];
    let compute = || -> Result<J, Error> {
        let data = match client.fetch_text("roofline_model.json", &session, &params)? {
            Some(data) => Some(data),
            None => client.fetch_text("roofline_model", &session, &params)?,
        };
        let Some(data) = data else {
            return Ok(obj! {"status" => "NO_DATA", "message" => format!("No roofline model data found for session {}.", py_repr(&session))});
        };
        let parsed = J::parse(&data).ok_or_else(|| Error::new(Kind::Value, "Expecting value: line 1 column 1 (char 0)"))?;
        let table = parsed.items().first().ok_or_else(|| Error::new(Kind::Value, "Unexpected roofline model data format: expected non-empty list"))?;
        let info = device_info(table.at("p"));
        let rows: Vec<J> = table.at("rows").items().iter().map(|row| row_dict(table, row)).collect();
        let Some(program) = rows.first() else {
            return Ok(obj! {"status" => "NO_DATA", "message" => "Roofline model table has no rows", "device_info" => info});
        };
        let number = |row: &J, key: &str| row.at(key).float().unwrap_or(0.0);
        let utilization = |util: &str, bandwidth: &str, peaks: [&str; 2]| {
            let value = program.at(util);
            if *value != J::Null && number(program, util) > 0.0 {
                return percent(number(program, util));
            }
            let mut measured = number(program, bandwidth);
            if measured == 0.0 && bandwidth.starts_with("hbm_") {
                measured = number(program, "hbm_bw");
            }
            match peaks.iter().map(|key| number(&info, key)).find(|peak| *peak > 0.0) {
                Some(peak) => format!("{:.2}%", measured / peak * 100.0),
                None if *value != J::Null => percent(number(program, util)),
                None => "N/A".into(),
            }
        };
        let mut summary = obj! {
            "bound_by" => program.get("bound_by").cloned().unwrap_or_else(|| J::from("Unknown")),
            "operational_intensity_flop_per_byte" => round(number(program, "operational_intensity"), 4),
            "bottleneck_operational_intensity_flop_per_byte" => round(number(program, "bottleneck_operational_intensity"), 4),
            "roofline_efficiency_percent" => percent(number(program, "roofline_efficiency")),
            "compute_efficiency_percent" => percent(number(program, "compute_efficiency")),
            "max_mem_bw_utilization_percent" => percent(number(program, "max_mem_bw_utilization")),
            "optimal_flop_rate_gflops" => round(number(program, "optimal_flop_rate"), 2),
            "dma_stall_percent" => percent(number(program, "dma_stall_percent")),
            "measured_flop_rate_gflops" => round(number(program, "measured_flop_rate"), 2),
            "model_flop_rate_gflops" => round(number(program, "model_flop_rate"), 2),
            "measured_memory_bw_gibs" => round(number(program, "measured_memory_bw"), 2),
            "hbm_bw_gibs" => round(number(program, "hbm_bw"), 2),
            "hbm_read_bw_utilization_percent" => utilization("hbm_read_bw_utilization", "hbm_read_bw", ["peak_hbm_read_bw", "peak_hbm_bw"]),
            "hbm_write_bw_utilization_percent" => utilization("hbm_write_bw_utilization", "hbm_write_bw", ["peak_hbm_write_bw", "peak_hbm_bw"]),
            "cmem_read_bw_utilization_percent" => utilization("cmem_read_bw_utilization", "cmem_read_bw", ["peak_cmem_read_bw", "peak_cmem_bw"]),
            "cmem_write_bw_utilization_percent" => utilization("cmem_write_bw_utilization", "cmem_write_bw", ["peak_cmem_write_bw", "peak_cmem_bw"]),
            "vmem_read_bw_utilization_percent" => utilization("vmem_read_bw_utilization", "vmem_read_bw", ["peak_vmem_read_bw", "peak_vmem_bw"]),
            "vmem_write_bw_utilization_percent" => utilization("vmem_write_bw_utilization", "vmem_write_bw", ["peak_vmem_write_bw", "peak_vmem_bw"]),
            "total_time_ms" => round(number(program, "total_time") / 1000.0, 3),
        };
        let mut seen = Vec::new();
        let mut operations: Vec<J> = Vec::new();
        for row in rows.iter().skip(1).filter(|row| number(row, "total_self_time") > 0.0) {
            let name = Some(row.at("operation")).filter(|value| value.truthy()).unwrap_or_else(|| row.get("hlo_name").unwrap_or(&J::Null)).clone();
            let category = Some(row.at("category")).filter(|value| value.truthy()).unwrap_or_else(|| row.get("hlo_category").unwrap_or(&J::Null)).clone();
            let mut bound = Some(row.at("bound_by")).filter(|value| value.truthy()).cloned().unwrap_or_else(|| J::from("Unknown"));
            if bound == J::from("Unknown") && (name.text().starts_with("custom-call") || matches!(category.text().to_lowercase().as_str(), "custom-call" | "custom_call")) {
                bound = J::from("CustomCall (opaque)");
            }
            let rank = number(row, "rank").trunc() as i64;
            if seen.contains(&(rank, name.clone())) {
                continue;
            }
            seen.push((rank, name.clone()));
            operations.push(obj! {
                "rank" => rank,
                "name" => name,
                "category" => category,
                "total_self_time_ms" => round(number(row, "total_self_time") / 1000.0, 3),
                "total_self_time_percent" => percent(number(row, "total_self_time_percent")),
                "operational_intensity_flop_per_byte" => round(number(row, "operational_intensity"), 4),
                "bottleneck_operational_intensity_flop_per_byte" => round(number(row, "bottleneck_operational_intensity"), 4),
                "roofline_efficiency_percent" => percent(number(row, "roofline_efficiency")),
                "compute_efficiency_percent" => percent(number(row, "compute_efficiency")),
                "max_mem_bw_utilization_percent" => percent(number(row, "max_mem_bw_utilization")),
                "optimal_flop_rate_gflops" => round(number(row, "optimal_flop_rate"), 2),
                "dma_stall_percent" => percent(number(row, "dma_stall_percent")),
                "bound_by" => bound,
                "hlo_module_id" => row.get("hlo_module_id").map_or(String::new(), J::text),
                "source_info" => strip_html(row.at("source_info")),
            });
        }
        let total = operations.len();
        operations.sort_by(|left, right| right.at("total_self_time_ms").float().partial_cmp(&left.at("total_self_time_ms").float()).unwrap_or(std::cmp::Ordering::Equal));
        operations.truncate(top_n.max(0) as usize);
        let custom = operations.iter().any(|operation| operation.at("bound_by").str() == Some("CustomCall (opaque)"));
        if custom && matches!(summary.at("bound_by"), J::Null) || custom && matches!(summary.at("bound_by").str(), Some("Unknown" | "")) {
            summary.set("bound_by", "CustomCall (opaque)");
        }
        let mut output = obj! {"program" => summary, "device_info" => info, "top_operations" => operations, "total_operations_analyzed" => total};
        if custom {
            output.set("guidance", CUSTOM_CALL_GUIDANCE);
        }
        Ok(output)
    };
    Ok(compute()
        .map_err(|error| {
            rethrow(error, |error| {
                if error.message.is_empty() { format!("Error fetching roofline model: {}", error.name()) } else { format!("Error fetching roofline model: {}: {}", error.name(), error.message) }
            })
        })?
        .into())
}

fn loose_float(value: &J, default: f64) -> f64 {
    match value {
        J::Str(text) if matches!(text.trim().to_lowercase().as_str(), "nan" | "none" | "" | "null") => default,
        value => value.float().unwrap_or(default),
    }
}

fn percentage(rows: &[&J]) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    let peaked: Vec<f64> = rows.iter().filter(|row| loose_float(row.at("Peak"), 0.0) > 0.0).map(|row| loose_float(row.at("Achieved"), 0.0) * 100.0 / loose_float(row.at("Peak"), 1.0)).collect();
    if !peaked.is_empty() {
        return Some(round(fsum(peaked.iter().copied()) / peaked.len() as f64, 2));
    }
    rows.iter().all(|row| loose_float(row.at("Achieved"), 0.0) == 0.0).then_some(0.0)
}

pub fn utilization(raw: &str, host: i64, device: i64, node: i64) -> J {
    let parsed = J::parse(raw.trim()).filter(|table| table.has("cols"));
    let (rows, fields): (Vec<J>, Vec<String>) = match &parsed {
        Some(table) => {
            let labels: Vec<String> = table
                .at("cols")
                .items()
                .iter()
                .enumerate()
                .map(|(index, column)| Some(column.at("label")).filter(|label| label.truthy()).or(column.get("id")).map_or(format!("col_{index}"), J::text))
                .collect();
            let rows = table
                .at("rows")
                .items()
                .iter()
                .map(|row| J::Map(labels.iter().cloned().zip(row.at("c").items().iter().map(|cell| if let J::Map(_) = cell { cell.at("v").clone() } else { cell.clone() })).collect()))
                .collect();
            (rows, labels)
        }
        None => {
            let mut lines = raw.lines();
            let header: Vec<String> = lines.next().unwrap_or_default().split(',').map(|field| field.trim().to_string()).collect();
            let rows = lines.map(|line| J::Map(header.iter().cloned().zip(line.split(',').map(|value| J::from(value.trim()))).filter(|(key, _)| !key.is_empty()).collect())).collect();
            (rows, header.into_iter().filter(|field| !field.is_empty()).collect())
        }
    };
    if rows.is_empty() {
        return obj! {"status" => "NO_DATA", "message" => "No hardware performance counter events found in trace"};
    }
    let mut warnings = Vec::new();
    let mut selected: Vec<&J> = rows.iter().collect();
    for (column, wanted) in [("Host", host), ("Device", device), ("Node", node)] {
        if fields.iter().any(|field| field == column) || rows.iter().any(|row| row.has(column)) {
            selected.retain(|row| {
                let value = row.at(column);
                *value != J::Null && !matches!(value.text().to_lowercase().as_str(), "nan" | "none" | "") && loose_float(value, -1.0).trunc() as i64 == wanted
            });
        } else if wanted != 0 {
            warnings.push(format!("{column} column missing; ignoring {}={wanted} filter", column.to_lowercase()));
        }
    }
    if selected.is_empty() {
        return obj! {"status" => "NO_DATA", "message" => format!("No data found for Host {host} Device {device} Node {node}")};
    }
    let named = |name: &str| percentage(&selected.iter().copied().filter(|row| row.at("Name").str() == Some(name)).collect::<Vec<_>>());
    let mut names: Vec<String> = Vec::new();
    for row in &selected {
        if row.at("Name").truthy() && !names.contains(&row.at("Name").text()) {
            names.push(row.at("Name").text());
        }
    }
    let average = |prefix: &str| {
        let values: Vec<f64> = names.iter().filter(|name| NUMBERED.is_match(name) && name.starts_with(prefix)).filter_map(|name| named(name)).collect();
        (!values.is_empty()).then(|| round(fsum(values.iter().copied()) / values.len() as f64, 2))
    };
    let mxu_values: Vec<f64> = names.iter().filter(|name| NUMBERED.is_match(name) && name.starts_with("MXU")).filter_map(|name| named(name)).collect();
    let hbm = percentage(&selected.iter().copied().filter(|row| matches!(row.at("Name").str(), Some("HBM Rd+Wr (per chip)" | "HBM Rd+Wr"))).collect::<Vec<_>>());
    let idleness = if selected.iter().any(|row| row.at("Name").str() == Some("No MXU Busy")) {
        named("No MXU Busy")
    } else if !mxu_values.is_empty() {
        Some(round(100.0 - mxu_values.iter().cloned().fold(f64::MIN, f64::max), 2).max(0.0))
    } else {
        Some(named("Avg MXU Busy").map_or(100.0, |busy| round(100.0 - busy, 2)))
    };
    let results = [
        ("hbm_bandwidth_utilization_percent", hbm),
        ("ici_read_utilization_percent", named("ICI (Read)")),
        ("ici_write_utilization_percent", named("ICI (Write)")),
        ("vector_alu_utilization_percent", named("Vector ALUs")),
        ("scalar_unit_utilization_percent", named("Scalar Unit")),
        ("vmem_cmem_stores_utilization_percent", named("Vmem/Cmem Stores")),
        ("vmem_loads_utilization_percent", named("Vmem Loads")),
        ("cmem_loads_utilization_percent", named("Cmem Loads")),
        ("xlu_utilization_percent", average("XLU")),
        ("mxu_utilization_percent", named("Avg MXU Busy").or_else(|| average("MXU"))),
        ("idleness_percent", idleness),
    ];
    let mut output = J::Map(results.into_iter().filter_map(|(key, value)| Some((key.to_string(), J::Float(value?)))).collect());
    let metrics: Vec<(String, J)> = names.iter().filter_map(|name| Some((name.clone(), J::Float(named(name)?)))).collect();
    if !metrics.is_empty() {
        output.set("metrics", J::Map(metrics));
    }
    if !warnings.is_empty() {
        output.set("warnings", warnings);
    }
    output
}

pub fn utilization_viewer(client: &dyn Client, session: &str, host: i64, device: i64, node: i64, bypass: bool) -> Result<J, Error> {
    let mut params = vec![("tqx", "out:csv".to_string())];
    params.push(super::bypass(bypass));
    let raw = client
        .fetch_text("utilization_viewer.json", session, &params)
        .map_err(|error| rethrow(error, |error| format!("Error fetching utilization_viewer.json for session {session}: {}", error.repr())))?;
    Ok(match raw {
        None => obj! {"status" => "NO_DATA", "message" => format!("No data returned for session {session}")},
        Some(raw) => utilization(&raw, host, device, node),
    })
}

pub fn get_utilization_viewer(client: &dyn Client, args: &Args) -> Result<Out, Error> {
    let index = |name: &str| args.int(name, 0).map_err(|_| Error::new(Kind::Value, format!("invalid literal for int() with base 10: {}", args.get(name).map_or_else(String::new, J::repr))));
    Ok(utilization_viewer(client, &args.session(), index("host")?, index("device")?, index("node")?, args.flag("bypass_cache", false))?.into())
}
