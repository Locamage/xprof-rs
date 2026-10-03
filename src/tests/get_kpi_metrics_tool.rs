use super::cli_support::{Fake, args, json, parse};
use crate::cli::overview::get_kpi_metrics;
use crate::cli::{Error, Kind};

fn kpi_fake(memory: Result<Option<&'static str>, Error>) -> Fake {
    Fake::new(move |tool, _| {
        match tool {
        "overview_page" | "overview_page.json" => Ok(Some(
            r#"[{"p": {"steptime_ms_average": "10.5", "device_duty_cycle_percent": "95.0", "mxu_utilization_percent": "80.0", "flop_rate_utilization_relative_to_roofline": "45.0", "device_type": "TPU", "device_core_count": "8"}}]"#
                .into(),
        )),
        "memory_profile.json" => memory.clone().map(|data| data.map(|data| data.as_bytes().to_vec())),
        _ => Ok(None),
    }
    })
}

#[test]
fn test_get_kpi_metrics_success() {
    let memory = r#"[{"memoryProfilePerAllocator": {"default": {"profileSummary": {"memoryCapacity": "34359738368", "peakStats": {"peakBytesInUse": "13421772800"}}}}}]"#;
    let result = json(get_kpi_metrics(&kpi_fake(Ok(Some(memory))), &args("test_session", &[])));
    let expected = parse(
        r#"{"step_time_ms": "10.5", "duty_cycle_percent": "95.0", "mxu_utilization_percent": "80.0", "roofline_utilization": "45.0", "peak_hbm_gib": 12.5, "accelerator_info": {"device_type": "TPU", "device_core_count": "8"}}"#,
    );
    assert_eq!(result, expected);
}

#[test]
fn test_get_kpi_metrics_overview_error() {
    let fake = Fake::new(|_, _| Ok(Some(Vec::new())));
    let error = get_kpi_metrics(&fake, &args("test_session", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
}

#[test]
fn test_get_kpi_metrics_memory_error() {
    let result = json(get_kpi_metrics(&kpi_fake(Err(Error::new(Kind::Runtime, "Fetch failed"))), &args("test_session", &[])));
    assert!(!result.has("error"));
    assert_eq!(result.at("step_time_ms").str(), Some("10.5"));
    assert_eq!(result.at("peak_hbm_gib").str(), Some("N/A"));
}
