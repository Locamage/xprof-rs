use super::cli_support::{Fake, args, json, parse};
use crate::cli::Kind;
use crate::cli::overview::get_memory_profile;

const UNSET: &str = concat!(
    r#"{"memory_capacity_gib": -1.0, "peak_memory_usage_gib": -1.0, "peak_usage_details": "#,
    r#"{"stack_reservation_gib": -1.0, "heap_allocation_gib": -1.0, "free_memory_gib": -1.0, "#,
    r#""fragmentation_percent": -1.0, "utilization_percent": -1.0}}"#,
);
const SIXTEEN: &str = concat!(
    r#"{"memory_capacity_gib": 16.0, "peak_memory_usage_gib": 8.0, "peak_usage_details": "#,
    r#"{"stack_reservation_gib": 1.0, "heap_allocation_gib": 7.0, "free_memory_gib": 8.0, "#,
    r#""fragmentation_percent": 5.0, "utilization_percent": 50.0}}"#,
);

#[test]
fn test_get_memory_profile() {
    let cases = [
        (
            concat!(
                r#"[{"memoryProfilePerAllocator": {"0": {"profileSummary": "#,
                r#"{"memoryCapacity": "17179869184", "peakStats": {"peakBytesInUse": "#,
                r#""8589934592", "stackReservedBytes": "1073741824", "heapAllocatedBytes": "#,
                r#""7516192768", "freeMemoryBytes": "8589934592", "fragmentation": 0.05}}}}}]"#,
            ),
            SIXTEEN,
        ),
        (
            concat!(
                r#"[{"memoryProfileSummary": {"memoryCapacity": "34359738368", "peakStats": "#,
                r#"{"peakBytesUsageHbm": "17179869184", "stackReservedBytes": "2147483648", "#,
                r#""heapAllocatedBytes": "15032385536", "freeMemoryBytes": "17179869184", "#,
                r#""fragmentation": 0.10}}}]"#,
            ),
            concat!(
                r#"{"memory_capacity_gib": 32.0, "peak_memory_usage_gib": 16.0, "#,
                r#""peak_usage_details": {"stack_reservation_gib": 2.0, "#,
                r#""heap_allocation_gib": 14.0, "free_memory_gib": 16.0, "#,
                r#""fragmentation_percent": 10.0, "utilization_percent": 50.0}}"#,
            ),
        ),
        (
            r#"[{"peakMemoryUsageMiB": "4096.0"}]"#,
            concat!(
                r#"{"memory_capacity_gib": -1.0, "peak_memory_usage_gib": 4.0, "#,
                r#""peak_usage_details": {"stack_reservation_gib": -1.0, "#,
                r#""heap_allocation_gib": -1.0, "free_memory_gib": -1.0, "#,
                r#""fragmentation_percent": -1.0, "utilization_percent": -1.0}}"#,
            ),
        ),
        (
            concat!(
                r#"[{"memoryProfileSummary": {"memoryCapacity": "0", "peakStats": "#,
                r#"{"peakBytesUsageHbm": "0"}}}]"#,
            ),
            UNSET,
        ),
    ];
    for (response, expected) in cases {
        let result = json(get_memory_profile(
            &Fake::fixed(response).with_hosts(&[]),
            &args("test_session", &[]),
        ));
        assert!(!result.has("error"));
        assert_eq!(result, parse(expected), "{response}");
    }
}

#[test]
fn test_get_memory_profile_no_data() {
    let fake = Fake::tools(|_| None).with_hosts(&[]);
    assert_eq!(json(get_memory_profile(&fake, &args("test_session", &[]))), parse(UNSET));
}

#[test]
fn test_get_memory_profile_fallback() {
    let fake = Fake::new(|_, params| {
        let host = params.iter().find(|(key, _)| *key == "host").map(|(_, value)| value.as_str());
        Ok(Some(
            match host {
                Some("host-secondary") => concat!(
                    r#"[{"memoryProfileSummary": {"memoryCapacity": "17179869184", "#,
                    r#""peakStats": {"peakBytesUsageHbm": "8589934592", "#,
                    r#""stackReservedBytes": "1073741824", "heapAllocatedBytes": "#,
                    r#""7516192768", "freeMemoryBytes": "8589934592", "fragmentation": "#,
                    r#"0.05}}}]"#,
                ),
                _ => r#"[{"memoryProfileSummary": {"memoryCapacity": "0"}}]"#,
            }
            .as_bytes()
            .to_vec(),
        ))
    })
    .with_hosts(&["host-primary", "host-secondary"]);
    let result = json(get_memory_profile(&fake, &args("test_session", &[])));
    assert!(!result.has("error"));
    assert_eq!(result, parse(SIXTEEN));
    let format = ("format".to_string(), "json".to_string());
    let host = |name: &str| ("host".to_string(), name.to_string());
    let expected = vec![
        ("memory_profile.json".to_string(), vec![format.clone()]),
        ("memory_profile.json".to_string(), vec![format.clone(), host("host-primary")]),
        ("memory_profile.json".to_string(), vec![format, host("host-secondary")]),
    ];
    assert_eq!(*fake.calls.borrow(), expected);
}

#[test]
fn test_get_memory_profile_invalid_json() {
    let error = get_memory_profile(
        &Fake::fixed("invalid json data").with_hosts(&[]),
        &args("test_session", &[]),
    )
    .unwrap_err();
    assert_eq!(error.kind, Kind::Value);
}
