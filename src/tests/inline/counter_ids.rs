use super::*;

#[test]
fn embedded_headers_resolve_names() {
    assert_eq!(V6E_IDS["VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES"], 3257466888);
    assert!(names("v6e").unwrap().starts_with("[{\"name\": \"vf_chip_tc_tcs_tc_misc_tcs_stats_tcs_stats_counters_unprivileged_count_cycles\", \"val\": 3257466888}"));
    assert!(names("v7x").is_some());
    assert!(names("v4").is_none());
}
