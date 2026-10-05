mod capture;
mod check_host_boundness_tool;
mod cli_support;
mod compute_inference_latency;
mod counter_extractor;
mod csv_writer;
mod data_table_utils;
mod dcn_collective_stats;
mod delta;
mod derived_timeline;
mod duty_cycle_combiner;
mod duty_cycle_tracker;
mod e2e_agent_analysis_rubrics;
mod e2e_cross_cutting_contracts;
mod e2e_defect_regressions;
mod e2e_numerical_fidelity;
mod e2e_numerical_parity_and_scale;
mod e2e_oracles;
mod get_hlo_stats_tool;
mod get_kpi_metrics_tool;
mod get_memory_profile_tool;
mod get_overview_tool;
mod get_peak_allocations_tool;
mod get_roofline_model_tool;
mod get_step_trace_tool;
mod get_top_hlo_ops_tool;
mod get_utilization_viewer_tool;
mod gpu_parity;
mod group_events;
mod hardware_type_utils;
mod hlo_cost_analysis_wrapper;
mod hlo_fixture;
mod hlo_instruction;
mod hlo_module;
mod hlo_parser;
mod hlo_proto_map;
mod hlo_proto_to_graph_view;
mod hlo_proto_to_memory_visualization_utils;
mod hlo_proto_to_module;
mod hlo_sharding;
mod inference_stats;
mod inference_stats_grouping;
mod inference_stats_sampler;
mod json_parse;
mod kernel_stats_db;
mod kernel_stats_utils;

mod layout;
mod layout_util;
pub mod legacy;
mod literal;
mod malformed_input;
mod memory_profile;
pub(crate) mod mock_tool_data_provider;
mod multi_xspace_to_inference_stats;
mod op_metrics_db;
mod op_metrics_db_utils;
mod op_metrics_to_record;
mod op_stats;
mod op_stats_combiner;
mod op_stats_to_input_pipeline_analysis;
mod op_stats_to_op_profile;
mod op_stats_to_overview_page;
mod op_stats_to_pod_stats;
mod op_stats_to_pod_viewer;
mod op_stats_to_tf_stats;
mod opstats_adapter;
mod oss_graph_viewer_tool;
mod oss_hlo_tools;
mod oss_kernel_stats_tools;
mod oss_kernel_utilization_tool;
mod oss_upload_trace_tool;
mod oss_xplane_tools;
mod oss_xprof_client;
mod prefix_trie;
mod preprocess_xplane;
mod profile_io;
mod profile_plugin;
mod raw_to_tool_data;
mod roofline_model_utils;
mod rules;
mod server;
mod shape;
mod source_info_utils;
mod step_events;
mod step_intersection;
mod streaming_trace_viewer_processor;
mod tf_op_utils;
mod timespan;
mod tpu_counter_util;
mod tpu_xplane_utils;
mod trace_container;
mod trace_events;
mod trace_events_json;
mod trace_events_to_json;
mod trace_events_util;
mod trace_options;
mod trace_utils;
mod trace_view_options;
mod trace_viewer_visibility;
mod unified_graph_viewer_processor;
mod unified_memory_profile_processor;
mod unified_memory_viewer_processor;
mod unified_trace_viewer_processor;
mod verify_numerical_parity_tool;
mod xla_op_utils;
mod xplane_builder;
mod xplane_hlo_fixer;
mod xplane_to_hlo;
mod xplane_to_perf_counters;
mod xplane_to_tool_names;
mod xplane_to_tools_data;
mod xplane_to_trace_events;
mod xplane_to_utilization_viewer;
mod xplane_utils;
mod xplane_visitor;
mod xprof_cli;
mod xprof_data;
mod xprof_gpu_cost_analysis;
mod xspace;
mod xspace_to_event_time_fraction_analyzer;
mod zstd_compression;

/// Gives the directory of this test run. A run removes the directories of the runs that stopped, so the test files do not collect.
pub fn temp_dir() -> std::path::PathBuf {
    static ROOT: std::sync::LazyLock<std::path::PathBuf> = std::sync::LazyLock::new(|| {
        fn remove(path: &std::path::Path) {
            _ = std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o700));
            std::fs::read_dir(path).into_iter().flatten().map_while(Result::ok).filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir())).for_each(|entry| remove(&entry.path()));
            _ = std::fs::remove_dir_all(path);
        }
        let base = std::env::temp_dir();
        for entry in std::fs::read_dir(&base).into_iter().flatten().map_while(Result::ok) {
            if let Some(pid) = entry.file_name().to_str().and_then(|name| name.strip_prefix("xprof-rs-tests-"))
                && !std::path::Path::new("/proc").join(pid).exists()
            {
                remove(&entry.path());
            }
        }
        let root = base.join(format!("xprof-rs-tests-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        root
    });
    ROOT.clone()
}

pub fn scratch(name: &str) -> std::path::PathBuf {
    let dir = temp_dir().join(name);
    _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn with_file<T>(bytes: &[u8], load: impl FnOnce(&std::path::Path) -> T) -> T {
    static FILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = temp_dir().join(format!("xprof-rs-test-{}-{}.xplane.pb", std::process::id(), FILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::write(&path, bytes).unwrap();
    let value = load(&path);
    std::fs::remove_file(&path).unwrap();
    value
}
