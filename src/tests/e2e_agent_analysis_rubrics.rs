use super::cli_support::{parse, run, scratch};
use super::e2e_oracles::demo;
use crate::cli::json::J;

fn query(name: &str, command: &str, flags: &[&str]) -> J {
    let dir = scratch(name);
    let path = demo(&dir);
    let argv: Vec<&str> = [command, path.to_str().unwrap()].into_iter().chain(flags.iter().copied()).collect();
    let (code, out, _) = run(&argv);
    assert_eq!(code, 0, "{out}");
    parse(&out)
}

#[test]
fn test_a01_steptime_and_duty_cycle_rubric() {
    assert!(query("a01", "get_overview", &[]).has("performance_summary"));
}

#[test]
fn test_a02_roofline_bound_rubric() {
    assert!(matches!(query("a02", "get_roofline_model", &[]), J::Map(_)));
}

#[test]
fn test_a03_hottest_op_rubric() {
    let result = query("a03", "get_top_hlo_ops", &["--limit=1"]);
    let top = result.at("top_by_time").items();
    assert_eq!(top.len(), 1);
    assert!(top[0].has("name") && top[0].has("total_self_time_ms"));
}

#[test]
fn test_a04_source_provenance_attribution_rubric() {
    let result = query("a04", "get_top_hlo_ops", &["--limit=5"]);
    let top = result.at("top_by_time").items();
    assert!(top.iter().any(|op| op.has("source_file")));
    assert!(top.iter().filter_map(|op| op.get("source_file")).all(|file| file.str().is_some()));
}

#[test]
fn test_a05_hbm_headroom_rubric() {
    assert!(matches!(query("a05", "get_memory_profile", &[]), J::Map(_)));
}

#[test]
fn test_a06_hlo_op_profile_category_macro_breakdown_rubric() {
    let result = query("a06", "get_hlo_op_profile", &["--view=category"]);
    assert!(result.has("category_summary") && !result.has("grouped_operations"));
    let hints = result.at("navigation_hints");
    assert!(hints.has("available_categories") && hints.has("drill_down_into_top_category"));
    for category in result.at("category_summary").items() {
        assert!(category.has("category") && category.has("total_self_time_ms") && category.has("fraction_of_total_time"));
    }
}

#[test]
fn test_a07_hlo_op_profile_grouped_navigation_rubric() {
    let result = query("a07", "get_hlo_op_profile", &["--view=grouped", "--top_n=3"]);
    assert!(result.has("category_summary") && result.has("grouped_operations"));
    let hints = result.at("navigation_hints");
    assert!(hints.has("drill_down_category") && hints.has("available_categories"));
    for (_, ops) in result.at("grouped_operations").entries() {
        assert!(matches!(ops, J::List(_)));
        for op in ops.items() {
            assert!(["name", "category", "category_fraction", "total_self_time_ms"].iter().all(|key| op.has(key)));
        }
    }
}

#[test]
fn test_a08_hlo_op_profile_category_drill_down_rubric() {
    let dir = scratch("a08");
    let path = demo(&dir);
    let macro_view = parse(&run(&["get_hlo_op_profile", path.to_str().unwrap(), "--view=category"]).1);
    let target = macro_view.at("category_summary").items()[0].at("category").text();
    let result = parse(&run(&["get_hlo_op_profile", path.to_str().unwrap(), &format!("--category={target}"), "--top_n=5"]).1);
    assert_eq!(result.at("category").text(), target);
    assert!(["total_self_time_ms", "fraction_of_total_time", "operations", "navigation_hints"].iter().all(|key| result.has(key)));
    let hints = result.at("navigation_hints");
    assert!(hints.has("inspect_top_op_ast") && hints.has("inspect_graph") && hints.has("back_to_categories"));
    for op in result.at("operations").items() {
        assert!(op.has("name") && op.has("total_self_time_ms") && op.has("category_fraction"));
    }
}

#[test]
fn test_a09_hlo_op_profile_tree_view_rubric() {
    let result = query("a09", "get_hlo_op_profile", &["--view=tree", "--path=by_category", "--depth=2"]);
    assert!(["by_category", "by_program", "root"].contains(&result.at("current_path").text().as_str()));
    assert_eq!(result.at("depth_limit").int(), Some(2));
    let hints = result.at("navigation_hints");
    assert!(hints.has("available_child_paths") && hints.has("navigate_deeper_into_child"));
    assert!(result.at("tree").has("name") && result.at("tree").has("children"));
}

#[test]
fn test_a10_hlo_op_profile_flat_backward_compatibility_rubric() {
    let result = query("a10", "get_hlo_op_profile", &["--view=flat", "--top_n=5"]);
    let ops = result.items();
    assert!(matches!(result, J::List(_)) && ops.len() <= 5);
    for op in ops {
        assert!(op.has("name") && op.has("category") && op.has("total_self_time_ms"));
    }
}
