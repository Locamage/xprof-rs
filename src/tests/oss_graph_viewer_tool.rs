use super::cli_support::{Fake, args, text};
use crate::cli::hlo::get_graph_viewer;
use crate::cli::json::J;
use crate::cli::{Error, Kind};

fn options(fake: &Fake) -> Vec<(String, String)> {
    let calls = fake.calls.borrow();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "graph_viewer");
    let mut options = calls[0].1.clone();
    options.sort();
    options
}

fn expected(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = pairs.iter().map(|(key, value)| (key.to_string(), value.to_string())).collect();
    pairs.sort();
    pairs
}

#[test]
fn test_get_graph_viewer_missing_args() {
    let error = get_graph_viewer(&Fake::fixed("unused"), &args("", &[])).unwrap_err();
    assert_eq!(error, Error::new(Kind::Value, "Either session_id or symbol_id must be provided"));
}

#[test]
fn test_get_graph_viewer_both_args() {
    let error = get_graph_viewer(&Fake::fixed("unused"), &args("session_123", &[("symbol_id", J::from("symbol_123"))])).unwrap_err();
    assert_eq!(error, Error::new(Kind::Value, "Cannot set both session_id and symbol_id"));
}

#[test]
fn test_get_graph_viewer_with_session_id() {
    let fake = Fake::fixed("hlo content");
    assert_eq!(text(get_graph_viewer(&fake, &args("session_123", &[]))), "hlo content");
    assert_eq!(options(&fake), expected(&[("graph_type", "xla"), ("type", "short_txt"), ("show_metadata", "true")]));
}

#[test]
fn test_get_graph_viewer_with_symbol_id() {
    let fake = Fake::fixed("hlo content");
    assert_eq!(text(get_graph_viewer(&fake, &args("", &[("symbol_id", J::from("symbol_123"))]))), "hlo content");
    assert_eq!(options(&fake), expected(&[("graph_type", "xla"), ("type", "short_txt"), ("show_metadata", "true"), ("symbol_id", "symbol_123")]));
}

#[test]
fn test_get_graph_viewer_with_advanced_params() {
    let fake = Fake::fixed("graph content");
    let flags = [
        ("node_name", J::from("fusion.112")),
        ("module_name", J::from("jit_train_step")),
        ("graph_width", J::Int(2)),
        ("show_metadata", J::Bool(false)),
        ("merge_fusion", J::Bool(true)),
        ("graph_type", J::from("xla")),
        ("tag", J::from("graph_viewer")),
        ("tool", J::from("hlo_op_profile")),
        ("op_profile_limit", J::Int(1)),
        ("use_xplane", J::Int(1)),
        ("output_type", J::from("short_txt")),
    ];
    assert_eq!(text(get_graph_viewer(&fake, &args("session_123", &flags))), "graph content");
    assert_eq!(
        options(&fake),
        expected(&[
            ("graph_type", "xla"),
            ("type", "short_txt"),
            ("show_metadata", "false"),
            ("node_name", "fusion.112"),
            ("module_name", "jit_train_step"),
            ("graph_width", "2"),
            ("merge_fusion", "true"),
            ("tag", "graph_viewer"),
            ("tool", "hlo_op_profile"),
            ("op_profile_limit", "1"),
            ("use_xplane", "1"),
        ])
    );
}

#[test]
fn test_missing_hlo_proto_returns_clean_diagnostic() {
    let fake = Fake::new(|_, _| Err(Error::new(Kind::Value, "Can not load hlo proto from options.")));
    let error = get_graph_viewer(&fake, &args("session_without_hlo", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
    assert!(error.message.to_lowercase().contains("xla_flags"));
}

#[test]
fn test_empty_graph_viewer_data_raises_file_not_found() {
    let error = get_graph_viewer(&Fake::fixed(""), &args("session_empty", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
    assert!(error.message.contains("No graph_viewer data found"));
}
