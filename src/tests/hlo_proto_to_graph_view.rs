use crate::hlo::graph::serve;
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/graph_viewer")
}

fn view(options: &[(&str, &str)]) -> Result<String, String> {
    let params: HashMap<String, String> = options.iter().map(|&(key, value)| (key.to_string(), value.to_string())).collect();
    serve(&fixtures(), &params).map(|(body, _)| String::from_utf8(body).unwrap())
}

#[test]
fn graph_type() {
    let defaults = [("module_name", "fused_while"), ("type", "graph"), ("node_name", "fusion")];
    assert_eq!(view(&defaults), Err("Can't render as URL; no URL renderer was registered.".to_string()));
    let html = [&defaults[..], &[("format", "html")]].concat();
    let params1 = view(&html).unwrap();
    let explicit = [&html[..], &[("graph_width", "3"), ("show_metadata", "false"), ("merge_fusion", "false")]].concat();
    assert_eq!(view(&explicit).unwrap(), params1);
    assert!(params1.contains("Fused expression for <b>fusion</b>"));
    assert!(!params1.contains("op_type: Mul"));
    let options2 = [&html[..], &[("graph_width", "10"), ("show_metadata", "true"), ("merge_fusion", "true")]].concat();
    let params2 = view(&options2).unwrap();
    assert!(params2.contains("op_type: Mul"));
    assert!(!params2.contains("Fused expression for <b>fusion</b>"));
    let chain = |width: &'static str| view(&[("module_name", "chain"), ("type", "graph"), ("node_name", "a15"), ("format", "html"), ("graph_width", width)]).unwrap();
    assert_ne!(chain("10"), chain("3"));
    assert_eq!(chain("abc"), chain("3"));
}

#[test]
fn short_txt_type() {
    let params1 = view(&[("module_name", "fused_while"), ("type", "short_txt")]).unwrap();
    assert!(params1.contains("  mul = f32[4]{0} multiply(p0, p1)\n"));
    assert!(!params1.contains("metadata="));
    let params2 = view(&[("module_name", "fused_while"), ("type", "short_txt"), ("show_metadata", "true")]).unwrap();
    assert!(params2.contains("  mul = f32[4]{0} multiply(p0, p1)\n"));
    assert!(params2.contains("metadata={op_type=\"Mul\" op_name=\"jit(f)/while/body/mul\" source_file=\"f.py\" source_line=7}"));
}

#[test]
fn long_txt_type() {
    let params1 = view(&[("module_name", "fused_while"), ("type", "long_txt")]).unwrap();
    assert!(params1.contains("  %mul = f32[4]{0} multiply(%p0, %p1)\n"));
    assert!(!params1.contains("metadata="));
    let params2 = view(&[("module_name", "fused_while"), ("type", "long_txt"), ("show_metadata", "true")]).unwrap();
    assert!(params2.contains("  %mul = f32[4]{0} multiply(%p0, %p1)\n"));
    assert!(params2.contains("metadata={op_type=\"Mul\" op_name=\"jit(f)/while/body/mul\" source_file=\"f.py\" source_line=7}"));
}

#[test]
fn adj_nodes_type() {
    assert_eq!(view(&[("module_name", "fused_while"), ("type", "adj_nodes"), ("node_name", "fusion")]).unwrap(), r#"{"consumer_names":["t"],"operand_names":["x","x"]}"#);
}

#[test]
fn other_types() {
    assert!(view(&[("module_name", "fused_while")]).unwrap_err().contains("Graph viewer must provide a type option"));
    assert!(view(&[("module_name", "fused_while"), ("type", "abcd")]).unwrap_err().contains("Unknown graph viewer type option: abcd"));
}

#[test]
fn graph_viewer_matches_xprof_on_synthetic_modules() {
    let mut text = String::new();
    GzDecoder::new(&include_bytes!("../../tests/data/graph_viewer/expected.json.gz")[..]).read_to_string(&mut text).unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let dir = fixtures();
    for case in &cases {
        let options: Vec<(&str, &str)> = case["params"].as_object().unwrap().iter().map(|(key, value)| (key.as_str(), value.as_str().unwrap())).collect();
        let expected = case["body"].as_str().unwrap().replace("{dir}", &dir.display().to_string());
        let actual = match view(&options) {
            Ok(body) => (200, body),
            Err(message) => (500, message),
        };
        assert_eq!(actual, (case["status"].as_u64().unwrap(), expected), "{options:?}");
    }
    let params = HashMap::from([("module_name".to_string(), "ops".to_string()), ("type".to_string(), "pb".to_string())]);
    assert_eq!(serve(&dir, &params).unwrap().0, std::fs::read(dir.join("ops.hlo_proto.pb")).unwrap());
}
