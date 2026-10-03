use super::cli_support::{args, run};
use crate::cli::Kind;
use crate::cli::client::Local;
use crate::cli::json::J;
use crate::cli::xplane::verify_numerical_parity;

const DEPENDENCIES: &str = "Required numerical dependencies are not installed in current environment: No module named 'ml_dtypes'. Please install numpy and ml_dtypes (e.g. 'pip install numpy ml_dtypes') to use verify_numerical_parity.";

fn unavailable(argv: &[&str]) {
    let (code, out, err) = run(argv);
    assert_eq!(code, 1, "{out}");
    let payload = J::parse(&out).unwrap();
    assert_eq!(payload.at("reason").str(), Some("INTERNAL_ERROR"));
    assert!(payload.at("error").text().starts_with(DEPENDENCIES));
    assert!(payload.at("traceback").text().contains(&format!("ImportError: {DEPENDENCIES}")));
    assert!(err.contains("INTERNAL_ERROR"));
}

#[test]
fn test_verify_with_string_dotted_import_paths() {
    unavailable(&["verify_numerical_parity", "numpy.sin", "numpy.sin", "[(8, 8)]", "--dtype_str=float32", "--tier=fast_agent"]);
}

#[test]
fn test_verify_with_string_colon_import_paths() {
    unavailable(&["verify_numerical_parity", "numpy:cos", "numpy:cos", "[(8, 8)]", "--dtype_str=float32", "--tier=fast_agent"]);
}

#[test]
fn test_verify_with_string_shapes_literal() {
    unavailable(&["verify_numerical_parity", "--kernel_ref=numpy.sin", "--kernel_candidate=numpy.cos", "--shapes=[(16, 32)]", "--dtype_str=float32"]);
}

#[test]
fn test_resolve_callable_invalid_module_raises() {
    let error = verify_numerical_parity(&Local::default(), &args("", &[("kernel_ref", J::from("non_existent_module_xyz.some_fn"))])).unwrap_err();
    assert_eq!(error.kind, Kind::Import);
    assert_eq!(error.message, DEPENDENCIES);
}
