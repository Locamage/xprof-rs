use super::cli_support::{parse, run, scratch};
use super::e2e_oracles::{TRAINING_STEP_PS, demo, session, training_trace};
use crate::cli::client::{Client, Local};
use std::time::{Duration, Instant};

const SMALL_BUDGET: Duration = Duration::from_secs(10);
const MEDIUM_BUDGET: Duration = Duration::from_secs(35);
const PARITY_ARGS: [&str; 4] =
    ["verify_numerical_parity", "module.ref_kernel", "module.candidate_kernel", "[(4, 16)]"];

#[test]
fn test_p03_ulp_ground_truth_oracle() {
    let (code, out, err) = run(&PARITY_ARGS);
    assert_eq!(code, 1);
    let report = parse(&out);
    assert_eq!(report.at("reason").str(), Some("INTERNAL_ERROR"));
    assert!(report.at("error").text().contains("No module named 'ml_dtypes'"), "{out}");
    assert!(err.starts_with("INTERNAL_ERROR: Required numerical dependencies are not installed"));
}

#[test]
fn test_p06_random_seed_determinism() {
    let seeded: Vec<&str> =
        PARITY_ARGS.iter().copied().chain(["--seed=12345", "--dtype_str=float32"]).collect();
    assert_eq!(run(&seeded), run(&seeded));
}

#[test]
fn test_s01_scale_budget_small_trace() {
    let path = session(&scratch("s01"), "host.xplane.pb", &training_trace(TRAINING_STEP_PS));
    let start = Instant::now();
    let (code, out, _) = run(&["get_kpi_metrics", path.to_str().unwrap()]);
    assert!(start.elapsed() < SMALL_BUDGET);
    assert_eq!(code, 0);
    assert!(!parse(&out).has("error"));
}

#[test]
fn test_s02_scale_budget_medium_trace() {
    let path = demo(&scratch("s02"));
    let start = Instant::now();
    let (code, out, _) = run(&["get_kpi_metrics", path.to_str().unwrap()]);
    assert!(start.elapsed() < MEDIUM_BUDGET);
    assert_eq!(code, 0);
    assert!(!parse(&out).has("error"));
}

#[test]
fn test_s03_timeout_surfacing() {
    let path = session(&scratch("s03"), "host.xplane.pb", &training_trace(TRAINING_STEP_PS));
    assert!(
        Local::default()
            .fetch("overview_page", path.to_str().unwrap(), &[("rpc_deadline_s", "600".into())])
            .unwrap()
            .is_some()
    );
}
