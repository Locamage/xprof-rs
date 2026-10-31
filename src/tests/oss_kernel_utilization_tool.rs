use super::cli_support::{Fake, args, json, scratch, text};
use super::xspace::{V, XSpace};
use crate::cli::json::J;
use crate::cli::xplane::get_kernel_utilization;
use crate::cli::{Error, Kind};
use prost::Message;

const MATMUL_JSON: &str = r#"{"status": "SUCCESS", "devices": [{"device_id": 0, "device_type": "TPU v7x", "kernels": [{"kernel_name": "matmul", "mxu_utilization": 25.0, "mxu_is_anomaly": false, "mxu_cycles_breakdown": {"BF16": 100.0, "Int8": 0.0, "Int4": 0.0, "FP8": 0.0}, "other_metrics": {}}]}]}"#;

fn sample() -> Vec<u8> {
    let mut space = XSpace::default();
    let plane = space.plane("/device:TPU:0");
    plane.add_stat("device_id", 0i64);
    plane.add_stat("device_type_string", "TPU v7x");
    plane.named_line(0, "counters_matmul");
    for (name, id, value) in [
        ("CYCLES", 2847490056u64, 2000.0),
        ("MXU_BUSY_1", 2791875024, 800.0),
        ("MATMUL_VREG_BF16_MXU_0", 2791874968, 400.0),
    ] {
        plane.event(
            0,
            name,
            0,
            0,
            &[
                ("performance_counter_id", id.into()),
                ("counter_value", V::Double(value)),
            ],
        );
    }
    space.encode_to_vec()
}

fn trace_file(name: &str) -> String {
    let path = scratch(name).join("test_trace.xplane.pb");
    std::fs::write(&path, sample()).unwrap();
    path.to_string_lossy().into_owned()
}

fn kernel(result: &J) -> &J {
    &result.at("devices").items()[0].at("kernels").items()[0]
}

#[test]
fn test_get_kernel_utilization_from_raw_bytes() {
    let parsed = J::parse(&text(get_kernel_utilization(
        &Fake::fixed(""),
        &args(&trace_file("ku-raw"), &[("output_format", J::from("json"))]),
    )))
    .unwrap();
    assert_eq!(parsed.at("status").str(), Some("SUCCESS"));
    assert_eq!(parsed.at("devices").items().len(), 1);
    let device = &parsed.at("devices").items()[0];
    assert_eq!(device.at("device_type").str(), Some("TPU v7x"));
    assert_eq!(device.at("device_id"), &J::Int(0));
    assert_eq!(device.at("kernels").items().len(), 1);
    let kernel = kernel(&parsed);
    assert_eq!(kernel.at("kernel_name").str(), Some("matmul"));
    assert!(kernel.at("mxu_utilization").float().unwrap() > 0.0);
    assert_eq!(
        kernel.at("mxu_cycles_breakdown").at("BF16").float(),
        Some(100.0)
    );
    assert_eq!(kernel.at("mxu_is_anomaly"), &J::Bool(false));
}

#[test]
fn test_get_kernel_utilization_from_local_file() {
    let result = json(get_kernel_utilization(
        &Fake::fixed(""),
        &args(
            &trace_file("ku-file"),
            &[("output_format", J::from("dict"))],
        ),
    ));
    assert_eq!(result.at("status").str(), Some("SUCCESS"));
    assert_eq!(kernel(&result).at("kernel_name").str(), Some("matmul"));
}

#[test]
fn test_get_kernel_utilization_from_local_directory() {
    let dir = scratch("ku-dir");
    std::fs::write(dir.join("test.xplane.pb"), sample()).unwrap();
    let result = json(get_kernel_utilization(
        &Fake::fixed(""),
        &args(
            &dir.to_string_lossy(),
            &[("output_format", J::from("dict"))],
        ),
    ));
    assert_eq!(result.at("status").str(), Some("SUCCESS"));
    assert_eq!(kernel(&result).at("kernel_name").str(), Some("matmul"));
}

#[test]
fn test_get_kernel_utilization_from_empty_directory_raises_file_not_found() {
    let error = get_kernel_utilization(
        &Fake::fixed(""),
        &args(&scratch("ku-empty").to_string_lossy(), &[]),
    )
    .unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
}

#[test]
fn test_get_kernel_utilization_needs_the_host_name_in_the_file_name() {
    let mut space = XSpace::decode(&sample()[..]).unwrap();
    space.hostnames = vec!["worker-0".into()];
    for (name, ok) in [
        ("worker-0.xplane.pb", true),
        ("job.worker-0.1.xplane.pb", true),
        ("host.xplane.pb", false),
        ("worker-0.xplane.pb.xplane.pb", true),
    ] {
        let path = scratch(&format!("ku-host-{name}")).join(name);
        std::fs::write(&path, space.encode_to_vec()).unwrap();
        let result = get_kernel_utilization(&Fake::fixed(""), &args(&path.to_string_lossy(), &[]));
        assert_eq!(result.is_ok(), ok, "{name}");
        if !ok {
            assert!(result.unwrap_err().message.ends_with(": no data returned."));
        }
    }
}

#[test]
fn test_get_kernel_utilization_session_ignores_host_filename() {
    let logdir = scratch("ku-session-host");
    let run = logdir.join("plugins/profile/s2");
    std::fs::create_dir_all(&run).unwrap();
    let mut space = XSpace::decode(&sample()[..]).unwrap();
    space.hostnames = vec!["worker-0".into()];
    std::fs::write(run.join("host.xplane.pb"), space.encode_to_vec()).unwrap();
    let client = crate::cli::client::Local {
        logdir: Some(logdir),
        ..Default::default()
    };
    let result = json(get_kernel_utilization(&client, &args("s2", &[])));
    assert_eq!(result.at("status").str(), Some("SUCCESS"));
    assert_eq!(kernel(&result).at("kernel_name").str(), Some("matmul"));
}

#[test]
fn test_get_kernel_utilization_dangling_trace_link_raises_path_error() {
    let dir = scratch("ku-dangling");
    let link = dir.join("host.xplane.pb");
    std::os::unix::fs::symlink(dir.join("missing"), &link).unwrap();
    let error =
        get_kernel_utilization(&Fake::fixed(""), &args(&dir.to_string_lossy(), &[])).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
    assert_eq!(
        error.message,
        format!("[Errno 2] No such file or directory: '{}'", link.display())
    );
}

#[test]
fn test_get_kernel_utilization_from_session_id_with_client() {
    let fake = Fake::fixed(MATMUL_JSON);
    let result = json(get_kernel_utilization(
        &fake,
        &args(
            "session_12345",
            &[
                ("kernel_name", J::from("matmul")),
                ("output_format", J::from("dict")),
                ("bypass_cache", J::Bool(true)),
            ],
        ),
    ));
    let calls = fake.calls.borrow();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0],
        (
            "kernel_utilization.json".to_string(),
            vec![
                ("kernel".to_string(), "matmul".to_string()),
                ("bypass_cache".to_string(), "True".to_string())
            ]
        )
    );
    assert_eq!(result.at("status").str(), Some("SUCCESS"));
    assert_eq!(kernel(&result).at("kernel_name").str(), Some("matmul"));
}

#[test]
fn test_get_kernel_utilization_duration_override() {
    let flags = [
        ("duration_us", J::Float(20.0)),
        ("force_duration", J::Bool(true)),
        ("output_format", J::from("dict")),
    ];
    let result = json(get_kernel_utilization(
        &Fake::fixed(""),
        &args(&trace_file("ku-duration"), &flags),
    ));
    assert_eq!(result.at("status").str(), Some("SUCCESS"));
    assert!((kernel(&result).at("duration_us").float().unwrap() - 20.0).abs() < 1e-3);
}

#[test]
fn test_get_kernel_utilization_empty_session_and_raw_bytes_raises_value_error() {
    let error = get_kernel_utilization(&Fake::fixed(""), &args("", &[])).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
}

#[test]
fn test_get_kernel_utilization_client_no_data_raises_file_not_found_error() {
    let error = get_kernel_utilization(
        &Fake::new(|_, _| Ok(None)),
        &args("non_existent_session", &[]),
    )
    .unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
}

#[test]
fn test_get_kernel_utilization_client_error_raises_runtime_error() {
    let error = get_kernel_utilization(
        &Fake::new(|_, _| Err(Error::new(Kind::Os, "Network error"))),
        &args("session_err", &[]),
    )
    .unwrap_err();
    assert_eq!(error.kind, Kind::Runtime);
}
