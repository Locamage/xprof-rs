use super::cli_support::{json, run, scratch};
use crate::cli::client::Local;
use crate::cli::json::J;
use crate::cli::xplane::upload_trace;
use crate::cli::{Args, Error, Kind, Out};
use std::path::{Path, PathBuf};

const TRACE: &[u8] = b"mock_xplane_binary_payload_12345";

fn setup(name: &str) -> (PathBuf, PathBuf) {
    let dir = scratch(&format!("upload-{name}"));
    let source = dir.join("source");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join("test_trace.xplane.pb"), TRACE).unwrap();
    (dir.join("logdir"), source)
}

fn upload(logdir: Option<&Path>, file: &Path, run_name: Option<&str>) -> Result<Out, Error> {
    let mut values = vec![("file_path".to_string(), J::from(file.to_string_lossy().into_owned()))];
    values.extend(run_name.map(|run| ("run_name".to_string(), J::from(run))));
    upload_trace(&Local { logdir: logdir.map(Path::to_path_buf), ..Default::default() }, &Args { values })
}

#[test]
fn test_upload_trace_success_with_custom_run_name() {
    let (logdir, source) = setup("custom");
    let result = json(upload(Some(&logdir), &source.join("test_trace.xplane.pb"), Some("my_custom_run")));
    assert_eq!(result.at("status").str(), Some("success"));
    assert_eq!(result.at("run_name").str(), Some("my_custom_run"));
    assert_eq!(std::fs::read(logdir.join("plugins/profile/my_custom_run/test_trace.xplane.pb")).unwrap(), TRACE);
}

#[test]
fn test_upload_trace_default_run_name() {
    let (logdir, source) = setup("default");
    let result = json(upload(Some(&logdir), &source.join("test_trace.xplane.pb"), None));
    assert_eq!(result.at("status").str(), Some("success"));
    assert_eq!(result.at("run_name").str(), Some("imported_trace"));
    assert_eq!(std::fs::read(logdir.join("plugins/profile/imported_trace/test_trace.xplane.pb")).unwrap(), TRACE);
}

#[test]
fn test_upload_trace_missing_logdir_raises_value_error() {
    let (_, source) = setup("nologdir");
    let error = upload(None, &source.join("test_trace.xplane.pb"), Some("test_run")).unwrap_err();
    assert_eq!(error.kind, Kind::Value);
    assert!(error.message.contains("Logdir not set in client"));
}

#[test]
fn test_upload_trace_nonexistent_file_raises_file_not_found() {
    let (logdir, source) = setup("missing");
    let error = upload(Some(&logdir), &source.join("does_not_exist.xplane.pb"), Some("test_run")).unwrap_err();
    assert_eq!(error.kind, Kind::FileNotFound);
    assert!(error.message.contains("does not exist"));
}

#[test]
fn test_upload_trace_invalid_extension_rejected() {
    let (logdir, source) = setup("extension");
    for name in ["trace.trace.json.gz", "trace.txt", "trace.json"] {
        std::fs::write(source.join(name), b"dummy_data").unwrap();
        let error = upload(Some(&logdir), &source.join(name), Some("test_run")).unwrap_err();
        assert_eq!(error.kind, Kind::Value);
        assert!(error.message.contains("Unsupported file format"), "{name}");
    }
}

#[test]
fn test_upload_trace_valid_xspace_pb_accepted() {
    let (logdir, source) = setup("xspace");
    std::fs::write(source.join("trace.xspace.pb"), b"mock_xspace_binary_data").unwrap();
    let result = json(upload(Some(&logdir), &source.join("trace.xspace.pb"), Some("xspace_run")));
    assert_eq!(result.at("status").str(), Some("success"));
    assert_eq!(result.at("run_name").str(), Some("xspace_run"));
    assert!(logdir.join("plugins/profile/xspace_run/trace.xspace.pb").exists());
}

#[test]
fn test_upload_trace_path_traversal_rejected() {
    let (logdir, source) = setup("traversal");
    for run in ["/tmp/ESCAPED", "../../ESCAPED", "nested/sub_run", r"nested\sub_run", ".."] {
        let error = upload(Some(&logdir), &source.join("test_trace.xplane.pb"), Some(run)).unwrap_err();
        assert_eq!(error.kind, Kind::Value, "{run}");
        assert!(error.message.contains("Invalid run_name"), "{run}");
    }
}

#[test]
fn test_upload_trace_copy_failure_raises_os_error() {
    let (logdir, source) = setup("copyfail");
    std::fs::create_dir_all(logdir.join("plugins/profile")).unwrap();
    std::fs::write(logdir.join("plugins/profile/fail_run"), b"a file, not a directory").unwrap();
    let error = upload(Some(&logdir), &source.join("test_trace.xplane.pb"), Some("fail_run")).unwrap_err();
    assert_eq!(error.kind, Kind::Os);
}

#[test]
fn test_upload_trace_ignores_extra_kwargs() {
    let (logdir, source) = setup("kwargs");
    let file = source.join("test_trace.xplane.pb").to_string_lossy().into_owned();
    let logdir_flag = format!("--logdir={}", logdir.display());
    let (code, out, _) = run(&["upload_trace", &file, &logdir_flag, "--ttl=3600", "--tag=tag1", "--run_name=tagged_run", "--extra_param=ignored_value"]);
    assert_eq!(code, 0, "{out}");
    let result = J::parse(&out).unwrap();
    assert_eq!(result.at("status").str(), Some("success"));
    assert_eq!(result.at("run_name").str(), Some("tagged_run"));
}
