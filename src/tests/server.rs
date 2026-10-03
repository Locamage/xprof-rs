use crate::{DEFAULT_PORT, Settings, arguments};
use std::path::PathBuf;

fn run(list: &[&str]) -> Result<Settings, String> {
    arguments(list.iter().map(|argument| argument.to_string()))
}

fn logdir_of(logdir: &str) -> PathBuf {
    run(&["--logdir", logdir]).unwrap().logdir
}

#[test]
fn test_get_abs_path() {
    let absolute = std::env::temp_dir().canonicalize().unwrap();
    assert_eq!(logdir_of(&absolute.to_string_lossy()), absolute);
    assert_eq!(logdir_of("~"), PathBuf::from(std::env::var("HOME").unwrap()).canonicalize().unwrap());
    assert_eq!(logdir_of("src/tests"), std::env::current_dir().unwrap().join("src/tests").canonicalize().unwrap());
}

#[test]
fn test_start_server() {
    let dir = std::env::temp_dir().canonicalize().unwrap();
    let path = dir.to_string_lossy().into_owned();
    assert_eq!(run(&["--port", "1234", "--grpc_port", "50051"]), Ok(Settings { port: 1234, ..Default::default() }));
    assert_eq!(run(&["--logdir", &path, "--port", "5678"]), Ok(Settings { logdir: dir.clone(), port: 5678, ..Default::default() }));
    assert_eq!(
        run(&["--port", "1234", "--hide_capture_profile_button", "--src_prefix", ""]),
        Ok(Settings { port: 1234, hide_capture_profile_button: true, src_prefix: Some(String::new()), ..Default::default() })
    );
}

#[test]
fn test_start_server_errors() {
    assert!(run(&["--port", "50051", "--grpc_port", "50051"]).unwrap_err().contains("The main server port"));
    assert!(run(&["--logdir", "/nonexistent/xprof-rs/log", "--port", "3456"]).unwrap_err().contains("Log directory"));
}

#[test]
fn test_main_intercepts_cli_subcommand() {
    for subcommand in ["get_overview", "get_roofline_model", "check_host_boundness", "get_top_hlo_ops"] {
        let (code, out, _) = crate::cli::execute(&[subcommand.to_string(), "/path/to/trace".into()]).unwrap();
        assert_eq!(code, 3);
        assert!(String::from_utf8_lossy(&out).contains("Trace path '/path/to/trace' does not exist."));
    }
    assert!(crate::cli::execute(&["--logdir".into(), "/tmp".into()]).is_none());
}

#[test]
fn test_main_launches_server() {
    let dir = std::env::temp_dir().canonicalize().unwrap();
    assert_eq!(run(&["--logdir", &dir.to_string_lossy(), "--port", "1234"]), Ok(Settings { logdir: dir, port: 1234, ..Default::default() }));
    assert_eq!(run(&[]).map(|settings| settings.port), Ok(DEFAULT_PORT));
}

#[test]
fn test_server_subcommand_starts_server() {
    assert_eq!(run(&["server", "--port", "1234"]), Ok(Settings { port: 1234, ..Default::default() }));
    assert!(crate::cli::execute(&["server".into()]).is_none());
}
