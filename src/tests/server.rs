use crate::{DEFAULT_PORT, Settings, arguments};
use std::path::PathBuf;

fn run(list: &[&str]) -> Result<Settings, String> {
    arguments(list.iter().map(std::string::ToString::to_string))
}

fn logdir_of(logdir: &str) -> PathBuf {
    run(&["--logdir", logdir]).unwrap().logdir
}

#[test]
fn test_get_abs_path() {
    let absolute = std::env::temp_dir().canonicalize().unwrap();
    assert_eq!(logdir_of(&absolute.to_string_lossy()), absolute);
    assert_eq!(
        logdir_of("~"),
        PathBuf::from(std::env::var("HOME").unwrap())
            .canonicalize()
            .unwrap()
    );
    assert_eq!(
        logdir_of("src/tests"),
        std::env::current_dir()
            .unwrap()
            .join("src/tests")
            .canonicalize()
            .unwrap()
    );
}

#[test]
fn test_start_server() {
    let dir = std::env::temp_dir().canonicalize().unwrap();
    let path = dir.to_string_lossy().into_owned();
    assert_eq!(
        run(&["--port", "1234", "--grpc_port", "50051"]),
        Ok(Settings {
            port: 1234,
            ..Default::default()
        })
    );
    assert_eq!(
        run(&["--logdir", &path, "--port", "5678"]),
        Ok(Settings {
            logdir: dir,
            port: 5678,
            ..Default::default()
        })
    );
    assert_eq!(
        run(&[
            "--port",
            "1234",
            "--hide_capture_profile_button",
            "--src_prefix",
            ""
        ]),
        Ok(Settings {
            port: 1234,
            hide_capture_profile_button: true,
            src_prefix: Some(String::new()),
            ..Default::default()
        })
    );
}

#[test]
fn test_start_server_errors() {
    assert!(
        run(&["--port", "50051", "--grpc_port", "50051"])
            .unwrap_err()
            .contains("The main server port")
    );
    assert!(
        run(&["--logdir", "/nonexistent/xprof-rs/log", "--port", "3456"])
            .unwrap_err()
            .contains("Log directory")
    );
}

#[test]
fn test_open_uri_pattern_is_validated_and_anchored() {
    let settings = run(&["--open_uri_pattern", "s3://[^/]+/profile"]).unwrap();
    assert_eq!(
        settings.open_uri_pattern.as_deref(),
        Some("s3://[^/]+/profile")
    );
    let server = crate::state(&settings);
    let pattern = server.open_uri_pattern.as_ref().unwrap();
    assert!(pattern.is_match("s3://bucket/profile"));
    assert!(!pattern.is_match("prefix-s3://bucket/profile"));
    assert!(
        run(&["--open_uri_pattern", "("])
            .unwrap_err()
            .contains("invalid regex")
    );
}

#[test]
fn test_main_intercepts_cli_subcommand() {
    for subcommand in [
        "get_overview",
        "get_roofline_model",
        "check_host_boundness",
        "get_top_hlo_ops",
    ] {
        let (code, out, _) =
            crate::cli::execute(&[subcommand.to_string(), "/path/to/trace".into()]).unwrap();
        assert_eq!(code, 3);
        assert!(
            String::from_utf8_lossy(&out).contains("Trace path '/path/to/trace' does not exist.")
        );
    }
    assert!(crate::cli::execute(&["--logdir".into(), "/tmp".into()]).is_none());
}

#[test]
fn test_main_launches_server() {
    let dir = std::env::temp_dir().canonicalize().unwrap();
    assert_eq!(
        run(&["--logdir", &dir.to_string_lossy(), "--port", "1234"]),
        Ok(Settings {
            logdir: dir,
            port: 1234,
            ..Default::default()
        })
    );
    assert_eq!(run(&[]).map(|settings| settings.port), Ok(DEFAULT_PORT));
}

#[test]
fn test_server_subcommand_starts_server() {
    assert_eq!(
        run(&["server", "--port", "1234"]),
        Ok(Settings {
            port: 1234,
            ..Default::default()
        })
    );
    assert!(crate::cli::execute(&["server".into()]).is_none());
}

#[test]
fn test_value_flag_does_not_take_the_next_flag() {
    assert_eq!(
        run(&["--logdir", "--port", "1234"]),
        Err("argument --logdir: expected one argument".into())
    );
    assert_eq!(
        run(&["--src_prefix"]),
        Err("argument --src_prefix: expected one argument".into())
    );
    assert_eq!(
        run(&["--src_prefix", "-1.5"]).map(|settings| settings.src_prefix),
        Ok(Some("-1.5".into()))
    );
    assert_eq!(
        run(&["--src_prefix", "-a b"]).map(|settings| settings.src_prefix),
        Ok(Some("-a b".into()))
    );
    assert_eq!(
        run(&["--src_prefix", "-"]).map(|settings| settings.src_prefix),
        Ok(Some("-".into()))
    );
    assert_eq!(
        run(&["--src_prefix=--x"]).map(|settings| settings.src_prefix),
        Ok(Some("--x".into()))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn open_stages_allowlisted_profiles_with_and_without_a_logdir() {
    use crate::{Body, app, state};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;
    let demo = include_bytes!("../../tests/data/demo.xplane.pb");
    for local_logdir in [None, Some(crate::tests::legacy::logdir("open-local"))] {
        let root = crate::tests::legacy::logdir(if local_logdir.is_some() {
            "open-local-store"
        } else {
            "open-no-logdir-store"
        });
        for host in ["h0", "h1"] {
            std::fs::write(
                root.join(format!("run/plugins/profile/s/{host}.xplane.pb")),
                demo,
            )
            .unwrap();
        }
        let uri = format!("file://{}", root.join("run").display());
        let state = state(&Settings {
            logdir: local_logdir.unwrap_or_default(),
            open_uri_pattern: Some("file://.*".into()),
            ..Default::default()
        });
        let opened = crate::tests::legacy::fetch(
            &state,
            &format!(
                "/open?uri={}&tool=trace_viewer",
                url::form_urlencoded::byte_serialize(uri.as_bytes()).collect::<String>()
            ),
        )
        .await;
        assert!(
            matches!(opened.0, StatusCode::ACCEPTED | StatusCode::SEE_OTHER),
            "{}",
            opened.2
        );
        let stage = state
            .staging
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .clone();
        if opened.0 == StatusCode::ACCEPTED {
            assert!(
                opened.2.contains("./progress?uri=") && opened.2.contains("tool=trace_viewer"),
                "{}",
                opened.2
            );
        }
        let progress_uri = format!(
            "/progress?uri={}&tool=trace_viewer",
            url::form_urlencoded::byte_serialize(uri.as_bytes()).collect::<String>()
        );
        let ready = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let reply = crate::tests::legacy::fetch(&state, &progress_uri).await;
                let value: serde_json::Value = serde_json::from_str(&reply.2).unwrap();
                if value["state"] == "ready" {
                    break value;
                }
                assert!(
                    matches!(value["state"].as_str(), Some("starting" | "downloading")),
                    "{value}: {:?}",
                    stage.result.lock().unwrap()
                );
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            }
        })
        .await
        .unwrap();
        let location = url::Url::parse("http://localhost/")
            .unwrap()
            .join(ready["location"].as_str().unwrap())
            .unwrap();
        assert_eq!(
            location
                .query_pairs()
                .find(|(key, _)| key == "tool")
                .unwrap()
                .1,
            "trace_viewer"
        );
        let profile = location
            .query_pairs()
            .find(|(key, _)| key == "run_path")
            .unwrap()
            .1
            .into_owned();
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        query.append_pair("run_path", &profile);
        assert_eq!(
            crate::tests::legacy::fetch(&state, &format!("/runs?{}", query.finish()))
                .await
                .2,
            "[\"s\"]"
        );
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        query
            .append_pair("run_path", &profile)
            .append_pair("run", "s")
            .append_pair("tag", "overview_page")
            .append_pair("host", "ALL_HOSTS");
        let served =
            crate::tests::legacy::fetch(&state, &format!("/data?{}", query.finish())).await;
        assert_eq!(served.0, StatusCode::OK, "{}", served.2);
        let stage_mirror = state
            .staging
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .remote
            .mirror
            .clone();
        std::fs::remove_dir_all(stage_mirror).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    let unrestricted = state(&Settings::default());
    assert_eq!(
        crate::tests::legacy::fetch(&unrestricted, "/open?uri=file%3A%2F%2Ftmp%2Fprofile")
            .await
            .0,
        StatusCode::FORBIDDEN
    );
    let restricted = state(&Settings {
        open_uri_pattern: Some("gs://.*".into()),
        ..Default::default()
    });
    assert_eq!(
        crate::tests::legacy::fetch(&restricted, "/open?uri=file%3A%2F%2Ftmp%2Fprofile")
            .await
            .0,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        crate::tests::legacy::fetch(&restricted, "/open").await.0,
        StatusCode::BAD_REQUEST
    );
    let post = app(restricted)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/open?uri=gs%3A%2F%2Fbucket%2Fprofile")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(post.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn hidden_capture_routes_do_not_connect_to_the_requested_address() {
    use crate::{Settings, state};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let state = state(&Settings {
        hide_capture_profile_button: true,
        ..Default::default()
    });
    for path in ["/capture_profile", "/data/plugin/profile/capture_profile"] {
        let reply =
            crate::tests::legacy::fetch(&state, &format!("{path}?service_addr={address}")).await;
        assert_eq!(
            (reply.0, reply.2.as_str()),
            (axum::http::StatusCode::NOT_FOUND, "Not Found\n")
        );
    }
    let error = listener.accept().unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
}

#[tokio::test]
async fn healthz_returns_plain_text_ok() {
    let state = crate::state(&Settings::default());
    let reply = crate::tests::legacy::fetch(&state, "/healthz").await;
    assert_eq!(reply.0, axum::http::StatusCode::OK);
    assert_eq!(
        reply.1[axum::http::header::CONTENT_TYPE],
        "text/plain; charset=utf-8"
    );
    assert_eq!(reply.2, "ok\n");
}

#[test]
fn top_level_help_lists_every_command() {
    let help = crate::top_help();
    assert!(help.starts_with("usage: xprof-rs "));
    for (name, synopsis, _) in crate::cli::COMMANDS {
        assert!(help.contains(&format!("  {name} {synopsis}")), "{name}");
    }
    assert!(help.ends_with("xprof-rs <command> --help shows its parameters\n"));
}

#[test]
fn release_version_has_a_changelog_section() {
    let section = format!("## {}", env!("CARGO_PKG_VERSION"));
    assert!(
        include_str!("../../CHANGELOG.md")
            .lines()
            .any(|line| line == section)
    );
}
