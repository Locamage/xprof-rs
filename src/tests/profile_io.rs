use super::legacy::fetch;
use crate::{Settings, Shared, replace_file, state, xplanes};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

fn temp_dir(name: &str) -> PathBuf {
    crate::tests::scratch(&format!("profile-io-{name}")).canonicalize().unwrap()
}

fn served(dir: &std::path::Path) -> Shared {
    state(&Settings { logdir: dir.to_path_buf(), ..Default::default() })
}

#[test]
#[ignore = "differs from upstream: xprof-rs reads only .xplane.pb files, it has no riegeli decoder, so 3.xplane.riegeli is not a profile"]
fn test_get_xplane_basenames() {
    let dir = temp_dir("basenames");
    for (name, content) in [("1.xplane.pb", "test"), ("2.txt", "test2"), ("3.xplane.riegeli", "test3")] {
        std::fs::write(dir.join(name), content).unwrap();
    }
    let mut names: Vec<String> = xplanes(&dir).iter().map(|path| path.file_name().unwrap().to_string_lossy().into_owned()).collect();
    names.sort();
    assert_eq!(names, ["1.xplane.pb", "3.xplane.riegeli"]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dir_has_xplane_files() {
    let dir = temp_dir("has-xplane");
    let state = served(&dir);
    let uri = format!("/data/plugin/profile/runs?session_path={}", dir.display());
    assert_eq!(fetch(&state, &uri).await.2, "[]");
    std::fs::write(dir.join("1.xplane.pb"), "test").unwrap();
    assert_eq!(fetch(&state, &uri).await.2, format!("[\"{}\"]", dir.file_name().unwrap().to_string_lossy()));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_session_paths() {
    let dir = temp_dir("session-paths");
    std::fs::create_dir(dir.join("session1")).unwrap();
    std::fs::write(dir.join("session1/1.xplane.pb"), "test").unwrap();
    std::fs::create_dir(dir.join("session2")).unwrap();
    assert_eq!(fetch(&served(&dir), &format!("/data/plugin/profile/runs?run_path={}", dir.display())).await.2, "[\"session1\"]");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_get_session_paths_oserror() {
    let dir = temp_dir("session-paths-oserror");
    let runs = dir.join("runs");
    std::fs::create_dir_all(runs.join("session1")).unwrap();
    std::fs::write(runs.join("session1/1.xplane.pb"), "test").unwrap();
    std::fs::set_permissions(&runs, std::fs::Permissions::from_mode(0o000)).unwrap();
    let listed = fetch(&served(&dir), &format!("/data/plugin/profile/runs?run_path={}", runs.display())).await.2;
    std::fs::set_permissions(&runs, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert_eq!(listed, "[]");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_replaced_file_has_the_new_bytes_and_no_partial_file_stays() {
    let dir = temp_dir("replace");
    let path = dir.join("out.pb");
    replace_file(&path, b"new file").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"new file");
    let contents = [b"first".repeat(1000), b"second".repeat(1000)];
    std::thread::scope(|scope| {
        for bytes in &contents {
            scope.spawn(|| replace_file(&path, bytes).unwrap());
        }
    });
    assert!(contents.contains(&std::fs::read(&path).unwrap()));
    assert!(replace_file(&dir.join("missing/out.pb"), b"x").is_err());
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    std::fs::remove_dir_all(&dir).unwrap();
}
