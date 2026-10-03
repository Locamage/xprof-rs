use super::*;
use crate::tests::legacy::{fetch, logdir};
use crate::{Settings, arguments, state, xplanes};
use object_store::memory::InMemory;
use std::os::unix::fs::MetadataExt;

const DEMO: &[u8] = include_bytes!("../../../tests/data/demo.xplane.pb");

fn memory(name: &str) -> (Arc<InMemory>, Remote) {
    let store = Arc::new(InMemory::new());
    let remote = Remote::new(&format!("memory:///{name}-{}", std::process::id()), store.clone(), Key::from("logs")).unwrap();
    _ = std::fs::remove_dir_all(&remote.mirror);
    std::fs::create_dir_all(&remote.mirror).unwrap();
    (store, remote)
}

async fn put(store: &InMemory, path: &str, bytes: Vec<u8>) {
    store.put(&Key::from(path), PutPayload::from(bytes)).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn listing_mirrors_sessions_and_sessions_download_on_demand() {
    let (store, remote) = memory("listing");
    let big: Vec<u8> = (0..PART_BYTES * 2 + 12345).map(|index| (index % 251) as u8).collect();
    put(&store, "logs/plugins/profile/s0/h.xplane.pb", vec![1; 10]).await;
    put(&store, "logs/a/plugins/profile/s1/h1.xplane.pb", big.clone()).await;
    put(&store, "logs/a/plugins/profile/s1/jit_f.hlo_proto.pb", vec![2; 3]).await;
    put(&store, "logs/a/plugins/profile/s1/empty.xplane.pb", Vec::new()).await;
    put(&store, "logs/a/plugins/profile/s1/notes.txt", vec![3; 3]).await;
    put(&store, "logs/b/c/plugins/profile/s2/x.xplane.pb", vec![4; 4]).await;
    put(&store, "other/plugins/profile/s3/x.xplane.pb", vec![5; 4]).await;
    remote.sync(&remote.mirror, false).await.unwrap();
    let mut names: Vec<String> = sessions(&remote.mirror).into_iter().map(|(name, _)| name).collect();
    names.sort();
    assert_eq!(names, ["a/s1", "b/c/s2", "s0"]);
    let s1 = remote.mirror.join("a/plugins/profile/s1");
    assert!(list(&s1, |_| true).is_empty());
    remote.sync(&s1, true).await.unwrap();
    let mut files: Vec<String> = list(&s1, |_| true).iter().map(|path| path.file_name().unwrap().to_string_lossy().into_owned()).collect();
    files.sort();
    assert_eq!(files, ["empty.xplane.pb", "h1.xplane.pb", "jit_f.hlo_proto.pb"]);
    assert_eq!(std::fs::read(s1.join("h1.xplane.pb")).unwrap(), big);
    let inode = std::fs::metadata(s1.join("h1.xplane.pb")).unwrap().ino();
    remote.checked.lock().unwrap().clear();
    remote.sync(&s1, true).await.unwrap();
    assert_eq!(std::fs::metadata(s1.join("h1.xplane.pb")).unwrap().ino(), inode);
    put(&store, "logs/a/plugins/profile/s1/h1.xplane.pb", vec![9; 5]).await;
    remote.sync(&s1, true).await.unwrap();
    assert_eq!(std::fs::read(s1.join("h1.xplane.pb")).unwrap(), big);
    remote.checked.lock().unwrap().clear();
    remote.sync(&s1, true).await.unwrap();
    assert_eq!(std::fs::read(s1.join("h1.xplane.pb")).unwrap(), [9; 5]);
    assert!(remote.key(&remote.mirror.join("a/../../escape")).is_none());
    assert!(remote.key(Path::new("/elsewhere")).is_none());
    std::fs::remove_dir_all(&remote.mirror).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn least_recently_used_sessions_are_evicted_beyond_the_budget() {
    let (store, mut remote) = memory("evict");
    for (session, len) in [("s1", 100), ("s2", 200), ("s3", 300)] {
        put(&store, &format!("logs/plugins/profile/{session}/h.xplane.pb"), vec![1; len]).await;
    }
    remote.budget = 500;
    remote.sync(&remote.mirror, false).await.unwrap();
    let dir = |session: &str| remote.mirror.join("plugins/profile").join(session);
    let held = |session: &str| dir(session).join("h.xplane.pb").exists();
    for session in ["s1", "s2"] {
        remote.sync(&dir(session), true).await.unwrap();
    }
    let long_ago = Instant::now().checked_sub(IN_USE * 2).unwrap();
    remote.used.lock().unwrap().insert(dir("s1"), long_ago);
    remote.used.lock().unwrap().insert(dir("s2"), long_ago + Duration::from_secs(1));
    remote.sync(&dir("s3"), true).await.unwrap();
    assert_eq!([held("s1"), held("s2"), held("s3")], [false, true, true]);
    assert!(dir("s1").is_dir() && !remote.checked.lock().unwrap().contains_key(&dir("s1")));
    remote.sync(&dir("s1"), true).await.unwrap();
    assert_eq!([held("s1"), held("s2"), held("s3")], [true, false, true]);
    std::fs::remove_dir_all(&remote.mirror).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_file_url_logdir_serves_what_the_same_local_logdir_serves() {
    let bucket = logdir("remote-bucket");
    std::fs::create_dir_all(bucket.join("nested/run/plugins/profile/t")).unwrap();
    std::fs::write(bucket.join("run/plugins/profile/s/host-a.xplane.pb"), DEMO).unwrap();
    std::fs::write(bucket.join("run/plugins/profile/s/host-b.xplane.pb"), DEMO).unwrap();
    std::fs::write(bucket.join("nested/run/plugins/profile/t/h.xplane.pb"), DEMO).unwrap();
    let local = state(&Settings { logdir: bucket.clone(), ..Default::default() });
    let settings = arguments(["--logdir".to_string(), format!("file://{}", bucket.display())]).unwrap();
    let mirror = settings.remote.as_ref().unwrap().mirror.clone();
    _ = std::fs::remove_dir_all(&mirror);
    std::fs::create_dir_all(&mirror).unwrap();
    assert_eq!(settings.logdir, mirror);
    let remote = state(&settings);
    let base = "/data/plugin/profile";
    assert_eq!(fetch(&remote, &format!("{base}/runs")).await.2, "[\"run/s\", \"nested/run/t\"]");
    assert!(xplanes(&mirror.join("run/plugins/profile/s")).is_empty());
    for uri in [
        format!("{base}/runs"),
        format!("{base}/run_tools?run=run/s"),
        format!("{base}/hosts?run=run/s&tag=overview_page"),
        format!("{base}/hosts?run=nested/run/t&tag=trace_viewer@"),
        format!("{base}/data?run=run/s&tag=trace_viewer@&host=host-a"),
        format!("{base}/data?run=run/s&tag=overview_page&host=ALL_HOSTS"),
        format!("{base}/data?run=run/s&tag=op_profile&host=host-b"),
        format!("{base}/module_list?run=nested/run/t"),
        format!("{base}/runs?run_path=run/plugins/profile"),
        format!("{base}/data?run=s&run_path=run/plugins/profile&tag=hlo_stats&host=host-a"),
        format!("{base}/data?run=s&session_path=run/plugins/profile/s&tag=framework_op_stats&host=host-a"),
        format!("{base}/data?run=../x/s&tag=trace_viewer@&host=h"),
        format!("{base}/data?run=s&session_path=../../etc&tag=trace_viewer@"),
    ] {
        let (expected, served) = (fetch(&local, &uri).await, fetch(&remote, &uri).await);
        assert_eq!((served.0, &served.2), (expected.0, &expected.2), "{uri}");
    }
    assert_eq!(std::fs::read(mirror.join("run/plugins/profile/s/host-b.xplane.pb")).unwrap(), DEMO);
    std::fs::remove_dir_all(&bucket).unwrap();
    std::fs::remove_dir_all(&mirror).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unreadable_store_answers_bad_gateway() {
    let bucket = logdir("remote-unreadable");
    let file = bucket.join("locked");
    std::fs::create_dir(&file).unwrap();
    std::fs::set_permissions(&file, std::os::unix::fs::PermissionsExt::from_mode(0o0)).unwrap();
    let settings = arguments(["--logdir".to_string(), format!("file://{}", file.display())]).unwrap();
    let state = state(&settings);
    let (status, _, message) = fetch(&state, "/data/plugin/profile/runs").await;
    assert_eq!(status, StatusCode::BAD_GATEWAY, "{message}");
    assert!(message.starts_with(&format!("Cannot read file://{}", file.display())), "{message}");
    assert_eq!(fetch(&state, "/data/plugin/profile/version").await.0, StatusCode::OK);
    assert!(arguments(["--logdir".to_string(), "gs://".to_string()]).unwrap_err().starts_with("Cannot open gs://: "));
    std::fs::set_permissions(&file, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    std::fs::remove_dir_all(&bucket).unwrap();
    std::fs::remove_dir_all(&settings.remote.unwrap().mirror).unwrap();
}
