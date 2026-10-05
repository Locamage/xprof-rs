use crate::server::capture::{ProfileRequest, ProfileResponse, ToolData, handle};
use axum::http::{Request, Response};
use std::convert::Infallible;
use std::future::{Ready, ready};
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::body::Body;
use tonic::server::{Grpc, NamedService, UnaryService};
use tonic_prost::ProstCodec;

#[derive(Clone)]
struct Profiler;

struct Profile;

impl UnaryService<ProfileRequest> for Profile {
    type Response = ProfileResponse;
    type Future = Ready<Result<tonic::Response<ProfileResponse>, tonic::Status>>;

    fn call(&mut self, request: tonic::Request<ProfileRequest>) -> Self::Future {
        let request = request.into_inner();
        let data = format!("{}|{}|{}", request.host_name, request.duration_ms, request.opts.unwrap().host_tracer_level).into_bytes();
        ready(Ok(tonic::Response::new(ProfileResponse { tool_data: vec![ToolData { name: "xplane.pb".into(), data }], empty_trace: false })))
    }
}

impl NamedService for Profiler {
    const NAME: &'static str = "tensorflow.ProfilerService";
}

impl tower::Service<Request<Body>> for Profiler {
    type Response = Response<Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Infallible>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        Box::pin(async move { Ok(Grpc::new(ProstCodec::default()).unary(Profile, request).await) })
    }
}

async fn serve() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    tokio::spawn(tonic::transport::Server::builder().add_service(Profiler).serve_with_incoming(TcpListenerStream::new(listener)));
    address
}

fn query(pairs: &[(&str, &str)]) -> crate::Params {
    pairs.iter().map(|(key, value)| (key.to_string(), value.to_string())).collect()
}

async fn capture(logdir: &Path, pairs: &[(&str, &str)]) -> (u16, String) {
    let reply = handle(logdir, None, &query(pairs)).await;
    (reply.status().as_u16(), String::from_utf8(axum::body::to_bytes(reply.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap())
}

#[tokio::test]
async fn capture_saves_the_remote_xplane_under_the_plugin_directory() {
    let (address, logdir) = (serve().await, super::legacy::logdir("capture"));
    let (status, body) = capture(&logdir, &[("service_addr", &format!("grpc://{address}")), ("duration", "50"), ("host_tracer_level", "3")]).await;
    assert_eq!((status, body.as_str()), (200, "{\"result\": \"Capture profile successfully. Please refresh.\"}"));
    let sessions: Vec<_> = std::fs::read_dir(logdir.join("plugins/profile")).unwrap().map(|entry| entry.unwrap().path()).collect();
    let saved = std::fs::read_to_string(sessions.last().unwrap().join(format!("{}.xplane.pb", address.replace(':', "_")))).unwrap();
    assert_eq!(saved, format!("{address}|50|3"));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test]
async fn capture_reports_invalid_requests_like_xprof() {
    let logdir = super::legacy::logdir("capture-errors");
    for (pairs, status, message) in [
        (vec![], 500, "'NoneType' object has no attribute 'removeprefix'"),
        (vec![("service_addr", "bad")], 500, "INVALID_ARGUMENT: Could not interpret \\\"bad\\\" as a host-port pair."),
        (vec![("service_addr", "h:1"), ("duration", "0")], 500, "INVALID_ARGUMENT: duration_ms must be greater than zero."),
        (vec![("service_addr", "h:1"), ("is_tpu_name", "true")], 500, "TensorFlow is not installed, but is required to use TPU names."),
        (vec![("service_addr", "127.0.0.1:1"), ("duration", "10")], 500, "UNAVAILABLE: No trace event was collected"),
    ] {
        let (code, body) = capture(&logdir, &pairs).await;
        assert!(code == status && body.contains(message), "{pairs:?}: {code} {body}");
    }
    assert_eq!(capture(&logdir, &[("service_addr", "h:1"), ("duration", "x")]).await.0, 500);
    assert!(capture(&logdir, &[("service_addr", "127.0.0.1:1"), ("duration", " 1_0 "), ("num_retry", "-1")]).await.1.contains("UNAVAILABLE"));
    std::fs::remove_dir_all(&logdir).unwrap();
}

#[tokio::test]
async fn capture_into_a_remote_logdir_uploads_the_session() {
    let (address, bucket) = (serve().await, super::legacy::logdir("capture-remote"));
    let url = format!("file://{}", bucket.display());
    let remote = crate::server::remote::Remote::open(&url).unwrap();
    let reply = handle(&remote.mirror, Some(&remote), &query(&[("service_addr", &address), ("duration", "50")])).await;
    assert_eq!(reply.status().as_u16(), 200);
    let name = format!("{}.xplane.pb", address.replace(':', "_"));
    let session = std::fs::read_dir(bucket.join("plugins/profile")).unwrap().next().unwrap().unwrap().path();
    assert_eq!(std::fs::read_to_string(session.join(&name)).unwrap(), format!("{address}|50|2"));
    let mirrored = remote.mirror.join("plugins/profile").join(session.file_name().unwrap()).join(&name);
    assert_eq!(std::fs::metadata(mirrored).unwrap().modified().unwrap(), std::fs::metadata(session.join(&name)).unwrap().modified().unwrap());
    std::fs::remove_dir_all(&bucket).unwrap();
    std::fs::remove_dir_all(&remote.mirror).unwrap();
}
