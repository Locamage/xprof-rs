use crate::server::remote::Remote;
use crate::{Params, response};
use axum::http::StatusCode;
use axum::response::Response;
use futures_util::future::join_all;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tonic::client::Grpc;
use tonic::transport::Endpoint;
use tonic::{Code, Request, Status};
use tonic_prost::ProstCodec;

const PROFILE: &str = "/tensorflow.ProfilerService/Profile";
const NEW_SESSION: &str = "/tensorflow.ProfileAnalysis/NewSession";
const GRACE: Duration = Duration::from_secs(60);
const MULTI_HOST_DELAY_MS: u64 = 3000;
const MAX_EVENTS: u64 = 1_000_000;
const TOOLS: [&str; 10] = ["trace_viewer", "op_profile", "input_pipeline", "kernel_stats", "memory_viewer", "memory_profile", "overview_page", "pod_viewer", "tensorflow_stats", "xplane.pb"];
const TF_MISSING: &str = "TensorFlow is not installed, but is required to use TPU names.";
const SERVER_ERROR: &str = "<!doctype html>\n<html lang=en>\n<title>500 Internal Server Error</title>\n<h1>Internal Server Error</h1>\n<p>The server encountered an internal error and was unable to complete your request. Either the server is overloaded or there is an error in the application.</p>\n";

#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct ProfileOptions {
    #[prost(uint32, tag = "5")]
    pub version: u32,
    #[prost(bool, tag = "1")]
    pub include_dataset_ops: bool,
    #[prost(uint32, tag = "2")]
    pub host_tracer_level: u32,
    #[prost(uint32, tag = "3")]
    pub device_tracer_level: u32,
    #[prost(uint32, tag = "4")]
    pub python_tracer_level: u32,
    #[prost(bool, tag = "7")]
    pub enable_hlo_proto: bool,
    #[prost(uint64, tag = "8")]
    pub start_timestamp_ns: u64,
    #[prost(uint64, tag = "9")]
    pub duration_ms: u64,
    #[prost(string, tag = "10")]
    pub repository_path: String,
}

#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct ProfileRequest {
    #[prost(uint64, tag = "1")]
    pub duration_ms: u64,
    #[prost(uint64, tag = "2")]
    pub max_events: u64,
    #[prost(string, repeated, tag = "3")]
    pub tools: Vec<String>,
    #[prost(message, optional, tag = "4")]
    pub opts: Option<ProfileOptions>,
    #[prost(string, tag = "5")]
    pub repository_root: String,
    #[prost(string, tag = "6")]
    pub session_id: String,
    #[prost(string, tag = "7")]
    pub host_name: String,
}

#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct ToolData {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(bytes = "vec", tag = "2")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct ProfileResponse {
    #[prost(message, repeated, tag = "6")]
    pub tool_data: Vec<ToolData>,
    #[prost(bool, tag = "7")]
    pub empty_trace: bool,
}

#[derive(Clone, PartialEq, prost::Message)]
struct NewSessionRequest {
    #[prost(message, optional, tag = "1")]
    request: Option<ProfileRequest>,
    #[prost(string, tag = "2")]
    repository_root: String,
    #[prost(string, repeated, tag = "3")]
    hosts: Vec<String>,
    #[prost(string, tag = "4")]
    session_id: String,
}

#[derive(Clone, PartialEq, prost::Message)]
struct NewSessionResponse {
    #[prost(string, tag = "1")]
    error_message: String,
    #[prost(bool, tag = "2")]
    empty_trace: bool,
}

fn json(code: StatusCode, key: &str, message: &str) -> Response {
    response(code, "application/json", crate::server::run_tools::python_value(&serde_json::json!({ key: message })))
}

fn unix_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_nanos() as u64)
}

fn timestamp() -> String {
    let seconds = unix_ns() as i64 / 1_000_000_000;
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&raw const seconds, &raw mut local) };
    format!("{:04}_{:02}_{:02}_{:02}_{:02}_{:02}", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec)
}

fn status_text(status: &Status) -> String {
    let name = format!("{:?}", status.code());
    let shouted: String =
        name.chars().enumerate().flat_map(|(index, character)| (index > 0 && character.is_ascii_uppercase()).then_some('_').into_iter().chain([character.to_ascii_uppercase()])).collect();
    format!("{shouted}: {}", status.message())
}

async fn call<Req: prost::Message + Send + Sync + 'static, Rep: prost::Message + Default + Send + Sync + 'static>(
    address: &str,
    path: &'static str,
    message: Req,
    deadline: Duration,
) -> Result<Rep, Status> {
    let channel = Endpoint::from_shared(format!("http://{address}"))
        .map_err(|error| Status::unavailable(error.to_string()))?
        .connect_timeout(deadline)
        .connect()
        .await
        .map_err(|error| Status::unavailable(error.to_string()))?;
    let mut request = Request::new(message);
    request.set_timeout(deadline);
    let mut client = Grpc::new(channel).max_decoding_message_size(i32::MAX as usize);
    client.ready().await.map_err(|error| Status::unavailable(error.to_string()))?;
    let reply = client.unary(request, path.parse().unwrap(), ProstCodec::<Req, Rep>::default()).await?;
    Ok(reply.into_inner())
}

fn save(root: &Path, session: &str, address: &str, reply: &ProfileResponse) -> std::io::Result<()> {
    if reply.tool_data.is_empty() {
        return Ok(());
    }
    let dir = root.join(session);
    std::fs::create_dir_all(&dir)?;
    let host = address.replace(':', "_");
    for tool in &reply.tool_data {
        if tool.name.ends_with("_helper") {
            continue;
        }
        let prefix = if host.is_empty() { String::new() } else { format!("{host}.") };
        std::fs::write(dir.join(format!("{prefix}{}", tool.name)), &tool.data)?;
    }
    Ok(())
}

pub async fn handle(logdir: &Path, remote: Option<&Remote>, params: &Params) -> Response {
    let int = |key: &str, default: i64| params.get(key).map_or(Some(default), |value| crate::cli::json::py_int(value).and_then(|number| i64::try_from(number).ok()));
    let (Some(duration), Some(retries), Some(host_level), Some(device_level), Some(python_level), Some(delay)) =
        (int("duration", 1000), int("num_retry", 0), int("host_tracer_level", 2), int("device_tracer_level", 1), int("python_tracer_level", 0), int("delay", 0))
    else {
        return response(StatusCode::INTERNAL_SERVER_ERROR, "text/html; charset=utf-8", SERVER_ERROR);
    };
    let worker_list = params.get("worker_list").filter(|list| !list.is_empty());
    if params.get("is_tpu_name").is_some_and(|value| value == "true") {
        return json(StatusCode::INTERNAL_SERVER_ERROR, "error", TF_MISSING);
    }
    if logdir.as_os_str().is_empty() {
        return json(StatusCode::INTERNAL_SERVER_ERROR, "error", "logdir is not set, abort capturing.");
    }
    let Some(service) = params.get("service_addr") else { return json(StatusCode::INTERNAL_SERVER_ERROR, "error", "'NoneType' object has no attribute 'removeprefix'") };
    let service = service.strip_prefix("grpc://").unwrap_or(service);
    let addresses: Vec<String> = worker_list.map_or(service, |list| list).split(',').map(String::from).collect();
    let created = unix_ns();
    let duration_ms = duration.max(0) as u64;
    let delay_ms = if addresses.len() > 1 { MULTI_HOST_DELAY_MS } else { delay.max(0) as u64 };
    let attempts = retries.saturating_add(1);
    let invalid = if duration_ms == 0 {
        Some("duration_ms must be greater than zero.".to_string())
    } else {
        addresses
            .iter()
            .find(|address| !address.split_once(':').is_some_and(|(host, port)| !host.is_empty() && !host.contains('/') && port.parse::<u32>().is_ok()))
            .map(|address| format!("Could not interpret \"{address}\" as a host-port pair."))
    };
    if let Some(message) = invalid {
        return json(StatusCode::INTERNAL_SERVER_ERROR, "error", &format!("INVALID_ARGUMENT: {message}"));
    }
    let session = timestamp();
    let root = logdir.join("plugins/profile");
    let repository = remote.map_or_else(|| logdir.to_string_lossy().into_owned(), |remote| remote.url.trim_end_matches('/').to_string());
    let repository_root = format!("{repository}/plugins/profile");
    let mut remaining = attempts;
    let status = loop {
        let start = unix_ns().saturating_add(delay_ms.saturating_mul(1_000_000));
        remaining -= 1;
        let options = ProfileOptions {
            version: 1,
            include_dataset_ops: true,
            host_tracer_level: host_level as u32,
            device_tracer_level: device_level as u32,
            python_tracer_level: python_level as u32,
            enable_hlo_proto: true,
            start_timestamp_ns: start,
            duration_ms,
            repository_path: repository.clone(),
        };
        let request = |host: &str| ProfileRequest {
            duration_ms,
            max_events: MAX_EVENTS,
            tools: TOOLS.map(String::from).to_vec(),
            opts: Some(options.clone()),
            repository_root: repository_root.clone(),
            session_id: session.clone(),
            host_name: host.to_string(),
        };
        let grace = GRACE.max(Duration::from_millis(duration_ms) * 2) + Duration::from_nanos(start.saturating_sub(created));
        let outcome = if worker_list.is_some() {
            let message = NewSessionRequest { request: Some(request(&addresses[0])), repository_root: repository_root.clone(), hosts: addresses.clone(), session_id: session.clone() };
            call::<_, NewSessionResponse>(&addresses[0], NEW_SESSION, message, grace)
                .await
                .and_then(|reply| if reply.empty_trace { Err(Status::unavailable("No trace event is collected")) } else { Ok(()) })
        } else {
            let replies = join_all(addresses.iter().map(|address| call::<_, ProfileResponse>(address, PROFILE, request(address), grace))).await;
            let mut traced = false;
            let mut failure = None;
            for (address, reply) in addresses.iter().zip(replies) {
                if let Ok(reply) = reply.as_ref().inspect_err(|status| eprintln!("{address} returned {}", status_text(status)))
                    && !reply.empty_trace
                {
                    traced = true;
                    if let Err(error) = save(&root, &session, address, reply) {
                        failure = Some(Status::internal(error.to_string()));
                    }
                }
            }
            if traced
                && failure.is_none()
                && let Some(remote) = remote
                && let Err(message) = remote.upload(&root.join(&session)).await
            {
                failure = Some(Status::internal(message));
            }
            match failure {
                Some(status) => Err(status),
                None if traced => Ok(()),
                None => Err(Status::unavailable("No trace event was collected because there were no responses from clients or the responses did not have trace data.")),
            }
        };
        match outcome {
            Err(status) if remaining > 0 && (matches!(status.code(), Code::Unavailable | Code::AlreadyExists) || (status.code() == Code::Unknown && status.message() == "Stream removed")) => continue,
            other => break other,
        }
    };
    match status {
        Ok(()) => json(StatusCode::OK, "result", "Capture profile successfully. Please refresh."),
        Err(status) => json(StatusCode::INTERNAL_SERVER_ERROR, "error", &status_text(&status)),
    }
}
