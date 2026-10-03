# xprof-rs

xprof-rs is a fast Rust backend for the [XProf](https://github.com/openxla/xprof) trace viewer. It replaces the XProf server. It gives the same responses as XProf 2.23.2, and tests compare them with the output of XProf.

XProf needs 10 to 25 seconds to convert a large `.xplane.pb` file for each cold request. xprof-rs reads the file into memory, converts the events in parallel, caches the result, and renders only the time window that you zoom to.

| Trace | Size | Events | Cold view | Warm view | Zoom |
|---|---|---|---|---|---|
| TPU v4-8 | 279 MB | 2.95 M | 0.63 s load | 128 ms | 5 ms |

The test machine has 4 cores. The load takes 2.0 s on one core. XProf takes 11 to 25 s for the same file.

## Install and start

You need a stable Rust toolchain (edition 2024) on a 64-bit Unix system. You do not need Python, a C++ toolchain, or `protoc`.

```bash
git clone <repository-url> && cd xprof-rs
cargo install --path .
xprof-rs --logdir ~/logs          # open http://localhost:8791
xprof-rs get_overview ~/logs/run1 # the same binary runs the XProf agent CLI
```

The release profile uses fat LTO. A full build takes several minutes. For a quick build, set `CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`. A `v*` tag publishes an `x86_64-linux` tarball (`.github/workflows/release.yml`).

## Server

```bash
xprof-rs [--logdir DIR|URL] [--port 8791] [--host ADDRESS] [--src_prefix PREFIX] [--hide_capture_profile_button] [--enable_tab_name_label]
```

- The interface is the prebuilt frontend of the XProf 2.23.2 Python package (Apache License 2.0, copyright The TensorFlow Authors). The binary includes the unchanged files from `static/`. To serve other files, set `XPROF_STATIC_DIR` to a directory.
- Each route answers at the root and under `/data/plugin/profile`.
- The server listens on all interfaces on port 8791 and has no authentication. To keep it local, use `--host 127.0.0.1`.
- With `--logdir`, a request can read only files in the log directory. A `run`, `session_path`, or `module_name` that points outside it gets a 400 response. The directory walk does not follow symbolic links to directories.
- Without `--logdir`, the server lists no runs, but `session_path` and `run_path` requests can still open a directory.
- `--grpc_port`, `--worker_service_address`, and `--max_concurrent_worker_requests` are accepted. The server does not use them.

All tools are native Rust: `trace_viewer@` (zoom, search, details, DMA, `format=pb`), `trace_viewer`, `overview_page`, `op_profile`, `hlo_stats`, `framework_op_stats`, `input_pipeline_analyzer`, `roofline_model`, `memory_profile`, `memory_viewer`, `graph_viewer`, `module_list`, `kernel_stats`, `pod_viewer`, `megascale_stats`, `smart_suggestion`, `perf_counters`, `utilization_viewer`, `kernel_utilization`, `inference_profile`, `runs`, `run_tools`, `hosts`, `data_csv`, `version`, `config`, `POST /generate_cache`, `/capture_profile` (gRPC client), and static files.

Time on the 279 MB v4 trace:

| Endpoint | XProf, cold | xprof-rs, first | xprof-rs, repeat |
|---|---|---|---|
| `trace_viewer@` | 11 to 25 s | 0.45 s | 2 ms |
| `overview_page` | 18 s | 2.8 s (builds the shared statistics) | under 1 ms |
| `op_profile` | 4.2 s | 0.35 s | 3 ms |
| `hlo_stats` | 2.4 s | 0.35 s | 3 ms |
| `framework_op_stats`, `input_pipeline_analyzer`, `roofline_model`, `memory_profile`, `kernel_stats` | 0.5 to 2.8 s | 1 to 75 ms | under 1 ms |
| `memory_viewer`, `graph_viewer`, `module_list` | 2.3 s | 0.25 s | |

The peak memory for all tools is about 2.4 GB (1.6 GB for a 197 MB trace).

## Remote log directories

`--logdir` accepts `gs://`, `s3://` (and S3-compatible stores), `az://`, `abfs://`, `http(s)://` (WebDAV), and `file://` URLs, through the [`object_store`](https://crates.io/crates/object_store) crate. Credentials come from the environment: `GOOGLE_APPLICATION_CREDENTIALS` or application default credentials for `gs://`, `AWS_*` for S3, `AZURE_*` for Azure.

- The server copies the store lazily to `xprof-rs-<hash of the URL>` in `XPROF_CACHE_DIR` (default: the temporary directory). All tools read this copy.
- A request that names a session downloads its `.xplane.pb` and `.hlo_proto.pb` files (four files at a time, eight 8 MiB ranges for each file). A file that has the same size and modification time as in the store is not downloaded again. The server asks the store at most one time each 5 s for each session.
- Each 30 s, a background poll checks the listing and the 8 sessions that were requested most recently.
- If the copy is larger than `XPROF_CACHE_BYTES` (default 20 GiB), the server removes the sessions that were requested least recently. It keeps a session that was requested in the last minute.
- A store that is not reachable, or credentials that are not correct, give a 502 response with the error of the store. A listing stops after 30 s.
- The server never writes the store, with one exception: `/capture_profile` uploads the captured files to `<URL>/plugins/profile/<session>/`.

Set the URL to the directory that holds the profiles. The listing visits each directory under it.

## Command line

`xprof-rs <command> <session> [--flag=value ...]` runs the agent CLI of XProf. The stdout and the exit codes are the same as in XProf 2.23.2. The session is a run directory, a log directory (the latest run), an `.xplane.pb` file, or a run name with `--logdir`. The flags follow XProf (Python Fire): `--name=value`, `--name value`, `--name` and `--noname` for booleans, positional parameters in the order of XProf, and the aliases `--session_dir`, `--session_path`, and `--source`.

An error prints `{"status": "ERROR", "reason": ...}` and exits with code 2 (usage), 3 (path), 4 (value not valid), or 1 (internal). A result larger than 10 MB goes to a file in `$TMPDIR`, as in XProf.

Commands: `get_overview`, `get_kpi_metrics`, `get_device_information`, `get_hosts`, `get_profile_summary`, `get_roofline_model`, `get_memory_profile`, `get_peak_allocations`, `get_top_hlo_ops`, `get_hlo_op_profile`, `get_hlo_stats`, `list_hlo_modules`, `get_hlo_module_content`, `get_hlo_neighborhood`, `get_hlo_text`, `get_graph_viewer`, `get_step_trace`, `check_host_boundness`, `get_utilization_viewer`, `get_kernel_utilization`, `compute_utilization`, `get_kernel_stats`, `get_avg_step_time`, `list_xplane_events`, `aggregate_xplane_events`, `get_xspace_proto`, `upload_trace`, `get_llo_analysis`, `get_llo_debug_string`, `verify_numerical_parity`. The flags of each command are the flags of XProf.

```bash
xprof-rs get_top_hlo_ops ~/logs/run1 --limit=5 --category_filter=fusion
xprof-rs get_hlo_op_profile ~/logs --view=tree --path=by_program --depth=3
xprof-rs list_xplane_events ~/logs/run1 --plane_regex='TPU:0$' --event_regex=all-reduce --max_events=20
```

Time on the 279 MB v4 trace: `get_overview` 2.3 s (XProf 18 s), `get_hlo_op_profile` 1.8 s (25 s), `aggregate_xplane_events` 2.3 s (25 s).

Differences from the Python CLI:

- There is no result cache. Each call computes the result again, as with `--bypass_cache`, and there are no `__cached__` fields.
- Regex errors are the errors of Rust. `--help` prints a short synopsis.
- `get_graph_viewer --output_type=pb` returns the bytes of the module file. XProf serializes the module again with hash-ordered maps.
- For multi-host sessions, tools that use combined op statistics report no data, and `memory_profile` fails. XProf needs its worker service for these sessions.

## Missing and not supported

xprof-rs is a rewrite of the backend. It is not a port of the complete XProf distribution.

- **Inputs:** `.xplane.riegeli` sessions are not read. They come only from the continuous-profiling mode of the gRPC capture client. `jax.profiler.start_trace` (used by MaxText, Levanter, and marin) writes `.xplane.pb`. A profile of 4 GiB or more is rejected.
- **Worker mode:** There is no distributed worker mode and no worker gRPC service. `/capture_profile` is a client only.
- **Deployment:** There is no TLS, no authentication, and no TensorBoard plugin loader. xprof-rs runs on 64-bit Unix only. There is no Windows build.
- **Tools:** `get_llo_analysis` and `get_llo_debug_string` return the LLO-not-available report of XProf (the LLO analysis is closed source). `verify_numerical_parity` returns only the missing-dependencies error (XProf runs Python callables). The `detect_*` commands from the README of XProf are not commands (XProf 2.23.2 does not register them). `mpmd_pipeline_view=true` is accepted and has no effect, as in XProf.
- **Python part of the XProf package:** `xprof.convert`, the `ProfileData` library, the `xprof.api` snapshot client, `install_and_run.py`, the labs, and the agent skills are not available.
- **Frontend:** This repository does not contain the sources of the interface.

These outputs of XProf are different from run to run, so xprof-rs cannot copy them:

- The order of tied rows in `hlo_stats`, `op_profile`, `memory_viewer`, and `get_peak_allocations` (hash order with a seed for each process), and the order of processes and threads in the protobuf.
- The allocator peaks and fragmentation snapshot of the memory profile.
- The `bind_id` flow ids of the trace viewer, the graph node ids (heap pointers), and the hover CSS rules and caller lists of the graph viewer.
- The `Launch Stats` line of a GPU device. XProf writes it in hash order, so the `uid` of host events at the same timestamp can differ.
- The statistics of multi-host sessions. XProf caches the first op statistics under `ALL_HOSTS` and sometimes leaves the duty cycle at 0. xprof-rs answers each host selection as a new XProf process does.
- The tensor pattern rows, the ties between requests of equal latency, and the order of the sums in the latency averages of `inference_profile`.

## Test coverage

- The repository has one trace: a demo trace in `tests/data` (a small synthetic workload on a TPU v4-8, with the host name, the script path, and the host system information replaced), the outputs of XProf for it, and synthetic GPU profiles and HLO fixtures in `tests/data/gpu`. `cargo test` compares each tool and the trace viewer with these outputs.
- Other tests used TPU v4 and v6e traces, one 8-GPU trace, and synthetic multi-host (2, 4, and 8 hosts), GPU (1, 2, and 8 GPUs; NVIDIA and AMD; TensorFlow, JAX, and eager steps; NCCL; CUDA graphs), and serving sessions. These traces are not in the repository. No real serving profile was available.
- The real-world coverage is limited. If a response differs from XProf 2.23.2, open an issue. Do not include private data.

## Memory

- Three LRU caches remove an entry after one hour without use: converted traces (4 GiB), op statistics (4 GiB), and rendered responses (1 GiB, compressed). The total is about 9 GiB. A trace that is larger than its budget is still served and cached alone.
- The server loads a file again when its size, modification time, or inode changes. It does not cache a file that changed in the last 2 s.
- At startup and when a session changes, a background task loads the trace and op statistics of the 8 newest sessions that did not change for 30 s. This keeps about 0.7 GB more in memory for a 279 MB trace.
- A 4-host session of 279 MB v4 traces takes 1.8 s for the trace viewer of all hosts and 3.8 s for `overview_page` (4.5 GB peak). For 8 hosts, the times are 3.8 s and 5.2 s (7.6 GB peak).

## Development

```bash
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release
```

The code is in `src/`: `main.rs` (flags, routes, caches), `xplane.rs` and `trace.rs` (reader and conversion), one module for each tool, `cli/` (agent CLI), `remote.rs` (object stores), and `tests/` (tests from XProf and XLA).

## License

Apache-2.0. The algorithms follow Apache-licensed code from [OpenXLA XProf](https://github.com/openxla/xprof) and [XLA/TSL](https://github.com/openxla/xla). See `NOTICE`.
