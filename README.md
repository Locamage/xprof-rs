# xprof-rs

xprof-rs is a fast Rust backend for the [XProf](https://github.com/openxla/xprof) trace viewer. It is a replacement for the XProf server.

XProf converts a large `.xplane.pb` file in Python and C++ for each cold request. This takes 10 to 25 seconds. xprof-rs does the same work in Rust:

1. It reads the file into memory.
2. It converts the events in parallel.
3. It keeps the result in a cache.
4. It renders only the time window that you zoom to.

xprof-rs follows the trace-viewer pipeline of XProf 2.23.2 step by step. Tests compare the output with the output of XProf event by event.

| Trace | Size | Events | Cold view | Warm view | Zoom |
|---|---|---|---|---|---|
| TPU v4-8 | 279 MB | 2.95 M | 0.63 s load | 128 ms | 5 ms |

The test machine has 4 cores. The load takes 2.0 s on one core. XProf takes 11 to 25 s for the same file.

## Install

xprof-rs is one Rust binary. You need a stable Rust toolchain (edition 2024) on a 64-bit Unix system. You do not need Python, a C++ toolchain, or `protoc`.

```bash
git clone <repository-url> && cd xprof-rs
cargo install --path .
```

You can also run `cargo build --release` and use `target/release/xprof-rs`.

The release profile uses fat LTO and one codegen unit. A full build takes several minutes. For a quick build, set these variables:

```bash
CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release
```

A release tag publishes an `x86_64-linux` tarball. The workflow is in `.github/workflows/release.yml`.

## Start

Start the server and open `http://localhost:8791`:

```bash
xprof-rs --logdir ~/logs
```

The same binary runs the agent CLI of XProf:

```bash
xprof-rs get_overview ~/logs/run1
```

## What the server gives

Everything is native Rust. There is no Python and no fallback. The responses are the same as the responses of XProf 2.23.2 for TPU, GPU, TensorFlow, and host-only traces.

| Endpoint | XProf, cold (279 MB v4 trace) | xprof-rs, first request | xprof-rs, repeat |
|---|---|---|---|
| Trace viewer (`trace_viewer@`: zoom, search, details, DMA) | 11 to 25 s | 0.45 s | 2 ms |
| `overview_page` | 18 s | 2.8 s (builds the shared statistics) | under 1 ms |
| `op_profile` (by program, category, provenance) | 4.2 s | 0.35 s | 3 ms |
| `hlo_stats` | 2.4 s | 0.35 s | 3 ms |
| `framework_op_stats`, `input_pipeline_analyzer`, `roofline_model`, `memory_profile`, `kernel_stats` | 0.5 to 2.8 s | 1 to 75 ms | under 1 ms |
| `memory_viewer`, `graph_viewer`, `module_list` (from HLO protos) | 2.3 s | 0.25 s | |
| `trace_viewer` (JSON, no streaming), `trace_viewer@&format=pb` (zstd delta-series protobuf), `pod_viewer`, `megascale_stats`, `smart_suggestion`, `perf_counters`, `utilization_viewer`, `kernel_utilization` | | native | |
| `runs`, `run_tools`, `hosts`, `data_csv`, `version`, `config`, `POST /generate_cache`, `/capture_profile` (gRPC client), static files | | native | |

The peak memory for all tools on the 279 MB trace is about 2.4 GB. A 197 MB trace uses 1.6 GB.

Each route answers at the root and under `/data/plugin/profile`.

## Web interface

xprof-rs has no user interface of its own. It serves the interface of XProf. The binary includes the prebuilt frontend files of the XProf 2.23.2 Python package (Apache License 2.0, copyright The TensorFlow Authors). These files are in `static/` and are not changed. This project replaces only the backend that answers the requests of the interface.

To serve other frontend files, set `XPROF_STATIC_DIR` to a directory.

## Server options

```bash
xprof-rs [--logdir DIR|URL] [--port 8791] [--host ADDRESS] [--src_prefix PREFIX] [--hide_capture_profile_button] [--enable_tab_name_label]
```

- Without `--logdir`, the server lists no runs, as `xprof` does. A `session_path` or `run_path` request can still open a directory.
- `--grpc_port`, `--worker_service_address`, and `--max_concurrent_worker_requests` are accepted. The server does not use them.
- The server listens on all interfaces (`::`, or `0.0.0.0` if IPv6 is not available) on port 8791. It has no authentication. To keep the server local, use `--host 127.0.0.1`.
- With `--logdir`, a request can read only the profiles in the log directory. A `run`, `session_path`, or `module_name` that points outside it gets a 400 response. Absolute paths, `..`, and symbolic links to other places point outside it.
- The directory walk does not follow symbolic links to directories.

## Remote log directories

`--logdir` also accepts an object store URL, for example `gs://bucket/dir`. These schemes work, through the [`object_store`](https://crates.io/crates/object_store) crate:

- `gs://`
- `s3://` (and S3-compatible stores, for example R2)
- `az://` and `abfs://`
- `http(s)://` (WebDAV)
- `file://`

The credentials come from the environment, as the `object_store` crate reads them:

- `gs://`: `GOOGLE_APPLICATION_CREDENTIALS` or application default credentials.
- S3: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT`, and `AWS_ALLOW_HTTP`.
- Azure: the `AZURE_*` variables.

The server copies the store lazily to a local directory. The directory name is `xprof-rs-<hash of the URL>` in `XPROF_CACHE_DIR`. The default is the temporary directory of the operating system. All tools read this copy.

- `runs` lists `plugins/profile/` under the URL and each directory in it. It makes the session directories in the copy and downloads nothing.
- A request that names a session downloads the `.xplane.pb` and `.hlo_proto.pb` files of the session. The names are `run`, `session_path`, or `run_path` with `run`. A `run_path` alone fetches all sessions under it. The server downloads four files at a time, with eight 8 MiB ranges for each file. It writes the files in place.
- The server does not download a file again if the size and the modification time are the same as in the store. It asks the store at most one time each 5 s for each session and for the listing.
- Each 30 s, a background poll checks the listing again. It also checks the 8 sessions that were requested most recently and are still in the copy.
- If the copy is larger than `XPROF_CACHE_BYTES` (default 20 GiB), the server removes the sessions that were requested least recently. It keeps a session that was requested in the last minute. It downloads a removed session again when a request names it.
- If the store is not reachable or the credentials are not correct, the answer is 502 with the error of the store. A listing stops after 30 s.
- The server writes the copy. It never writes the store, with one exception: `/capture_profile` uploads the captured files to `<URL>/plugins/profile/<session>/` and gives the URL to the profiler as its repository.
- The `run_tools` cache file, the HLO protos that `module_list` extracts, and the rendered responses stay local.

Set the URL to the directory that holds the profiles. The listing visits each directory under the URL.

## Command line

The binary also runs the agent CLI of XProf: `xprof-rs <command> <session> [--flag=value ...]`. The output on stdout and the exit codes are the same as the output and exit codes of XProf 2.23.2.

The session can be one of these:

- a run directory
- a log directory (the CLI uses the latest run)
- an `.xplane.pb` file
- a run name, if you set `--logdir`

The flags follow XProf (Python Fire):

- `--name=value` and `--name value`
- `--name` and `--noname` for booleans
- positional parameters in the order of XProf
- the aliases `--session_dir`, `--session_path`, and `--source`

The output is JSON or text of XProf on stdout. An error prints `{"status": "ERROR", "reason": ...}` and exits with one of these codes:

| Code | Meaning |
|---|---|
| 1 | Internal error |
| 2 | Usage error |
| 3 | Path error |
| 4 | Value not valid |

A result larger than 10 MB goes to a file in `$TMPDIR`, and the CLI reports the file, as XProf does.

`xprof-rs` and `xprof-rs server` with server flags start the server.

Speed on the 279 MB v4 trace:

| Command | XProf CLI | xprof-rs |
|---|---|---|
| `get_overview` | 18 s | 2.3 s |
| `get_hlo_op_profile` | 25 s | 1.8 s |
| `aggregate_xplane_events` | 25 s | 2.3 s |

| Command | Result |
|---|---|
| `get_overview` | Performance summary and run environment, with roofline fallbacks (`--include_command`) |
| `get_kpi_metrics` | Step time, duty cycle, MXU and roofline utilization, peak HBM, accelerator |
| `get_device_information` | Device type, peak FLOP rate, bandwidths (GiB/s), ridge points |
| `get_hosts` | Hosts of the session |
| `get_profile_summary` | Text table of the top operations by self time |
| `get_roofline_model` | Program roofline metrics, device information, top operations (`--top_n`) |
| `get_memory_profile` | HBM capacity, peak usage, stack, heap, free memory, fragmentation |
| `get_peak_allocations` | Modules by peak HBM with their largest buffers (`--limit --min_size_mib --output_format=markdown --noaggregate_instructions --noinclude_summary`) |
| `get_top_hlo_ops` | Top HLO ops by time, FLOPs, and bytes (`--limit --category_filter`) |
| `get_hlo_op_profile` | Grouped, category, flat, or tree views of the HLO op profile (`--view --category --path --depth --sort_by --top_n`) |
| `get_hlo_stats` | HLO statistics records (`--limit --sort_by --category_filter`) |
| `list_hlo_modules` | HLO modules of the session (the CLI extracts them from the trace if necessary) |
| `get_hlo_module_content` | Module text (`--module_name --max_lines --print_metadata`) |
| `get_hlo_neighborhood` | Operands and users of an instruction (`--instruction_name --radius --module_name`) |
| `get_hlo_text` | Module text or the neighborhood of an op, with an option to save it (`--op_name --module_name --path`) |
| `get_graph_viewer` | Graph viewer output: `short_txt`, `long_txt`, `pbtxt`, `pb`, `graph`, or `adj_nodes` (`--output_type --node_name --module_name ...`) |
| `get_step_trace` | Breakdown and summary for each step, from the pod viewer, input pipeline, or overview (`--step_num --limit --device_core --noinclude_summary`) |
| `check_host_boundness` | Host-boundness result from the overview, op profile, barrier-cores, and utilization |
| `get_utilization_viewer` | Hardware counter utilization of one host, device, and node (`--host --device --node`) |
| `get_kernel_utilization`, `compute_utilization` | Counter utilization of each kernel (`--kernel_name --device --duration_us --force_duration`) |
| `get_kernel_stats` | Device kernel durations, with the disjoint interval union as an option (`--limit --kernel_name --include_summary --trace_matchers --output_format`) |
| `get_avg_step_time` | Average step time of `XLA Modules` (`--func_name`) |
| `list_xplane_events` | Timeline events, filtered by plane, event regex, and time window (`--plane_regex --event_regex --start_time_ps --end_time_ps --max_events --offset`) |
| `aggregate_xplane_events` | Count, total, average, minimum, maximum, and standard deviation for each event name |
| `get_xspace_proto` | The raw XSpace bytes, or a saved file (`--output_path`) |
| `upload_trace` | Copies a trace into `--logdir` under `--run_name` |
| `get_llo_analysis`, `get_llo_debug_string` | The LLO-not-available report of XProf (the LLO analysis is closed source) |
| `verify_numerical_parity` | The missing-dependencies error of XProf (the command runs Python callables) |

Examples:

```bash
xprof-rs get_overview ~/logs/run1
xprof-rs get_top_hlo_ops ~/logs/run1 --limit=5 --category_filter=fusion
xprof-rs get_hlo_op_profile ~/logs --view=tree --path=by_program --depth=3
xprof-rs get_hlo_neighborhood run1.xplane.pb --instruction_name=fusion.12 --radius=1
xprof-rs list_xplane_events ~/logs/run1 --plane_regex='TPU:0$' --event_regex=all-reduce --max_events=20
```

Differences from the Python CLI:

- xprof-rs has no result cache. Each call computes the result again, as with `--bypass_cache`. There are no `__cached__` fields.
- XProf orders tied rows in random hash order. xprof-rs can give a different order for equal-size buffers in `get_peak_allocations`, equal-time ops in `get_hlo_op_profile`, and equal allocator peaks in `get_memory_profile`.
- `get_graph_viewer --output_type=pb` returns the bytes of the module file. XProf serializes the module again with hash-ordered maps.
- Error messages for regex are the messages of Rust.
- `--help` prints a short synopsis.
- For multi-host sessions, tools that use combined op statistics report no data, as in the CLI of XProf 2.23.2. XProf needs its worker service for these sessions. `memory_profile` fails.
- The README of XProf lists `detect_*` commands for HLO. XProf 2.23.2 does not register them in its CLI, and xprof-rs does not have them.

## Source layout

xprof-rs is one binary crate. The only `unsafe` code is a few libc calls (`snprintf`, `localtime_r`, `getuid`) and the `mi_collect` call of mimalloc.

| Path | Purpose |
|---|---|
| `src/main.rs` | Flags, axum routes, caches (converted traces, op statistics, rendered responses), background prefetch |
| `src/xplane.rs`, `src/trace.rs`, `src/derive.rs`, `src/group.rs` | XSpace reader, event conversion, derived timelines, grouping of steps and ops |
| `src/json.rs`, `src/legacy_trace.rs`, `src/delta.rs` | Trace viewer renderers: streaming JSON, JSON without streaming, zstd delta protobuf |
| `src/opstats.rs` and the tool modules (`op_profile.rs`, `hlo_stats.rs`, `roofline.rs`, `overview_page.rs`, `memory_profile.rs`, and others) | Tools that use the shared op statistics |
| `src/hlo*.rs`, `src/graph_viewer.rs`, `src/memory_viewer.rs`, `src/gpu_cost.rs`, `src/pbtext.rs` | HLO parser and printer, graph viewer, memory viewer, GPU cost analysis |
| `src/gpu.rs`, `src/inference_profile.rs`, `src/pod_viewer.rs`, `src/megascale*.rs`, `src/utilization.rs`, `src/smart_suggestion.rs` | GPU, inference, pod, megascale, and utilization tools |
| `src/remote.rs`, `src/capture.rs`, `src/run_tools.rs` | Object store copies, the `/capture_profile` gRPC client, `run_tools` |
| `src/cli/` | The agent CLI of XProf |
| `src/tests/` | Unit tests that come from XProf and XLA, and golden comparisons |

## Exactness

The repository has a demo trace and the outputs of XProf for it in `tests/data`. The demo trace is a small synthetic workload on a TPU v4-8. The host name, the script path, and the host system information are replaced. `cargo test` compares each tool and the trace viewer with these outputs.

Some fields cannot match one specific run, because XProf is not deterministic for them. See "Output differences that cannot be closed" below.

## Missing and not supported

xprof-rs is a rewrite of the backend. It is not a port of the complete XProf distribution. These items are not available.

**Inputs**

- xprof-rs does not read `.xplane.riegeli` sessions. XProf lists them. They come only from the continuous-profiling mode of the gRPC capture client. `jax.profiler.start_trace` (used by MaxText, Levanter, and marin) writes `.xplane.pb`.
- xprof-rs rejects a profile of 4 GiB or more with the error `profile larger than 4 GiB`.

**Server and deployment**

- There is no distributed worker mode. The flags `--grpc_port`, `--worker_service_address`, and `--max_concurrent_worker_requests` have no effect. There is no worker gRPC service and no fan-out of requests. For this reason, tools that use combined op statistics report no data for multi-host sessions in the CLI. The CLI of XProf does the same without its worker service.
- `/capture_profile` is a gRPC client only. xprof-rs does not host a profiler service.
- There is no authentication and no TLS, as in `xprof`.
- There is no TensorBoard plugin loader. xprof-rs is a standalone server. You cannot start it with `tensorboard --logdir`.
- xprof-rs runs on 64-bit Unix only. The CI tests Linux. There is no Windows build.

**Tools**

- `get_llo_analysis` and `get_llo_debug_string` return the LLO-not-available report of XProf. The LLO analysis is closed source.
- `verify_numerical_parity` returns only the missing-dependencies error of XProf. The command runs Python callables.
- The `detect_*` commands from the README of XProf are not commands in xprof-rs. XProf 2.23.2 does not register them in its CLI.
- `mpmd_pipeline_view=true` is accepted and has no effect. XProf 2.23.2 also ignores it. `custom_call_graph` is not a graph type in XProf 2.23.2.
- There is no CLI result cache.

**The Python part of the XProf package**

- These parts are not available: `xprof.convert`, the `ProfileData` Python library, the `xprof.api` client for continuous-profiling snapshots, `install_and_run.py`, the labs, and the agent skills.
- Only the server routes and the CLI commands are available.

**Output differences that cannot be closed**

XProf gives different output from run to run in these cases. xprof-rs cannot copy this.

- Tied rows in `hlo_stats`, `op_profile`, `memory_viewer`, and `get_peak_allocations`. XProf orders them in absl hash-map order. The seed changes for each process.
- The allocator peaks and the fragmentation snapshot of the memory profile.
- The order of processes and threads in the protobuf.
- The `bind_id` flow ids of the trace viewer. XProf computes them with a hash that has a seed for each process.
- Graph node ids. They are heap pointers.
- The hover CSS rules and the caller lists of the graph viewer.
- The `Launch Stats` line of a GPU device. XProf writes it in hash order and merges it as if it was sorted. For launches with an API name that does not start with `cu`, the `uid` of host events at the same timestamp differs from run to run. Detail lookups of these events differ also.
- The statistics of a multi-host session. XProf caches the first op statistics under `ALL_HOSTS` and serves them for each host selection. It sometimes leaves the high-confidence duty cycle at 0. This depends on what its process served before. xprof-rs answers each selection as a new XProf process does. For this reason, `roofline_model` follows the host selection.
- `inference_profile` and the inference latency table of `overview_page`. XProf orders the tensor pattern rows in random hash-map order. It does the same for ties between requests of equal latency from different hosts or models, and for the order of the sums in the average latency of many models.

**Frontend**

- The interface is the prebuilt frontend of XProf. This repository does not contain its sources. To change the interface, use the XProf frontend sources.

## Test coverage

- xprof-rs follows XProf 2.23.2.
- TPU: Tests compare TPU v4 and v6e traces with XProf for each tool. The demo trace in `tests/data` is the only trace in the repository.
- Multi-host TPU: Tests use synthetic sessions of 2, 4, and 8 hosts. They come from the traces above and have these changes: renamed hosts, shifted clocks, fewer steps, a host without `Steps` or `XLA Modules`, and hosts with fewer chips. The tests check each tool for `ALL_HOSTS` and for each host. They check the trace viewer for each host and for sets of hosts. The flow ids use the XSpace host name, as in XProf. Flows of different hosts stay apart.
- GPU: Tests use one 8-GPU trace and synthetic GPU profiles. The synthetic profiles cover 1, 2, and 8 GPUs, NVIDIA from Kepler to Blackwell, AMD, TensorFlow, JAX, and eager steps, memcpy, memset, P2P, NCCL, CUDA graphs, and one or two hosts. The tests check each tool. For the GPU cost analysis and the graph viewer, the tests use XLA modules compiled for sm_90a. They also use variants of the cuDNN and cuBLAS backend config in `graph_viewer`. The synthetic GPU profiles and the HLO fixtures are in `tests/data/gpu`.
- Inference: Tests use only synthetic serving profiles (TensorFlow, batching, TFRT, Pathways, Orbax, user-defined requests, one or more hosts). No real serving profile was available.
- The real-world coverage is limited to these traces. If a response differs from XProf 2.23.2, open an issue. Do not include private data in the trace.

## Memory and caches

- The server has three LRU caches. Each cache removes an entry after one hour without use.
  - Converted traces: up to 4 GiB (file and events).
  - Op statistics: up to 4 GiB (weighted by file size).
  - Rendered responses: up to 1 GiB (compressed).
  - The total is about 9 GiB.
- A trace that is larger than its budget is still served. The server caches it alone.
- The server loads a file again when the size, the modification time, or the inode changes. The server does not cache a file that changed in the last 2 s.
- A background task loads the trace and the op statistics of the 8 newest sessions. It does this at startup and when a session changes. It loads one file at a time and only files that did not change for 30 s. The first request for these sessions then hits the cache. This keeps about 0.7 GB more in memory for a 279 MB trace than loading on demand.
- A 4-host session of 279 MB v4 traces loads in 1.8 s for the trace viewer of all hosts. `overview_page` takes 3.8 s and a peak of 4.5 GB. For 8 hosts, the times are 3.8 s and 5.2 s, and the peak is 7.6 GB.
- mimalloc is the global allocator. The server asks it to return free memory after large evictions.

## Development

Run these commands before you commit:

```bash
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release
```

`cargo test` runs the unit tests that come from XProf and XLA (in `src/tests`). It also compares each tool and the trace viewer with the outputs of XProf for the demo trace.

## License

Apache-2.0. The algorithms follow Apache-licensed code from [OpenXLA XProf](https://github.com/openxla/xprof) and [XLA/TSL](https://github.com/openxla/xla). See `NOTICE`.
