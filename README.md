# xprof-rs

A fast, drop-in Rust backend for the [XProf](https://github.com/openxla/xprof) trace viewer.

XProf (the TensorBoard profiler) converts a multi-hundred-megabyte `.xplane.pb` into trace-viewer JSON in
Python and C++ on every cold request, which takes 10 to 25 seconds. xprof-rs serves the same `/data` responses
natively: it reads the file into memory, converts it in parallel, caches the result and renders only the window you
zoom to. It reproduces XProf 2.23.2's trace-viewer pipeline step by step and is checked against it
event by event.

| Trace | Size | Events | Cold view | Warm view | Zoom |
|---|---|---|---|---|---|
| TPU v4-8 | 279 MB | 2.95 M | 0.63 s load | 128 ms | 5 ms |

(4 cores; 2.0 s load on one core. XProf takes 11 to 25 s for the same file.)

## Install

xprof-rs is a single Rust binary (edition 2024, stable toolchain, Unix 64-bit). Building needs no Python, no C++ toolchain and no `protoc`.

```bash
git clone <repository-url> && cd xprof-rs
cargo install --path .
```

or `cargo build --release` and use `target/release/xprof-rs`. The release profile uses fat LTO and one codegen unit, so a
full build takes several minutes; for a quick iteration build use
`CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release`. Tagged releases publish an
`x86_64-linux` tarball (`.github/workflows/release.yml`).

```bash
xprof-rs --logdir ~/logs            # then open http://localhost:8791
xprof-rs get_overview ~/logs/run1   # the same binary is XProf's agent CLI
```

## What it serves

Everything is native Rust; there is no Python and no fallback. Same responses as XProf 2.23.2 for TPU, GPU, TensorFlow and host-only traces:

| Endpoint | Python XProf, cold (279 MB v4 trace) | xprof-rs first request | xprof-rs repeat |
|---|---|---|---|
| Trace viewer (`trace_viewer@`: zoom, search, details, DMA) | 11 to 25 s | 0.45 s | 2 ms |
| `overview_page` | 18 s | 2.8 s (builds the shared stats) | under 1 ms |
| `op_profile` (by program, category, provenance) | 4.2 s | 0.35 s | 3 ms |
| `hlo_stats` | 2.4 s | 0.35 s | 3 ms |
| `framework_op_stats`, `input_pipeline_analyzer`, `roofline_model`, `memory_profile`, `kernel_stats` | 0.5 to 2.8 s | 1 to 75 ms | under 1 ms |
| `memory_viewer`, `graph_viewer`, `module_list` (from HLO protos) | 2.3 s | 0.25 s | |
| `trace_viewer` (non-streaming JSON), `trace_viewer@&format=pb` (zstd delta-series protobuf), `pod_viewer`, `megascale_stats`, `smart_suggestion`, `perf_counters`, `utilization_viewer`, `kernel_utilization` | | native | |
| `runs`, `run_tools`, `hosts`, `data_csv`, `version`, `config`, `module_list`, `POST /generate_cache`, `/capture_profile` (gRPC client), static files | | native | |

Peak memory while serving every tool on that trace: about 2.4 GB (1.6 GB for a 197 MB trace).

## Web interface

xprof-rs does not contain its own UI. The web interface is XProf's: the prebuilt frontend files from the XProf 2.23.2
Python package (Apache License 2.0, copyright The TensorFlow Authors) are embedded unchanged in the binary from `static/`
and served as they are. Only the backend that answers the UI's requests is rewritten here. Point `XPROF_STATIC_DIR` at a
directory to serve different frontend files.

## Usage

```bash
xprof-rs [--logdir DIR|URL] [--port 8791] [--host ADDRESS] [--src_prefix PREFIX] [--hide_capture_profile_button] [--enable_tab_name_label]
```

Without `--logdir` there are no runs to list (as in `xprof`), but `session_path` and `run_path` requests still open any
directory. `XPROF_STATIC_DIR`, when it names a directory, overrides the embedded frontend files like it does for `xprof`.

`--grpc_port`, `--worker_service_address` and `--max_concurrent_worker_requests` are accepted and ignored (there is no distributed worker mode). Every route answers both at the root and under `/data/plugin/profile`.

Like `xprof`, it listens on every interface (`::`, or `0.0.0.0` without IPv6) on port 8791 and has no authentication;
pass `--host 127.0.0.1` to keep it local. With `--logdir`, requests can only read profiles inside it: `run`, `session_path` and
`module_name` that resolve outside it (absolute paths, `..`, symlinks pointing elsewhere) get a 400, and the logdir walk
does not follow directory symlinks.

## Remote logdirs

`--logdir` also takes an object store URL, like `xprof --logdir gs://bucket/dir`: `gs://`, `s3://` (and S3-compatible
stores such as R2), `az://`/`abfs://`, `http(s)://` (WebDAV) and `file://`, through the
[`object_store`](https://crates.io/crates/object_store) crate. Credentials come from the environment as that crate reads
them: `GOOGLE_APPLICATION_CREDENTIALS` or application default credentials for `gs://`; `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT` and `AWS_ALLOW_HTTP` for S3 and S3-compatible stores; `AZURE_*`
for Azure.

The store is mirrored lazily into a local directory, `xprof-rs-<hash of the URL>` under `XPROF_CACHE_DIR` (default the
OS temp dir), and every tool reads the mirror:

- `runs` lists `plugins/profile/` under the URL and every nested directory, as for a local logdir, and creates the
  session directories in the mirror without downloading anything.
- A request naming a session (`run`, `session_path`, or `run_path` with `run`; `run_path` alone fetches every session
  under it) first downloads that session's `.xplane.pb` and `.hlo_proto.pb` files, eight 8 MiB ranges per file and
  four files at a time, written in place. A file whose size and modification time match the store is not fetched
  again, and the store is asked again at most every 5 s per session and for the listing.
- The background poll re-checks the listing and the 8 most recently requested sessions that are still mirrored, every
  30 s.
- Sessions beyond `XPROF_CACHE_BYTES` (default 20 GiB) are evicted least recently requested first; a session requested
  in the last minute is kept, and an evicted one is downloaded again when it is next requested.
- An unreachable store or bad credentials answer 502 with the store's error (a listing gives up after 30 s).
- The mirror is written, the store never is, except by `/capture_profile`: captured files are uploaded to
  `<URL>/plugins/profile/<session>/`, and the profiler is given the URL as its repository. The `run_tools` cache file,
  the HLO protos `module_list` extracts and the rendered responses stay local.

Point the URL at the directory that holds the profiles: like the local walk, the listing visits every directory under it.

## Command line

The same binary also runs XProf's agent CLI (`xprof <command> <path> [flags]`), natively and with XProf 2.23.2's output:
`xprof-rs <command> <session> [--flag=value ...]`. The session is a run directory, a logdir (its latest run), an
`.xplane.pb` file or, with `--logdir`, a run name. Flags follow XProf (Python Fire): `--name=value`, `--name value`,
`--name`/`--noname` for booleans, positional parameters in XProf's order, and the `--session_dir`, `--session_path` and
`--source` aliases. Output is XProf's JSON or text on stdout; errors print XProf's `{"status": "ERROR", "reason": ...}`
with exit code 2 (usage), 3 (path), 4 (invalid value) or 1 (internal). Results over 10 MB are written to a file in
`$TMPDIR` and reported, like XProf does. `xprof-rs` and `xprof-rs server` with server flags still start the server.
On the 279 MB v4 trace, `get_overview` takes 2.3 s (18 s with XProf's CLI), `get_hlo_op_profile` 1.8 s (25 s) and
`aggregate_xplane_events` 2.3 s (25 s).

| Command | What it returns |
|---|---|
| `get_overview` | Performance summary and run environment, with roofline fallbacks (`--include_command`) |
| `get_kpi_metrics` | Step time, duty cycle, MXU and roofline utilization, peak HBM, accelerator |
| `get_device_information` | Device type, peak FLOP rate, bandwidths (GiB/s) and ridge points |
| `get_hosts` | Hosts of the session |
| `get_profile_summary` | Text table of the top operations by self time |
| `get_roofline_model` | Program roofline metrics, device info and top operations (`--top_n`) |
| `get_memory_profile` | HBM capacity, peak usage, stack, heap, free memory and fragmentation |
| `get_peak_allocations` | Modules by peak HBM with their largest buffers (`--limit --min_size_mib --output_format=markdown --noaggregate_instructions --noinclude_summary`) |
| `get_top_hlo_ops` | Top HLO ops by time, FLOPs and bytes (`--limit --category_filter`) |
| `get_hlo_op_profile` | Grouped, category, flat or tree views of the HLO op profile (`--view --category --path --depth --sort_by --top_n`) |
| `get_hlo_stats` | HLO statistics records (`--limit --sort_by --category_filter`) |
| `list_hlo_modules` | HLO modules of the session (extracting them from the trace when needed) |
| `get_hlo_module_content` | Module text (`--module_name --max_lines --print_metadata`) |
| `get_hlo_neighborhood` | Operands and users of an instruction (`--instruction_name --radius --module_name`) |
| `get_hlo_text` | Module text or an op's neighborhood, optionally saved (`--op_name --module_name --path`) |
| `get_graph_viewer` | Graph viewer output: `short_txt`, `long_txt`, `pbtxt`, `pb`, `graph` or `adj_nodes` (`--output_type --node_name --module_name ...`) |
| `get_step_trace` | Per-step breakdown and summary from pod viewer, input pipeline or overview (`--step_num --limit --device_core --noinclude_summary`) |
| `check_host_boundness` | Host-boundness verdict from overview, op profile, barrier-cores and utilization |
| `get_utilization_viewer` | Hardware counter utilization of one host, device and node (`--host --device --node`) |
| `get_kernel_utilization`, `compute_utilization` | Per-kernel counter utilization (`--kernel_name --device --duration_us --force_duration`) |
| `get_kernel_stats` | Device kernel durations, optionally with the disjoint interval union (`--limit --kernel_name --include_summary --trace_matchers --output_format`) |
| `get_avg_step_time` | Average `XLA Modules` step time (`--func_name`) |
| `list_xplane_events` | Timeline events filtered by plane and event regex and time window (`--plane_regex --event_regex --start_time_ps --end_time_ps --max_events --offset`) |
| `aggregate_xplane_events` | Count, total, average, min, max and standard deviation per event name |
| `get_xspace_proto` | The raw XSpace bytes, or saved with `--output_path` |
| `upload_trace` | Copies a trace into `--logdir` under `--run_name` |
| `get_llo_analysis`, `get_llo_debug_string` | XProf's LLO-unavailable report (LLO analysis is closed source) |
| `verify_numerical_parity` | XProf's missing-numerical-dependencies error (it runs Python callables) |

```bash
xprof-rs get_overview ~/logs/run1
xprof-rs get_top_hlo_ops ~/logs/run1 --limit=5 --category_filter=fusion
xprof-rs get_hlo_op_profile ~/logs --view=tree --path=by_program --depth=3
xprof-rs get_hlo_neighborhood run1.xplane.pb --instruction_name=fusion.12 --radius=1
xprof-rs list_xplane_events ~/logs/run1 --plane_regex='TPU:0$' --event_regex=all-reduce --max_events=20
```

Differences from the Python CLI: there is no result cache (every call computes fresh, as with `--bypass_cache`), so no
`__cached__` fields; XProf's own hash-order ties (equal-size buffers in `get_peak_allocations`, equal-time ops in
`get_hlo_op_profile`, equal allocator peaks in `get_memory_profile`) can come out in a different order, and
`get_graph_viewer --output_type=pb` returns the module file's bytes where XProf re-serializes it with hash-ordered maps; regex error
messages are Rust's; `--help` prints a short synopsis. Like XProf 2.23.2's CLI, tools built on combined op statistics
report no data for multi-host sessions (XProf needs its worker service for those), and `memory_profile` fails on them.
The HLO detectors listed in XProf's README (`detect_*`) are not registered by XProf 2.23.2's CLI and are not commands here either.

## Source layout

One binary crate. `unsafe` is limited to a few libc calls (`snprintf`, `localtime_r`, `getuid`) and mimalloc's `mi_collect`.

| Path | Role |
|---|---|
| `src/main.rs` | Flags, axum routes, caches (converted traces, op statistics, rendered responses), background prefetch |
| `src/xplane.rs`, `src/trace.rs`, `src/derive.rs`, `src/group.rs` | XSpace reader, event conversion, derived timelines, step and op grouping |
| `src/json.rs`, `src/legacy_trace.rs`, `src/delta.rs` | Trace viewer renderers: streaming JSON, non-streaming JSON, zstd delta protobuf |
| `src/opstats.rs` and the per-tool modules (`op_profile.rs`, `hlo_stats.rs`, `roofline.rs`, `overview_page.rs`, `memory_profile.rs`, ...) | Tools built on the shared op statistics |
| `src/hlo*.rs`, `src/graph_viewer.rs`, `src/memory_viewer.rs`, `src/gpu_cost.rs`, `src/pbtext.rs` | HLO parser/printer, graph and memory viewers, GPU cost analysis |
| `src/gpu.rs`, `src/inference_profile.rs`, `src/pod_viewer.rs`, `src/megascale*.rs`, `src/utilization.rs`, `src/smart_suggestion.rs` | GPU, inference, pod, megascale and utilization tools |
| `src/remote.rs`, `src/capture.rs`, `src/run_tools.rs` | Object store mirrors, `/capture_profile` gRPC client, `run_tools` |
| `src/cli/` | XProf's agent CLI |
| `src/tests/` | Ported XProf/XLA unit tests and golden comparisons |

## Exactness

The demo trace (a small synthetic workload on a TPU v4-8, with host name, script path and host system information replaced) and XProf's outputs for it are checked in under `tests/data`; `cargo test` compares every tool and the
trace viewer against them. A few fields cannot match any single run because XProf itself is not deterministic there
(tied rows in absl hash-map order, which `memory_profile` snapshot is kept, buffer order in `memory_viewer`).

## Missing and not supported

This is a backend rewrite, not a port of the whole XProf distribution. What is not here:

**Inputs**
- `.xplane.riegeli` sessions, which XProf also lists, are not read. Those files only come from the gRPC capture client's continuous-profiling mode; `jax.profiler.start_trace` (used by MaxText, Levanter and marin) writes `.xplane.pb`.
- Profiles of 4 GiB or more are rejected (`profile larger than 4 GiB`).

**Server and deployment**
- No distributed worker mode: `--grpc_port`, `--worker_service_address` and `--max_concurrent_worker_requests` are accepted and ignored, there is no worker gRPC service and no request fan-out. Consequently tools built on combined op statistics report no data for multi-host sessions through the CLI, exactly like XProf's CLI without its worker service.
- `/capture_profile` is a gRPC client only; xprof-rs does not host a profiler service.
- No authentication or TLS, as in `xprof`.
- No TensorBoard plugin loader: it is a standalone server, not `tensorboard --logdir`.
- Unix, 64-bit only (Linux is what CI runs); no Windows build.

**Tools**
- `get_llo_analysis`, `get_llo_debug_string` return XProf's "LLO unavailable" report; the LLO analysis is closed source.
- `verify_numerical_parity` only returns XProf's missing-dependencies error, because it executes Python callables.
- The `detect_*` commands from XProf's README are not registered by XProf 2.23.2's CLI and are not commands here either.
- `mpmd_pipeline_view=true` is accepted and ignored (XProf 2.23.2 ignores it too); `custom_call_graph` is not a graph type in XProf 2.23.2 either.
- No CLI result cache: every call computes fresh as with `--bypass_cache`, so there are no `__cached__` fields.
- `--help` of the CLI prints a short synopsis, and regex errors are Rust's.

**The Python surface of the XProf package**
- `xprof.convert` and the `ProfileData` Python library, the `xprof.api` continuous-profiling snapshot client, `install_and_run.py`, the labs, and the agent skills are not provided. Only the server routes and the CLI commands are.

**Output differences that cannot be closed**
- XProf's own run-to-run randomness cannot be reproduced: tie order in `hlo_stats`, `op_profile`, `memory_viewer` and `get_peak_allocations` (absl hash-map order, seeded per process), memory profile allocator peaks and fragmentation snapshots, the order of processes and threads in the protobuf, `bind_id` flow ids of the trace viewer (a process-seeded hash), graph node ids (heap pointers) and the hover CSS rules and caller lists of the graph viewer.
- `get_graph_viewer --output_type=pb` returns the module file's bytes where XProf re-serializes it with hash-ordered maps.
- XProf writes the `Launch Stats` line of a GPU device in absl hash order and merges it as if sorted, so for launches whose API name does not start with `cu` the `uid` of host events sharing a timestamp with it, and detail lookups of those events, differ from run to run.
- XProf caches the first op statistics it computes for a multi-host session under `ALL_HOSTS` and serves them for every host selection, and sometimes leaves the high-confidence duty cycle of a multi-host session at 0 depending on what its process served before. xprof-rs answers each selection as a fresh XProf process does (so `roofline_model` follows the host selection).
- `inference_profile` (and the inference latency table of `overview_page`): XProf orders tensor pattern rows, ties between requests of equal latency from different hosts or models, and the summation order of multi-model latency averages in randomized hash-map order.

**Frontend**
- The UI is XProf's prebuilt frontend, embedded unchanged. There is nothing to extend in this repository; changes to the UI need the XProf frontend sources.

## Coverage of the checks

- Pinned to XProf 2.23.2 semantics.
- TPU: TPU v4 and v6e traces are compared with XProf on every tool; the demo trace in `tests/data` is the only trace checked into the repository. Multi-host TPU sessions are checked on synthetic sessions of 2, 4 and 8 hosts derived from those traces (renamed hosts, shifted clocks, truncated step counts, a host without `Steps` or `XLA Modules`, hosts with fewer chips) on every tool for `ALL_HOSTS` and each host, and on the trace viewer for each host and host subsets. Flow ids mix in the XSpace hostname as XProf does, so flows of different hosts stay apart.
- GPU: one 8-GPU trace and synthetic GPU profiles (1, 2 and 8 GPUs; NVIDIA from Kepler to Blackwell and AMD; TensorFlow, JAX and eager steps; memcpy, memset, P2P and NCCL; CUDA graphs; one or two hosts) on every tool; XLA modules compiled for sm_90a for the GPU cost analysis and the graph viewer, and cuDNN/cuBLAS backend-config variants in `graph_viewer`. The synthetic GPU profiles and HLO fixtures are checked in under `tests/data/gpu`.
- Serving (`inference_profile`) is checked on synthetic serving profiles only (TensorFlow, batching, TFRT, Pathways, Orbax and user-defined requests, one or more hosts); no real serving profile was available.
- Real-world coverage is therefore limited to the traces above. Please file an issue with a failing trace (without private data) when a response differs from XProf 2.23.2.

## Resource use

- Three LRU caches, each evicting after an hour idle: converted traces up to 4 GiB (file plus events), op statistics up to 4 GiB (weighted by file size) and rendered responses up to 1 GiB (compressed), so about 9 GiB in total. A single trace larger than its budget is still served and cached alone. A file is reloaded when its size, modification time or inode changes, and a file modified in the last 2 s is served without being cached.
- A background task loads the trace and the op statistics of the 8 newest sessions (files untouched for 30 s) at startup and whenever a session changes, one file at a time, so their first request is a cache hit; this keeps about 0.7 GB more resident for a 279 MB trace than loading on demand.
- Loading a 4-host session of 279 MB v4 traces takes 1.8 s for the trace viewer of all hosts and 3.8 s for `overview_page` (4.5 GB peak); 8 hosts take 3.8 s and 5.2 s (7.6 GB peak).
- mimalloc is the global allocator and is asked to return freed memory after large evictions.

## Development

```bash
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release
```

`cargo test` runs the ported XProf and XLA unit tests under `src/tests` and compares every tool and the trace viewer with
XProf's outputs for the demo trace in `tests/data`.

## License

Apache-2.0. The algorithms follow Apache-licensed code from [OpenXLA XProf](https://github.com/openxla/xprof) and
[XLA/TSL](https://github.com/openxla/xla); see `NOTICE`.
