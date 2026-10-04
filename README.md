# xprof-rs

xprof-rs is a fast backend for the [XProf](https://github.com/openxla/xprof) profile viewer. It is one Rust binary. It replaces the XProf server and the XProf agent CLI, and it serves the same user interface. Its responses are the same as the responses of XProf 2.23.2. The tests compare them with the output of XProf.

| 279 MB TPU v4 profile, 2.95 M events | XProf | xprof-rs |
|---|---|---|
| Open the trace viewer | 12 s | 0.73 s |
| Open the overview page | 18 s | 1.3 s |
| Open the op profile after the overview | 4.0 s | 0.13 s |
| `get_overview` in the CLI | 18 s | 1.0 s |

We measured all times in this file with 4 cores of a larger machine, unless the text gives a different number of cores. More cores make xprof-rs faster. [`examples/benchmark.py`](examples/benchmark.py) measures these times on your own profile.

XProf converts the `.xplane.pb` file again for each cold request. xprof-rs does these steps:

1. It reads the file into memory.
2. It converts the events in parallel.
3. It keeps the result in a cache.
4. It renders only the time window that you zoom to. A new window takes about 10 ms, and the same window again takes about 3 ms.

## Install and start

The [releases](https://github.com/Locamage/xprof-rs/releases) have binaries for Linux on `x86_64` and `aarch64`. They need glibc 2.28 or newer (RHEL 8, Debian 10, Ubuntu 20.04, or newer). You do not need Python.

```bash
version=v0.1.0 target=x86_64-linux   # or aarch64-linux
curl -LO https://github.com/Locamage/xprof-rs/releases/download/$version/xprof-rs-$version-$target.tar.gz
tar -xzf xprof-rs-$version-$target.tar.gz
xprof-rs-$version-$target/xprof-rs --logdir ~/logs          # open http://localhost:8791
xprof-rs-$version-$target/xprof-rs get_overview ~/logs/run1 # the same binary runs the XProf agent CLI
```

To build from source, you need Rust 1.95 or newer and a 64-bit Unix system. You do not need a C++ toolchain or `protoc`.

```bash
git clone https://github.com/Locamage/xprof-rs && cd xprof-rs
cargo install --locked --path .
```

The release profile uses fat LTO and one codegen unit. A full build takes about 80 s on a machine with 240 cores. It uses about 10 CPU minutes. For a quick build, set `CARGO_PROFILE_RELEASE_LTO=false` and `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`.

## Server

```bash
xprof-rs [--logdir DIR|URL] [--port 8791] [--host ADDRESS] [--src_prefix PREFIX] [--hide_capture_profile_button] [--enable_tab_name_label]
```

- The user interface is the prebuilt frontend of the XProf 2.23.2 Python package. It uses the Apache License 2.0. The copyright holder is The TensorFlow Authors.
- The binary includes the unchanged files from `static/`. To serve other files, set `XPROF_STATIC_DIR` to a directory.
- Each route answers at the root and under `/data/plugin/profile`.
- The server listens on `127.0.0.1` port 8791. There is no authentication. To let other computers connect, use `--host 0.0.0.0` (all IPv4 interfaces) or `--host ::` (all interfaces). XProf listens on all interfaces.
- With `--logdir`, a request can read only the files in the log directory.
- A `run`, `session_path`, or `module_name` that points outside the log directory gets a 400 response.
- The directory walk does not follow symbolic links to directories.
- Without `--logdir`, the server lists no runs. A `session_path` or `run_path` request can still open a directory.
- The server accepts `--grpc_port`, `--worker_service_address`, and `--max_concurrent_worker_requests`. It does not use them.

The server has these tools. They are all written in Rust.

- Trace viewer: `trace_viewer@` (zoom, search, details, DMA, `format=pb`) and `trace_viewer`.
- Statistics: `overview_page`, `op_profile`, `hlo_stats`, `framework_op_stats`, `input_pipeline_analyzer`, `roofline_model`, `kernel_stats`, `inference_profile`.
- Memory and graphs: `memory_profile`, `memory_viewer`, `graph_viewer`, `module_list`.
- Hardware: `pod_viewer`, `megascale_stats`, `smart_suggestion`, `perf_counters`, `utilization_viewer`, `kernel_utilization`.
- Session: `runs`, `run_tools`, `hosts`, `data_csv`, `version`, `config`, `POST /generate_cache`, `/capture_profile` (gRPC client), and the static files.

The table shows the time on the 279 MB v4 trace. [`examples/benchmark.py`](examples/benchmark.py) measures these columns.

| Endpoint | XProf, cold | xprof-rs, first | xprof-rs, repeat |
|---|---|---|---|
| `trace_viewer@` | 12 s | 0.73 s | 2 ms |
| `overview_page` | 18 s | 1.3 s (builds the shared statistics) | under 1 ms |
| `op_profile` | 4.0 s | 0.13 s | 3 ms |
| `hlo_stats` | 2.1 s | 0.09 s | 3 ms |
| `framework_op_stats`, `input_pipeline_analyzer`, `roofline_model`, `memory_profile`, `kernel_stats`, `pod_viewer` | 0.35 to 2.7 s | 0 to 0.03 s | under 1 ms |
| `memory_viewer`, `graph_viewer`, `module_list` | 2.3 s | 0.3 s | 3 ms |

The peak memory of xprof-rs for the tools in the table is 1.5 GB. XProf used 4.8 GB for all of its tools. A machine with more cores uses more memory, because more work runs at the same time.

## Remote log directories

`--logdir` accepts these URL schemes: `gs://`, `s3://` (also S3-compatible stores), `az://`, `abfs://`, `http(s)://` (WebDAV), and `file://`. The [`object_store`](https://crates.io/crates/object_store) crate does the access.

The credentials come from the environment:

- `gs://`: `GOOGLE_APPLICATION_CREDENTIALS` or the application default credentials.
- S3: the `AWS_*` variables.
- Azure: the `AZURE_*` variables.

The server copies the store to a local directory. The directory name is `xprof-rs-<hash of the URL>`. It is in `XPROF_CACHE_DIR`. The default is the temporary directory. All tools read this copy.

- A request that names a session downloads the `.xplane.pb` and `.hlo_proto.pb` files of the session.
- The server downloads 4 files at a time. It reads each file in eight ranges of 8 MiB.
- The server does not download a file again when the size and the modification time are the same as in the store.
- The server asks the store at most one time in 5 s for each session.
- Each 30 s, a background poll checks the list of sessions. It also checks the 8 sessions with the newest requests.
- If the copy is larger than `XPROF_CACHE_BYTES`, the server removes sessions. The default is 20 GiB. The server removes the session with the oldest request first. It keeps a session with a request in the last minute.
- If the store is not reachable, the response is 502 with the error of the store. The response is the same for credentials that are not correct. A list request stops after 30 s.
- The server does not write to the store. There is one exception. `/capture_profile` uploads the captured files to `<URL>/plugins/profile/<session>/`.

Set the URL to the directory that holds the profiles. The server visits each directory under it.

## Command line

Use `xprof-rs <command> <session> [--flag=value ...]` to run the agent CLI of XProf. The stdout and the exit codes are the same as in XProf 2.23.2.

The session is one of these:

- a run directory
- a log directory (the CLI uses the latest run)
- an `.xplane.pb` file
- a run name, with `--logdir`

The flags follow XProf (Python Fire):

- `--name=value` and `--name value`
- `--name` and `--noname` for booleans
- positional parameters in the order of XProf
- the aliases `--session_dir`, `--session_path`, and `--source`

An error prints `{"status": "ERROR", "reason": ...}`. The exit code is 2 (usage), 3 (path), 4 (value not valid), or 1 (internal). A result larger than 10 MB goes to a file in `$TMPDIR`. This is the same as in XProf.

The commands are:

- Summary: `get_overview`, `get_kpi_metrics`, `get_device_information`, `get_hosts`, `get_profile_summary`, `get_avg_step_time`, `get_step_trace`, `check_host_boundness`.
- Operations: `get_roofline_model`, `get_top_hlo_ops`, `get_hlo_op_profile`, `get_hlo_stats`, `get_kernel_stats`.
- Memory: `get_memory_profile`, `get_peak_allocations`.
- HLO: `list_hlo_modules`, `get_hlo_module_content`, `get_hlo_neighborhood`, `get_hlo_text`, `get_graph_viewer`.
- Utilization: `get_utilization_viewer`, `get_kernel_utilization`, `compute_utilization`.
- Events and files: `list_xplane_events`, `aggregate_xplane_events`, `get_xspace_proto`, `upload_trace`.
- Not available: `get_llo_analysis`, `get_llo_debug_string`, `verify_numerical_parity`.

The flags of each command are the flags of XProf. Examples:

```bash
xprof-rs get_top_hlo_ops ~/logs/run1 --limit=5 --category_filter=fusion
xprof-rs get_hlo_op_profile ~/logs --view=tree --path=by_program --depth=3
xprof-rs list_xplane_events ~/logs/run1 --plane_regex='TPU:0$' --event_regex=all-reduce --max_events=20
```

The table shows the time on the 279 MB v4 trace.

| Command | XProf | xprof-rs |
|---|---|---|
| `get_overview` | 18 s | 1.0 s |
| `get_hlo_op_profile` | 25 s | 1.4 s |
| `aggregate_xplane_events` | 25 s | 0.65 s |

More cores make a command faster, up to about 16 cores. `get_overview` takes 1.9 s on 2 cores, 0.62 s on 8 cores, 0.40 s on 16 cores, and 0.38 s on 32 cores. Its peak memory is 0.96 GB on 4 cores and 1.6 GB on 32 cores.

The CLI is different from the Python CLI in these points:

- There is no result cache. Each call computes the result again, as with `--bypass_cache`. There are no `__cached__` fields.
- Regex errors are the errors of Rust.
- `--help` prints a short synopsis.
- `get_graph_viewer --output_type=pb` returns the bytes of the module file. XProf serializes the module again with hash-ordered maps.
- For a multi-host session, the tools that use combined op statistics report no data. `memory_profile` fails. XProf needs its worker service for these sessions.

## Not available and not supported

xprof-rs replaces the backend. It does not replace the complete XProf distribution.

**Inputs**

- xprof-rs does not read `.xplane.riegeli` sessions. The continuous capture mode of the gRPC client is the only source of these files. The `jax.profiler.start_trace` function writes `.xplane.pb`. MaxText, Levanter, and marin use this function.
- xprof-rs rejects a profile of 4 GiB or more. A profile file is one protobuf message, and protobuf limits a message to 2 GiB. Thus XProf cannot read a profile of 2 GiB or more.

**Server**

- There is no distributed worker mode. There is no worker gRPC service. `/capture_profile` is a client.
- There is no TLS. There is no authentication. There is no TensorBoard plugin loader.
- xprof-rs runs on 64-bit Unix. CI tests Linux on x86_64 and aarch64. There is no Windows build.

**Tools**

- `get_llo_analysis` and `get_llo_debug_string` return the LLO-not-available report of XProf. The source code of the LLO analysis is not public.
- `verify_numerical_parity` returns the error of XProf for dependencies that are not installed. XProf runs Python callables for this command.
- The `detect_*` commands from the README of XProf are not available. XProf 2.23.2 does not register them.
- `mpmd_pipeline_view=true` has no effect. XProf has the same behavior.

**Python part of the XProf package**

- These parts are not available: `xprof.convert`, the `ProfileData` library, the `xprof.api` snapshot client, `install_and_run.py`, the labs, and the agent skills.

**Frontend**

- This repository does not contain the sources of the user interface.

**Output that changes from run to run in XProf**

xprof-rs cannot copy these outputs.

- The order of tied rows in `hlo_stats`, `op_profile`, `memory_viewer`, and `get_peak_allocations`. XProf uses hash order. The seed changes for each process.
- The order of processes and threads in the protobuf.
- The allocator peaks and the fragmentation snapshot of the memory profile.
- The `bind_id` flow ids of the trace viewer.
- The graph node ids. They are heap pointers.
- The hover CSS rules and the caller lists of the graph viewer.
- The `Launch Stats` line of a GPU device. XProf writes it in hash order. The `uid` of host events at the same timestamp can differ.
- The statistics of a multi-host session. XProf caches the first op statistics under `ALL_HOSTS`. It sometimes leaves the duty cycle at 0. xprof-rs answers each host selection as a new XProf process does.
- The tensor pattern rows of `inference_profile`.
- The order of requests with equal latency in `inference_profile`.
- The order of the sums in the latency averages.

## Test coverage

- `tests/data` has a demo trace. It is a small synthetic workload on a TPU v4-8. We replaced the host name, the script path, and the host system information.
- `tests/data` also has the outputs of XProf for the demo trace. It has synthetic GPU profiles and HLO fixtures in `tests/data/gpu`.
- `cargo test` compares each tool and the trace viewer with these outputs.
- Other tests used TPU v4 and v6e traces and one 8-GPU trace. They also used synthetic sessions of 2, 4, and 8 hosts.
- The synthetic GPU sessions use 1, 2, or 8 GPUs from NVIDIA or AMD. They cover TensorFlow, JAX, eager steps, NCCL, and CUDA graphs.
- The synthetic inference sessions cover TensorFlow, request batches, TFRT, Pathways, and Orbax. No real inference profile was available.
- These traces are not in the repository.
- The real-world coverage is small. If a response is different from XProf 2.23.2, open an issue. Do not include private data.

## Memory

- Three LRU caches hold converted traces, op statistics, and rendered responses (compressed). Each of the first two holds an eighth of the physical memory, at most 4 GiB. The third holds a quarter of that. On a machine with 32 GiB or more, the total is about 9 GiB.
- The server loads 2 files at once on a machine with 32 GiB or more, and 1 file otherwise. A load needs about four to six times the size of its file. More cores need more memory.
- The server and the CLI use one thread per core, at most 32.
- A cache removes an entry after one hour without use.
- A trace that is larger than its cache is still served. The server caches it alone.
- The server loads a file again when the size, the modification time, or the inode changes. It does not cache a file that changed in the last 2 s.
- A background task loads the 8 newest sessions at startup and when a session changes. It loads only the files that did not change for 30 s and that are smaller than a thirty-second of the physical memory. This task keeps about 0.7 GB more in memory for a 279 MB trace.
- A session of 4 hosts has 279 MB v4 traces. The trace viewer of all hosts takes 1.8 s. `overview_page` takes 3.8 s and a peak of 4.5 GB.
- For 8 hosts, the times are 3.8 s and 5.2 s. The peak is 7.6 GB.

## Development

Run these commands before each commit:

```bash
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release
```

A tag that starts with `v` makes a draft release with `x86_64-linux` and `aarch64-linux` archives and SHA-256 files. The steps are in `RELEASING.md`.

The code is in `src/`:

- `main.rs`: flags, routes, and caches.
- `xplane.rs` and `trace.rs`: reader and conversion.
- One module for each tool.
- `cli/`: the agent CLI.
- `remote.rs`: object stores.
- `tests/`: tests from XProf and XLA.

## License

Apache-2.0. The algorithms follow Apache-licensed code from [OpenXLA XProf](https://github.com/openxla/xprof) and [XLA/TSL](https://github.com/openxla/xla). See `NOTICE`.
