# xprof-rs

[![crates.io](https://img.shields.io/crates/v/xprof-rs.svg)](https://crates.io/crates/xprof-rs)
[![CI](https://github.com/Locamage/xprof-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Locamage/xprof-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

xprof-rs is a fast backend for the [XProf](https://github.com/openxla/xprof) profile viewer. It is one Rust binary. It replaces the XProf server and the XProf agent CLI, and it serves the same user interface. Its responses are the same as the responses of XProf 2.23.2. The tests compare them with the output of XProf.

xprof-rs is an independent project. Google and the OpenXLA project do not maintain or endorse it.

We wrote xprof-rs with [Claude](https://www.anthropic.com/claude), an AI model from Anthropic. The tests compare the output of each tool with the output of XProf 2.23.2.

On an 80 MB TPU v4 profile of a job that trains a CLIP model, the trace viewer opens in 0.30 s (XProf: 5.6 s). After the first request, it opens in approximately 2 ms (XProf: 0.62 s). The peak memory of the server is 0.42 GB (XProf: 1.42 GB).

xprof-rs also reads profiles from S3-compatible stores such as Cloudflare R2. XProf 2.23.2 cannot read them. The [performance](#performance) section has the times of all tools, also on a disk and on R2.

We measured all times in this file with 4 cores of a larger machine, unless the text gives a different number of cores.

XProf converts the `.xplane.pb` file again for each cold request. xprof-rs does these steps:

1. It reads the file into memory.
2. It converts the events in parallel.
3. It keeps the result in a cache.
4. It renders only the time window that you zoom to. A new window takes about 10 ms, and the same window again takes about 3 ms.

## Performance

The tables show the times on an 80 MB TPU v4 profile of a job that trains a CLIP model. Each server and each CLI process used 4 cores.

- Server, cold: the first request to a new server on a new copy of the profile. There is no cache on the disk.
- Server, warm: the mean of 5 more requests to the same server.
- CLI, cold: one process on a new copy of the profile, with no result cache.
- CLI, warm: the same command again.
- Peak memory: the peak resident memory of the server over the cold and the warm requests, or of the cold CLI process. It includes the pages of the profile that the process maps.

Each cell is the mean and the standard deviation of 8 trials, after 1 warmup trial that is not counted, as in pyperf. The trials of XProf and xprof-rs alternate, so a change of the machine load affects the two the same. The mark (!) shows a standard deviation of more than 10% of the mean.

| Tool | XProf, cold | xprof-rs, cold | Speed-up | XProf, warm | xprof-rs, warm | Speed-up | XProf, peak memory | xprof-rs, peak memory |
|---|---|---|---|---|---|---|---|---|
| `trace_viewer@` | 5.58 ± 0.05 s | 300 ± 5 ms | 19× | 618 ± 2 ms | 1.99 ± 0.06 ms | 310× | 1.42 ± 0.01 GB | 419 ± 10 MB |
| `overview_page` | 3.92 ± 0.02 s | 193 ± 6 ms | 20× | 56.2 ± 0.4 ms | 1.02 ± 0.64 ms (!) | 55× | 1.24 ± 0.00 GB | 330 ± 7 MB |
| `op_profile` | 4.39 ± 0.02 s | 211 ± 6 ms | 21× | 572 ± 6 ms | 3.08 ± 4.43 ms (!) | 186× | 1.30 ± 0.02 GB | 327 ± 6 MB |
| `hlo_stats` | 3.99 ± 0.01 s | 202 ± 3 ms | 20× | 177 ± 3 ms | 0.519 ± 0.016 ms | 341× | 1.26 ± 0.03 GB | 327 ± 10 MB |
| `framework_op_stats` | 3.93 ± 0.09 s | 194 ± 9 ms | 20× | 97.9 ± 1.5 ms | 0.750 ± 0.549 ms (!) | 131× | 1.24 ± 0.00 GB | 326 ± 8 MB |
| `input_pipeline_analyzer` | 3.86 ± 0.05 s | 191 ± 6 ms | 20× | 54.2 ± 0.7 ms | 0.427 ± 0.044 ms (!) | 127× | 1.24 ± 0.00 GB | 328 ± 10 MB |
| `roofline_model` | 4.11 ± 0.04 s | 206 ± 7 ms | 20× | 269 ± 2 ms | 0.582 ± 0.017 ms | 462× | 1.32 ± 0.01 GB | 325 ± 3 MB |
| `memory_profile` | 1.68 ± 0.01 s | 102 ± 2 ms | 16× | 1.52 ± 0.00 s | 0.558 ± 0.321 ms (!) | 2724× | 3.92 ± 0.00 GB | 267 ± 2 MB |
| `pod_viewer` | 3.90 ± 0.01 s | 189 ± 5 ms | 21× | 56.5 ± 1.9 ms | 0.402 ± 0.013 ms | 141× | 1.24 ± 0.00 GB | 329 ± 9 MB |
| `memory_viewer` | 85.7 ± 0.6 ms | 19.3 ± 0.8 ms | 4.4× | 79.5 ± 1.1 ms | 0.359 ± 0.020 ms | 221× | 240 ± 10 MB | 76 ± 1 MB |

| Command | XProf, cold | xprof-rs, cold | Speed-up | XProf, warm | xprof-rs, warm | Speed-up | XProf, peak memory | xprof-rs, peak memory |
|---|---|---|---|---|---|---|---|---|
| `get_overview` | 4.20 ± 0.01 s | 190 ± 5 ms | 22× | 224 ± 1 ms | 188 ± 5 ms | 1.2× | 1.20 ± 0.00 GB | 300 ± 9 MB |
| `get_top_hlo_ops` | 5.40 ± 0.02 s | 111 ± 3 ms | 49× | 225 ± 1 ms | 108 ± 2 ms | 2.1× | 1.21 ± 0.00 GB | 214 ± 2 MB |
| `get_hlo_op_profile` | 5.41 ± 0.01 s | 112 ± 5 ms | 48× | 228 ± 1 ms | 108 ± 3 ms | 2.1× | 1.20 ± 0.00 GB | 214 ± 1 MB |
| `get_hlo_stats` | 4.11 ± 0.02 s | 107 ± 9 ms | 38× | 224 ± 1 ms | 105 ± 8 ms | 2.1× | 1.20 ± 0.00 GB | 202 ± 2 MB |
| `get_roofline_model` | 4.21 ± 0.02 s | 178 ± 1 ms | 24× | 224 ± 1 ms | 178 ± 3 ms | 1.3× | 1.21 ± 0.00 GB | 299 ± 4 MB |
| `get_step_trace` | 4.05 ± 0.02 s | 193 ± 4 ms | 21× | 224 ± 1 ms | 183 ± 6 ms | 1.2× | 1.20 ± 0.00 GB | 301 ± 8 MB |
| `check_host_boundness` | 66.5 ± 0.4 s | 231 ± 7 ms | 288× | 223 ± 1 ms | 229 ± 5 ms | 1.0× | 7.89 ± 0.01 GB | 331 ± 4 MB |
| `get_memory_profile` | 1.82 ± 0.01 s | 113 ± 1 ms | 16× | 224 ± 1 ms | 112 ± 2 ms | 2.0× | 860 ± 0 MB | 256 ± 6 MB |
| `list_hlo_modules` | 224 ± 1 ms | 5.92 ± 0.16 ms | 38× | 223 ± 2 ms | 5.30 ± 0.18 ms | 42× | 60 ± 0 MB | 23 ± 0 MB |
| `aggregate_xplane_events` | 21.0 ± 0.1 s | 106 ± 3 ms | 198× | 239 ± 1 ms | 104 ± 2 ms | 2.3× | 2.52 ± 0.00 GB | 154 ± 2 MB |
| `compute_utilization` | 3.08 ± 0.02 s | 50.1 ± 1.6 ms | 62× | 223 ± 1 ms | 46.5 ± 0.5 ms | 4.8× | 1.26 ± 0.00 GB | 91 ± 2 MB |
| `get_avg_step_time` | 1.33 ± 0.00 s | 53.4 ± 1.6 ms | 25× | 223 ± 0 ms | 52.6 ± 1.2 ms | 4.2× | 668 ± 0 MB | 93 ± 1 MB |
| `get_device_information` | 4.14 ± 0.01 s | 87.5 ± 2.2 ms | 47× | 223 ± 1 ms | 85.9 ± 1.9 ms | 2.6× | 1.20 ± 0.00 GB | 187 ± 2 MB |
| `get_graph_viewer` | 338 ± 1 ms | 25.7 ± 0.8 ms | 13× | 338 ± 1 ms | 25.9 ± 1.0 ms | 13× | 102 ± 0 MB | 59 ± 2 MB |
| `get_hlo_module_content` | 345 ± 1 ms | 25.9 ± 0.7 ms | 13× | 224 ± 1 ms | 26.3 ± 1.1 ms | 8.5× | 102 ± 0 MB | 59 ± 1 MB |
| `get_hlo_text` | 346 ± 2 ms | 26.2 ± 0.8 ms | 13× | 225 ± 1 ms | 26.4 ± 1.1 ms | 8.5× | 102 ± 0 MB | 59 ± 1 MB |
| `get_hosts` | 223 ± 1 ms | 5.74 ± 0.36 ms | 39× | 222 ± 1 ms | 5.33 ± 0.30 ms | 42× | 60 ± 0 MB | 23 ± 0 MB |
| `get_kernel_stats` | 11.5 ± 0.0 s | 72.1 ± 1.9 ms | 159× | 223 ± 1 ms | 69.3 ± 1.9 ms | 3.2× | 1.62 ± 0.00 GB | 106 ± 2 MB |
| `get_kernel_utilization` | 3.08 ± 0.01 s | 49.6 ± 1.0 ms | 62× | 223 ± 2 ms | 46.6 ± 0.6 ms | 4.8× | 1.26 ± 0.00 GB | 89 ± 1 MB |
| `get_kpi_metrics` | 5.40 ± 0.02 s | 211 ± 5 ms | 26× | 226 ± 1 ms | 206 ± 4 ms | 1.1× | 1.21 ± 0.00 GB | 301 ± 8 MB |
| `get_llo_analysis` | 1.29 ± 0.01 s | 6.01 ± 0.24 ms | 215× | 1.29 ± 0.00 s | 5.27 ± 0.17 ms | 245× | 587 ± 0 MB | 23 ± 0 MB |
| `get_llo_debug_string` | 1.29 ± 0.00 s | 5.93 ± 0.34 ms | 218× | 1.29 ± 0.01 s | 5.39 ± 0.37 ms | 239× | 588 ± 0 MB | 23 ± 0 MB |
| `get_peak_allocations` | 297 ± 1 ms | 26.7 ± 1.0 ms | 11× | 223 ± 1 ms | 27.6 ± 1.1 ms | 8.1× | 90 ± 0 MB | 61 ± 0 MB |
| `get_profile_summary` | 5.39 ± 0.01 s | 106 ± 2 ms | 51× | 224 ± 1 ms | 103 ± 4 ms | 2.2× | 1.21 ± 0.00 GB | 210 ± 2 MB |
| `get_utilization_viewer` | 2.16 ± 0.00 s | 49.8 ± 0.9 ms | 44× | 223 ± 1 ms | 47.4 ± 1.2 ms | 4.7× | 972 ± 0 MB | 91 ± 1 MB |
| `get_xspace_proto` | 347 ± 3 ms | 85.5 ± 2.0 ms | 4.1× | 788 ± 49 ms | 95.7 ± 0.9 ms | 8.2× | 140 ± 0 MB | 96 ± 0 MB |
| `list_xplane_events` | 20.8 ± 0.1 s | 73.9 ± 1.7 ms | 281× | 224 ± 1 ms | 74.4 ± 1.6 ms | 3.0× | 2.50 ± 0.00 GB | 99 ± 1 MB |

Some warm XProf commands are as fast as xprof-rs or faster. XProf keeps each result in a cache in `$TMPDIR`, and a second call reads this cache. xprof-rs has no result cache. It reads the profile again for each call.

The table has all the commands that need only a session. `get_hlo_neighborhood`, `verify_numerical_parity`, and `upload_trace` need more input, so the table does not show them. `get_xspace_proto` writes the 80 MB profile to `/tmp` on a disk. On a warm call, XProf writes over the file of the cold call, and the write waits until the disk has the data.

The next table shows five profiles of jobs that train CLIP models. The trace viewer time is for a cold server. The peak memory is the peak of the server process after the trace viewer and seven other tools.

| Profile | XProf, trace viewer | xprof-rs, trace viewer | XProf, peak memory | xprof-rs, peak memory |
|---|---|---|---|---|
| 43 MB | 3.0 s | 186 ms | 1.21 GB | 0.43 GB |
| 43 MB | 2.6 s | 195 ms | 1.23 GB | 0.44 GB |
| 52 MB | 3.4 s | 295 ms | 1.50 GB | 0.57 GB |
| 80 MB | 6.0 s | 453 ms | 2.09 GB | 0.75 GB |
| 117 MB | 8.1 s | 678 ms | 3.09 GB | 1.05 GB |

These numbers are for the 80 MB profile:

| Item | XProf | xprof-rs |
|---|---|---|
| Trace viewer, zoom to a new time window | 95 ms | 1.9 ms |
| Trace viewer, the same window again | 88 ms | 0.3 ms |
| Server start, until the first response | 0.41 s | 0.05 s |
| Size of the installed files | 127 MB (Python packages, without Python) | 27 MB (one binary) |
| Size of the download | 52 Python packages | 12.6 MB archive |

More cores make xprof-rs faster. The next table shows the cold times of xprof-rs on the 80 MB profile. XProf on 4 cores takes 5.4 s for the trace viewer and 3.9 s for the overview page.

| Cores | Trace viewer | Overview page | Peak memory |
|---|---|---|---|
| 1 | 1.04 s | 0.89 s | 0.62 GB |
| 2 | 0.61 s | 0.47 s | 0.65 GB |
| 4 | 0.40 s | 0.28 s | 0.69 GB |
| 8 | 0.32 s | 0.18 s | 0.76 GB |
| 16 | 0.36 s | 0.16 s | 0.79 GB |
| 32 | 0.36 s | 0.16 s | 0.81 GB |
| 64 | 0.35 s | 0.14 s | 0.80 GB |

To measure the times of the first two tables on your profile, run [`examples/benchmark.py`](examples/benchmark.py) `SESSION_DIR --xprof PATH --cores 0-3 --trials 8`.

The next table shows the cold times of the overview page on a disk and on Cloudflare R2. Each time starts with a new server and stops at the end of the first response. XProf 2.23.2 cannot read `s3://` URLs, so it reads R2 through `rclone mount`. The section [Times on a disk and on R2](#times-on-a-disk-and-on-r2) has all the tools and the method.

| Profile | Source | XProf | xprof-rs |
|---|---|---|---|
| 80 MB | Disk of a cloud VM | 5.79 s | 1.11 s |
| 80 MB | R2, `rclone mount` | 6.06 s | 3.49 s |
| 80 MB | R2, `s3://` URL | not available | 1.45 s |
| 279 MB | Disk of a cloud VM | 25.2 s | 3.30 s |
| 279 MB | R2, `rclone mount` | 24.9 s | 5.81 s |
| 279 MB | R2, `s3://` URL | not available | 3.47 s |

## Install and start

The [releases](https://github.com/Locamage/xprof-rs/releases) have binaries for Linux on `x86_64` and `aarch64`, and for macOS on Apple silicon. The Linux binaries need glibc 2.28 or newer (RHEL 8, Debian 10, Ubuntu 20.04, or newer). You do not need Python.

```bash
version=v0.1.3 target=x86_64-linux   # or aarch64-linux, aarch64-macos
curl -LO https://github.com/Locamage/xprof-rs/releases/download/$version/xprof-rs-$version-$target.tar.gz
tar -xzf xprof-rs-$version-$target.tar.gz
xprof-rs-$version-$target/xprof-rs --logdir ~/logs          # open http://localhost:8791
xprof-rs-$version-$target/xprof-rs get_overview ~/logs/run1 # the same binary runs the XProf agent CLI
```

To build from source, you need Rust 1.95 or newer, a C compiler, and a 64-bit Unix system. You do not need a C++ toolchain or `protoc`.

```bash
cargo install --locked xprof-rs
```

The release profile uses fat LTO and one codegen unit. A full build takes about 80 s on a machine with 240 cores. It uses about 10 CPU minutes. On 8 cores, it takes about 2 minutes. For a quick build, set `CARGO_PROFILE_RELEASE_LTO=false` and `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`.

## Server

```bash
xprof-rs [--logdir DIR|URL] [--port 8791] [--host ADDRESS] [--src_prefix PREFIX] [--hide_capture_profile_button] [--enable_tab_name_label]
```

- The user interface is the prebuilt frontend of the XProf 2.23.2 Python package. It uses the Apache License 2.0. The copyright holder is The TensorFlow Authors.
- The binary includes the files from `ui/`. To serve other files, set `XPROF_STATIC_DIR` to a directory.
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

A machine with more cores uses more memory, because more work runs at the same time.

## Remote log directories

`--logdir` accepts these URL schemes: `gs://`, `s3://` (also S3-compatible stores), `az://`, `abfs://`, `http(s)://` (WebDAV), and `file://`. The [`object_store`](https://crates.io/crates/object_store) crate does the access.

The credentials come from the environment:

- `gs://`: `GOOGLE_APPLICATION_CREDENTIALS` or the application default credentials.
- S3: the `AWS_*` variables.
- Azure: the `AZURE_*` variables.

The server copies the store to a local directory, and all tools read this copy. The directory is `xprof-rs-<hash of the URL>` in `XPROF_CACHE_DIR`. The default is the temporary directory. Only the user of the server can read the directory. The server does not start if a different user owns it.

- A request that names a session downloads the `.xplane.pb` and `.hlo_proto.pb` files of the session.
- The server downloads 4 files at a time. It reads each file in ranges of 8 MiB, and it reads 8 ranges at a time.
- The server does not download a file again when the size and the modification time are the same as in the store.
- The server asks the store at most one time in 5 s for each session.
- Each 30 s, a background poll checks the list of sessions. It also checks the 8 sessions with the newest requests.
- If the copy is larger than `XPROF_CACHE_BYTES`, the server removes sessions. The default is 20 GiB. The server removes the session with the oldest request first. It keeps a session with a request in the last minute.
- If the store is not reachable, the response is 502 with the error of the store. The response is the same for credentials that are not correct. A list request stops after 30 s.
- The server does not write to the store. There is one exception. `/capture_profile` uploads the captured files to `<URL>/plugins/profile/<session>/`.

Set the URL to the directory that holds the profiles. The server visits each directory under it.

### Times on a disk and on R2

XProf 2.23.2 does not read `s3://` URLs. Its server and its file code know only `gs://` and local paths. To read R2, XProf needs a FUSE mount of the bucket, for example `rclone mount`. xprof-rs can read R2 through the `s3://` URL or through the mount.

The next tables show the cold times of a new server with 4 cores. Each time starts with the process and stops at the end of the first response.

- tmpfs: the files are in memory.
- Disk: the persistent disk of a cloud VM. Before each trial, the page cache did not have the files.
- R2 mount: a new `rclone mount --vfs-cache-mode writes` of Cloudflare R2 for each trial. XProf writes its cache files into the session directory, so the mount must accept writes.
- R2: xprof-rs reads the `s3://` URL. Before each trial, the cache directory was empty.

Before each trial, we removed the files that the servers wrote. Each cell is the mean of 3 trials on the 80 MB profile and of 2 trials on the 279 MB profile.

80 MB profile:

| Tool | XProf, tmpfs | xprof-rs, tmpfs | XProf, disk | xprof-rs, disk | XProf, R2 mount | xprof-rs, R2 mount | xprof-rs, R2 |
|---|---|---|---|---|---|---|---|
| Trace viewer | 5.71 s | 0.46 s | 7.33 s | 1.48 s | 7.32 s | 2.40 s | 1.77 s |
| Overview page | 4.31 s | 0.26 s | 5.79 s | 1.11 s | 6.06 s | 3.49 s | 1.45 s |
| Op profile | 4.83 s | 0.32 s | 6.27 s | 0.81 s | 6.34 s | 2.72 s | 1.43 s |
| HLO op stats | 4.41 s | 0.28 s | 5.93 s | 1.21 s | 5.92 s | 4.64 s | 1.80 s |
| Framework op stats | 4.31 s | 0.27 s | 5.75 s | 0.79 s | 5.84 s | 2.25 s | 1.57 s |
| Input pipeline | 4.26 s | 0.26 s | 5.88 s | 0.74 s | 6.92 s | 2.58 s | 1.37 s |
| Roofline model | 4.49 s | 0.29 s | 6.00 s | 0.78 s | 6.14 s | 2.52 s | 1.45 s |
| Memory profile | 2.09 s | 0.16 s | 3.71 s | 0.83 s | 4.31 s | 2.01 s | 1.47 s |
| Pod viewer | 4.32 s | 0.26 s | 5.90 s | 0.75 s | 6.38 s | 2.38 s | 1.44 s |
| Memory viewer | 0.46 s | 0.05 s | 0.58 s | 0.11 s | 1.52 s | 1.00 s | 1.15 s |

279 MB profile:

| Tool | XProf, tmpfs | xprof-rs, tmpfs | XProf, disk | xprof-rs, disk | XProf, R2 mount | xprof-rs, R2 mount | xprof-rs, R2 |
|---|---|---|---|---|---|---|---|
| Trace viewer | 14.7 s | 0.84 s | 18.0 s | 3.51 s | 20.0 s | 7.76 s | 3.02 s |
| Overview page | 21.3 s | 1.05 s | 25.2 s | 3.30 s | 24.9 s | 5.81 s | 3.47 s |
| Op profile | 27.3 s | 1.29 s | 31.0 s | 3.59 s | 33.5 s | 5.56 s | 2.97 s |
| HLO op stats | 23.2 s | 1.21 s | 26.4 s | 3.56 s | 28.8 s | 4.73 s | 3.29 s |
| Framework op stats | 21.3 s | 1.03 s | 24.8 s | 3.34 s | 24.2 s | 7.02 s | 2.83 s |
| Input pipeline | 21.3 s | 1.01 s | 24.6 s | 3.27 s | 25.4 s | 8.37 s | 2.71 s |
| Roofline model | 21.6 s | 1.06 s | 25.0 s | 3.33 s | 25.2 s | 7.12 s | 2.90 s |
| Memory profile | 3.88 s | 0.28 s | 7.18 s | 2.61 s | 7.89 s | 6.62 s | 1.96 s |
| Pod viewer | 21.3 s | 1.05 s | 24.7 s | 3.27 s | 26.6 s | 6.14 s | 2.70 s |
| Memory viewer | 2.06 s | 0.33 s | 2.53 s | 1.17 s | 3.16 s | 1.84 s | 2.09 s |

- After the first request, xprof-rs answers in 0.4 ms to 6 ms from all sources. XProf answers in 55 ms to 2.2 s on the 80 MB profile and in 0.6 s to 6.9 s on the 279 MB profile.
- The responses of xprof-rs were the same for all sources. For 5 tools, the bytes of XProf change from one process to the next: the order of map keys and of equal items, and the `bind_id` values of the trace viewer. Without these changes, the responses of XProf and xprof-rs have the same data.
- The disk gave approximately 150 MB/s. This is the limit of the disk, and it changes with the load of the disk.
- A read through `rclone mount` is approximately 46 MB/s, because the mount reads a file in order with one stream. On a FUSE mount, xprof-rs reads each file in order. On other file systems, it reads parts of the file in parallel.
- With the `s3://` URL, the first list of the store takes approximately 0.3 s. The download is approximately 210 MB/s, approximately the same as `rclone` with 8 to 32 streams. The server downloads all the files of the session, also for the memory viewer, which needs only the HLO file.
- The R2 times change with the network. One XProf trial of the op profile on the mount took 149 s, and we measured it again.
- When the server starts again with the same cache directory, it does not download the files again. The first overview page of the 279 MB profile then takes 1.05 s.

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

More cores make a command faster, up to about 16 cores.

The CLI is different from the Python CLI in these points:

- There is no result cache. Each call computes the result again, as with `--bypass_cache`. There are no `__cached__` fields.
- The regex options find text as Python `re` does, and a bad pattern gives the error message of Python. But `\N{name}` always gives the error `undefined character name`, and a pattern with more than approximately 60 levels of nested groups gives an error.
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
- xprof-rs runs on 64-bit Unix. CI tests Linux on x86_64 and aarch64, and macOS on Apple silicon. There is no Windows build.

**Tools**

- `get_llo_analysis` and `get_llo_debug_string` return the LLO-not-available report of XProf. The source code of the LLO analysis is not public.
- `verify_numerical_parity` returns the error of XProf for dependencies that are not installed. XProf runs Python callables for this command.
- The `detect_*` commands from the README of XProf are not available. XProf 2.23.2 does not register them.
- `mpmd_pipeline_view=true` has no effect. XProf has the same behavior.
- `perf_counters` with `names_only` and `utilization_viewer` support only TPU v6e and v7x. XProf 2.23.2 has counter tables only for these two chips. The tables are in `src/data/`.

**Python part of the XProf package**

- These parts are not available: `xprof.convert`, the `ProfileData` library, the `xprof.api` snapshot client, `install_and_run.py`, the labs, and the agent skills.

**Frontend**

- This repository does not contain the sources of the user interface.

**Output that changes from run to run in XProf**

xprof-rs cannot copy these outputs.

- The order of tied rows in `hlo_stats`, `op_profile`, `memory_viewer`, and `get_peak_allocations`. XProf uses hash order. The seed changes for each process. A node of `op_profile` shows at most 100 children. When children have the same time, the tie order selects the children that show. The category sums of `get_hlo_op_profile` add the children that show, so these sums can also change.
- The tied ops of `get_top_hlo_ops`. This command reads the same children, so its order of tied ops and its list of ops can also change.
- The order of processes and threads in the protobuf.
- The allocator peaks and the fragmentation snapshot of the memory profile.
- The `bind_id` flow ids of the trace viewer.
- The order of the `stackFrames` entries of the trace viewer. XProf writes them in hash order.
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
- To report a security problem, follow `SECURITY.md`. Do not open a public issue.

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

Run these commands before each commit. `Cargo.toml` turns on the pedantic lints of clippy.

```bash
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release
```

A tag that starts with `v` makes a draft release with `x86_64-linux`, `aarch64-linux`, and `aarch64-macos` archives and SHA-256 files. The steps are in `RELEASING.md`.

The code is in `src/`:

- `main.rs`: flags, routes, and caches.
- `xplane/`: the `.xplane.pb` reader, derived lines, event groups, and steps.
- `trace/`: the trace viewer conversion and its JSON and protobuf output.
- `hlo/`: the HLO module, its text form, the GPU cost model, the graph viewer, and the memory viewer.
- `tools/`: one module for each other tool.
- `server/`: object stores, profile capture, and the tool list of a run.
- `cli/`: the agent CLI.
- `data/`: the protobuf descriptors, the TPU counter tables, and the HTML page of the graph viewer.
- `tests/`: tests from XProf and XLA. `tests/inline/` has the unit tests of the modules, in the same folders.

## License

Apache-2.0. The algorithms follow Apache-licensed code from [OpenXLA XProf](https://github.com/openxla/xprof) and [XLA/TSL](https://github.com/openxla/xla). See `NOTICE`.
