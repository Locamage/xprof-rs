# Changelog

## 0.1.2

The tools and the CLI commands are faster, and they use less memory. The responses do not change.

- On an 80 MB TPU v4 profile with 4 cores, `get_hlo_stats`, `get_top_hlo_ops`, `get_hlo_op_profile` and `get_profile_summary` take 0.14 s to 0.15 s. In 0.1.1, they took 0.29 s to 0.31 s. Their peak memory is approximately 40% less.
- `get_roofline_model` takes 0.22 s, and `get_device_information` takes 0.20 s. In 0.1.1, they took 0.30 s and 0.29 s.
- The first request of the trace viewer takes 0.37 s. In 0.1.1, it took 0.41 s. The first request of the other tools takes 0.24 s to 0.26 s. In 0.1.1, it took 0.27 s to 0.32 s.
- The `trace_viewer` tool checks the file at the same time as it reads the trace. Before, the check ran first.

## 0.1.1

The tools and the CLI commands are faster, and they use less memory. The responses do not change.

- On an 80 MB TPU v4 profile with 4 cores, the first request of the overview page takes 0.27 s. In 0.1.0, it took 0.52 s. The memory profile takes 0.13 s. In 0.1.0, it took 0.53 s.
- The CLI commands that read op statistics take 0.27 s to 0.32 s. In 0.1.0, they took 0.50 s to 0.57 s. `get_kernel_stats` takes 77 ms. In 0.1.0, it took 0.79 s.
- The peak memory of these CLI commands is approximately 20% less.
- xprof-rs uses version 2 of mimalloc.

## 0.1.0

First release.

- Rust backend for the XProf trace viewer. The responses are the same as the responses of XProf 2.23.2.
- All XProf tools and the XProf agent CLI.
- Binaries for Linux on `x86_64` and `aarch64`, and for macOS on Apple silicon.
- The crate on crates.io: `cargo install --locked xprof-rs`.
- Remote log directories (`gs://`, `s3://`, `az://`, `http(s)://`, `file://`).
- The server listens on `127.0.0.1` by default. XProf listens on all interfaces. Use `--host` to change the address.
- The README lists the features that are not available.
