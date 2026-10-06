# Changelog

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
