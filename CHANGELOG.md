# Changelog

## 0.1.0

First release.

- Rust backend for the XProf trace viewer. The responses are the same as the responses of XProf 2.23.2.
- All XProf tools and the XProf agent CLI.
- Remote log directories (`gs://`, `s3://`, `az://`, `http(s)://`, `file://`).
- The server listens on `127.0.0.1` by default. XProf listens on all interfaces. Use `--host` to change the address.
- See the README for the list of missing features.
