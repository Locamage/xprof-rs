# Release steps

1. Check that the working tree is clean and CI is green on `main`.
2. Change `version` in `Cargo.toml`. Run `cargo check --locked` to update `Cargo.lock`.
3. Add the changes to `CHANGELOG.md`.
4. Run the full build and checks on the release profile:
   - `cargo fmt --check`
   - `cargo clippy --release --all-targets -- -D warnings`
   - `cargo test --release`
5. Commit the changes with the message `Release vX.Y.Z`.
6. Make the tag: `git tag vX.Y.Z`. Push the commit and the tag: `git push origin main vX.Y.Z`.
7. The `release` workflow does these steps:
   - It checks that the tag and the `Cargo.toml` version are the same.
   - It builds `x86_64-linux` and `aarch64-linux` binaries.
   - It makes a GitHub release with the archives and the SHA-256 files.
8. Optional: publish the crate with `cargo publish --locked`. The package must be smaller than 10 MiB.

The minimum Rust version is 1.95. The `msrv` job in CI checks it. Change `rust-version` in `Cargo.toml` and the `msrv` job together.
