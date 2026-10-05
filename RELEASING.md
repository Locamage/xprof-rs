# Release steps

Each release of xprof-rs is a GitHub release with prebuilt binaries. We do not publish xprof-rs to crates.io (`publish = false`).

1. Make sure that `git status` shows no changes and that CI passes on `main`.
2. Change `version` in `Cargo.toml`. Run `cargo check --locked` to update `Cargo.lock`.
3. Add a `## X.Y.Z` section to `CHANGELOG.md`. The section becomes the text of the release.
4. Run the full build and checks on the release profile:
   - `cargo fmt --check`
   - `cargo clippy --release --all-targets -- -D warnings`
   - `cargo test --release`
5. Optional dry run: start the `release` workflow by hand (`gh workflow run release`). It builds both binaries and does not publish.
6. Commit the changes with the message `Release vX.Y.Z`.
7. Make the tag: `git tag vX.Y.Z`. Push the commit and the tag: `git push origin main vX.Y.Z`.
8. The `release` workflow does these steps:
   - It checks that the tag and the `Cargo.toml` version are the same, and that `CHANGELOG.md` has a section for the version.
   - It builds `x86_64-linux` and `aarch64-linux` binaries in the `manylinux_2_28` image. These binaries need glibc 2.28 or newer. It builds the `aarch64-macos` binary on a macOS runner.
   - It makes a draft GitHub release with the archives, the SHA-256 files, and the changelog section.
9. Download an archive from the draft. Check it with `sha256sum -c`, and run it on a profile. Then publish the draft: `gh release edit vX.Y.Z --draft=false`.

The minimum Rust version is 1.95. The `msrv` job in CI checks it. Change `rust-version` in `Cargo.toml` and the `msrv` job together.
