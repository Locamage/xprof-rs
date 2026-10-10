# Changelog

## 0.1.3

The user interface and the trace viewer are faster, and they look the same. The JSON text of the trace viewer is now the same as the text of XProf.

- The trace viewer writes the times as XProf does. XProf writes `%.17g` of the time in microseconds, for example `660215.41857099999`. Before, xprof-rs wrote six decimal places, for example `660215.418571`. The two texts have the same value, but the bytes were different. The response is approximately 4% larger, and its first render takes approximately 5% more time.
- Charts that did not show now show. Before, a tool sometimes got its data before Google Charts loaded its packages. Then some charts stayed empty. Examples are the charts of the HLO op stats, the heap chart of the memory viewer, and the device charts of the framework op stats. Now the interface starts after Google Charts is ready.
- The browser keeps the files of the user interface. When the page opens again, the server sends a short "not modified" response, not 1.3 MB of files. The browser also keeps its compiled scripts, so the trace viewer opens approximately 20% faster when you open it again. The page looks the same.
- The user interface is faster, and the pages look the same. On a 280 MB TPU v4 profile, the roofline model is ready in 2.6 s on a new server (before: 5.1 s), and in 1.6 s when the server has the data (before: 3.9 s). A chart draws one time when its data and its filters change, not two times. Ready means that the page draws all charts and responds to input.
- On a 280 MB TPU v4 profile, the HLO op stats page responds to input after 4.5 s. Before, it did not respond until 8.2 s. The charts show at the same time as before, after approximately 3.1 s. Google Charts makes a hidden table of all the rows of each chart for screen readers. The browser does not do the layout of these tables until they come into view. Screen readers can still read them.
- The trace viewer opens faster, and it shows the same trace. On a 280 MB TPU v4 profile, it opens in 2.0 s on a new server (before: 2.8 s), and in 1.8 s when the server has the trace (before: 2.1 s). When the browser asks for the hosts of a profile with one host, the server starts to load the profile. Thus the load and the start of the trace viewer occur at the same time.
- When you zoom the trace viewer, it gets new data 200 ms after the last change of the view. Before, it waited 500 ms. On a CLIP profile, the zoomed view shows in 245 ms. Before, it showed in 540 ms.
- When you zoom the trace viewer on a 280 MB TPU v4 profile, the new view shows approximately 60 ms faster.
- When you zoom the trace viewer during a load, it gets the data for the new view after the load. Before, it kept the data of the old view until you zoomed again.
- On a 280 MB TPU v4 profile, the first request of the op statistics tools is 2% to 3% faster. The server does not do a text comparison for each event, and it checks valid text faster.
- On a 280 MB TPU v4 profile with 4 cores, the CLI commands that read op statistics are 4% to 6% faster. On an 80 MB profile, they are 1% to 5% faster. For the time span of the XLA ops on a line, xprof-rs reads only the first and the last events of the line. Before, it read the stats of all the events.
- `get_xspace_proto` removes the old file in `/tmp` before it writes the new file. On ext4, a write over a large file waits until the disk has the data. When you call the command again for the same session on an 80 MB profile, it takes 0.10 s. Before, it took 0.60 s. The first call does not change. The file does not change.
- `get_hlo_stats` does not make the statistics of the instructions in fusions, because its output does not show them. On a 280 MB TPU v4 profile with 4 cores, it takes 0.58 s. Before, it took 0.80 s. On an 80 MB profile, it takes 0.12 s. Before, it took 0.14 s.
- The CLI commands that scan all the events start with the largest lines. Before, one thread sometimes scanned all of them. On a 280 MB TPU v4 profile with 4 cores, `get_kernel_stats` takes 0.37 s, `aggregate_xplane_events` takes 0.42 s, and `list_xplane_events` takes 0.21 s. Before, they took 0.59 s, 0.58 s and 0.32 s. On an 80 MB profile, these commands and `get_avg_step_time` are 2% to 10% faster.
- When a profile has more than 5,000,000 events, `aggregate_xplane_events` uses the results of the parallel scan. Before, it scanned the profile again on one thread. On a 1.7 GB profile with 4 cores, it takes 2.2 s. Before, it took 6.4 s.
- `check_host_boundness` gets the times of the op profile from the op statistics. Before, it made the JSON text of the op profile and then read the text again. On a 280 MB TPU v4 profile with 4 cores, it takes 1.15 s. Before, it took 1.25 s. On an 80 MB profile, it takes 0.30 s. Before, it took 0.31 s.
- `get_device_information` makes only the properties of the roofline table. It does not make the rows or the programs of the steps. On a 280 MB TPU v4 profile with 4 cores, it takes 0.37 s. Before, it took 0.59 s. On an 80 MB profile, it takes 0.10 s. Before, it took 0.20 s.
- The trace viewer finds the events with a flow when it gives the flows their numbers. Before, it read all the events again on one thread. On a 1.7 GB profile with 90,000,000 events, the first request of the trace viewer takes 13.2 s. Before, it took 13.6 s.
- When a trace has more than 16,777,216 events, the trace viewer sorts the events in place. Before, it kept two copies of the events during the sort. On a 1.7 GB profile with 8 cores, the first request of the trace viewer uses 14 GB of memory and takes 11.9 s. Before, it used 22 GB and took 10.2 s. A smaller trace does not change.
- When the trace viewer finds the names of the stack frames, each thread reads a flag before it writes the flag. Before, all the threads wrote the same flags. On a 1.7 GB profile, this step takes 0.13 s. Before, it took 1.5 s.
- The legacy JSON trace of the `trace_viewer` tool keeps the first 5,000,000 events in time. For a larger trace, it now finds the time of the last kept event first, and it makes only the events up to that time. Before, it made all the events. On a 1.7 GB profile with 4 cores, it takes 14 s. Before, it took 132 s.
- The barrier times of `check_host_boundness` use the same method, and they make only the barrier events. On a 1.7 GB profile with 4 cores, `check_host_boundness` takes 9.8 s and 8 GB of memory. Before, it took 32 s and 40 GB.
- The README shows the times of all the CLI commands that need only a session.
- On a FUSE mount, for example `rclone mount` of a bucket, xprof-rs reads each file in order. Before, it read 4 parts of the file at the same time, and `rclone mount` then gave approximately 1 MB/s. On an 80 MB profile on an `rclone mount` of Cloudflare R2, the first overview page takes 2 s to 5 s. Before, it took 66 s. On other file systems, the read does not change.
- The README compares XProf and xprof-rs on tmpfs, on a disk, and on Cloudflare R2.

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
