#!/usr/bin/env python3
"""Measure the cold and warm times of XProf and xprof-rs on one profile, and print Markdown tables.

Usage: benchmark.py SESSION_DIR --xprof PATH [--rs PATH] [--cores 0-3] [--trials N]

SESSION_DIR holds one `.xplane.pb` file and optional `.hlo_proto.pb` files.
Server, cold: the first request to a new server on a new copy of the files. Warm: the best of 3 more requests.
CLI, cold: one process on a new copy with an empty TMPDIR (the XProf result cache). Warm: the same command again.
Each cold time is the best of N trials. Python 3 standard library only.
"""

import argparse
import http.client
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

TOOLS = ["trace_viewer@", "overview_page", "op_profile", "hlo_stats", "framework_op_stats", "input_pipeline_analyzer",
         "roofline_model", "memory_profile", "pod_viewer", "memory_viewer"]
COMMANDS = ["get_overview", "get_top_hlo_ops", "get_hlo_op_profile", "get_hlo_stats", "get_roofline_model", "get_step_trace",
            "check_host_boundness", "get_memory_profile", "list_hlo_modules", "aggregate_xplane_events"]


def fresh(session, scratch):
    shutil.rmtree(scratch, ignore_errors=True)
    target = scratch / "logs/run/plugins/profile/s"
    target.mkdir(parents=True)
    (scratch / "tmp").mkdir()
    for source in session.iterdir():
        if source.name.endswith((".xplane.pb", ".hlo_proto.pb")):
            shutil.copy2(source, target)
    env = {key: value for key, value in os.environ.items() if key != "LD_PRELOAD"}
    return scratch / "logs", {**env, "TMPDIR": str(scratch / "tmp")}


def fetch(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    start = time.perf_counter()
    connection.request("GET", path, headers={"Accept-Encoding": "gzip"})
    response = connection.getresponse()
    response.read()
    if response.status != 200:
        raise RuntimeError(f"{path}: HTTP {response.status}")
    return time.perf_counter() - start


def server(command, session, scratch, port, path):
    logs, env = fresh(session, scratch)
    process = subprocess.Popen(command + ["--logdir", str(logs), "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        for _ in range(600):
            try:
                fetch(port, "/data/plugin/profile/runs")
                break
            except OSError:
                time.sleep(0.1)
        return fetch(port, path), min(fetch(port, path) for _ in range(3))
    finally:
        process.terminate()
        process.wait()


def cli(command, session, scratch, name):
    logs, env = fresh(session, scratch)
    times = []
    for _ in range(2):
        start = time.perf_counter()
        done = subprocess.run(command + [name, str(logs / "run")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        times.append(time.perf_counter() - start)
        if done.returncode != 0:
            raise RuntimeError(f"{name}: exit code {done.returncode}")
    return times


def best(measure, trials):
    pairs = [measure() for _ in range(trials)]
    return min(pair[0] for pair in pairs), min(pair[1] for pair in pairs)


def duration(seconds):
    if seconds < 0.01:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def ratio(slow, fast):
    value = slow / fast
    return f"{value:.0f}×" if value >= 10 else f"{value:.1f}×"


def table(title, rows):
    print(f"| {title} | XProf, cold | xprof-rs, cold | Speed-up | XProf, warm | xprof-rs, warm | Speed-up |")
    print("|---|---|---|---|---|---|---|")
    for name, ((old_cold, old_warm), (new_cold, new_warm)) in rows.items():
        print(f"| `{name}` | {duration(old_cold)} | {duration(new_cold)} | {ratio(old_cold, new_cold)} "
              f"| {duration(old_warm)} | {duration(new_warm)} | {ratio(old_warm, new_warm)} |")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--xprof", required=True, help="path of the XProf command")
    parser.add_argument("--rs", default="target/release/xprof-rs")
    parser.add_argument("--cores", help="CPU list for taskset, for example 0-3")
    parser.add_argument("--trials", type=int, default=2)
    arguments = parser.parse_args()
    pin = ["taskset", "-c", arguments.cores] if arguments.cores else []
    commands = {"xprof": pin + [arguments.xprof], "xprof-rs": pin + [str(Path(arguments.rs).resolve())]}
    host = next(arguments.session.glob("*.xplane.pb")).name.removesuffix(".xplane.pb")
    modules = [path.name.removesuffix(".hlo_proto.pb") for path in arguments.session.glob("*.hlo_proto.pb")]
    tools = [tool for tool in TOOLS if tool != "memory_viewer" or modules]
    with tempfile.TemporaryDirectory(prefix="xprof-benchmark-") as scratch:
        scratch = Path(scratch) / "work"
        servers, clis = {}, {}
        for tool in tools:
            extra = "&resolution=8000" if tool == "trace_viewer@" else f"&module_name={modules[0]}" if tool == "memory_viewer" else ""
            path = f"/data/plugin/profile/data?run=run/s&tag={tool}&host={host}{extra}"
            servers[tool] = [best(lambda: server(command, arguments.session, scratch, 8981 + index, path), arguments.trials)
                             for index, command in enumerate(commands.values())]
        for name in COMMANDS:
            clis[name] = [best(lambda: cli(command, arguments.session, scratch, name), arguments.trials) for command in commands.values()]
    table("Tool", servers)
    table("Command", clis)


if __name__ == "__main__":
    main()
