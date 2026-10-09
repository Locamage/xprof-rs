#!/usr/bin/env python3
"""Measure the cold and warm times of XProf and xprof-rs on one profile, and print Markdown tables.

Usage: benchmark.py SESSION_DIR --xprof PATH [--rs PATH] [--cores 0-3] [--trials N] [--warmups N] [--json FILE]

SESSION_DIR holds one `.xplane.pb` file and optional `.hlo_proto.pb` files.
Server, cold: the first request to a new server on a new copy of the files. Warm: the mean of 5 more requests.
CLI, cold: one process on a new copy with an empty TMPDIR (the XProf result cache). Warm: the same command again.
As in pyperf, the warmup trials are not counted, and each cell is the mean and the standard deviation of the trials.
A cell with a standard deviation of more than 10% of the mean has the mark (!).
The trials of XProf and xprof-rs alternate, so a change of the machine load affects the two the same.
Python 3 standard library only.
"""

import argparse
import http.client
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLS = ["trace_viewer@", "overview_page", "op_profile", "hlo_stats", "framework_op_stats", "input_pipeline_analyzer",
         "roofline_model", "memory_profile", "pod_viewer", "memory_viewer"]
COMMANDS = ["get_overview", "get_top_hlo_ops", "get_hlo_op_profile", "get_hlo_stats", "get_roofline_model", "get_step_trace",
            "check_host_boundness", "get_memory_profile", "list_hlo_modules", "aggregate_xplane_events", "compute_utilization",
            "get_avg_step_time", "get_device_information", "get_graph_viewer", "get_hlo_module_content", "get_hlo_text",
            "get_hosts", "get_kernel_stats", "get_kernel_utilization", "get_kpi_metrics", "get_llo_analysis",
            "get_llo_debug_string", "get_peak_allocations", "get_profile_summary", "get_utilization_viewer",
            "get_xspace_proto", "list_xplane_events"]
WARM_REQUESTS = 5
UNSTABLE = 0.10


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
        return fetch(port, path), statistics.fmean(fetch(port, path) for _ in range(WARM_REQUESTS))
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


def trials(measures, count, warmups):
    """Runs the measures in turns. Gives, for each measure, the cold and the warm values of the counted trials."""
    values = [([], []) for _ in measures]
    for trial in range(warmups + count):
        order = list(enumerate(measures))
        for index, measure in order if trial % 2 == 0 else reversed(order):
            cold, warm = measure()
            if trial >= warmups:
                values[index][0].append(cold)
                values[index][1].append(warm)
    return values


def cell(values):
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    unit, scale = ("ms", 1000) if mean < 1 else ("s", 1)
    digits = max(0, 2 - math.floor(math.log10(mean * scale)))
    warning = "" if deviation <= UNSTABLE * mean else " (!)"
    return f"{mean * scale:.{digits}f} ± {deviation * scale:.{digits}f} {unit}{warning}"


def ratio(slow, fast):
    value = statistics.fmean(slow) / statistics.fmean(fast)
    return f"{value:.0f}×" if value >= 10 else f"{value:.1f}×"


def table(title, rows):
    print(f"| {title} | XProf, cold | xprof-rs, cold | Speed-up | XProf, warm | xprof-rs, warm | Speed-up |")
    print("|---|---|---|---|---|---|---|")
    for name, ((old_cold, old_warm), (new_cold, new_warm)) in rows.items():
        print(f"| `{name}` | {cell(old_cold)} | {cell(new_cold)} | {ratio(old_cold, new_cold)} "
              f"| {cell(old_warm)} | {cell(new_warm)} | {ratio(old_warm, new_warm)} |")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--xprof", required=True, help="path of the XProf command")
    parser.add_argument("--rs", default="target/release/xprof-rs")
    parser.add_argument("--cores", help="CPU list for taskset, for example 0-3")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--json", type=Path, help="file for the values of all trials")
    arguments = parser.parse_args()
    pin = ["taskset", "-c", arguments.cores] if arguments.cores else []
    commands = [pin + [arguments.xprof], pin + [str(Path(arguments.rs).resolve())]]
    host = next(arguments.session.glob("*.xplane.pb")).name.removesuffix(".xplane.pb")
    modules = [path.name.removesuffix(".hlo_proto.pb") for path in arguments.session.glob("*.hlo_proto.pb")]
    tools = [tool for tool in TOOLS if tool != "memory_viewer" or modules]
    servers, clis = {}, {}
    with tempfile.TemporaryDirectory(prefix="xprof-benchmark-") as scratch:
        scratch = Path(scratch) / "work"
        for tool in tools:
            extra = "&resolution=8000" if tool == "trace_viewer@" else f"&module_name={modules[0]}" if tool == "memory_viewer" else ""
            path = f"/data/plugin/profile/data?run=run/s&tag={tool}&host={host}{extra}"
            measures = [lambda command=command, port=8981 + index: server(command, arguments.session, scratch, port, path) for index, command in enumerate(commands)]
            servers[tool] = trials(measures, arguments.trials, arguments.warmups)
            print(f"{tool}: done", file=sys.stderr, flush=True)
        for name in COMMANDS:
            measures = [lambda command=command: cli(command, arguments.session, scratch, name) for command in commands]
            clis[name] = trials(measures, arguments.trials, arguments.warmups)
            print(f"{name}: done", file=sys.stderr, flush=True)
    if arguments.json:
        arguments.json.write_text(json.dumps({"servers": servers, "clis": clis}, indent=1))
    table("Tool", servers)
    table("Command", clis)


if __name__ == "__main__":
    main()
