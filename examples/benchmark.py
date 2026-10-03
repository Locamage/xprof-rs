#!/usr/bin/env python3
"""Measure the time of xprof-rs and XProf on one logdir.

Usage: benchmark.py LOGDIR [--rs PATH] [--xprof PATH] [--repeat N] [--tools a,b,c]

Each server runs on a fresh copy of the profile files, so no cache is warm.
"first" is the first request after the server starts. "repeat" is the best of N later requests.
The CLI rows show the wall time of one process, best of N. Python 3 standard library only.
"""

import argparse
import gzip
import http.client
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLS = ["overview_page", "framework_op_stats", "input_pipeline_analyzer", "hlo_stats", "op_profile", "roofline_model", "memory_profile", "kernel_stats", "pod_viewer", "trace_viewer@"]
COMMANDS = ["get_overview", "get_top_hlo_ops", "get_hlo_op_profile", "get_step_trace", "check_host_boundness"]
PORTS = {"xprof-rs": 8981, "xprof": 8982}


def fetch(port, path):
    start = time.perf_counter()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    connection.request("GET", path, headers={"Accept-Encoding": "gzip"})
    response = connection.getresponse()
    size = len(response.read())
    return (time.perf_counter() - start) * 1000, size, response.status


def get_json(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port)
    connection.request("GET", path)
    body = connection.getresponse().read()
    return json.loads(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)


def peak_memory_mb(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM"):
            return int(line.split()[1]) // 1024
    return 0


def copy_profiles(logdir, target):
    for source in Path(logdir).rglob("*"):
        if source.is_file() and source.name.endswith((".xplane.pb", ".hlo_proto.pb")):
            destination = target / source.relative_to(logdir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def serve(name, command, logdir, tools, repeat):
    port = PORTS[name]
    with tempfile.TemporaryDirectory(prefix="bench-") as scratch:
        copy_profiles(logdir, Path(scratch))
        server = subprocess.Popen(command + ["--logdir", scratch, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(600):
                try:
                    status = fetch(port, "/data/plugin/profile/runs")
                    if status[2] == 200:
                        break
                except OSError:
                    time.sleep(0.1)
            run = get_json(port, "/data/plugin/profile/runs")[0]
            hosts = get_json(port, f"/data/plugin/profile/hosts?run={run}&tag=overview_page")
            host = next((entry["hostname"] for entry in hosts if entry["hostname"] != "ALL_HOSTS"), hosts[0]["hostname"])
            rows = {}
            for tool in tools:
                extra = "&resolution=8000" if tool.startswith("trace_viewer") else ""
                path = f"/data/plugin/profile/data?run={run}&tag={tool}&host={host}{extra}"
                first, size, status = fetch(port, path)
                best = min(fetch(port, path)[0] for _ in range(repeat))
                rows[tool] = (first, best, size, status)
            return rows, peak_memory_mb(server.pid)
        finally:
            server.terminate()
            server.wait()


def command_line(binary, logdir, command, repeat):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        subprocess.run([binary, command, str(logdir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        best = min(best, time.perf_counter() - start)
    return best * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir")
    parser.add_argument("--rs", default="target/release/xprof-rs")
    parser.add_argument("--xprof", help="path of the XProf command; skip XProf when missing")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--tools", default=",".join(TOOLS))
    arguments = parser.parse_args()
    tools = arguments.tools.split(",")
    results = {"xprof-rs": serve("xprof-rs", [arguments.rs], arguments.logdir, tools, arguments.repeat)}
    if arguments.xprof:
        results["xprof"] = serve("xprof", [arguments.xprof, "--grpc_port", "8983"], arguments.logdir, tools, arguments.repeat)
    names = list(results)
    print(f"{'tool':26}" + "".join(f"{name + ' first':>18}{name + ' repeat':>18}" for name in names) + f"{'bytes':>12}")
    for tool in tools:
        cells = "".join(f"{results[name][0][tool][0]:>15.0f} ms{results[name][0][tool][1]:>15.1f} ms" for name in names)
        print(f"{tool:26}{cells}{results[names[0]][0][tool][2]:>12}")
    print("peak memory " + ", ".join(f"{name} {results[name][1]} MB" for name in names))
    print("\nCLI wall time of xprof-rs, best of", arguments.repeat)
    for command in COMMANDS:
        print(f"{command:26}{command_line(arguments.rs, arguments.logdir, command, arguments.repeat):>10.0f} ms")


if __name__ == "__main__":
    sys.exit(main())
