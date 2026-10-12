#!/usr/bin/env python3
"""Measure the cold and warm times of XProf and xprof-rs on one profile, and print Markdown tables.

Usage: benchmark.py SESSION_DIR --xprof PATH [--rs PATH] [--cores 0-3] [--trials N] [--warmups N] [--tools A,B] [--commands A,B] [--timeout S] [--memory-limit GB] [--json FILE]

SESSION_DIR holds one `.xplane.pb` file and optional `.hlo_proto.pb` files.

Server, cold: the first request to a new server on a new copy of the files. Warm: the mean of 5 more requests.

CLI, cold: one process on a new copy with an empty TMPDIR (the XProf result cache). Warm: the same command again.

Peak memory: the peak resident memory of the server process over the cold and the warm requests, or of the cold CLI process.
The system gives this value when the process stops (`ru_maxrss` of `wait4`). It includes the pages of the profile that the process maps.

With --timeout, a cold request or a cold command that takes longer stops. Its row shows the limit with a mark (*), and it has no more trials.
With --memory-limit, a process with more resident memory stops. Its row shows the limit with a mark (**), and it has no more trials.

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
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
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
TIMEOUT = None
MEMORY_LIMIT = None
# A measure that stops at a limit gives one of these texts. The mark of the text shows in the table.
MARKS = {"time": "*", "memory": "**"}
# ru_maxrss is in bytes on macOS and in KiB on Linux.
RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def reap(process):
    """Waits for the process to stop, and gives its peak resident memory in bytes."""
    _, status, usage = os.wait4(process.pid, 0)
    process.returncode = os.waitstatus_to_exitcode(status)
    return usage.ru_maxrss * RSS_UNIT


def send(process, number):
    """Sends a signal. Popen.kill and Popen.terminate can collect a stopped process, and then reap does not find it."""
    try:
        os.kill(process.pid, number)
    except ProcessLookupError:
        pass


def resident(pid):
    """The resident memory of the process in bytes, or 0 if the process stopped."""
    try:
        if sys.platform == "darwin":
            return int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout or 0) * 1024
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return 0


class Guard:
    """Stops the process when its resident memory is more than MEMORY_LIMIT."""

    def __init__(self, process):
        self.process, self.exceeded, self.done = process, False, threading.Event()
        if MEMORY_LIMIT:
            threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while not self.done.wait(0.05):
            if resident(self.process.pid) > MEMORY_LIMIT:
                self.exceeded = True
                send(self.process, signal.SIGKILL)
                return

    def stop(self):
        self.done.set()


def fresh(session, scratch):
    shutil.rmtree(scratch, ignore_errors=True)
    target = scratch / "logs/run/plugins/profile/s"
    target.mkdir(parents=True)
    (scratch / "tmp").mkdir()
    for source in session.iterdir():
        if source.name.endswith((".xplane.pb", ".hlo_proto.pb")):
            shutil.copy2(source, target)
    # get_xspace_proto writes this file. A cold call must not find the file of the last trial.
    Path("/tmp", "xspace_" + re.sub(r"[^A-Za-z0-9_-]", "_", str(scratch / "logs/run")) + ".pb").unlink(missing_ok=True)
    env = {key: value for key, value in os.environ.items() if key != "LD_PRELOAD"}
    return scratch / "logs", {**env, "TMPDIR": str(scratch / "tmp")}


def fetch(port, path, timeout=3600):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
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
    guard = Guard(process)
    try:
        for _ in range(600):
            try:
                fetch(port, "/data/plugin/profile/runs")
                break
            except OSError:
                time.sleep(0.1)
        cold = fetch(port, path, TIMEOUT or 3600)
        warm = statistics.fmean(fetch(port, path) for _ in range(WARM_REQUESTS))
    except (OSError, http.client.HTTPException) as error:
        if guard.exceeded:
            return "memory"
        if isinstance(error, TimeoutError):
            return "time"
        raise
    finally:
        send(process, signal.SIGTERM)
        peak = reap(process)
        guard.stop()
    return cold, warm, peak


def cli(command, session, scratch, name):
    logs, env = fresh(session, scratch)
    times, peaks = [], []
    for run in range(2):
        start = time.perf_counter()
        process = subprocess.Popen(command + [name, str(logs / "run")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        guard = Guard(process)
        timer = threading.Timer(TIMEOUT, send, (process, signal.SIGKILL)) if TIMEOUT and run == 0 else None
        if timer:
            timer.start()
        peaks.append(reap(process))
        times.append(time.perf_counter() - start)
        guard.stop()
        if guard.exceeded:
            return "memory"
        if timer:
            timer.cancel()
            if times[0] >= TIMEOUT:
                return "time"
        if process.returncode != 0:
            raise RuntimeError(f"{name}: exit code {process.returncode}")
    return times[0], times[1], peaks[0]


def trials(measures, count, warmups):
    """Runs the measures in turns. Gives, for each measure, the cold times, the warm times and the peak memory of the counted trials.
    A measure that reaches a limit gives the text of the limit, and it does not run again."""
    values = [([], [], []) for _ in measures]
    for trial in range(warmups + count):
        order = list(enumerate(measures))
        for index, measure in order if trial % 2 == 0 else reversed(order):
            if isinstance(values[index], str):
                continue
            measured = measure()
            if isinstance(measured, str):
                values[index] = measured
            elif trial >= warmups:
                for column, value in zip(values[index], measured):
                    column.append(value)
    return values


def cell(values):
    if values == "time":
        return f">{TIMEOUT:g} s*"
    if values == "memory":
        return "not measured**"
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    unit, scale = ("ms", 1000) if mean < 1 else ("s", 1)
    digits = max(0, 2 - math.floor(math.log10(mean * scale)))
    warning = "" if deviation <= UNSTABLE * mean else " (!)"
    return f"{mean * scale:.{digits}f} ± {deviation * scale:.{digits}f} {unit}{warning}"


def memory(values):
    if values == "memory":
        return f">{MEMORY_LIMIT / 1e9:g} GB**"
    if values == "time":
        return "not measured*"
    mean, deviation = statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0
    unit, scale = ("MB", 1e6) if mean < 1e9 else ("GB", 1e9)
    warning = "" if deviation <= UNSTABLE * mean else " (!)"
    return f"{mean / scale:.{0 if unit == 'MB' else 2}f} ± {deviation / scale:.{0 if unit == 'MB' else 2}f} {unit}{warning}"


def ratio(slow, fast):
    if isinstance(fast, str) or slow == "memory":
        return "—"
    value = (TIMEOUT if slow == "time" else statistics.fmean(slow)) / statistics.fmean(fast)
    text = f"{value:.0f}×" if value >= 10 else f"{value:.1f}×"
    return f">{text}" if slow == "time" else text


def table(title, rows):
    print(f"| {title} | XProf, cold | xprof-rs, cold | Speed-up | XProf, warm | xprof-rs, warm | Speed-up | XProf, peak memory | xprof-rs, peak memory |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, (old, new) in rows.items():
        (old_cold, old_warm, old_peak), (new_cold, new_warm, new_peak) = [(values,) * 3 if isinstance(values, str) else values for values in (old, new)]
        warm = lambda values: "not measured" + MARKS[values] if isinstance(values, str) else cell(values)
        measured = not isinstance(old, str) and not isinstance(new, str)
        print(f"| `{name}` | {cell(old_cold)} | {cell(new_cold)} | {ratio(old_cold, new_cold)} "
              f"| {warm(old_warm)} | {warm(new_warm)} | {ratio(old_warm, new_warm) if measured else '—'} | {memory(old_peak)} | {memory(new_peak)} |")
    if any("time" in pair for pair in rows.values()):
        print(f"\n\\* The cold call took longer than {TIMEOUT:g} s. Because it took so long, the script stopped it and did not measure the row further.")
    if any("memory" in pair for pair in rows.values()):
        print(f"\n\\*\\* The process used more than {MEMORY_LIMIT / 1e9:g} GB of memory. Because it used so much, the script stopped it and did not measure the row further.")
    print()


def chosen(names, text):
    """The names in `text`, in the order of `names`. No text gives all names."""
    if text is None:
        return names
    wanted = [name for name in text.split(",") if name]
    unknown = set(wanted) - set(names)
    if unknown:
        raise SystemExit(f"unknown names: {', '.join(sorted(unknown))}")
    return [name for name in names if name in wanted]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--xprof", required=True, help="path of the XProf command")
    parser.add_argument("--rs", default="target/release/xprof-rs")
    parser.add_argument("--cores", help="CPU list for taskset, for example 0-3")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--tools", help="comma-separated server tools, or an empty text for none (default: all)")
    parser.add_argument("--commands", help="comma-separated CLI commands, or an empty text for none (default: all)")
    parser.add_argument("--timeout", type=float, help="seconds after which a cold request or command stops (default: no limit)")
    parser.add_argument("--memory-limit", type=float, help="GB of resident memory after which a server or a command stops (default: no limit)")
    parser.add_argument("--json", type=Path, help="file for the values of all trials")
    arguments = parser.parse_args()
    global TIMEOUT, MEMORY_LIMIT
    TIMEOUT = arguments.timeout
    MEMORY_LIMIT = arguments.memory_limit and arguments.memory_limit * 1e9
    pin = ["taskset", "-c", arguments.cores] if arguments.cores else []
    commands = [pin + [arguments.xprof], pin + [str(Path(arguments.rs).resolve())]]
    host = next(arguments.session.glob("*.xplane.pb")).name.removesuffix(".xplane.pb")
    modules = [path.name.removesuffix(".hlo_proto.pb") for path in arguments.session.glob("*.hlo_proto.pb")]
    tools = [tool for tool in chosen(TOOLS, arguments.tools) if tool != "memory_viewer" or modules]
    servers, clis = {}, {}
    with tempfile.TemporaryDirectory(prefix="xprof-benchmark-") as scratch:
        scratch = Path(scratch) / "work"
        for tool in tools:
            extra = "&resolution=8000" if tool == "trace_viewer@" else f"&module_name={modules[0]}" if tool == "memory_viewer" else ""
            path = f"/data/plugin/profile/data?run=run/s&tag={tool}&host={host}{extra}"
            measures = [lambda command=command, port=8981 + index: server(command, arguments.session, scratch, port, path) for index, command in enumerate(commands)]
            servers[tool] = trials(measures, arguments.trials, arguments.warmups)
            print(f"{tool}: done", file=sys.stderr, flush=True)
        for name in chosen(COMMANDS, arguments.commands):
            measures = [lambda command=command: cli(command, arguments.session, scratch, name) for command in commands]
            clis[name] = trials(measures, arguments.trials, arguments.warmups)
            print(f"{name}: done", file=sys.stderr, flush=True)
    if arguments.json:
        arguments.json.write_text(json.dumps({"servers": servers, "clis": clis}, indent=1))
    if servers:
        table("Tool", servers)
    if clis:
        table("Command", clis)


if __name__ == "__main__":
    main()
