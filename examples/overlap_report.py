import argparse
import json
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description="Report how much of the device communication of a trace overlaps with compute. It uses the xprof-rs agent CLI.")
parser.add_argument("session", help="A run directory or an .xplane.pb file.")
parser.add_argument("--xprof-rs", default="xprof-rs")
parser.add_argument("--device", default="/device:TPU:0$", help="Regex for the plane of the device.")
parser.add_argument("--page", type=int, default=40000)
args = parser.parse_args()

COLLECTIVE = re.compile(r"all-gather|reduce-scatter|all-reduce|all-to-all|collective-permute|collective-broadcast|ragged-all-to-all")
TRANSFER = re.compile(r"copy-start|copy-done|infeed|outfeed|send|recv|host-offload|memcpy", re.IGNORECASE)
CONTAINER = re.compile(r"^%?(while|conditional|call)(\.\d+)?\s")
PS = 1e-9


def run(*flags):
    output = subprocess.run([args.xprof_rs, "list_xplane_events", args.session, *flags], capture_output=True, text=True, check=True).stdout
    result = json.loads(output)
    if result.get("status") == "SAVED_TO_FILE":
        with open(result["file_path"]) as file:
            result = json.load(file)
    return result


def events():
    offset = 0
    while True:
        page = run(f"--plane_regex={args.device}", "--event_regex=.", f"--max_events={args.page}", f"--offset={offset}")
        yield from page["events"]
        if not page["truncated"]:
            return
        offset += page["returned"]


def label(event):
    name, _, rest = event.partition(" = ")
    called = re.search(r"calls=%([\w.\-]+)", rest)
    return name + " " + (called.group(1) if called else "")


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def total(intervals):
    return sum(end - start for start, end in intervals)


def overlap(left, right):
    result, index = 0, 0
    for start, end in left:
        while index < len(right) and right[index][1] <= start:
            index += 1
        cursor = index
        while cursor < len(right) and right[cursor][0] < end:
            result += min(end, right[cursor][1]) - max(start, right[cursor][0])
            cursor += 1
    return result


lines = {}
for event in events():
    lines.setdefault(event["line_id"], []).append((event["offset_ps"], event["offset_ps"] + event["duration_ps"], label(event["event"])))
steps = sorted(lines.get("Steps", []))
if not steps:
    sys.exit("The trace has no Steps line for the device.")

rows = []
for start, end, name in steps:
    inside = lambda line: [(s, e, n) for s, e, n in lines.get(line, []) if s >= start and s < end]
    ops, asynchronous = [op for op in inside("XLA Ops") if not CONTAINER.match(op[2])], inside("Async XLA Ops")
    exposed = merge([(s, e) for s, e, n in ops if COLLECTIVE.search(n)])
    compute = merge([(s, e) for s, e, n in ops if not COLLECTIVE.search(n) and not TRANSFER.search(n)])
    in_flight = merge([(s, e) for s, e, n in asynchronous if COLLECTIVE.search(n)])
    copies = merge([(s, e) for s, e, n in asynchronous if TRANSFER.search(n)])
    busy = merge([(s, e) for s, e, n in ops])
    rows.append(
        {
            "step": name,
            "step_ms": (end - start) * PS,
            "busy_ms": total(busy) * PS,
            "compute_ms": total(compute) * PS,
            "exposed_collective_ms": total(exposed) * PS,
            "async_collective_ms": total(in_flight) * PS,
            "async_collective_hidden_ms": overlap(in_flight, compute) * PS,
            "async_copy_ms": total(copies) * PS,
            "async_copy_hidden_ms": overlap(copies, compute) * PS,
        }
    )

print(f"{'step':>5} {'step ms':>9} {'busy ms':>9} {'compute':>9} {'exposed':>9} {'async coll':>11} {'hidden':>9} {'async copy':>11} {'hidden':>9}")
for row in rows:
    print(f"{row['step']:>5} {row['step_ms']:9.2f} {row['busy_ms']:9.2f} {row['compute_ms']:9.2f} {row['exposed_collective_ms']:9.2f} {row['async_collective_ms']:11.2f} {row['async_collective_hidden_ms']:9.2f} {row['async_copy_ms']:11.2f} {row['async_copy_hidden_ms']:9.2f}")

steady = rows[1:] or rows
mean = lambda key: sum(row[key] for row in steady) / len(steady)
step, busy, exposed, hidden, flight = mean("step_ms"), mean("busy_ms"), mean("exposed_collective_ms"), mean("async_collective_hidden_ms"), mean("async_collective_ms")
communication = exposed + hidden
print()
print(f"steady steps: {len(steady)} of {len(rows)} (the first step is not used)")
print(f"mean step: {step:.2f} ms. Device busy: {100 * busy / step:.1f}%. Device idle: {100 * (step - busy) / step:.1f}%")
print(f"mean exposed collective time: {exposed:.2f} ms ({100 * exposed / step:.1f}% of the step)")
print(f"mean asynchronous collective time: {flight:.2f} ms. Hidden behind compute: {hidden:.2f} ms")
print(f"communication overlap: {100 * hidden / communication if communication else 0:.1f}% (hidden / (hidden + exposed))")
