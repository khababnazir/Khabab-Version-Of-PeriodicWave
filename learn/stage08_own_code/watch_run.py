"""
Stage 8 -- a watchdog for one training run, on the server itself (2026-11-09).

Checks from the app stop when nobody is watching (the app's link can hang), so this runs on
the server, started with setsid nohup, and appends one line every --every seconds to
<system>/training/logs/watch.log.  It only reads: it never stops, starts or restarts anything.

Each line: the time (UTC); whether train_moire.py or evaluate_weights.py is running for this
run; the last iteration, the iterations since the last line, seconds per iteration and the
expected end of training; the mean energy per electron of the last 500 iterations (meV),
acceptance and variance; non-finite numbers in the new rows; S(kappa) and its static part
|<rho_kappa>|^2 / N at the newest checkpoint's walkers; the GPU's temperature and use.
ALERT lines: non-finite numbers; no new iteration while training runs; no process running
before the target is reached; a queue log line "ended with exit code N", N not 0; GPU at
88 C or more.  It stops after the queue log reports the run's evaluation ended, or when
nothing runs and training has stopped.

Run, from the repository:
    setsid nohup .venv/bin/python learn/stage08_own_code/watch_run.py RUN --every 1800 \\
        >> <system>/training/logs/watch.log 2>&1 < /dev/null &
    .venv/bin/python learn/stage08_own_code/watch_run.py RUN --once     # one line, now
RUN is the training folder, e.g. runs/stage08/18_electrons_27_sites_paper_network_real_gradient/training/attention_18e_eps5.
"""
import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

ENERGY_UNIT_MEV = 3.375559                      # code unit hbar^2 / (m* a_M^2)
KAPPAS = 4 * np.pi / 3 * np.array([[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)])
WINDOW = 500


def utc(t=None):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t))


def s_kappa(path):
    """S(kappa) = <|rho_kappa|^2> / N and its static part |<rho_kappa>|^2 / N, over the 6 kappas."""
    with np.load(path) as saved:
        R = saved["walkers"].astype(np.float64)                     # (walkers, N, 2), a_M = 1
    rho = np.exp(-1j * np.einsum("wnd,qd->wnq", R, KAPPAS)).sum(axis=1)   # (walkers, 6)
    n = R.shape[1]
    return float(np.mean(np.abs(rho) ** 2) / n), float(np.mean(np.abs(rho.mean(axis=0)) ** 2) / n)


def processes(run):
    """'train', 'evaluate' or both, for the python processes whose command line names this run."""
    found = set()
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            words = cmdline.read_bytes().split(b"\0")
        except OSError:
            continue
        text = " ".join(w.decode(errors="replace") for w in words)
        if f"{run.parent.parent.name}/training/{run.name}" not in text:
            continue
        if "train_moire.py" in text:
            found.add("train")
        if "evaluate_weights.py" in text:
            found.add("evaluate")
    return found


def rows_after(path, offset):
    """New complete rows of train_stats.csv after byte offset (header skipped), and the new offset."""
    with open(path, "rb") as handle:
        header_line = handle.readline()
        start = max(offset, len(header_line))
        handle.seek(start)
        data = handle.read()
    complete = data[: data.rfind(b"\n") + 1]                       # a row being written is left for later
    header = header_line.decode().strip().split(",")
    rows = list(csv.DictReader(complete.decode().splitlines(), fieldnames=header))
    return rows, start + len(complete)


def last_rows(path, n):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-n:]


def gpu_state(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--format=csv,noheader,nounits",
                              "--query-gpu=temperature.gpu,utilization.gpu"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        temperature, use = (int(x) for x in out.split(","))
        return temperature, use
    except Exception:                                                  # noqa: BLE001 -- never die on nvidia-smi
        return None, None


def check(run, state):
    """One line (and ALERT lines) about the run; updates state.  Returns (lines, finished)."""
    lines, alerts = [], []
    settings = json.loads((run / "settings.json").read_text())
    target, n = settings["iterations"], settings["electrons"]
    starts = (run / "starts.log").read_text().strip().splitlines()
    gpu = int(starts[-1].split("GPU")[1].split()[0])
    queue_log = run.parent / "logs" / f"gpu{gpu}.log"
    queue = queue_log.read_text(errors="replace") if queue_log.exists() else ""
    running = processes(run)

    new, state["offset"] = rows_after(run / "train_stats.csv", state.get("offset", 0))
    columns = ("energy", "variance", "pmove", "step_width", "learning_rate", "fisher_step_sq")
    bad = [r["step"] for r in new if any(not math.isfinite(float(r[c])) for c in columns if r.get(c) not in (None, ""))]
    recent = last_rows(run / "train_stats.csv", WINDOW)
    step = int(recent[-1]["step"]) if recent else -1
    now = time.time()
    previous_step, previous_time = state.get("step"), state.get("time")
    rate = ""
    if previous_step is not None and step > previous_step:
        per = (now - previous_time) / (step - previous_step)
        rate = f" (+{step - previous_step} since the last line, {per:.2f} s/it, training ends about {utc(now + per * (target - 1 - step))})"
    energy = np.mean([float(r["energy"]) for r in recent]) * ENERGY_UNIT_MEV / n
    pmove = np.mean([float(r["pmove"]) for r in recent])
    variance = np.mean([float(r["variance"]) for r in recent])

    checkpoints = sorted(run.glob("checkpoint_*.npz"))
    structure = ""
    if checkpoints and checkpoints[-1].name != state.get("checkpoint"):
        S, static = s_kappa(checkpoints[-1])
        state["checkpoint"], state["structure"] = checkpoints[-1].name, f"S(kappa) {S:.3f}, static {static:.3f}"
    if checkpoints:
        structure = f" | {state['checkpoint']}: {state['structure']}"
    temperature, use = gpu_state(gpu)

    lines.append(f"{utc(now)} | running: {', '.join(sorted(running)) or 'nothing'} | iteration {step} of {target}{rate}"
                 f" | last {len(recent)}: E/N {energy:.3f} meV, pmove {pmove:.3f}, variance {variance:.1f}"
                 f" | non-finite rows {len(bad)}{structure} | GPU {gpu}: {temperature} C, {use} %")
    if bad:
        alerts.append(f"non-finite numbers in iterations {bad[:5]}")
    if "train" in running and previous_step is not None and step <= previous_step:
        alerts.append(f"no new iteration since the last line (still {step})")
    finished_training = (run / f"checkpoint_{target:06d}.npz").exists()
    if not running and not finished_training:
        alerts.append(f"nothing runs for this run and it stopped at iteration {step} of {target}")
    for line in queue.splitlines():
        if run.name in line and "ended with exit code" in line and not line.rstrip().endswith("exit code 0"):
            if line not in state.setdefault("reported", []):
                state["reported"].append(line)
                alerts.append(f"queue log: {line.strip()}")
    if temperature is not None and temperature >= 88:
        alerts.append(f"GPU {gpu} at {temperature} C")
    lines += [f"{utc(now)} ALERT {a}" for a in alerts]
    state["step"], state["time"] = step, now
    # Stop after the evaluation that follows the latest start has ended, or after two checks in a
    # row with nothing running (one could fall between the end of training and the evaluation).
    evaluation_ended = f"evaluation of {run.name} ended" in queue.split(f"start {run.name}")[-1]
    state["idle"] = 0 if running else state.get("idle", 0) + 1
    return lines, evaluation_ended or state["idle"] >= 2


def main():
    parser = argparse.ArgumentParser(description="watchdog for one training run (reads only)")
    parser.add_argument("run", type=Path)
    parser.add_argument("--every", type=int, default=1800, help="seconds between lines")
    parser.add_argument("--once", action="store_true", help="one line, then exit")
    args = parser.parse_args()
    run, state = args.run.resolve(), {}
    print(f"{utc()} | watch_run.py started for {run} (every {args.every} s; reads only)", flush=True)
    while True:
        try:
            lines, finished = check(run, state)
        except Exception as error:                                     # noqa: BLE001 -- report, keep watching
            lines, finished = [f"{utc()} ALERT the check itself failed: {error!r}"], False
        print("\n".join(lines), flush=True)
        if args.once or finished:
            print(f"{utc()} | watch_run.py stops" + ("" if args.once else ": the run's evaluation ended or the run stopped"),
                  flush=True)
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
