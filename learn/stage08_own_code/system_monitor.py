"""Record CPU and memory use, of the machine and of one job, every few seconds (Stage 8).

Usage:  python system_monitor.py --parent PID --out FILE.csv [--every 10]

The launchers start it next to nvidia-smi, so every run has a GPU record
(gpuN_monitor_<date>_<time>.csv) and this one (gpuN_system_<date>_<time>.csv).
One row per sample:

    timestamp            server time
    cpu_percent          all cores together, 0-100
    load_1min            Linux load average over the last minute
    ram_total_MiB        installed memory
    ram_used_MiB         total - available
    ram_available_MiB    what new programs could still get
    swap_used_MiB
    job_pids             the processes started by PARENT (e.g. train_moire.py), nvidia-smi not counted
    job_cpu_percent      their CPU use; 100 = one core fully busy
    job_rss_MiB          their memory (resident set size)

It stops by itself when PARENT has ended.  Linux only (it reads /proc);
nothing to install.
"""
import argparse
import csv
import os
import time

TICKS = os.sysconf("SC_CLK_TCK")        # /proc counts CPU time in these
PAGE_MiB = os.sysconf("SC_PAGE_SIZE") / 2**20
COLUMNS = ["timestamp", "cpu_percent", "load_1min", "ram_total_MiB", "ram_used_MiB",
           "ram_available_MiB", "swap_used_MiB", "job_pids", "job_cpu_percent", "job_rss_MiB"]


def cpu_times():
    """(busy, total) CPU time of the whole machine since boot, from /proc/stat."""
    with open("/proc/stat") as handle:
        fields = [int(x) for x in handle.readline().split()[1:9]]   # user nice system idle iowait irq softirq steal
    idle = fields[3] + fields[4]
    return sum(fields) - idle, sum(fields)


def memory_MiB():
    info = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            name, value = line.split(":")
            info[name] = int(value.split()[0]) / 1024              # kB -> MiB
    return info


def processes():
    """{pid: (parent pid, CPU ticks used so far, resident MiB, name)} for every process."""
    table = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as handle:
                text = handle.read()
        except OSError:                                             # it ended meanwhile
            continue
        name = text[text.index("(") + 1:text.rindex(")")]
        rest = text[text.rindex(")") + 2:].split()                  # rest[0] is field 3 of proc(5)
        table[int(entry)] = (int(rest[1]), int(rest[11]) + int(rest[12]), int(rest[21]) * PAGE_MiB, name)
    return table


def descendants(table, parent):
    """Every process below parent: its children, their children, ..."""
    found, todo = set(), [parent]
    while todo:
        pid = todo.pop()
        for child, info in table.items():
            if info[0] == pid and child not in found:
                found.add(child)
                todo.append(child)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=float, default=10.0)
    args = parser.parse_args()
    me = os.getpid()
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if new_file:
            writer.writerow(COLUMNS)
        busy, total = cpu_times()
        then = time.time()
        ticks_then = {pid: info[1] for pid, info in processes().items()}
        while os.path.exists(f"/proc/{args.parent}"):
            time.sleep(args.every)
            table, now = processes(), time.time()
            busy_now, total_now = cpu_times()
            job = sorted(pid for pid in descendants(table, args.parent)
                         if pid != me and table[pid][3] != "nvidia-smi")
            # A process that started since the last sample counts from the next one.
            job_ticks = sum(table[pid][1] - ticks_then.get(pid, table[pid][1]) for pid in job)
            memory = memory_MiB()
            writer.writerow([
                time.strftime("%Y/%m/%d %H:%M:%S"),
                round(100.0 * (busy_now - busy) / max(total_now - total, 1), 1),
                open("/proc/loadavg").read().split()[0],
                round(memory["MemTotal"]),
                round(memory["MemTotal"] - memory["MemAvailable"]),
                round(memory["MemAvailable"]),
                round(memory["SwapTotal"] - memory["SwapFree"]),
                " ".join(str(pid) for pid in job),
                round(100.0 * job_ticks / TICKS / (now - then), 1),
                round(sum(table[pid][2] for pid in job)),
            ])
            handle.flush()
            busy, total, then = busy_now, total_now, now
            ticks_then = {pid: info[1] for pid, info in table.items()}


if __name__ == "__main__":
    main()
