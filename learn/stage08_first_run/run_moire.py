"""
Stage 8 -- train the authors' own code (PeriodicWave) on the paper's moire system.

Run (one GPU per run):
    CUDA_VISIBLE_DEVICES=0 python run_moire.py --network CustomPsiformer --eps 5 \
        --out ../../runs/stage08/psiformer_eps5

Options: --network CustomPsiformer | SlaterNet,  --eps 5 | 10,
         --iterations N   (default: the authors' 200000),
         --laplacian folx | default   (default: folx, as the authors run it;
                          'default' returns the complex E_loc, see test_stage08.py.
                          Use a separate folder for it.)

Why not simply run  run_2deg_triangular_potential.py ?
That script builds the configuration and immediately starts training.  It
writes to a folder named after the parameters, relative to wherever it was
started, and seeds the random numbers from the clock.  We want:

  1. exactly the authors' configuration.  So we execute THEIR script, but
     intercept its call to train.train and take the configuration it was
     about to use;
  2. a run folder we choose.  FermiNet resumes from the last checkpoint it
     finds in that folder, so after a crash the same command continues the
     run (checkpoints are written every 30 minutes);
  3. a fixed seed (cfg.debug.deterministic: seed 42), so a run can be repeated.

Those three, plus the two options above when used, are the ONLY changes.
They are listed in overrides.json, and the full configuration is written to
config.json in the run folder.

Safety rules, all checked before JAX starts, so a refused start never
touches a GPU that a real run is using:
  - only one process may train into a folder.  The process takes an
    exclusive lock on <folder>/.training.lock and holds it until it ends;
    the operating system releases it when the process exits for any reason,
    a crash included, so no stale lock is ever left behind.  Runs started
    before locks existed hold none, so a scan of the running processes
    catches those as well.  Either way the exit code is 75 (ALREADY_RUNNING),
    which launch_fig5.sh treats as "this GPU is busy", not as a failure;
  - a resumed run must be the same experiment: network, eps and laplacian
    must match what the folder was started with.  Only the iteration count
    may change, and each start is appended to starts.log;
  - a run whose newest COMPLETE checkpoint is its final step exits at once.
    A damaged checkpoint is moved into unreadable_checkpoints/ and the run
    resumes from the one before it.
"""
import argparse
import fcntl
import json
import os
import runpy
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

AUTHORS_CONFIG = (Path(__file__).resolve().parents[2] / "external" / "PeriodicWave"
                  / "periodicwave" / "configs" / "run_2deg_triangular_potential.py")


def authors_command_line(network, dielectric_constant):
    """Their arguments: spins, sites, m*/m_e, 1/eps, a_M (nm), V (meV), phi (degrees), network."""
    return ["6_0", "9", "0.35", repr(1.0 / dielectric_constant), "8.031", "15", "45", network]


def authors_configuration(command_line):
    """Run the authors' config script up to the point where it would start training."""
    from periodicwave import train

    captured = {}
    real_train = train.train
    train.train = lambda cfg, *args, **kwargs: captured.setdefault("cfg", cfg)

    saved_argv, saved_cwd = sys.argv, os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)          # their script writes a config.json and device_info.log next to it
        sys.argv = [str(AUTHORS_CONFIG)] + command_line
        try:
            runpy.run_path(str(AUTHORS_CONFIG), run_name="__main__")
        finally:
            sys.argv = saved_argv
            os.chdir(saved_cwd)
            train.train = real_train
    return captured["cfg"]


def this_process_and_its_parents():
    """Process ids from this one up to init, read from /proc/<pid>/stat (Linux)."""
    chain = set()
    pid = os.getpid()
    while pid > 1:
        chain.add(pid)
        stat = Path(f"/proc/{pid}/stat").read_text()
        pid = int(stat.rsplit(")", 1)[1].split()[1])     # the field after "(name) state" is the parent
    return chain


ALREADY_RUNNING = 75        # exit code: another process is training into this folder


def refuse(message, code):
    print(message, file=sys.stderr)
    sys.exit(code)


def hold_folder_lock(folder):
    """Take the exclusive lock on <folder>/.training.lock, or stop.

    The returned file must stay open for as long as the training runs:
    closing it, or the process ending, releases the lock.
    """
    handle = open(folder / ".training.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        refuse(f"refusing to start: another process holds the lock on {folder}", ALREADY_RUNNING)
    return handle


def out_argument(argv):
    """The value of --out in a command line, written as '--out PATH' or '--out=PATH'."""
    for i, argument in enumerate(argv):
        if argument == "--out" and i + 1 < len(argv):
            return argv[i + 1]
        if argument.startswith("--out="):
            return argument.split("=", 1)[1]
    return None


def older_runs_training_into(folder):
    """Process ids of run_moire.py processes that write into this folder without a lock.

    Only needed for runs started before the lock existed.  Reads each
    process's command line and working directory from /proc (Linux).  This
    process and its parents are skipped: a wrapper such as
    `timeout 600 python run_moire.py --out ...` carries the same command line.
    A relative --out is resolved against the other process's working
    directory, which is only reliable once that process is training; a
    process that holds the lock is caught by hold_folder_lock instead.
    """
    ours = this_process_and_its_parents()
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) in ours:
            continue
        try:
            argv = (process / "cmdline").read_bytes().decode().split("\0")
            working_directory = os.readlink(process / "cwd")
        except OSError:                      # the process ended, or is not ours
            continue
        value = out_argument(argv)
        if value is None or not any(a.endswith("run_moire.py") for a in argv):
            continue
        if (Path(working_directory) / value).resolve() == folder:
            found.append(int(process.name))
    return found


def checkpoint_problem(path, step):
    """None if the checkpoint is complete, otherwise what is wrong with it.

    Complete means: every member of the .npz (a zip archive) passes its CRC
    check, and the step saved inside it ('t') equals the step in its name.
    Nothing is unpickled, so this needs no JAX.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            damaged = archive.testzip()
            if damaged is not None:
                return f"damaged member {damaged}"
            if "t.npy" not in archive.namelist():
                return "no saved step in it"
        with np.load(path, allow_pickle=False) as contents:
            saved_step = int(contents["t"])
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as error:
        return f"{type(error).__name__}: {error}"
    if saved_step != step:
        return f"it holds step {saved_step}, not {step}"
    return None


def newest_valid_checkpoint(folder):
    """The step of the newest complete checkpoint, or None.  Damaged ones are moved aside.

    The authors' restore (checkpoint.find_last_checkpoint) already skips a
    file numpy cannot open, such as an empty or truncated one, and falls back
    to the one before.  But np.load reads only the archive's directory, so it
    would accept a file whose directory is intact and whose data is damaged,
    and the restore would then fail.  A file that fails checkpoint_problem is
    therefore moved into unreadable_checkpoints/ -- never deleted -- so that
    neither this check nor the authors' restore can pick it.  Called only
    while this process holds the folder lock, so no run can be writing to it.
    """
    for path in sorted(folder.glob("qmcjax_ckpt_*.npz"), reverse=True):
        step = int(path.stem.rsplit("_", 1)[1])
        problem = checkpoint_problem(path, step)
        if problem is None:
            return step
        quarantine = folder / "unreadable_checkpoints"
        quarantine.mkdir(exist_ok=True)
        path.rename(quarantine / path.name)
        print(f"warning: {path.name} is unusable ({problem}); moved to "
              f"unreadable_checkpoints/, resuming from the checkpoint before it", file=sys.stderr)
    return None


def already_finished(folder, iterations):
    """True if the newest complete checkpoint is the run's final step.

    Training writes a checkpoint at step iterations - 1.  If --iterations was
    not given, the target is the one the folder was last started with
    (config.json).  Reads files only, so it needs no JAX.
    """
    if iterations is None:
        config_file = folder / "config.json"
        if not config_file.exists():
            return False
        iterations = json.loads(config_file.read_text())["optim"]["iterations"]
    step = newest_valid_checkpoint(folder)
    return step is not None and step >= iterations - 1


def check_same_experiment(folder, command_line, laplacian):
    """A folder that already holds a run may only be resumed with the same settings."""
    saved_file = folder / "overrides.json"
    if not saved_file.exists():
        return
    saved = json.loads(saved_file.read_text())
    saved_laplacian = saved["overrides"].get(
        "system.make_local_energy_kwargs.kinetic_kwargs.laplacian_method", "folx")
    if saved["authors_command_line"] != command_line or saved_laplacian != laplacian:
        refuse(f"refusing to resume {folder}: it was started with "
               f"{saved['authors_command_line']} and laplacian {saved_laplacian}, "
               f"this command asks for {command_line} and laplacian {laplacian}. "
               f"Use a new --out folder for a different experiment.", 1)


def main():
    # allow_abbrev=False: only --out and --out= are accepted, never a
    # shortened --ou, so older_runs_training_into can find the folder.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--network", choices=["CustomPsiformer", "SlaterNet"], required=True)
    parser.add_argument("--eps", type=float, choices=[5.0, 10.0], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--laplacian", choices=["folx", "default"], default="folx")
    args = parser.parse_args()

    # Safety checks first, before anything imports JAX or touches a GPU.
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = hold_folder_lock(out)                       # noqa: F841 -- held until the process ends
    older = older_runs_training_into(out)
    if older:
        refuse(f"refusing to start: process {older} is already training into {out}", ALREADY_RUNNING)
    command_line = authors_command_line(args.network, args.eps)
    check_same_experiment(out, command_line, args.laplacian)
    if already_finished(out, args.iterations):
        print(f"{out} has already finished: its last checkpoint is the final step")
        return

    cfg = authors_configuration(command_line)

    overrides = {"log.save_path": str(out), "debug.deterministic": True}
    cfg.log.save_path = str(out)
    cfg.debug.deterministic = True
    if args.iterations is not None:
        overrides["optim.iterations"] = args.iterations
        cfg.optim.iterations = args.iterations
    if args.laplacian != "folx":
        overrides["system.make_local_energy_kwargs.kinetic_kwargs.laplacian_method"] = args.laplacian
        cfg.system.make_local_energy_kwargs["kinetic_kwargs"]["laplacian_method"] = args.laplacian

    (out / "overrides.json").write_text(json.dumps(
        {"authors_command_line": command_line, "overrides": overrides}, indent=2))
    from periodicwave.utils import custom_logging
    custom_logging.save_config_dict_as_json(cfg, str(out / "config.json"))
    with open(out / "starts.log", "a") as log:
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}  start, iterations "
                  f"{cfg.optim.iterations}, laplacian {args.laplacian}\n")

    from periodicwave import train
    train.train(cfg)


if __name__ == "__main__":
    main()
