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

Two safety rules, checked before JAX starts (so a refused start never
touches a GPU that a real run is using):
  - only one process may train into a folder: if another run_moire.py is
    already writing there, this one stops;
  - a resumed run must be the same experiment: network, eps and laplacian
    must match what the folder was started with.  Only the iteration count
    may change, and each start is appended to starts.log.
"""
import argparse
import json
import os
import runpy
import sys
import tempfile
import time
from pathlib import Path

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


def other_processes_training_into(folder):
    """Process ids of other run_moire.py processes whose --out is this folder.

    Reads each process's command line and working directory from /proc
    (Linux), so it also finds runs started by an older version of this
    script, or by hand.  This process and its parents are skipped: a wrapper
    such as `timeout 600 python run_moire.py --out ...` carries the same
    command line, but it is this run, not another one.
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
        if not any(a.endswith("run_moire.py") for a in argv) or "--out" not in argv:
            continue
        their_folder = (Path(working_directory) / argv[argv.index("--out") + 1]).resolve()
        if their_folder == folder:
            found.append(int(process.name))
    return found


def check_same_experiment(folder, command_line, laplacian):
    """A folder that already holds a run may only be resumed with the same settings."""
    saved_file = folder / "overrides.json"
    if not saved_file.exists():
        return
    saved = json.loads(saved_file.read_text())
    saved_laplacian = saved["overrides"].get(
        "system.make_local_energy_kwargs.kinetic_kwargs.laplacian_method", "folx")
    if saved["authors_command_line"] != command_line or saved_laplacian != laplacian:
        sys.exit(f"refusing to resume {folder}: it was started with "
                 f"{saved['authors_command_line']} and laplacian {saved_laplacian}, "
                 f"this command asks for {command_line} and laplacian {laplacian}. "
                 f"Use a new --out folder for a different experiment.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=["CustomPsiformer", "SlaterNet"], required=True)
    parser.add_argument("--eps", type=float, choices=[5.0, 10.0], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--laplacian", choices=["folx", "default"], default="folx")
    args = parser.parse_args()

    # Safety checks first, before anything imports JAX or touches a GPU.
    out = args.out.resolve()
    command_line = authors_command_line(args.network, args.eps)
    running = other_processes_training_into(out)
    if running:
        sys.exit(f"refusing to start: process {running} is already training into {out}")
    check_same_experiment(out, command_line, args.laplacian)

    cfg = authors_configuration(command_line)

    out.mkdir(parents=True, exist_ok=True)
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
