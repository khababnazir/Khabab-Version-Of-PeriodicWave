"""
Stage 8 -- train the authors' own code (PeriodicWave) on the paper's moire system.

Run (one GPU per run):
    CUDA_VISIBLE_DEVICES=0 python run_moire.py --network CustomPsiformer --eps 5 \
        --out ../../runs/stage08/psiformer_eps5

Options: --network CustomPsiformer | SlaterNet,  --eps 5 | 10,
         --iterations N   (default: the authors' 200000),
         --laplacian folx | default   (default: folx, as the authors run it;
                          'default' keeps Im E_loc, see test_stage08.py).

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
"""
import argparse
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

AUTHORS_CONFIG = (Path(__file__).resolve().parents[2] / "external" / "PeriodicWave"
                  / "periodicwave" / "configs" / "run_2deg_triangular_potential.py")


def authors_configuration(network, dielectric_constant):
    """Run the authors' config script up to the point where it would start training."""
    from periodicwave import train

    captured = {}
    real_train = train.train
    train.train = lambda cfg, *args, **kwargs: captured.setdefault("cfg", cfg)

    # Their command line: spins, sites, m*/m_e, 1/eps, a_M (nm), V (meV), phi (degrees), network.
    argv = [str(AUTHORS_CONFIG), "6_0", "9", "0.35", repr(1.0 / dielectric_constant),
            "8.031", "15", "45", network]
    saved_argv, saved_cwd = sys.argv, os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)          # their script writes a config.json and device_info.log next to it
        sys.argv = argv
        try:
            runpy.run_path(str(AUTHORS_CONFIG), run_name="__main__")
        finally:
            sys.argv = saved_argv
            os.chdir(saved_cwd)
            train.train = real_train
    return captured["cfg"], argv[1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=["CustomPsiformer", "SlaterNet"], required=True)
    parser.add_argument("--eps", type=float, choices=[5.0, 10.0], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--laplacian", choices=["folx", "default"], default="folx")
    args = parser.parse_args()

    cfg, authors_argv = authors_configuration(args.network, args.eps)

    out = args.out.resolve()
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
        {"authors_command_line": authors_argv, "overrides": overrides}, indent=2))
    from periodicwave.utils import custom_logging
    custom_logging.save_config_dict_as_json(cfg, str(out / "config.json"))

    from periodicwave import train
    train.train(cfg)


if __name__ == "__main__":
    main()
