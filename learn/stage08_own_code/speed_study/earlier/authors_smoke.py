"""Smoke test of the authors' code at 18 electrons on 27 sites: speed and memory only.  Writes into /tmp."""
import fcntl, os, sys
from pathlib import Path
sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/archive/stage08_authors_code")
import run_moire
batch, iterations, out, network = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
locks = []
for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(","):      # one job per GPU, as for every run
    handle = open(f"/home/khabab/moire-nnvmc/runs/gpu_locks/gpu{gpu}.lock", "w")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(handle)
cfg = run_moire.authors_configuration(["18_0", "27", "0.35", repr(1.0 / 5.0), "8.031", "15", "45", network])
cfg.batch_size = batch
cfg.optim.iterations = iterations
cfg.mcmc.burn_in = 10              # shorter burn-in; does not change the time per iteration
cfg.log.save_path = str(out)
cfg.debug.deterministic = True
out.mkdir(parents=True, exist_ok=True)
import jax
print("devices:", jax.devices(), "batch:", cfg.batch_size, flush=True)
from periodicwave import train
train.train(cfg)
