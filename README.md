# moire-nnvmc

Reproduction of Geier, Nazaryan, Zaklama & Fu, *Self-attention neural network for
solving correlated electron problems in solids*, Phys. Rev. B 112, 045119 (2025).

This is a **learning** repository. The goal is not only the reproduced numbers but
understanding every piece that produces them. See `docs/learning_path.md` for the
eleven stages and `docs/reproduction_map.md` for the paper-and-code audit.

## Layout

    docs/       the plan and the paper/code audit
    env/        environment definition and setup script
    learn/      one directory per learning stage: code written from scratch
    external/   upstream PeriodicWave clone -- read, never edit
    bped/       band-projected exact diagonalization, written from scratch (Stage 9)
    runs/       training outputs and checkpoints (gitignored)
    figures/    reproduced figures (committed)
    notes/      notes and checkpoint answers, one file per stage

## Convention

One commit per completed stage, tagged `stage-01`, `stage-02`, ... so any stage can
be recovered. Run outputs are never committed; the config that produced them is.

## Working on this repo

    cd ~/moire-nnvmc
    source .venv/bin/activate          # Python 3.13, jax 0.8.0, both GPUs visible

`uv` lives in `~/.local/bin`; add it to PATH with `source ~/.local/bin/env`.
Rebuild the environment from scratch at any time with `bash env/setup_env.sh`
(idempotent). `env/requirements-gpu.txt` is the pinned input;
`env/freeze-actual.txt` is what actually got installed.

Long runs must outlive the SSH session -- there is no scheduler on this host:

    tmux new -s run8
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python external/PeriodicWave/periodicwave/configs/run_2deg_triangular_potential.py ...
    # Ctrl-b d to detach, `tmux attach -t run8` to return
