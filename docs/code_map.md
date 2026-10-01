# Code map — which code does what, and where it came from

## Three codebases, three jobs

| location | whose | job | stages |
|---|---|---|---|
| `learn/` | ours | minimal re-implementations, one idea at a time, written to be read | 1–7 |
| `external/PeriodicWave` | the authors (Geier & Nazaryan), a fork of DeepMind's FermiNet | the reproduction runs that produce the paper's numbers | 8, 10 |
| `bped/` | ours | band-projected exact diagonalization — the authors never published it | 9 |

We do **not** run DeepMind's FermiNet directly. PeriodicWave *is* FermiNet, forked and adapted: upstream has no 2D Ewald Coulomb sum, no moiré potential, no SlaterNet, and its Psiformer lacks the paper's periodic-solid changes. The paper's numbers came from PeriodicWave, so that is what we run. `learn/` exists so that by the time we run it, every file in it is readable.

## PeriodicWave vs upstream FermiNet

Diffed file-by-file against `google-deepmind/ferminet` at `c4312c3` (2026-05-21).
Caveat: that is *today's* upstream, not the commit PeriodicWave forked from, so part of every "modified" count is FermiNet's own evolution since the fork.

| PeriodicWave file | vs FermiNet | what it does | our stage |
|---|---|---|---|
| `network_blocks.py` | identical | linear layers, `slogdet`, `logdet_matmul` (log-sum-exp over determinants) | 3 |
| `networks.py` | modified, heavily trimmed | input features, orbital projection (`make_orbitals`), network assembly | 3 |
| `SlaterNet.py` | **new** | per-electron residual MLP → Hartree-Fock reference | 3 |
| `pbc/feature_layer.py` | modified (8 lines added) | periodic sin/cos features, Eq. (7) | 1 |
| `pbc/lattices.py` | **new** | C₆-symmetric supercells (3, 12, 21, 27 sites) | 3, 7 |
| `CustomPsiformer.py` | **new**, replaces FermiNet's `psiformer.py` | the self-attention network | 5 |
| `mcmc.py` | modified | Metropolis–Hastings walkers | 4 |
| `hamiltonian.py` | modified | kinetic-energy machinery | 4 |
| `pbc/hamiltonian.py` | modified (183 lines added) | 2D Ewald Coulomb, moiré potential, local energy via folx (folx path returns Re E_loc only: reproduction_map §12 item 7) | 4, 6, 7 |
| `loss.py` | modified | VMC energy and gradient estimator, local-energy clipping (complex branch: exact with clipping on, counts the phase term twice with clipping off; verified in learn/stage06_kfac) | 4, 6 |
| `curvature_tags_and_blocks.py` | modified (2 lines) | registers layers with KFAC | 6 |
| `train.py` | modified | the optimisation loop | 6, 8 |
| `base_config.py` | modified | default hyperparameters | 8 |
| `configs/run_2deg*.py` | **new** | entry points for the 2DEG and the moiré system | 8 |
| `jastrows.py` | modified | optional 2D cusp Jastrow, paper App. B, off by default | — |
| `envelopes.py` | modified | null envelope for periodic systems, paper Sec. VI | — |
| `checkpoint.py`, `main.py`, `constants.py`, `utils/*` | minor or identical | I/O and logging | — |

**Dropped from FermiNet** (35 files): molecular configs, `pretrain.py`, pseudopotentials, Gaussian-basis and SCF utilities, excited states, `density.py`/`observables.py`, and FermiNet's own `psiformer.py` and 3D `pbc/envelopes.py`. Dropping `pretrain.py` is consistent with the paper's statement (Sec. VI) that every run starts from a randomly initialised network, with no Hartree-Fock pretraining.

## Where `learn/` differs from PeriodicWave, deliberately

1. **Spin input.** The authors append each electron's spin as a fifth input feature. For spin-polarised electrons that is a constant, so it acts as a bias on the embedding and adds d_L parameters per network beyond the paper's Appendix C formula. `learn/` follows the paper: four inputs, no bias.
2. **Residual block.** Theirs allows several perceptrons per residual block, optional layer norm, and GELU/ELU. Set to one perceptron, no layer norm, `TANH`, it is exactly Eq. (9), which is what `learn/` implements.

Identical in substance: the orbital projection (one linear layer, real and imaginary parts interleaved, no bias) and the log-sum-exp sum over determinants.
