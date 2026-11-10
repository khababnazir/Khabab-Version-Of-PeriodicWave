# runs/stage08/18_electrons_27_sites_paper_network_real_gradient — the paper's printed attention network at ε = 5

**What differs.** The attention network as the paper prints it (`train_moire.py --network-form
paper`; Stage 5's network, `learn/stage05_attention/psiformer.py`, last section): four inputs,
tanh residual blocks (Eq. 20), no layer norm, one key, query and value matrix per head, and
softmax_j(k_i . q_j) followed by 1 / sqrt(d_values) (Eq. 18). Every earlier Stage 8 run used
the authors' released network (GELU, layer norm, softmax(q . k / sqrt d), a constant spin input).
Initial weights: Stage 5's (0.1 x normal; the paper gives none), where the released network
starts from normal / sqrt(inputs). So against
`../18_electrons_27_sites_real_gradient/training/attention_18e_eps5` this compares two
implementations, their initial weights included; it cannot isolate tanh against GELU.

**Why.** The paper's Fig. 6(b) shows a honeycomb crystal at eps = 5; our released network
shows none (S(kappa) 0.726), and Appendix C says the Table II settings made the main-text
figures. The network form is the one documented difference from the paper we had not tested.
This run tests it; it is not a known cure.

**What stays the same**: eps = 5, seed 0, the authors' gradient (`--gradient real`), Table II
(4096 walkers, 150,000 iterations, learning rate 10 / (1 + t/100000), KFAC norm constraint
1e-3, damping 1e-3, clipping 5), 3 layers, 6 heads, 1 perceptron, 4 determinants (95,680
parameters, as Appendix C counts them), folx + slogdet. `settings.json` records it
(`"network_form": "paper"`).

**Checked before the run** (commit ff8ff05): it is Stage 5's network (the same wave function
and derivatives at 6 and 18 electrons); kfac-jax sees only dense layers (Stage 5's layout as
written gave kfac-jax's diagonal fallback for every parameter); fast and reference local
energies agree at 27 sites; a restart continues exactly; 7 mutants caught. Smoke run, 300
iterations: finite, 1.08 s per iteration.

**Two steps, one run** (GPU 0 only; GPU 1 stays free):
1. `bash learn/stage08_own_code/launch_table1.sh paper5 5000` (commit e5ecc44), started
   2026-11-08 06:44:30 server time (15:44 Korean time): 5,000 iterations, about 1.5 hours, no
   frozen evaluation. Then a look: finite numbers, Metropolis acceptance, the energy against the
   released run at the same iterations, S(kappa) at the checkpoints (no crystal is expected so
   early).
2. If it is healthy: `bash learn/stage08_own_code/launch_table1.sh paper5 150000` continues the
   same run from its checkpoint (about 44 hours more), then the frozen-weight evaluation (about
   1.3 hours). The first 5,000 iterations count.

Folders: `training/` (the run and `logs/`); later `evaluation/`, `results/`, `figures/`.

## After 5,000 iterations (reviewed 2026-11-09)

Step 1 ended on 2026-11-08 at 08:17:51 server time (exit code 0; 5,000 iterations in 5,487 s,
1.09 s per iteration). All 5,000 rows are there, in order, with no non-finite number. Against
the released run (`attention_18e_eps5` of the real-gradient folder) at the same iterations,
means over 500 iterations, energies in meV per electron:

| iterations | paper network E/N | released E/N | paper variance | released variance | paper acceptance | released acceptance |
|---|---|---|---|---|---|---|
| 0-499 | -36.61 | -49.30 | 2812 | 1416 | 0.21 | 0.09 |
| 1000-1499 | -49.63 | -57.59 | 1504 | 171 | 0.49 | 0.43 |
| 2500-2999 | -56.63 | -58.35 | 376 | 96 | 0.51 | 0.53 |
| 4500-4999 | -57.64 | -58.66 | 211 | 63 | 0.52 | 0.54 |

Healthy: finite, the acceptance settled at 0.52 as in the released run, the energy still
falling. But it is 1.0 meV per electron above the released network at the same iteration, with
3.3 times its variance.

Where the electrons are (`results/checkpoint_structure_attention_18e_eps5.csv`,
`figures/checkpoint_structure.png`):

| iteration | S(kappa) | connected part | share on the three sublattices | walkers with an emptiest sublattice of <= 3 electrons |
|---|---|---|---|---|
| 1,626 | 0.699 | 0.570 | 0.263 / 0.361 / 0.376 | 14 % |
| 3,273 | 0.806 | 0.527 | 0.226 / 0.386 / 0.388 | 27 % |
| 4,922 | 0.940 | 0.551 | 0.207 / 0.398 / 0.395 | 45 % |
| 5,000 | 0.951 | 0.550 | 0.205 / 0.403 / 0.392 | 45 % |

The released run at its checkpoints 1,627, 3,283 and 4,933: S(kappa) 0.791, 0.635, 0.619, its
emptiest sublattice at 0.242, 0.318, 0.330. Both started with one sublattice emptier; the
released network lost that, the paper network deepens it, towards the 0 / 1/2 / 1/2 of a
honeycomb that stays put. The rise of S(kappa) is in its static part (S(kappa) minus the
connected part is 0.40 at 5,000): the density itself begins to show one honeycomb. It is early:
S(kappa) is still below 1 (electrons fixed on a perfect honeycomb give 4.5), and the energy is
above the released network's. A trend, not a crystal.

**Step 2**: `bash learn/stage08_own_code/launch_table1.sh paper5 150000` started on 2026-11-09
at 01:08:59 server time and continued from `checkpoint_005000.npz` (GPU 0; about 44 hours,
then the frozen-weight evaluation). `learn/stage08_own_code/watch_run.py` watches it from the
server, one line every 30 minutes into `training/logs/watch.log`: iteration and speed, energy,
acceptance, variance, S(kappa) at the newest checkpoint, the GPU; ALERT lines for non-finite
numbers, a stall, an early stop, a non-zero exit code, or the GPU at 88 C. It only reads.

**Progress at 30,000 iterations** (2026-11-09, 08:50 UTC): the imbalance seen at 5,000 has
faded steadily since. S(kappa) went from 0.951 to 0.703 (checkpoint 29,814), its static part
from 0.40 to 0.077, the emptiest sublattice's share from 0.205 to 0.278
(`results/checkpoint_structure_attention_18e_eps5.csv`; `figures/checkpoint_structure.png`, now
with the released run beside it). Over the same stretch the gap to the released network's energy
at the same iterations closed from 1.01 to 0.24 meV per electron (4,500-4,999: -57.644 against
-58.658; 29,000-29,499: -58.791 against -59.033). Both networks started with one sublattice
emptier at their first checkpoint (shares 0.263 and 0.242); the paper network kept it longer
and is now losing it too. So far the honeycomb trend of the first 5,000 iterations looks
transient.

**Progress at 84,600 iterations** (2026-11-10, 01:24 UTC; `training/logs/watch.log`, no ALERT):
S(kappa) has been between 0.70 and 0.72 since about 45,000, its static part 0.001 to 0.004,
as in the released network's run: no honeycomb so far. Energy over the last 500 iterations
-59.032 meV per electron (variance 25). At checkpoint 86,034, 97.0(9) % of the wave function's
weight is on the six total momenta of the open shell (`momentum_sectors.py`; see
`../18_electrons_27_sites_eps3_real_gradient/README.md`), as for the released network.
This is one seed, at 84,600 of 150,000 iterations, with other initial weights than the released
network: it does not show that the network form plays no role.

**2026-11-10, 02:25:35: GPU 1 fell off the bus** (`../18_electrons_27_sites_authors_code/README.md`).
This run, on GPU 0, goes on (iteration 89,261 at 02:39). But no new program can start on either
GPU until the server is rebooted, so the frozen-weight evaluation that follows the training
will refuse to start. After a reboot, `bash learn/stage08_own_code/launch_table1.sh paper5 150000`
continues the training from its newest checkpoint if it was cut short, and runs the evaluation.

## Results

After the run.
