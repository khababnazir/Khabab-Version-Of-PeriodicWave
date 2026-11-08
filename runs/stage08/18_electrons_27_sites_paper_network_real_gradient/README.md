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

## After 5,000 iterations

To come.

## Results

After the run.
