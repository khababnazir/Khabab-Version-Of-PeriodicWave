# results — the main numbers of Stage 8

Two tables, one row per run, written by
`learn/archive/stage08_authors_code/analysis/plot_weights.py`. Every error is one
standard error from a blocking analysis.

## energies_meV_per_electron.csv

All energies in **meV per electron** (the paper's unit), except where noted.

| Column | Meaning |
|---|---|
| `run`, `network`, `eps` | the run folder in `../training/`, the network, the dielectric constant ε |
| `E_training_last20pct`, `_error` | mean energy over the last 20 % of training (iterations 160,000–199,999) |
| `training_drift`, `_error` | mean of the first half of that window minus the second half; positive = still falling. Only suggestive at 1–2 errors. |
| `E_frozen_weights`, `_error` | energy of the final weights, frozen, over 10,000 evaluation steps × 1024 walkers. **The cleanest number per run.** |
| `E_fig5_read_by_eye` | the paper's Fig. 5 value as read off the plot, good to about ±0.05 |
| `kinetic`, `moire`, `coulomb` | the frozen-weight energy split into its three parts; they add up to `E_frozen_weights` |
| `sigma_local_energy_meV` | spread (standard deviation) of the local energy over the walkers, in meV for the **whole system**. Zero for an exact eigenstate; a measure of wave-function quality. |
| `pmove` | fraction of proposed walker moves accepted |
| `evaluation_steps`, `configurations` | how much was sampled (configurations = steps × 1024) |
| `steps_per_second` | evaluation speed |

## electron_structure.csv

Where the electrons are, from the same 10,240,000 configurations per run.

| Column | Meaning |
|---|---|
| `P_honeycomb` | fraction of configurations that are one of the three honeycomb crystals (each electron assigned to its nearest site) |
| `P_largest_honeycomb` | the most frequent of the three |
| `P_honeycomb_if_random` | what `P_honeycomb` would be if the one-electron-per-site arrangements were random (3 of 84) |
| `P_two_nearest_one_site` | fraction with two electrons nearest the same site (mostly an electron between sites) |
| `g_site_neighbours`, `g_site_sqrt3` | ⟨n_k n_l⟩ / (⟨n_k⟩⟨n_l⟩) for site pairs at distance 1 and √3. An equal mixture of the three honeycombs gives 0.75 and 1.5; random filling 0.94 and 0.94. A single honeycomb that stays in place gives 1 for its filled pairs, so these two numbers alone cannot rule out a crystal — read them with `P_honeycomb` and `S_kappa`. |
| `occupation_min`, `occupation_max` | least and most filled site (2/3 each if uniform) |
| `S_kappa` | structure factor S(q) = ⟨\|ρ_q\|²⟩ / N at the corner κ of the moiré Brillouin zone, averaged over its 6 copies. A honeycomb crystal gives more than 1 (1.5 for electrons exactly on sites). |
| `S_kappa_connected` | the same without the static part \|⟨ρ_q⟩\|² / N |
| `S_kappa_first_half`, `_second_half` | `S_kappa` over each half of the evaluation; equal if the spatial pattern is settled |
