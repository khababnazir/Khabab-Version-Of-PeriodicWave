"""The paper's printed attention network at eps = 5 (this folder) against the released network:
frozen-weight energy, S(kappa), sublattices, site pattern.  Only calls functions of
learn/stage08_own_code/analysis/18_electrons_27_sites/table1_results.py and eps3_results.py.

Run, from learn/stage08_own_code/analysis/18_electrons_27_sites:
    PYTHONPATH=$PWD uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock \
        python ../../../../runs/stage08/18_electrons_27_sites_paper_network_real_gradient/results/paper_network_results.py
"""
import pandas as pd
import eps3_results as e3

t1 = e3.t1
PAPER = e3.STAGE08 / "18_electrons_27_sites_paper_network_real_gradient"
runs = [dict(e3.load(PAPER, "attention_18e_eps5"), folder="paper_network"),
        dict(e3.load(e3.REAL, "attention_18e_eps5"), folder="released_network")]
energies, structure = e3.energy_table(runs), t1.structure_table(runs)
pattern = e3.site_pattern_table(runs)
energies.to_csv(PAPER / "results" / "energies.csv", index=False)
structure.to_csv(PAPER / "results" / "structure.csv", index=False)
pattern.to_csv(PAPER / "results" / "site_pattern.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(energies[["run", "checkpoint", "E_frozen", "E_frozen_error", "E_training_last20pct",
                    "E_training_last20pct_error", "halves_apart_in_errors", "settled"]].to_string(index=False))
    print(structure.drop(columns=["electrons", "sites", "network"]).to_string(index=False))
    print(pattern.groupby("run", sort=False)["site_pattern"].sum().round(4).to_string())
