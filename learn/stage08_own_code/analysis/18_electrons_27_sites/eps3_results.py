"""
Stage 8 -- the attention network at eps = 3 (our diagnostic, not in the paper) against the
same network at eps = 5 and 10, all with the authors' gradient (18 electrons, 27 sites).

Does a stronger repulsion give the honeycomb crystal that the paper's Fig. 6(b) shows at
eps = 5 and our network does not?  Reads the frozen-weight evaluations
    runs/stage08/18_electrons_27_sites_eps3_real_gradient/evaluation/attention_18e_eps3
    runs/stage08/18_electrons_27_sites_real_gradient/evaluation/attention_18e_eps5, _eps10
with table1_results.load (the same checks: finished, sanity check passed, authors' gradient).

Writes into runs/stage08/18_electrons_27_sites_eps3_real_gradient/:
  results/energies.csv     frozen-weight and training energies in meV per electron, the
                           kinetic, moire and Coulomb parts, and whether the evaluation settled
                           (its two halves within two combined blocking errors)
  results/structure.csv    S(kappa), its connected part and its two halves, the share of the
                           electrons on each sublattice, the site-occupation range
  results/off_site.csv     the share of the density near the sites, between them, and at the
                           centres of the triangles (the potential's second, shallower minima)
  results/density_modulation.csv   the static part |<rho_q>|^2 / N of S(q): summed over all
                           wave vectors but the moire lattice's own, at kappa, and the largest six
  results/site_pattern.csv the pattern of the 27 mean site occupations at its 27 wave vectors,
                           without the aliasing of the continuous density (2026-11-08)
  figures/1_density_pair_correlation.png   density and pair correlation at eps = 10, 5, 3
  figures/2_sublattices_during_evaluation.png   the three sublattices step by step, eps = 3

Run, from this folder:
    uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python eps3_results.py
"""
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import table1_results as t1

plt = t1.plt
STAGE08 = t1.HERE.parents[3] / "runs" / "stage08"
EPS3 = STAGE08 / "18_electrons_27_sites_eps3_real_gradient"
REAL = STAGE08 / "18_electrons_27_sites_real_gradient"


def load(data, folder):
    """table1_results.load on the evaluation `folder` of the system folder `data`."""
    t1.DATA, t1.TRAINING, t1.EVALUATION, t1.GRADIENT = data, data / "training", data / "evaluation", "real"
    run = t1.load(folder)
    assert run is not None, f"{data / 'evaluation' / folder}: no finished evaluation"
    assert run["network"] == "attention" and run["final"], f"{folder}: not the final attention weights"
    return run


def settled(run):
    """Blocked means of the two halves of the evaluation and their difference in combined errors."""
    first, second = np.array_split(run["series"], 2)
    (m1, e1), (m2, e2) = t1.blocked_mean(first), t1.blocked_mean(second)
    return m1, e1, m2, e2, (m2 - m1) / np.hypot(e1, e2)


def energy_table(runs):
    rows = []
    for r in runs:
        m1, e1, m2, e2, sigmas = settled(r)
        rows.append(dict(run=r["folder"], eps=r["eps"], checkpoint=r["step"],
                         E_frozen=r["frozen"], E_frozen_error=r["frozen_error"],
                         E_training_last20pct=r["training"], E_training_last20pct_error=r["training_error"],
                         kinetic=r["kinetic"], moire=r["moire"], coulomb=r["coulomb"],
                         first_half=m1, first_half_error=e1, second_half=m2, second_half_error=e2,
                         halves_apart_in_errors=sigmas, settled=abs(sigmas) <= 2,
                         S_kappa_first_half=r["S_kappa_halves"][0], S_kappa_second_half=r["S_kappa_halves"][1],
                         sigma_local_energy_meV=r["sigma_local_energy_meV"], pmove=r["pmove"],
                         configurations=r["n_configurations"]))
    return pd.DataFrame(rows)


def off_site_table(runs):
    """How much of the density lies away from the moire sites, from the density map.

    The moire potential (phi = 45 degrees) has its minima on the sites (-63.6 meV for one
    electron) and shallower minima at the centres of the 'down' triangles (-23.3 meV; the 'up'
    centres are maxima, +86.9 meV), all 0.577 a_M from the nearest site."""
    cell = runs[0]["supercell"]
    sites = t1.observables.moire_sites(cell)
    a1, a2 = t1.observables.MOIRE_LATTICE[:, 0], t1.observables.MOIRE_LATTICE[:, 1]
    up, down = sites + (a1 + (a1 + a2)) / 3, sites + ((a1 + a2) + a2) / 3

    def distance_to(points, targets):
        d = np.asarray(t1.observables.minimum_image(points[..., None, :] - targets, cell))
        return np.sqrt((d ** 2).sum(-1)).min(-1)

    rows = []
    for r in runs:
        n = t1.observables.density_map(r["totals"]["density"], r["n_configurations"], cell)
        s = (np.arange(n.shape[0]) + 0.5) / n.shape[0]
        centres = np.stack(np.meshgrid(s, s, indexing="ij"), -1) @ cell.T            # bin centres
        electrons = n * t1.observables.supercell_area(cell) / n.size
        assert np.isclose(electrons.sum(), t1.N), "the density map holds every electron"
        d_site, d_down, d_up = distance_to(centres, sites), distance_to(centres, down), distance_to(centres, up)
        share = lambda mask: electrons[mask].sum() / t1.N                              # noqa: E731
        rows.append(dict(run=r["folder"], eps=r["eps"], within_0_2_of_a_site=share(d_site < 0.2),
                         from_0_2_to_0_35=share((d_site >= 0.2) & (d_site < 0.35)), beyond_0_35=share(d_site >= 0.35),
                         within_0_15_of_a_down_centre=share(d_down < 0.15), within_0_15_of_an_up_centre=share(d_up < 0.15)))
    return pd.DataFrame(rows)


def modulation_table(runs, largest=6):
    """Where the density itself is modulated: the static part |<rho_q>|^2 / N of S(q).

    q is written as m1 b1 + m2 b2 (b: the moire reciprocal vectors), m = (a1 . q, a2 . q) / 2 pi.
    Integer m are the moire potential's own wave vectors, modulated in every state, so they are
    left out.  The honeycomb's wave vectors are the kappas, e.g. m = (1/3, 1/3)."""
    rows = []
    for r in runs:
        q = np.array(r["summary"]["q_vectors"])
        S, S_connected = t1.observables.structure_factor(r["totals"]["rho_q"], r["totals"]["rho_q_squared"],
                                                         r["n_configurations"], t1.N)
        static = S - S_connected
        m = q @ t1.observables.MOIRE_LATTICE / (2 * np.pi)
        other = ~np.all(np.isclose(m, np.round(m), atol=1e-6), axis=1)              # q = 0 is a moire one too
        at_kappa = np.isclose(np.linalg.norm(q, axis=1), t1.KAPPA, atol=1e-6)
        base = dict(run=r["folder"], eps=r["eps"], static_sum=static[other].sum(), static_at_kappa=static[at_kappa].mean())
        rows.append(dict(base, rank=0, m1=np.nan, m2=np.nan, q=np.nan, static=np.nan))
        for rank, i in enumerate(np.flatnonzero(other)[np.argsort(static[other])[::-1][:largest]], start=1):
            rows.append(dict(base, rank=rank, m1=m[i, 0], m2=m[i, 1], q=np.linalg.norm(q[i]), static=static[i]))
    return pd.DataFrame(rows)


def site_pattern_table(runs):
    """The pattern of the 27 mean site occupations <n_k>, by wave vector, without aliasing.

    On the sites R_k, q and q + G (G a moire reciprocal vector) give the same phase
    exp(-i q . R_k), so the continuous density's modulation shows a site pattern at several
    q that differ by a G (in density_modulation.csv, for example, (7/9, -5/9) and -(2/9, 5/9)).
    Here the site pattern has exactly 27 wave vectors, each reduced to its shortest form
    q_BZ (inside the moire Brillouin zone): A(q) = sum_k (<n_k> - 2/3) exp(-i q . R_k),
    and |A(q)|^2 / N.  A(-q) = conj A(q), so the q come in +- pairs.  Rows with the same
    |q_BZ| are related by the lattice's rotations; how many of them carry the pattern tells
    whether it picks one direction.  The honeycomb is the pair q = +-kappa."""
    rows = []
    for r in runs:
        cell = r["supercell"]
        sites = t1.observables.moire_sites(cell)
        lattice = t1.observables.MOIRE_LATTICE
        b = 2 * np.pi * np.linalg.inv(lattice).T                              # columns: b1, b2
        q = np.array(r["summary"]["q_vectors"])
        m = q @ lattice / (2 * np.pi)
        reduced = {}
        for mi in np.round(np.mod(m, 1.0) * 27).astype(int) % 27:                # 1/27 is finer than needed
            reduced[tuple(mi)] = mi / 27
        assert len(reduced) == 27, f"expected the 27 wave vectors of the 27 sites, found {len(reduced)}"
        dn = r["occupation"] - t1.N / 27
        assert np.isclose(r["occupation"].sum(), t1.N)
        for mi in reduced.values():
            shifts = [(mi + np.array([i, j])) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            m_bz = min(shifts, key=lambda s: np.linalg.norm(b @ s))             # the shortest equivalent
            q_bz = b @ m_bz
            A = np.sum(dn * np.exp(-1j * sites @ q_bz))
            rows.append(dict(run=r["folder"], eps=r["eps"], m1=m_bz[0], m2=m_bz[1], q_bz=np.linalg.norm(q_bz),
                             at_kappa=bool(np.isclose(np.linalg.norm(q_bz), t1.KAPPA, atol=1e-6)),
                             site_pattern=abs(A) ** 2 / t1.N))
        mine = pd.DataFrame([row for row in rows if row["run"] == r["folder"]])
        # Parseval: summed over the 27 q, |A|^2 = 27 sum_k (<n_k> - 2/3)^2.
        assert np.isclose(mine["site_pattern"].sum() * t1.N, 27 * np.sum(dn ** 2)), "Parseval"
    table = pd.DataFrame(rows).sort_values(["eps", "site_pattern"], ascending=[False, False])
    return table


def density_pair_figure(runs, path):
    """Fig. 6's layout with one column per eps: density (top), pair correlation (bottom)."""
    fig, axes = plt.subplots(2, len(runs), figsize=(3.2 * len(runs) + 0.8, 6.0), constrained_layout=True)
    densities = [t1.observables.density_map(r["totals"]["density"], r["n_configurations"], r["supercell"])
                 for r in runs]
    pairs = [t1.observables.pair_correlation_map(r["totals"]["pairs"], r["n_configurations"], t1.N, r["supercell"])
             for r in runs]
    density_style = dict(cmap="magma", vmin=0, vmax=max(m.max() for m in densities))
    pair_style = dict(cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(m.max() for m in pairs)))
    for column, r in enumerate(runs):
        bins = r["summary"]["density_bins"]
        for row, (values, centred, style) in enumerate(((densities[column], False, density_style),
                                                         (pairs[column], True, pair_style))):
            ax = axes[row, column]
            for X, Y in t1.tiled_mesh(r["supercell"], bins, centred):
                image = ax.pcolormesh(X, Y, values, shading="flat", rasterized=True, **style)
            xy = t1.draw_cell(ax, r["supercell"], centred)
            ax.set_xlim(xy[:, 0].min() - 0.6, xy[:, 0].max() + 0.6)
            ax.set_ylim(xy[:, 1].min() - 0.6, xy[:, 1].max() + 0.6)
            ax.set_aspect("equal")
            if centred:
                ax.plot(0, 0, "+", color="k", ms=5, mew=0.8)           # the reference electron
            what = "density n(r)" if row == 0 else "pair correlation g(\u0394r)"
            ax.set_title(f"{'abcdef'[len(runs) * row + column]}   {what}, \u03b5 = {r['eps']:g}", fontsize=9)
            ax.set_xlabel("\u0394x / a$_M$" if centred else "x / a$_M$")
            ax.set_ylabel("\u0394y / a$_M$" if centred else "y / a$_M$")
            if column == len(runs) - 1:
                fig.colorbar(image, ax=axes[row, :], shrink=0.8,
                             label="n(r)  (a$_M^{-2}$)" if row == 0 else "g(\u0394r)  (1 = uniform, independent electrons)")
    fig.suptitle("Self-attention network, 27 sites, 18 electrons, authors' gradient (dashed: the supercell).  "
                 "\u03b5 = 10 and 5 as in the paper's Fig. 6; \u03b5 = 3 is our diagnostic, not in the paper.",
                 x=0.02, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    runs = [load(REAL, "attention_18e_eps10"), load(REAL, "attention_18e_eps5"), load(EPS3, "attention_18e_eps3")]
    assert [r["eps"] for r in runs] == [10.0, 5.0, 3.0]
    (EPS3 / "results").mkdir(exist_ok=True)
    (EPS3 / "figures").mkdir(exist_ok=True)
    energies, structure = energy_table(runs), t1.structure_table(runs)
    # Self-check: a perfect honeycomb (sublattice 0 empty) puts its whole pattern at +-kappa, 4.5 each.
    honeycomb = dict(runs[0], folder="honeycomb", eps=0.0,
                     occupation=(t1.sublattice_of_sites(runs[0]["supercell"]) != 0).astype(float))
    check = site_pattern_table([honeycomb])
    assert check["at_kappa"].sum() == 2 and np.allclose(check.loc[check["at_kappa"], "site_pattern"], 4.5), check
    assert np.isclose(check.loc[~check["at_kappa"], "site_pattern"].sum(), 0.0, atol=1e-9), check
    off_site, modulation, site_pattern = off_site_table(runs), modulation_table(runs), site_pattern_table(runs)
    site_pattern.to_csv(EPS3 / "results" / "site_pattern.csv", index=False)
    energies.to_csv(EPS3 / "results" / "energies.csv", index=False)
    structure.to_csv(EPS3 / "results" / "structure.csv", index=False)
    off_site.to_csv(EPS3 / "results" / "off_site.csv", index=False)
    modulation.to_csv(EPS3 / "results" / "density_modulation.csv", index=False)
    density_pair_figure(runs, EPS3 / "figures" / "1_density_pair_correlation.png")
    t1.sublattice_figure([runs[2]], EPS3 / "figures" / "2_sublattices_during_evaluation.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(energies[["run", "E_frozen", "E_frozen_error", "E_training_last20pct", "kinetic", "moire", "coulomb",
                        "halves_apart_in_errors", "settled", "pmove"]].to_string(index=False))
        print(structure.drop(columns=["electrons", "sites", "network"]).to_string(index=False))
        print(off_site.to_string(index=False))
        print(modulation[modulation["rank"] <= 3].to_string(index=False))
        for eps, group in site_pattern.groupby("eps", sort=False):
            print(f"eps {eps:g}: site pattern summed over q {group['site_pattern'].sum():.4f}")
            print(group.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
