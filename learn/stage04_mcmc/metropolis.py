"""
Stage 4, part 2 -- Metropolis-Hastings sampling from |Psi|^2.

Paper:   Sec. IV A
Authors: external/PeriodicWave/periodicwave/mcmc.py (mh_update, mh_accept),
         and the step-width rule in train.py

The problem.  We want configurations R distributed as |Psi(R)|^2.  But we
cannot normalise |Psi|^2 -- that would be the 2N-dimensional integral we are
trying to avoid.

The trick.  Metropolis-Hastings only ever needs the RATIO
|Psi(R')|^2 / |Psi(R)|^2, in which the normalisation cancels:

    1. propose R' = R + (small random Gaussian step), moving all electrons
    2. accept R' with probability  min(1, |Psi(R')|^2 / |Psi(R)|^2)
    3. if rejected, stay at R  (and count R again)

Because the Gaussian proposal is symmetric (R -> R' as likely as R' -> R),
accepting with this probability makes |Psi|^2 the distribution the walkers
settle into.  This is called detailed balance.

We run many independent walkers side by side -- a "batch" -- because the
GPU moves them all at once for about the price of one.
"""
import jax
import jax.numpy as jnp


def metropolis_step(log_abs_psi_batch, walkers, log_prob, key, step_width):
    """One Metropolis-Hastings step for every walker at once.

    log_abs_psi_batch: function (n_walkers, N, 2) -> (n_walkers,) giving log|Psi|
    walkers:           (n_walkers, N, 2) current electron positions
    log_prob:          (n_walkers,) log|Psi|^2 at the current walkers.  Passed in
                       so we do not recompute it every step.

    Returns (new_walkers, new_log_prob, acceptance_rate).
    """
    key_move, key_accept = jax.random.split(key)

    # 1. Propose: move every electron of every walker by a Gaussian step.
    noise = jax.random.normal(key_move, walkers.shape)
    proposal = walkers + step_width * noise

    # log|Psi|^2 = 2 log|Psi|
    proposal_log_prob = 2.0 * log_abs_psi_batch(proposal)

    # 2. Accept with probability min(1, |Psi'|^2 / |Psi|^2).
    #    In logarithms: accept if  log(uniform random number) < log|Psi'|^2 - log|Psi|^2.
    #    When the ratio is above 1 the right side is positive, and log(uniform)
    #    is always negative, so the move is always accepted -- as it should be.
    log_ratio = proposal_log_prob - log_prob
    uniform = jax.random.uniform(key_accept, log_ratio.shape)
    accepted = jnp.log(uniform) < log_ratio                # (n_walkers,) of True/False

    # 3. Where accepted take the proposal, elsewhere keep the old position.
    #    accepted[:, None, None] has shape (n_walkers, 1, 1), so it lines up
    #    with walkers of shape (n_walkers, N, 2).
    new_walkers = jnp.where(accepted[:, None, None], proposal, walkers)
    new_log_prob = jnp.where(accepted, proposal_log_prob, log_prob)

    acceptance_rate = jnp.mean(accepted)
    return new_walkers, new_log_prob, acceptance_rate


def adapt_step_width(step_width, acceptance_rate):
    """The authors' rule (train.py): keep 50-55% of moves accepted.

    Step too small: almost every move is accepted, but walkers barely move.
    Step too large: moves would go far, but almost all are rejected.
    Either way the samples change slowly.  About half accepted is a good middle.
    """
    if acceptance_rate > 0.55:
        return step_width * 1.1
    if acceptance_rate < 0.50:
        return step_width * 0.9
    return step_width


def run_metropolis(log_abs_psi_batch, walkers, key, n_steps, step_width, adapt_every=10):
    """Run n_steps Metropolis steps.  Returns (walkers, step_width, acceptance_rates).

    adapt_every=10:   adjust the step width every 10 steps, from the average
                      acceptance rate over those steps.  Use during burn-in.
    adapt_every=None: keep the step width fixed.  Use while measuring.

    Why freeze it?  The guarantee that walkers end up distributed as |Psi|^2
    belongs to a FIXED Metropolis rule.  A rule that keeps changing itself
    based on the walkers' own history is not covered by that guarantee.  So:
    tune first, then freeze.

    (The authors adjust the width every 100 training iterations, each of 30
    Metropolis steps, throughout training.  There Psi itself changes every
    iteration, so the target is moving anyway.  To measure the energy of one
    fixed Psi, a frozen width is the clean choice.)
    """
    log_prob = 2.0 * log_abs_psi_batch(walkers)
    acceptance_rates = []

    for step in range(n_steps):
        key, step_key = jax.random.split(key)
        walkers, log_prob, acceptance = metropolis_step(
            log_abs_psi_batch, walkers, log_prob, step_key, step_width)
        acceptance_rates.append(float(acceptance))

        if adapt_every is not None and (step + 1) % adapt_every == 0:
            recent_rate = sum(acceptance_rates[-adapt_every:]) / adapt_every
            step_width = adapt_step_width(step_width, recent_rate)

    return walkers, step_width, acceptance_rates
