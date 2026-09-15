"""Numeric core of the ringdown sensitive-volume calculation.

Deliberately free of bokeh so it can be imported and tested without
`plots.sif`; the figure lives in `plots.legacy.ringdown`.

The target population, stated once so it is never implicit (design
Ruling 3): log normal in source-frame remnant mass, with the spin
marginal equal to the transformed injection marginal, and the redshift
marginal proportional to dV_c/(1+z) so that it matches the measure
`utils.cosmology.get_astrophysical_volume` integrates.

Under that choice the quality and distance factors cancel between the
Jacobian and the injection density, leaving a one-dimensional weight

    w  =  p_M(M_src) * |dM_src/df| / q_f(f) * r_z(z)

with |dM_src/df| = M_src / f and r_z(z) = C_z / (1 + z).
"""

import numpy as np
from astropy.cosmology import z_at_value
from astropy.units import Mpc
from scipy.special import logsumexp

# Matches projects/plots/plots/legacy/tools.py:5, duplicated because that
# module imports bokeh and this one must not.
#
# NOTE: this is deliberately NOT ledger.events.SECONDS_IN_YEAR, which is
# 31556952 against this 31557600 -- a 648 s difference. EventSet.far
# (events.py:119,135) uses the ledger constant; the CBC plotting path uses
# this one. Adopting the ledger's value here would silently shift the
# ringdown FAR axis relative to the CBC plot it is meant to match. The
# inconsistency is pre-existing and out of scope; do not "fix" it.
SECONDS_PER_YEAR = 60 * 60 * 24 * 365.25


def redshift_bounds(source_prior, cosmology):
    """Redshift range implied by the source prior's distance bounds."""
    distance = source_prior["distance"]
    zmin = z_at_value(
        cosmology.luminosity_distance, distance.minimum * Mpc
    ).value
    zmax = z_at_value(
        cosmology.luminosity_distance, distance.maximum * Mpc
    ).value
    return float(zmin), float(zmax)


def redshift_ratio_norm(zmin, zmax, cosmology, n=2001):
    """Normalization C_z of the target/source redshift density ratio.

    The injection prior is uniform in comoving volume, q_z ~ dV_c/dz. The
    target is p_z ~ (dV_c/dz)/(1+z). C_z makes p_z/q_z = C_z/(1+z) a
    normalized ratio.
    """
    z = np.linspace(zmin, zmax, n)
    dv = cosmology.differential_comoving_volume(z).value
    return float(np.trapz(dv, z) / np.trapz(dv / (1 + z), z))


def log_importance_weights(
    mass_source, frequency, redshift, source_prior, target_prior, c_z
):
    """Log of the unnormalized importance weight, per injection.

    Evaluated in log space throughout: a narrow target underflows to
    exactly zero under direct density evaluation while remaining
    perfectly representable here.
    """
    log_p = target_prior["remnant_mass_source"].ln_prob(mass_source)
    # |dM_src/df| = M_src / f, since M_src is proportional to 1/f at
    # fixed quality and redshift.
    log_jacobian = np.log(mass_source) - np.log(frequency)
    log_q_f = source_prior["frequency"].ln_prob(frequency)
    log_r_z = np.log(c_z) - np.log1p(redshift)
    return log_p + log_jacobian - log_q_f + log_r_z


def normalize_log_weights(log_w):
    """Normalize log weights to a probability vector summing to one.

    Raises rather than emitting NaNs, so that an invalid weight vector is
    a loud failure. This is the arithmetic guard only — a vector that
    normalizes fine but concentrates on one draw is a *statistical*
    failure, caught by `effective_sample_size` against `ess_floor`.
    """
    log_w = np.asarray(log_w, dtype=float)
    if np.any(np.isnan(log_w)) or np.any(np.isposinf(log_w)):
        raise ValueError("importance weights are not finite")
    if not np.any(np.isfinite(log_w)):
        raise ValueError("importance weights have no finite entries")
    # Subtract the max into a new array before normalizing: logsumexp
    # sums stably, but its returned scalar still has to represent a large
    # common offset plus a small logarithmic correction, and that
    # correction rounds away once ln(n) falls below ulp(offset). Shifting
    # to a max of zero keeps the correction representable. A new array,
    # not `-=`, since callers keep their own reference to `log_w`.
    log_w = log_w - np.max(log_w)
    total = logsumexp(log_w)
    # Invariant assertion, not reachable today: the two guards above rule
    # out NaN and +inf entries and require at least one finite entry, and
    # subtracting that finite max pins the shifted max to exactly 0, so
    # logsumexp is bounded below by log(1) = 0 and `total` is always
    # finite. Kept in case a future change to the shift or the guards
    # above it stops guaranteeing that.
    if not np.isfinite(total):
        raise ValueError("importance weights have no finite normalization")
    return np.exp(log_w - total)


def _support_grid(source_prior, cosmology, n_q, n_z):
    """Mass support bounds and marginal weights on a (Q, z) grid.

    The transformed prior is not a box in (M, chi, z): for each (Q, z)
    the frequency bounds impose a different mass interval, so coverage
    has to be integrated rather than read off a range.
    """
    from ledger.injections import C, G, MSUN

    quality = source_prior["quality"]
    q = np.linspace(quality.minimum, quality.maximum, n_q)
    zmin, zmax = redshift_bounds(source_prior, cosmology)
    z = np.linspace(zmin, zmax, n_z)
    qq, zz = np.meshgrid(q, z, indexing="ij")

    k = C**3 / (2 * np.pi * G * MSUN)
    g = 1 - 0.63 * (2 / qq) ** (2 / 3)
    f_min = source_prior["frequency"].minimum
    f_max = source_prior["frequency"].maximum
    m_min = k * g / (f_max * (1 + zz))
    m_max = k * g / (f_min * (1 + zz))

    # Q keeps the source marginal; z uses the TARGET marginal, which is
    # what the weights actually impose (design Ruling 4).
    w_q = quality.prob(q)
    w_q = w_q / np.trapz(w_q, q)
    dv = cosmology.differential_comoving_volume(z).value
    p_z = dv / (1 + z)
    p_z = p_z / np.trapz(p_z, z)
    return q, z, qq, zz, m_min, m_max, w_q, p_z


def _target_mass_fraction(m_min, m_max, m0, sigma):
    """P(m_min <= M <= m_max) under LogNormal(ln m0, sigma), stably.

    A plain CDF difference loses the whole interval in the upper tail:
    both terms round to 1.0 and the difference is exactly 0.0 where the
    true probability is small but nonzero. At m0=1, sigma=0.3 over
    [24, 273] Msun the naive form gives 0.0 against 1.596885e-26 from the
    survival-function difference. That zero propagates to a zero coverage
    normalization and NaN conditioned means, with an ESS that still
    passes its floor -- so nothing downstream catches it.
    """
    from scipy.stats import norm

    lo = (np.log(m_min) - np.log(m0)) / sigma
    hi = (np.log(m_max) - np.log(m0)) / sigma
    # Subtract on whichever tail keeps the significant digits.
    upper = norm.sf(lo) - norm.sf(hi)
    lower = norm.cdf(hi) - norm.cdf(lo)
    return np.clip(np.where(lo > 0, upper, lower), 0.0, 1.0)


def support_coverage(m0, sigma, source_prior, cosmology, n_q=601, n_z=2001):
    """Fraction of the intended target lying inside the joint support.

    Independent of effective sample size: ESS is small when few samples
    carry most of the weight, coverage is small when part of the intended
    population has no samples at all. More injections raise ESS and never
    touch coverage.
    """
    q, z, _, _, m_min, m_max, w_q, p_z = _support_grid(
        source_prior, cosmology, n_q, n_z
    )
    a = _target_mass_fraction(m_min, m_max, m0, sigma)
    return float(np.trapz(np.trapz(a * p_z[None, :], z, axis=1) * w_q, q))


def conditioned_means(m0, sigma, source_prior, cosmology, n_q=601, n_z=2001):
    """Means of the support-conditioned population p_S.

    Self-normalizing the sampled weights estimates p_S for every target
    whose coverage is below one, so this — not the nominal marginal — is
    the reference a target-identity test compares against, regardless of
    any warning threshold.
    """
    q, z, qq, zz, m_min, m_max, w_q, p_z = _support_grid(
        source_prior, cosmology, n_q, n_z
    )
    a = _target_mass_fraction(m_min, m_max, m0, sigma)
    joint = a * w_q[:, None] * p_z[None, :]
    norm_c = np.trapz(np.trapz(joint, z, axis=1), q)
    if not np.isfinite(norm_c) or norm_c <= 0:
        raise ValueError(
            f"target M0={m0}, sigma={sigma} left no numerically "
            f"representable probability inside the joint support, so "
            f"conditioned means are undefined. A log-normal has positive "
            f"density at every positive mass, so this is underflow rather "
            f"than mathematically empty support; the point of raising is "
            f"to never write non-finite population metadata."
        )
    return {
        "quality": float(
            np.trapz(np.trapz(joint * qq, z, axis=1), q) / norm_c
        ),
        "redshift": float(
            np.trapz(np.trapz(joint * zz, z, axis=1), q) / norm_c
        ),
    }


def effective_sample_size(weights):
    """Kish effective sample size of a normalized weight vector."""
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / (weights**2).sum())


def sensitive_volume(statistic, detected, weights, thresholds):
    """Weighted detection fraction and its uncertainty, per threshold.

    Unlike `plots.legacy.compute.sensitive_volume`, the sum runs over
    EVERY draw, not only the foreground rows. Rejected injections and
    time-window misses carry their normalized weight with a detection
    indicator of zero, so they contribute w^2 mu^2 to the variance. The
    CBC implementation omits those terms and understates the error;
    it is left untouched here (design Ruling 6).

    Args:
        statistic:
            Detection statistic per draw, shape (n_draws,). Draws that
            were never injected take -inf so no threshold selects them.
        detected:
            Boolean per draw, shape (n_draws,). False for rejected draws
            and for recovered injections outside the dt window.
        weights:
            Normalized weights, shape (n_combos, n_draws), summing to one
            across draws for each combo.
        thresholds:
            Detection-statistic thresholds, shape (n_thresholds,).

    Returns:
        mu, err: each shape (n_combos, n_thresholds).
    """
    statistic = np.asarray(statistic, dtype=float)
    detected = np.asarray(detected, dtype=bool)
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    thresholds = np.asarray(thresholds, dtype=float)

    mu = np.empty((weights.shape[0], thresholds.size))
    err = np.empty_like(mu)
    for i, threshold in enumerate(thresholds):
        indicator = (detected & (statistic >= threshold)).astype(float)
        m = (weights * indicator).sum(axis=-1, keepdims=True)
        err[:, i] = ((weights * (indicator - m)) ** 2).sum(axis=-1) ** 0.5
        mu[:, i] = m[:, 0]
    return mu, err


def far_threshold_grid(background_statistic, tb_seconds, max_far):
    """False alarm rates and matching detection-statistic thresholds.

    Each returned FAR is the TRUE inclusive exceedance rate of its
    threshold, `count(background_statistic >= threshold) / Tb_years` --
    not a rank. Ranking consecutive sorted background events, as
    `plots/legacy/main.py:172` does (`np.arange(1, n + 1) / Tb` against
    `np.sort(background)[-n:][::-1]`), understates the FAR of any
    threshold inside a tie group: `sensitive_volume` selects with
    `statistic >= threshold`, which admits the WHOLE tie group, but a
    rank only credits the tie group's first member. E.g. background
    `[9.0, 9.0, 8.0]` over one year ranks 9.0 at FAR 1/yr, when the rate
    at which the pipeline actually reports something scoring >= 9.0 is
    2/yr. This function returns one (far, threshold) pair per DISTINCT
    background value, at that value's true rate, so a tie group is
    represented once, correctly, rather than once per member each
    under-reporting it.

    This also still validates the two silent failures Task 7 exists to
    prevent: a FAR axis with no acceptable rank pairs it with no
    thresholds instead of the CBC construction's `arr[-0:]` full-slice
    bug, and a threshold whose true rate exceeds `max_far` is dropped
    rather than kept under a rank-based rate that looked acceptable.
    """
    background_statistic = np.asarray(background_statistic, dtype=float)
    if not np.isfinite(tb_seconds) or tb_seconds <= 0:
        raise ValueError(
            f"background livetime must be positive and finite, "
            f"got {tb_seconds}"
        )
    if background_statistic.size == 0:
        raise ValueError("cannot build a FAR grid from an empty background")

    tb_years = tb_seconds / SECONDS_PER_YEAR
    sorted_ascending = np.sort(background_statistic)
    thresholds = np.unique(background_statistic)[::-1]
    # count(background_statistic >= value): everything from the index of
    # value's first occurrence in the ascending sort onward.
    counts = sorted_ascending.size - np.searchsorted(
        sorted_ascending, thresholds, side="left"
    )
    # thresholds descends, so counts -- and therefore fars -- strictly
    # ascend: each smaller threshold's tie group is a strict superset of
    # every larger one's.
    fars = counts / tb_years

    keep = fars <= max_far
    if not np.any(keep):
        # fars is sorted ascending, so fars[0] is the smallest rate any
        # threshold can carry; if even that exceeds max_far, no rank
        # satisfies it. More background samples cannot fix this -- they
        # can only add ties or raise counts further -- the remedy is a
        # longer livetime or a larger max_far.
        raise ValueError(
            f"no background rank satisfies max_far={max_far}/yr at "
            f"Tb={tb_years:.6g} yr (least available rate is "
            f"{fars[0]:.6g}/yr); a longer livetime or a larger max_far "
            f"is needed, not a larger background"
        )
    thresholds = thresholds[keep]
    fars = fars[keep]
    # Invariant assertion, not reachable today: `keep` is a single
    # boolean array applied to `thresholds` and `fars`, which are the
    # same length by construction (both derived elementwise from
    # `thresholds` before this point), so boolean-indexing both by the
    # same mask always leaves them the same length. Kept in case a
    # future refactor computes them independently instead.
    if len(fars) != len(thresholds):
        raise ValueError("FAR and threshold grids disagree in length")
    return fars, thresholds
