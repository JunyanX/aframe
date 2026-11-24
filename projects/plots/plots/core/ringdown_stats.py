"""Numerical part of the ringdown sensitive volume; `plots.ringdown`
draws the figure.

The volume is for a target population that each injection is reweighted
to. Its source-frame remnant mass is log normal about the target mass. Its
redshift density is proportional to dV_c/dz / (1 + z), the measure
`utils.cosmology.get_astrophysical_volume` integrates. Everything else,
including spin, keeps the injections' distribution.

The injections were drawn log-uniform in frequency and uniform in comoving
volume, so a draw with frequency f, source-frame mass M and redshift z has
weight

    w = p_M(M) * (M / f) / q_f(f) * C_z / (1 + z)

p_M is the target mass density, and M / f turns it into a density in
frequency, since M is proportional to 1 / f at fixed quality and redshift.
q_f is the injections' frequency density, and C_z normalizes the redshift
ratio.
"""

import numpy as np
from astropy.cosmology import z_at_value
from astropy.units import Mpc
from scipy.special import logsumexp

from plots.core.constants import SECONDS_PER_YEAR


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
    return float(np.trapezoid(dv, z) / np.trapezoid(dv / (1 + z), z))


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
    """Turn log weights into weights that sum to one.

    Works in log space, so weights too small to represent directly still
    come out right, and a -inf entry is a weight of zero. Raises if any
    entry is NaN or +inf, or if no entry is finite.

    It doesn't check that the weights are useful: weights dominated by a
    single draw normalize fine, and `plots.ringdown.main` rejects them
    through the effective sample size.
    """
    log_w = np.asarray(log_w, dtype=float)
    if np.any(np.isnan(log_w)) or np.any(np.isposinf(log_w)):
        raise ValueError("importance weights are not finite")
    if not np.any(np.isfinite(log_w)):
        raise ValueError("importance weights have no finite entries")
    # Shift so the largest is 0. With a large common offset, such as
    # -1e16, the small correction logsumexp adds would be lost to
    # rounding. A new array rather than `-=`, to leave the caller's
    # array alone.
    log_w = log_w - np.max(log_w)
    total = logsumexp(log_w)
    return np.exp(log_w - total)


def _support_grid(source_prior, cosmology, n_q, n_z):
    """Mass support bounds and marginal weights on a (Q, z) grid.

    The transformed prior is not a box in (M, chi, z): for each (Q, z)
    the frequency bounds impose a different mass interval, so coverage
    has to be integrated rather than read off a range.
    """
    from ledger.injections import MSUN, C, G

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
    # what the weights actually impose.
    w_q = quality.prob(q)
    w_q = w_q / np.trapezoid(w_q, q)
    dv = cosmology.differential_comoving_volume(z).value
    p_z = dv / (1 + z)
    p_z = p_z / np.trapezoid(p_z, z)
    return q, z, qq, zz, m_min, m_max, w_q, p_z


def _target_mass_fraction(m_min, m_max, m0, sigma):
    """Fraction of the target's mass distribution between m_min and m_max.

    The target is log normal with median m0 and log-width sigma. Far above
    the median, both cumulative probabilities round to exactly 1 and their
    difference to 0, so that side is computed from the upper tail instead.
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
    return float(
        np.trapezoid(np.trapezoid(a * p_z[None, :], z, axis=1) * w_q, q)
    )


def conditioned_means(m0, sigma, source_prior, cosmology, n_q=601, n_z=2001):
    """Mean quality and redshift of the population the volume describes.

    The weights are normalized over the injections, which cover only part
    of a target whose coverage is below one. The volume then describes the
    target restricted to that part, whose spin and redshift can differ from
    the target's. These are its means, written to the output beside the
    coverage.
    """
    q, z, qq, zz, m_min, m_max, w_q, p_z = _support_grid(
        source_prior, cosmology, n_q, n_z
    )
    a = _target_mass_fraction(m_min, m_max, m0, sigma)
    joint = a * w_q[:, None] * p_z[None, :]
    norm_c = np.trapezoid(np.trapezoid(joint, z, axis=1), q)
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
            np.trapezoid(np.trapezoid(joint * qq, z, axis=1), q) / norm_c
        ),
        "redshift": float(
            np.trapezoid(np.trapezoid(joint * zz, z, axis=1), q) / norm_c
        ),
    }


def effective_sample_size(weights):
    """Kish effective sample size of a normalized weight vector."""
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / (weights**2).sum())


def far_threshold_grid(background_statistic, tb_seconds, max_far):
    """The FAR axis of the sensitive volume: false alarm rates and the
    detection-statistic thresholds that give them.

    Returns one (FAR, threshold) pair for each distinct background value
    whose rate is at most `max_far`, in order of increasing FAR. A
    threshold's FAR is how often the background scores at least that much,
    `count(background_statistic >= threshold) / Tb_years`. Counting rather
    than ranking, as `plots.core.sv._far_grid` does, gives tied values
    their full rate: background `[9.0, 9.0, 8.0]` over one year puts 9.0
    at 2/yr, where a rank says 1/yr.

    Raises if no threshold is within `max_far`, or if the background holds
    a NaN or an infinity.
    """
    background_statistic = np.asarray(background_statistic, dtype=float)
    if not np.isfinite(tb_seconds) or tb_seconds <= 0:
        raise ValueError(
            f"background livetime must be positive and finite, "
            f"got {tb_seconds}"
        )
    # written this way round so that it catches NaN too
    if not max_far > 0:
        raise ValueError(f"max_far must be positive, got {max_far}")
    if background_statistic.size == 0:
        raise ValueError("cannot build a FAR grid from an empty background")
    # NaN sorts above every finite value, so it would become the top
    # threshold at the lowest FAR, where `statistic >= nan` selects nothing
    n_bad = int(np.count_nonzero(~np.isfinite(background_statistic)))
    if n_bad:
        raise ValueError(
            f"background statistic has {n_bad} non-finite entries (NaN or "
            f"+/-inf) out of {background_statistic.size}; a FAR grid over "
            f"them would be silently wrong, and dropping them would leave "
            f"Tb={tb_seconds} s describing a sample it no longer covers"
        )

    tb_years = tb_seconds / SECONDS_PER_YEAR
    sorted_ascending = np.sort(background_statistic)
    # Only the loudest events can have a rate within max_far: at most
    # max_far * tb_years of them. Take that many rounded up, plus one so
    # that rounding in the product can't lose an event, and leave the
    # exact cut to the comparison with max_far below. Looking only at
    # this tail, rather than np.unique over the whole background, saves
    # several full-length copies: ~13.6 GB at 4e8 events.
    k = int(min(np.ceil(max_far * tb_years) + 1, sorted_ascending.size))
    thresholds = np.unique(sorted_ascending[-k:])[::-1]
    # count(background_statistic >= value). Searched in the whole sorted
    # array, not the tail, so that ties reaching below the tail are
    # counted in full.
    counts = sorted_ascending.size - np.searchsorted(
        sorted_ascending, thresholds, side="left"
    )
    fars = counts / tb_years

    keep = fars <= max_far
    if not np.any(keep):
        # More background can't fix this: it only adds ties or raises
        # the counts.
        raise ValueError(
            f"no background rank satisfies max_far={max_far}/yr at "
            f"Tb={tb_years:.6g} yr (least available rate is "
            f"{fars[0]:.6g}/yr); a longer livetime or a larger max_far "
            f"is needed, not a larger background"
        )
    return fars[keep], thresholds[keep]
