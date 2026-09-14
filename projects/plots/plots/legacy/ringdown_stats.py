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
    if not np.isfinite(total):
        raise ValueError("importance weights have no finite normalization")
    return np.exp(log_w - total)
