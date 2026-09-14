import numpy as np
import pytest
from astropy.cosmology import Planck15
from bilby.core.prior import LogUniform, PriorDict, Uniform
from bilby.gw.prior import UniformComovingVolume

from plots.legacy.ringdown_stats import (
    log_importance_weights,
    normalize_log_weights,
    redshift_bounds,
    redshift_ratio_norm,
)


# Module-scoped: the population fixtures below are module-scoped and a
# module-scoped fixture cannot depend on a function-scoped one -- pytest
# raises ScopeMismatch at setup. These tests only read the prior.
@pytest.fixture(scope="module")
def source_prior():
    prior = PriorDict()
    prior["frequency"] = LogUniform(100, 1000)
    prior["quality"] = Uniform(8, 20)
    prior["distance"] = UniformComovingVolume(
        100, 1000, name="luminosity_distance", cosmology=Planck15
    )
    return prior


def test_redshift_bounds(source_prior):
    zmin, zmax = redshift_bounds(source_prior, Planck15)
    assert abs(zmin - 0.0222) < 1e-3
    assert abs(zmax - 0.1980) < 1e-3


def test_redshift_ratio_norm(source_prior):
    zmin, zmax = redshift_bounds(source_prior, Planck15)
    c_z = redshift_ratio_norm(zmin, zmax, Planck15)
    assert abs(c_z - 1.14582999) < 1e-6


def test_weight_is_proportional_to_closed_form(source_prior):
    """Design Ruling 3(a): constancy is the assertion, not the value."""
    from priors.priors import log_normal_remnant_mass

    rng = np.random.default_rng(0)
    f = np.exp(rng.uniform(np.log(100), np.log(1000), 500))
    q = rng.uniform(8, 20, 500)
    d = rng.uniform(100, 1000, 500)
    from astropy.units import Mpc
    from astropy.cosmology import z_at_value

    z = z_at_value(Planck15.luminosity_distance, d * Mpc).value
    from ledger.injections import C, G, MSUN

    m_det = (
        (1 / (2 * np.pi))
        * (C**3 / (G * f))
        * (1 - 0.63 * (2 / q) ** (2 / 3))
        / MSUN
    )
    m_src = m_det / (1 + z)

    target, _ = log_normal_remnant_mass(80.0, sigma=0.1)
    zmin, zmax = redshift_bounds(source_prior, Planck15)
    c_z = redshift_ratio_norm(zmin, zmax, Planck15)
    log_w = log_importance_weights(m_src, f, z, source_prior, target, c_z)
    oracle = target["remnant_mass_source"].prob(m_src) * m_src / (1 + z)
    ratio = np.exp(log_w) / oracle
    # Ruling 3(b): the constant is ln(10) * C_z under the stated
    # convention. Tolerance follows bilby's interpolated distance prior,
    # not machine precision.
    assert np.allclose(ratio, ratio[0], rtol=1e-6)
    assert abs(ratio[0] / (np.log(10) * c_z) - 1) < 1e-6


def test_normalize_log_weights_sums_to_one():
    log_w = np.array([-1.0, -2.0, -3.0])
    w = normalize_log_weights(log_w)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(w >= 0)


def test_normalize_log_weights_survives_extreme_lognormal():
    """Ruling 4: this case is numerically fine and statistically useless.

    It must NOT raise from the arithmetic guard; ESS rejects it (Task 5).
    """
    m = np.array([40.0, 80.0, 150.0, 240.0])
    m0, sigma = 1000.0, 0.01
    direct = np.exp(-((np.log(m) - np.log(m0)) ** 2) / (2 * sigma**2)) / (
        m * sigma * np.sqrt(2 * np.pi)
    )
    assert direct.sum() == 0.0  # underflow under direct evaluation

    log_w = -((np.log(m) - np.log(m0)) ** 2) / (2 * sigma**2) - np.log(
        m * sigma * np.sqrt(2 * np.pi)
    )
    w = normalize_log_weights(log_w)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-12
    np.testing.assert_allclose(w, np.array([0.0, 0.0, 0.0, 1.0]))


def test_normalize_log_weights_rejects_all_neg_inf():
    with pytest.raises(ValueError, match="no finite"):
        normalize_log_weights(np.array([-np.inf, -np.inf]))


def test_normalize_log_weights_rejects_nan():
    with pytest.raises(ValueError, match="not finite"):
        normalize_log_weights(np.array([-1.0, np.nan]))


def test_normalize_log_weights_equal_under_large_offset():
    """A shared large offset must not swamp the ln(n) correction."""
    w = normalize_log_weights(np.array([-1e16, -1e16]))
    np.testing.assert_allclose(w, np.array([0.5, 0.5]))


def test_normalize_log_weights_unequal_under_large_offset():
    """Same as above, but the two weights differ by a finite amount."""
    w = normalize_log_weights(np.array([-1e16, -1e16 - 2]))
    np.testing.assert_allclose(w, np.array([0.88079708, 0.11920292]))


def test_normalize_log_weights_moderate_offset():
    """A less extreme offset that still loses precision unshifted.

    Direct logsumexp-and-subtract gives sum=1.0000319 here -- close
    enough that a sum-only tolerance test could miss it, so this checks
    the values.
    """
    w = normalize_log_weights(np.array([-1e12, -1e12]))
    np.testing.assert_allclose(w, np.array([0.5, 0.5]))


def test_normalize_log_weights_single_element():
    w = normalize_log_weights(np.array([-5.0]))
    np.testing.assert_allclose(w, np.array([1.0]))


def test_normalize_log_weights_rejects_empty():
    with pytest.raises(ValueError, match="no finite"):
        normalize_log_weights(np.array([]))
