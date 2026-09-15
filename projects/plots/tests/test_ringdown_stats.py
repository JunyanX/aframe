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


def test_support_coverage_full_for_defaults(source_prior):
    from plots.legacy.ringdown_stats import support_coverage

    for m0 in (40.0, 80.0, 120.0, 150.0):
        assert support_coverage(m0, 0.1, source_prior, Planck15) > 0.9999


def test_support_coverage_reproduces_the_250_case(source_prior):
    """Design Established facts: 0.249103873 under the TARGET z measure."""
    from plots.legacy.ringdown_stats import support_coverage

    got = support_coverage(250.0, 0.1, source_prior, Planck15)
    assert abs(got - 0.249103873) < 1e-6
    got3 = support_coverage(250.0, 0.3, source_prior, Planck15)
    assert abs(got3 - 0.400856537) < 1e-6


def test_conditioning_shifts_marginals_above_the_warning_floor(source_prior):
    """Ruling 3(c): conditioning applies at any coverage < 1."""
    from plots.legacy.ringdown_stats import (
        conditioned_means,
        support_coverage,
    )

    cov = support_coverage(190.0, 0.1, source_prior, Planck15)
    assert abs(cov - 0.96035231) < 1e-6
    assert cov > 0.95  # above the default coverage_floor: no warning

    means = conditioned_means(190.0, 0.1, source_prior, Planck15)
    assert abs(means["quality"] - 14.11216) < 1e-4
    assert abs(means["redshift"] - 0.144986) < 1e-5


def test_conditioning_shifts_marginals_far_out(source_prior):
    from plots.legacy.ringdown_stats import conditioned_means

    means = conditioned_means(250.0, 0.1, source_prior, Planck15)
    assert abs(means["quality"] - 15.42325) < 1e-4
    assert abs(means["redshift"] - 0.129120) < 1e-5


def test_coverage_survives_the_upper_tail(source_prior):
    """Finite weights and a passing ESS, but naive CDF subtraction loses
    the interval probability entirely."""
    from scipy.stats import norm

    from plots.legacy.ringdown_stats import support_coverage

    lo = (np.log(24.0) - np.log(1.0)) / 0.3
    hi = (np.log(273.0) - np.log(1.0)) / 0.3
    assert norm.cdf(hi) - norm.cdf(lo) == 0.0  # the naive form
    assert norm.sf(lo) - norm.sf(hi) > 0.0  # the stable one

    cov = support_coverage(1.0, 0.3, source_prior, Planck15)
    assert np.isfinite(cov)
    assert cov > 0.0


def test_conditioned_means_refuses_unrepresentable_support(source_prior):
    """Named for what actually happens: the interval probabilities
    underflow, not that the support is mathematically empty."""
    from plots.legacy.ringdown_stats import conditioned_means

    with pytest.raises(
        ValueError, match="no numerically representable probability"
    ):
        conditioned_means(1000.0, 0.01, source_prior, Planck15)


def test_effective_sample_size():
    from plots.legacy.ringdown_stats import effective_sample_size

    assert abs(effective_sample_size(np.full(100, 0.01)) - 100.0) < 1e-9
    assert (
        abs(effective_sample_size(np.array([0.0, 0.0, 0.0, 1.0])) - 1.0) < 1e-9
    )


_Z_GRID_N = 10001


@pytest.fixture(scope="module")
def z_of_distance(source_prior):
    """Interpolated luminosity-distance -> redshift, validated in place.

    `z_at_value` is a scalar root-find: 1000 inversions take about 4.7 s,
    so the 400k draws below would cost roughly half an hour per target.
    An interpolation over the prior's distance range is exact enough --
    validated here against direct inversion on a coarse subsample, max
    absolute difference of order 1e-9.
    """
    from astropy.cosmology import z_at_value
    from astropy.units import Mpc

    d_lo = source_prior["distance"].minimum
    d_hi = source_prior["distance"].maximum
    d_grid = np.linspace(d_lo, d_hi, _Z_GRID_N)
    z_grid = z_at_value(Planck15.luminosity_distance, d_grid * Mpc).value

    # Validate BETWEEN nodes. np.linspace(d_lo, d_hi, 101) would be
    # exactly d_grid[::100] -- every check point a stored node, which
    # tests lookup rather than interpolation error.
    check_d = 0.5 * (d_grid[:-1:100] + d_grid[1::100])
    check_z = z_at_value(Planck15.luminosity_distance, check_d * Mpc).value
    assert np.max(np.abs(np.interp(check_d, d_grid, z_grid) - check_z)) < 1e-7

    return lambda d: np.interp(d, d_grid, z_grid)


@pytest.fixture(scope="module")
def population_sample(source_prior, z_of_distance):
    """One set of source-prior draws, shared across target masses."""
    from ledger.injections import C, G, MSUN

    rng = np.random.default_rng(3)
    n = 400000
    f = np.exp(rng.uniform(np.log(100), np.log(1000), n))
    q = rng.uniform(8, 20, n)
    # `.rescale` maps uniforms through the prior's inverse CDF, so the
    # seeded generator drives the distance draw too. `.sample(n)` would
    # use bilby's own RNG and make this fixture irreproducible.
    d = np.asarray(source_prior["distance"].rescale(rng.random(n)))
    z = z_of_distance(d)
    m_src = (
        (1 / (2 * np.pi))
        * (C**3 / (G * f))
        * (1 - 0.63 * (2 / q) ** (2 / 3))
        / MSUN
    ) / (1 + z)
    return f, q, z, m_src


@pytest.mark.parametrize("m0", [190.0, 250.0])
def test_weighted_population_matches_p_s(source_prior, population_sample, m0):
    """Design Ruling 3(c): the INDEPENDENT target-identity test.

    The quadrature tests above only compare `conditioned_means` with
    reference constants. This one compares the population the WEIGHTS
    actually induce against those means. Without it, nothing checks that
    `log_importance_weights` and `conditioned_means` describe the same
    population.

    Both parameters straddle the default warning floor: coverage is
    0.96035231 at M0=190 (no warning) and 0.249103873 at M0=250
    (warning). p_S is the oracle for both -- the floor is reporting only.
    """
    from plots.legacy.ringdown_stats import (
        conditioned_means,
        log_importance_weights,
        normalize_log_weights,
        redshift_bounds,
        redshift_ratio_norm,
    )
    from priors.priors import log_normal_remnant_mass

    f, q, z, m_src = population_sample
    target, _ = log_normal_remnant_mass(m0, sigma=0.1)
    zmin, zmax = redshift_bounds(source_prior, Planck15)
    c_z = redshift_ratio_norm(zmin, zmax, Planck15)
    w = normalize_log_weights(
        log_importance_weights(m_src, f, z, source_prior, target, c_z)
    )

    expected = conditioned_means(m0, 0.1, source_prior, Planck15)
    got_q = float((w * q).sum())
    got_z = float((w * z).sum())
    # Monte Carlo tolerance. If this proves too tight in practice, raise n
    # rather than loosening it: a loose tolerance would let the two code
    # paths describe different populations undetected.
    assert abs(got_q - expected["quality"]) < 0.10, (got_q, expected)
    assert abs(got_z - expected["redshift"]) < 0.003, (got_z, expected)
