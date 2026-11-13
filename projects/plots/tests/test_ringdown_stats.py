import numpy as np
import pytest
from astropy.cosmology import Planck15
from bilby.core.prior import LogUniform, PriorDict, Uniform
from bilby.gw.prior import UniformComovingVolume
from plots.core.ringdown_stats import (
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
    """The weight is the closed form times a constant. Constancy is the
    assertion, not the value."""
    from priors.priors import log_normal_remnant_mass

    rng = np.random.default_rng(0)
    f = np.exp(rng.uniform(np.log(100), np.log(1000), 500))
    q = rng.uniform(8, 20, 500)
    d = rng.uniform(100, 1000, 500)
    from astropy.cosmology import z_at_value
    from astropy.units import Mpc

    z = z_at_value(Planck15.luminosity_distance, d * Mpc).value
    from ledger.injections import MSUN, C, G

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
    # The constant is ln(10) * C_z under the stated
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
    """This case is numerically fine and statistically useless.

    It must NOT raise from the arithmetic guard; the effective sample
    size check in `plots.ringdown.main` rejects it.
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


@pytest.mark.parametrize(
    ("log_w", "message"),
    [
        ([-1.0, np.nan], "not finite"),
        ([-np.inf, -np.inf], "no finite"),
        ([], "no finite"),
    ],
)
def test_normalize_log_weights_rejects_unusable_weights(log_w, message):
    with pytest.raises(ValueError, match=message):
        normalize_log_weights(np.array(log_w))


def test_normalize_log_weights_unequal_under_large_offset():
    """Same as above, but the two weights differ by a finite amount."""
    w = normalize_log_weights(np.array([-1e16, -1e16 - 2]))
    np.testing.assert_allclose(w, np.array([0.88079708, 0.11920292]))


def test_support_coverage_full_for_defaults(source_prior):
    from plots.core.ringdown_stats import support_coverage

    for m0 in (40.0, 80.0, 120.0, 150.0):
        assert support_coverage(m0, 0.1, source_prior, Planck15) > 0.9999


def test_support_coverage_reproduces_the_250_case(source_prior):
    """Coverage is 0.249103873 under the TARGET z measure."""
    from plots.core.ringdown_stats import support_coverage

    got = support_coverage(250.0, 0.1, source_prior, Planck15)
    assert abs(got - 0.249103873) < 1e-6
    got3 = support_coverage(250.0, 0.3, source_prior, Planck15)
    assert abs(got3 - 0.400856537) < 1e-6


def test_conditioning_shifts_marginals_above_the_warning_floor(source_prior):
    """Conditioning applies at any coverage < 1."""
    from plots.core.ringdown_stats import (
        conditioned_means,
        support_coverage,
    )

    cov = support_coverage(190.0, 0.1, source_prior, Planck15)
    assert abs(cov - 0.96035231) < 1e-6
    assert cov > 0.95  # above the default coverage_floor: no warning

    means = conditioned_means(190.0, 0.1, source_prior, Planck15)
    assert abs(means["quality"] - 14.11216) < 1e-4
    assert abs(means["redshift"] - 0.144986) < 1e-5


def test_conditioned_means_refuses_unrepresentable_support(source_prior):
    """Named for what actually happens: the interval probabilities
    underflow, not that the support is mathematically empty."""
    from plots.core.ringdown_stats import conditioned_means

    with pytest.raises(
        ValueError, match="no numerically representable probability"
    ):
        conditioned_means(1000.0, 0.01, source_prior, Planck15)


def test_effective_sample_size():
    from plots.core.ringdown_stats import effective_sample_size

    assert abs(effective_sample_size(np.full(100, 0.01)) - 100.0) < 1e-9
    assert (
        abs(effective_sample_size(np.array([0.0, 0.0, 0.0, 1.0])) - 1.0) < 1e-9
    )


_Z_GRID_N = 10001


@pytest.fixture(scope="module")
def z_of_distance(source_prior):
    """Interpolated luminosity-distance -> redshift, validated in place.

    `z_at_value` is a scalar root-find: 1000 inversions take about 4.7 s,
    so the 400k draws below would cost roughly half an hour. The grid is
    built forwards instead: redshifts between the two ends of the prior,
    and the distance of each, so that only the ends need a root-find.
    """
    from astropy.cosmology import z_at_value
    from astropy.units import Mpc

    distance = source_prior["distance"]
    ends = [distance.minimum, distance.maximum] * Mpc
    z_lo, z_hi = z_at_value(Planck15.luminosity_distance, ends).value
    z_grid = np.linspace(z_lo, z_hi, _Z_GRID_N)
    d_grid = Planck15.luminosity_distance(z_grid).to_value(Mpc)

    # Validate BETWEEN nodes, against direct inversion.
    check_d = 0.5 * (d_grid[:-1:100] + d_grid[1::100])
    check_z = z_at_value(Planck15.luminosity_distance, check_d * Mpc).value
    assert np.max(np.abs(np.interp(check_d, d_grid, z_grid) - check_z)) < 1e-7

    return lambda d: np.interp(d, d_grid, z_grid)


@pytest.fixture(scope="module")
def population_sample(source_prior, z_of_distance):
    """One set of source-prior draws."""
    from ledger.injections import MSUN, C, G

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


def test_weighted_population_matches_p_s(source_prior, population_sample):
    """The INDEPENDENT target-identity test.

    The quadrature tests above only compare `conditioned_means` with
    reference constants. This one compares the population the WEIGHTS
    actually induce against those means. Without it, nothing checks that
    `log_importance_weights` and `conditioned_means` describe the same
    population.

    Coverage at M0=250 is 0.249103873, far below the default warning
    floor, so conditioning moves the means a long way. p_S is the oracle
    whatever the coverage -- the floor is reporting only.
    """
    m0 = 250.0
    from plots.core.ringdown_stats import (
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


def test_variance_includes_rejected_misses():
    """A rejected draw is never detected but keeps its weight, so it adds
    w^2 mu^2 to the variance. `plots.ringdown.main` passes those draws to
    the shared routine through `missed_w2`."""
    from plots.core.compute import sensitive_volume

    # one recovered and one rejected draw, each of weight 0.5
    mu, err = sensitive_volume(
        np.array([1.0]),
        np.array([[0.5]]),
        np.array([0.5]),
        missed_w2=np.array([0.25]),
    )
    assert abs(mu[0, 0] - 0.5) < 1e-12
    assert abs(err[0, 0] - 0.35355339) < 1e-7


def test_time_window_miss_keeps_its_weight():
    """A recovery outside the dt window is a miss carrying its original
    weight, not a zeroed weight. `plots.ringdown.main` gives it a
    statistic of -inf."""
    from plots.core.compute import sensitive_volume

    statistic = np.array([1.0, -np.inf])  # second is outside dt
    weights = np.array([[0.5, 0.5]])
    mu, err = sensitive_volume(statistic, weights, np.array([0.5]))
    assert abs(mu[0, 0] - 0.5) < 1e-12
    assert abs(err[0, 0] - 0.35355339) < 1e-7


def test_far_grid_lengths_match():
    from plots.core.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    bg = np.arange(1000, dtype=float)
    fars, thresholds = far_threshold_grid(bg, SECONDS_PER_YEAR, 365)
    assert len(fars) == len(thresholds) == 365
    assert thresholds[0] > thresholds[-1]  # descending


def test_far_grid_caps_at_available_background():
    """A cap that admits more events than the background holds."""
    from plots.core.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    bg = np.arange(10, dtype=float)
    fars, thresholds = far_threshold_grid(bg, SECONDS_PER_YEAR, 365)
    assert len(fars) == len(thresholds) == 10


def test_far_grid_rejects_short_livetime():
    """A livetime too short for any event's rate to be within the cap."""
    from plots.core.ringdown_stats import far_threshold_grid

    with pytest.raises(ValueError, match="no background rank"):
        far_threshold_grid(np.arange(1744, dtype=float), 13134.0, 365)


def test_far_grid_rejects_bad_livetime():
    from plots.core.ringdown_stats import far_threshold_grid

    bg = np.arange(100, dtype=float)
    with pytest.raises(ValueError, match="livetime"):
        far_threshold_grid(bg, 0.0, 365)
    with pytest.raises(ValueError, match="livetime"):
        far_threshold_grid(bg, -1.0, 365)
    with pytest.raises(ValueError, match="livetime"):
        far_threshold_grid(bg, np.inf, 365)
    with pytest.raises(ValueError, match="livetime"):
        far_threshold_grid(bg, np.nan, 365)


def test_far_grid_rejects_empty_background():
    from plots.core.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    with pytest.raises(ValueError, match="empty background"):
        far_threshold_grid(np.array([]), SECONDS_PER_YEAR, 365)


def test_far_grid_credits_tie_group_with_true_exceedance_rate():
    """A tie group must be reported once, at the rate of everything
    `sensitive_volume`'s `statistic >= threshold` would actually admit --
    not once per member at an understated rank-based rate."""
    from plots.core.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    bg = np.array([9.0, 9.0, 8.0])
    fars, thresholds = far_threshold_grid(bg, SECONDS_PER_YEAR, 3)
    # A list of (threshold, far) pairs, not a dict: a dict keyed on
    # threshold would silently collapse a rank-based result's duplicate
    # 9.0 entries down to whichever one is listed last, which can mask
    # exactly the bug this test exists to catch.
    pairs = list(zip(thresholds.tolist(), fars.tolist(), strict=True))
    assert pairs == [(9.0, 2.0), (8.0, 3.0)]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_far_grid_rejects_non_finite_background(bad):
    """NaN sorts ABOVE every finite value, so it would become the top
    threshold at the lowest FAR while `statistic >= nan` is false for
    every event. Without the check:

        bg = [1.0, 2.0, nan] -> thresholds [nan, 2, 1] at FARs [1, 2, 3]
                                actual inclusive counts:   [0, 1, 2]

    Every finite threshold's rate would be wrong too, not only the NaN's,
    and nothing downstream would notice.

    +inf and -inf are rejected as well, and none of the three is silently
    dropped: dropping would leave Tb describing a livetime the surviving
    sample no longer covers, understating every FAR.
    """
    from plots.core.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    with pytest.raises(ValueError, match="1 non-finite entries"):
        far_threshold_grid(np.array([1.0, 2.0, bad]), SECONDS_PER_YEAR, 3)


# tb_years * (1/tb_years) rounds to 0.9999999999999999 at this livetime,
# which is what makes the case below discriminate between
# bounds that differ only at a floating-point boundary.
_TB_BOUNDARY = 100006.0


def test_far_grid_three_event_boundary_exercises_the_bound():
    """The number of loudest events to look at, at a floating-point
    boundary.

    The cap is exactly the rate of two events, but
    `p = max_far * tb_years` comes out as 1.9999999999999998. Looking at
    `floor(p)` events would find one and lose the 8.0 threshold.
    """
    from plots.core.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    tb_years = _TB_BOUNDARY / SECONDS_PER_YEAR
    fars, thresholds = far_threshold_grid(
        np.array([7.0, 8.0, 9.0]), _TB_BOUNDARY, 2 / tb_years
    )
    np.testing.assert_array_equal(thresholds, np.array([9.0, 8.0]))
    assert abs(fars[0] - 315.55058696) < 1e-7
    assert abs(fars[1] - 631.10117393) < 1e-7


def test_far_grid_accepts_caps_that_overflow_a_naive_bound():
    """An infinite cap means no cap, and a huge one must not overflow.

    `int(np.ceil(max_far * tb_years))` raises OverflowError on both.
    """
    from plots.core.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    fars, thresholds = far_threshold_grid(
        np.array([9.0]), SECONDS_PER_YEAR, np.inf
    )
    np.testing.assert_array_equal(fars, np.array([1.0]))
    np.testing.assert_array_equal(thresholds, np.array([9.0]))

    fars, thresholds = far_threshold_grid(
        np.array([9.0]), 2 * SECONDS_PER_YEAR, 1e308
    )
    np.testing.assert_array_equal(fars, np.array([0.5]))
    np.testing.assert_array_equal(thresholds, np.array([9.0]))


@pytest.mark.parametrize("max_far", [np.nan, 0.0, -1.0])
def test_far_grid_rejects_a_cap_that_is_not_positive(max_far):
    from plots.core.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    with pytest.raises(ValueError, match="max_far must be positive"):
        far_threshold_grid(
            np.arange(100, dtype=float), SECONDS_PER_YEAR, max_far
        )
