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


def test_variance_includes_rejected_misses():
    """Design Established facts: compute.py gives 0.25 here; 0.35355339
    is right, because the rejected draw contributes w^2 mu^2."""
    from plots.legacy.ringdown_stats import sensitive_volume

    statistic = np.array([1.0, -np.inf])
    detected = np.array([True, False])
    weights = np.array([[0.5, 0.5]])
    mu, err = sensitive_volume(statistic, detected, weights, np.array([0.5]))
    assert abs(mu[0, 0] - 0.5) < 1e-12
    assert abs(err[0, 0] - 0.35355339) < 1e-7


def test_time_window_miss_keeps_its_weight():
    """A recovered-but-out-of-window injection is a miss carrying its
    original weight, not a zeroed weight."""
    from plots.legacy.ringdown_stats import sensitive_volume

    statistic = np.array([1.0, 1.0])
    detected = np.array([True, False])  # second is outside dt
    weights = np.array([[0.5, 0.5]])
    mu, err = sensitive_volume(statistic, detected, weights, np.array([0.5]))
    assert abs(mu[0, 0] - 0.5) < 1e-12
    assert abs(err[0, 0] - 0.35355339) < 1e-7


def test_all_misses_gives_zero_curve_not_an_error():
    from plots.legacy.ringdown_stats import sensitive_volume

    statistic = np.array([-np.inf, -np.inf])
    detected = np.array([False, False])
    weights = np.array([[0.5, 0.5]])
    mu, err = sensitive_volume(statistic, detected, weights, np.array([0.5]))
    assert mu[0, 0] == 0.0
    assert err[0, 0] == 0.0


def test_uncertainty_is_calibrated_under_the_stopping_rule():
    """Ruling 6 and review r5: an ESS threshold does not establish
    calibration.

    The reference is the FIXED population expectation
    `p_accept * p_detect`, not the realized acceptance fraction. Comparing
    against `p_detect * accepted / n` would divide out exactly the
    stopping-time variability this test exists to probe, and would pass
    even if the estimator ignored it.
    """
    from plots.legacy.ringdown_stats import sensitive_volume

    rng = np.random.default_rng(7)
    p_accept, p_detect, n_required = 0.4, 0.3, 200
    truth = p_accept * p_detect  # 0.12, fixed, independent of the draw
    covered, trials = 0, 2000
    for _ in range(trials):
        accepted, rejected = 0, 0
        while accepted < n_required:
            if rng.random() < p_accept:
                accepted += 1
            else:
                rejected += 1
        n = accepted + rejected
        detected = np.zeros(n, dtype=bool)
        detected[:accepted] = rng.random(accepted) < p_detect
        statistic = np.where(detected, 1.0, -np.inf)
        weights = np.full((1, n), 1.0 / n)
        mu, err = sensitive_volume(
            statistic, detected, weights, np.array([0.5])
        )
        if abs(mu[0, 0] - truth) <= err[0, 0]:
            covered += 1
    rate = covered / trials
    assert 0.60 < rate < 0.85, f"coverage rate {rate}"


def test_uncertainty_is_calibrated_with_heterogeneous_weights():
    """Uniform weights leave importance-weight heterogeneity untested.

    Here the weight and the detection probability both depend on a
    sampled parameter, so the weight vector is genuinely uneven — the
    regime the real campaign is in.
    """
    from plots.legacy.ringdown_stats import sensitive_volume

    rng = np.random.default_rng(11)
    n_draws, trials = 600, 1000

    def p_detect_of(x):
        return 0.2 + 0.5 * x

    # Population truth: the w-weighted mean detection probability.
    grid = np.linspace(0, 1, 20001)
    w_grid = 0.5 + grid
    truth = np.trapz(w_grid * p_detect_of(grid), grid) / np.trapz(w_grid, grid)

    covered = 0
    for _ in range(trials):
        x = rng.random(n_draws)
        detected = rng.random(n_draws) < p_detect_of(x)
        statistic = np.where(detected, 1.0, -np.inf)
        w = 0.5 + x
        w = w / w.sum()
        mu, err = sensitive_volume(
            statistic, detected, w[None, :], np.array([0.5])
        )
        if abs(mu[0, 0] - truth) <= err[0, 0]:
            covered += 1
    rate = covered / trials
    assert 0.55 < rate < 0.90, f"weighted coverage rate {rate}"


def test_far_grid_lengths_match():
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    bg = np.arange(1000, dtype=float)
    fars, thresholds = far_threshold_grid(bg, SECONDS_PER_YEAR, 365)
    assert len(fars) == len(thresholds) == 365
    assert thresholds[0] > thresholds[-1]  # descending


def test_far_grid_caps_at_available_background():
    """max_events > len(background): CBC silently returns 365 FARs
    against 10 thresholds."""
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    bg = np.arange(10, dtype=float)
    fars, thresholds = far_threshold_grid(bg, SECONDS_PER_YEAR, 365)
    assert len(fars) == len(thresholds) == 10


def test_far_grid_rejects_short_livetime():
    """max_events == 0: CBC returns an empty FAR axis against the whole
    threshold array, because [-0:] is the full slice."""
    from plots.legacy.ringdown_stats import far_threshold_grid

    with pytest.raises(ValueError, match="no background rank"):
        far_threshold_grid(np.arange(1744, dtype=float), 13134.0, 365)


def test_far_grid_rejects_bad_livetime():
    from plots.legacy.ringdown_stats import far_threshold_grid

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
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    with pytest.raises(ValueError, match="empty background"):
        far_threshold_grid(np.array([]), SECONDS_PER_YEAR, 365)


def test_far_grid_credits_tie_group_with_true_exceedance_rate():
    """A tie group must be reported once, at the rate of everything
    `sensitive_volume`'s `statistic >= threshold` would actually admit --
    not once per member at an understated rank-based rate."""
    from plots.legacy.ringdown_stats import (
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


def test_far_grid_rejects_tie_group_whose_true_rate_exceeds_cap():
    """The tie at 9.0 has a true rate of 2/yr. At max_far=1 that must
    raise, not silently advertise the rank-based rate of 1/yr."""
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    bg = np.array([9.0, 9.0, 8.0])
    with pytest.raises(ValueError, match="no background rank"):
        far_threshold_grid(bg, SECONDS_PER_YEAR, 1)


def test_far_grid_rejects_nan_background():
    """NaN sorts ABOVE every finite value, so it becomes the top
    threshold at the lowest FAR while `statistic >= nan` is false for
    every event. Reproduced against the committed helper:

        bg = [1.0, 2.0, nan] -> thresholds [nan, 2, 1] at FARs [1, 2, 3]
                                actual inclusive counts:   [0, 1, 2]

    Every finite threshold's rate was wrong too, not only the NaN's, and
    the run still exited 0 and wrote both output files. The raise has to
    come before any of that.
    """
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    with pytest.raises(ValueError, match="1 non-finite entries"):
        far_threshold_grid(np.array([1.0, 2.0, np.nan]), SECONDS_PER_YEAR, 3)


def test_far_grid_rejects_infinite_background_entries():
    """+inf and -inf are rejected as well, and none of the three is
    silently dropped: dropping would leave Tb describing a livetime the
    surviving sample no longer covers, understating every FAR."""
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    for bad in (np.inf, -np.inf):
        with pytest.raises(ValueError, match="non-finite"):
            far_threshold_grid(np.array([1.0, 2.0, bad]), SECONDS_PER_YEAR, 3)


def test_far_grid_non_finite_message_counts_every_offender():
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    bg = np.array([1.0, np.nan, 2.0, np.inf, -np.inf])
    with pytest.raises(ValueError, match="3 non-finite entries.* out of 5"):
        far_threshold_grid(bg, SECONDS_PER_YEAR, 3)


# tb_years * (1/tb_years) rounds to 0.9999999999999999 at this livetime,
# which is what makes the three cases below discriminate between
# candidate bounds that differ only at a floating-point boundary.
_TB_BOUNDARY = 100006.0


def test_far_grid_single_event_boundary():
    """One event, cap exactly its own rate.

    A `floor`-based candidate bound gives k=0 here -- and `s[-0:]` is the
    WHOLE array, the same slicing trap Task 7 exists for, so the wrong
    bound still answers correctly. A boundary regression, NOT the
    mutation gate.
    """
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    tb_years = _TB_BOUNDARY / SECONDS_PER_YEAR
    fars, thresholds = far_threshold_grid(
        np.array([9.0]), _TB_BOUNDARY, 1 / tb_years
    )
    np.testing.assert_array_equal(thresholds, np.array([9.0]))
    assert abs(fars[0] - 315.55706658) < 1e-7


def test_far_grid_two_event_boundary():
    """Two events, cap exactly the full-background rate.

    The cap admits the whole background, so the full-background branch
    short-circuits the candidate bound and a wrong bound is never
    evaluated. Also a boundary regression, also not the gate.
    """
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    tb_years = _TB_BOUNDARY / SECONDS_PER_YEAR
    fars, thresholds = far_threshold_grid(
        np.array([8.0, 9.0]), _TB_BOUNDARY, 2 / tb_years
    )
    np.testing.assert_array_equal(thresholds, np.array([9.0, 8.0]))
    assert abs(fars[-1] - 631.11413315) < 1e-7


def test_far_grid_three_event_boundary_exercises_the_bound():
    """THE mutation gate for the candidate bound.

    The full-background rate is 946.67/yr against a cap of 631.11/yr, so
    the full-background branch does not fire and the bound is genuinely
    evaluated. Re-derived against this implementation:

        [9.0]           committed [9.]     here [9.]     floor [9.]
        [8.0, 9.0]      committed [9. 8.]  here [9. 8.]  floor [9. 8.]
        [7.0, 8.0, 9.0] committed [9. 8.]  here [9. 8.]  floor [9.]

    `p = max_far * tb_years` is 1.9999999999999998 in the third case, so
    a `floor` bound takes k=1 and loses the 8.0 threshold.
    """
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    tb_years = _TB_BOUNDARY / SECONDS_PER_YEAR
    fars, thresholds = far_threshold_grid(
        np.array([7.0, 8.0, 9.0]), _TB_BOUNDARY, 2 / tb_years
    )
    np.testing.assert_array_equal(thresholds, np.array([9.0, 8.0]))
    assert abs(fars[0] - 315.55706658) < 1e-7
    assert abs(fars[1] - 631.11413315) < 1e-7


def test_far_grid_accepts_caps_that_overflow_a_naive_bound():
    """Caps the committed helper accepts and must keep accepting.

    `int(np.ceil(max_far * tb_years))` raises OverflowError on both.
    """
    from plots.legacy.ringdown_stats import (
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


def test_far_grid_nan_cap_keeps_the_committed_message():
    """The full-background branch is written as `not (... > ...)` so a
    NaN cap lands in it. Deleting that branch makes this die in
    `int(nan)` instead of raising the documented message."""
    from plots.legacy.ringdown_stats import (
        SECONDS_PER_YEAR,
        far_threshold_grid,
    )

    with pytest.raises(ValueError, match="no background rank"):
        far_threshold_grid(
            np.arange(100, dtype=float), SECONDS_PER_YEAR, np.nan
        )
