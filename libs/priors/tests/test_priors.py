import numpy as np
from bilby.core.prior import Cosine, LogUniform, PriorDict, Sine, Uniform
from bilby.gw.prior import UniformComovingVolume

from priors.priors import ringdown_prior


RINGDOWN_PARAMETERS = {
    "frequency",
    "quality",
    "epsilon",
    "phase",
    "inclination",
    "distance",
    "dec",
    "ra",
    "psi",
}


def test_ringdown_prior_parameters():
    prior, detector_frame_prior = ringdown_prior()

    assert isinstance(prior, PriorDict)
    assert detector_frame_prior is True
    assert set(prior) == RINGDOWN_PARAMETERS
    assert "theta_jn" not in prior
    assert "luminosity_distance" not in prior


def test_ringdown_prior_distributions():
    prior, _ = ringdown_prior()

    assert isinstance(prior["frequency"], LogUniform)
    assert (prior["frequency"].minimum, prior["frequency"].maximum) == (
        40,
        1000,
    )
    assert isinstance(prior["quality"], Uniform)
    assert (prior["quality"].minimum, prior["quality"].maximum) == (2, 20)
    assert isinstance(prior["epsilon"], Uniform)
    assert (prior["epsilon"].minimum, prior["epsilon"].maximum) == (0, 0.1)
    assert isinstance(prior["phase"], Uniform)
    assert (prior["phase"].minimum, prior["phase"].maximum) == (
        0,
        2 * np.pi,
    )
    assert isinstance(prior["inclination"], Sine)
    assert isinstance(prior["distance"], UniformComovingVolume)
    assert (prior["distance"].minimum, prior["distance"].maximum) == (
        100,
        20000,
    )
    # Bilby requires this internal name, while PriorDict.sample() preserves
    # the dictionary key ``distance`` expected by ml4gw.
    assert prior["distance"].name == "luminosity_distance"
    assert isinstance(prior["dec"], Cosine)
    assert isinstance(prior["ra"], Uniform)
    assert (prior["ra"].minimum, prior["ra"].maximum) == (0, 2 * np.pi)
    assert isinstance(prior["psi"], Uniform)
    assert (prior["psi"].minimum, prior["psi"].maximum) == (0, np.pi)


def test_ringdown_prior_samples():
    prior, _ = ringdown_prior()
    num_samples = 128

    samples = prior.sample(num_samples)

    assert set(samples) == RINGDOWN_PARAMETERS
    for values in samples.values():
        assert values.shape == (num_samples,)
        assert np.isfinite(values).all()

    assert (
        (40 <= samples["frequency"]) & (samples["frequency"] <= 1000)
    ).all()
    assert ((2 <= samples["quality"]) & (samples["quality"] <= 20)).all()
    assert ((0 <= samples["epsilon"]) & (samples["epsilon"] <= 0.1)).all()
    assert ((0 <= samples["phase"]) & (samples["phase"] <= 2 * np.pi)).all()
    assert (
        (0 <= samples["inclination"]) & (samples["inclination"] <= np.pi)
    ).all()
    assert (
        (100 <= samples["distance"]) & (samples["distance"] <= 20000)
    ).all()
    assert (
        (-np.pi / 2 <= samples["dec"]) & (samples["dec"] <= np.pi / 2)
    ).all()
    assert ((0 <= samples["ra"]) & (samples["ra"] <= 2 * np.pi)).all()
    assert ((0 <= samples["psi"]) & (samples["psi"] <= np.pi)).all()


def test_log_normal_remnant_mass_keys():
    from priors.priors import log_normal_remnant_mass

    prior, detector_frame_prior = log_normal_remnant_mass(80.0, sigma=0.1)
    assert set(prior) == {"remnant_mass_source"}
    # Source-frame mass, so the detector-frame flag is False, matching
    # log_normal_masses.
    assert detector_frame_prior is False


def test_log_normal_remnant_mass_median_is_m0():
    from priors.priors import log_normal_remnant_mass

    prior, _ = log_normal_remnant_mass(80.0, sigma=0.1)
    samples = prior["remnant_mass_source"].sample(200000)
    # M0 is the MEDIAN of the mass distribution, not the mean, and sigma
    # is the standard deviation of log mass, not a width in Msun.
    assert abs(np.median(samples) / 80.0 - 1) < 0.01
    assert abs(np.std(np.log(samples)) / 0.1 - 1) < 0.02


def test_log_normal_remnant_mass_density_matches_closed_form():
    from priors.priors import log_normal_remnant_mass

    prior, _ = log_normal_remnant_mass(80.0, sigma=0.3)
    m = np.array([40.0, 80.0, 160.0])
    expected = np.exp(-((np.log(m) - np.log(80.0)) ** 2) / (2 * 0.3**2)) / (
        m * 0.3 * np.sqrt(2 * np.pi)
    )
    np.testing.assert_allclose(
        prior["remnant_mass_source"].prob(m), expected, rtol=1e-10
    )
