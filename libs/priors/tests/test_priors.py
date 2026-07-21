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
        100,
        1000,
    )
    assert isinstance(prior["quality"], Uniform)
    assert (prior["quality"].minimum, prior["quality"].maximum) == (8, 20)
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
        1000,
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
        (100 <= samples["frequency"]) & (samples["frequency"] <= 1000)
    ).all()
    assert ((8 <= samples["quality"]) & (samples["quality"] <= 20)).all()
    assert ((0 <= samples["epsilon"]) & (samples["epsilon"] <= 0.1)).all()
    assert ((0 <= samples["phase"]) & (samples["phase"] <= 2 * np.pi)).all()
    assert (
        (0 <= samples["inclination"]) & (samples["inclination"] <= np.pi)
    ).all()
    assert ((100 <= samples["distance"]) & (samples["distance"] <= 1000)).all()
    assert (
        (-np.pi / 2 <= samples["dec"]) & (samples["dec"] <= np.pi / 2)
    ).all()
    assert ((0 <= samples["ra"]) & (samples["ra"] <= 2 * np.pi)).all()
    assert ((0 <= samples["psi"]) & (samples["psi"] <= np.pi)).all()
