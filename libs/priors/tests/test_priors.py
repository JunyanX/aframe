import numpy as np
from bilby.core.prior import Cosine, LogUniform, PriorDict, Sine, Uniform
from bilby.gw.prior import UniformComovingVolume
from priors.priors import ringdown_prior

# The distribution of each parameter, not its bounds, which are tuned from
# one campaign to the next. The distributions are what the ringdown
# sensitive volume takes for granted. Its weight is written for injections
# uniform in comoving volume, and it isn't reweighted in sky position or
# orientation, so those have to be isotropic.
RINGDOWN_DISTRIBUTIONS = {
    "frequency": LogUniform,
    "quality": Uniform,
    "epsilon": Uniform,
    "phase": Uniform,
    "inclination": Sine,
    "distance": UniformComovingVolume,
    "dec": Cosine,
    "ra": Uniform,
    "psi": Uniform,
}


def test_ringdown_prior_parameters():
    prior, detector_frame_prior = ringdown_prior()

    assert isinstance(prior, PriorDict)
    assert detector_frame_prior is True
    assert {k: type(v) for k, v in prior.items()} == RINGDOWN_DISTRIBUTIONS
    # Bilby requires this internal name, while PriorDict.sample() preserves
    # the dictionary key ``distance`` expected by ml4gw.
    assert prior["distance"].name == "luminosity_distance"


def test_log_normal_remnant_mass_keys():
    from priors.priors import log_normal_remnant_mass

    prior, detector_frame_prior = log_normal_remnant_mass(80.0, sigma=0.1)
    assert set(prior) == {"remnant_mass_source"}
    # Source-frame mass, so the detector-frame flag is False, as in
    # end_o3_ratesandpops.
    assert detector_frame_prior is False


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
