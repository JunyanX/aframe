import numpy as np
import pytest
import torch
from data.waveforms.rejection import rejection_sample
from ledger.injections import (
    RingdownInjectionParameterSet,
)
from priors.priors import ringdown_prior

IFOS = ["H1", "L1"]
# ringdown_prior draws frequency up to 1 kHz, so the sample rate has to keep
# the whole prior below Nyquist or generate_ringdown rejects the draw
SAMPLE_RATE = 2048
WAVEFORM_DURATION = 2
RIGHT_PAD = 1
NUM_SIGNALS = 4
SNR_THRESHOLD = 4.0


@pytest.fixture
def psd():
    """Flat PSD at roughly aLIGO's design amplitude, on the 1/duration grid."""
    num_freqs = int(SAMPLE_RATE * WAVEFORM_DURATION) // 2 + 1
    return 1e-46 * torch.ones((len(IFOS), num_freqs), dtype=torch.float64)


def sample_ringdown(psd):
    return rejection_sample(
        num_signals=NUM_SIGNALS,
        prior=ringdown_prior,
        ifos=IFOS,
        minimum_frequency=20,
        reference_frequency=50,
        sample_rate=SAMPLE_RATE,
        waveform_duration=WAVEFORM_DURATION,
        waveform_approximant="IMRPhenomXPHM",
        right_pad=RIGHT_PAD,
        highpass=32,
        lowpass=None,
        snr_threshold=SNR_THRESHOLD,
        psd=psd,
        max_num_samples=32,
        waveform_type="ringdown",
    )


def test_ringdown_rejection_sample_returns_projected_strain(psd):
    np.random.seed(0)
    torch.manual_seed(0)

    parameters, _ = sample_ringdown(psd)

    waveform_size = int(SAMPLE_RATE * WAVEFORM_DURATION)
    # validation waveforms are stored per interferometer, already projected
    for ifo in IFOS:
        strain = parameters[ifo.lower()]
        assert strain.shape == (NUM_SIGNALS, waveform_size)
        assert np.isfinite(strain).all()
        assert strain.any()

    assert parameters["snr"].shape == (NUM_SIGNALS,)
    assert parameters["ifo_snrs"].shape == (NUM_SIGNALS, len(IFOS))
    assert (parameters["snr"] >= SNR_THRESHOLD).all()

    assert parameters["sample_rate"] == SAMPLE_RATE
    assert parameters["duration"] == WAVEFORM_DURATION
    assert parameters["right_pad"] == RIGHT_PAD
    assert parameters["ifos"] == IFOS
    assert parameters["num_injections"] >= NUM_SIGNALS


def test_ringdown_rejection_sample_records_ringdown_parameters(psd):
    np.random.seed(1)
    torch.manual_seed(1)

    parameters, rejected = sample_ringdown(psd)

    # accepted parameters are the ones the ringdown prior drew
    for name in ("frequency", "quality", "epsilon", "inclination"):
        assert parameters[name].shape == (NUM_SIGNALS,)

    # rejected parameters use the ringdown ledger class, not the CBC one
    assert isinstance(rejected, RingdownInjectionParameterSet)
    assert "mass_1" not in rejected.__dataclass_fields__
