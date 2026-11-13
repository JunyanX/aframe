import numpy as np
from data.waveforms.training import training_waveforms
from ledger.injections import (
    RingdownWaveformPolarizationSet,
)
from priors.priors import ringdown_prior

SAMPLE_RATE = 2048
WAVEFORM_DURATION = 1
RIGHT_PAD = 0.5


def generate_training_waveforms(**kwargs):
    return training_waveforms(
        num_signals=4,
        sample_rate=SAMPLE_RATE,
        waveform_duration=WAVEFORM_DURATION,
        prior=ringdown_prior,
        minimum_frequency=20,
        reference_frequency=50,
        waveform_approximant="IMRPhenomXPHM",
        right_pad=RIGHT_PAD,
        waveform_type="ringdown",
        **kwargs,
    )


def test_generate_ringdown_training_waveforms():
    waveforms = generate_training_waveforms()
    waveform_size = int(SAMPLE_RATE * WAVEFORM_DURATION)
    onset = waveform_size - int(SAMPLE_RATE * RIGHT_PAD)

    assert isinstance(waveforms, RingdownWaveformPolarizationSet)
    assert len(waveforms) == waveforms.num_injections == 4
    assert waveforms.sample_rate == SAMPLE_RATE
    assert waveforms.duration == WAVEFORM_DURATION
    assert waveforms.right_pad == RIGHT_PAD
    assert waveforms.cross.shape == waveforms.plus.shape == (4, waveform_size)
    assert np.isfinite(waveforms.get_waveforms()).all()
    assert not waveforms.cross[:, :onset].any()
    assert not waveforms.plus[:, :onset].any()
