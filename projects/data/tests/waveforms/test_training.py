from unittest.mock import Mock, patch

import numpy as np
import pytest

from data.waveforms.training import training_waveforms
from ledger.injections import (
    BilbyParameterSet,
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


def test_ringdown_training_waveforms_hdf5_round_trip(tmp_path):
    waveforms = generate_training_waveforms()
    fname = tmp_path / "ringdown-training-waveforms.hdf5"
    waveforms.write(fname)

    loaded = RingdownWaveformPolarizationSet.read(fname)

    assert len(loaded) == len(waveforms)
    for name, field in waveforms.__dataclass_fields__.items():
        expected = getattr(waveforms, name)
        actual = getattr(loaded, name)
        if field.metadata["kind"] == "metadata":
            assert actual == expected
        else:
            np.testing.assert_array_equal(actual, expected)


def test_cbc_remains_default():
    samples = {
        name: np.ones(2)
        for name, field in BilbyParameterSet.__dataclass_fields__.items()
        if field.metadata["kind"] == "parameter"
    }
    prior = Mock()
    prior.sample.return_value = samples
    prior_factory = Mock(return_value=(prior, True))
    expected = Mock()

    with patch(
        "data.waveforms.training.WaveformPolarizationSet.from_parameters",
        return_value=expected,
    ) as generate:
        actual = training_waveforms(
            num_signals=2,
            sample_rate=SAMPLE_RATE,
            waveform_duration=WAVEFORM_DURATION,
            prior=prior_factory,
            minimum_frequency=20,
            reference_frequency=50,
            waveform_approximant="IMRPhenomXPHM",
            right_pad=RIGHT_PAD,
        )

    assert actual is expected
    generate.assert_called_once()


def test_rejects_unsupported_waveform_type():
    with pytest.raises(ValueError, match="waveform_type"):
        training_waveforms(
            num_signals=1,
            sample_rate=SAMPLE_RATE,
            waveform_duration=WAVEFORM_DURATION,
            prior=ringdown_prior,
            minimum_frequency=20,
            reference_frequency=50,
            waveform_approximant="IMRPhenomXPHM",
            right_pad=RIGHT_PAD,
            waveform_type="burst",
        )
