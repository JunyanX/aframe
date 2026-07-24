import numpy as np
import pytest
import torch
from ml4gw.waveforms.adhoc import Ringdown

from data.waveforms.ringdown import generate_ringdown


SAMPLE_RATE = 2048
WAVEFORM_DURATION = 1
RIGHT_PAD = 0.5


@pytest.fixture
def samples():
    return {
        "frequency": np.array([200, 400]),
        "quality": np.array([10, 15]),
        "epsilon": np.array([0.01, 0.02]),
        "phase": np.array([0, np.pi / 4]),
        "inclination": np.array([np.pi / 3, np.pi / 2]),
        "distance": np.array([100, 200]),
        "ra": np.array([0.1, 0.2]),
        "dec": np.array([-0.1, 0.1]),
        "psi": np.array([0.3, 0.4]),
    }


def test_generate_ringdown_is_aligned_with_ml4gw(samples):
    cross, plus = generate_ringdown(
        samples,
        sample_rate=SAMPLE_RATE,
        waveform_duration=WAVEFORM_DURATION,
        right_pad=RIGHT_PAD,
    )

    waveform_size = int(SAMPLE_RATE * WAVEFORM_DURATION)
    ringdown_size = int(SAMPLE_RATE * RIGHT_PAD)
    onset = waveform_size - ringdown_size

    assert cross.shape == plus.shape == (2, waveform_size)
    assert cross.dtype == plus.dtype == torch.float64
    assert torch.isfinite(cross).all()
    assert torch.isfinite(plus).all()
    assert torch.count_nonzero(cross[:, :onset]) == 0
    assert torch.count_nonzero(plus[:, :onset]) == 0

    generator = Ringdown(SAMPLE_RATE, RIGHT_PAD)
    parameters = {
        name: torch.as_tensor(samples[name], dtype=torch.float64)
        for name in (
            "frequency",
            "quality",
            "epsilon",
            "phase",
            "inclination",
            "distance",
        )
    }
    expected_cross, expected_plus = generator(**parameters)

    torch.testing.assert_close(cross[:, onset:], expected_cross)
    torch.testing.assert_close(plus[:, onset:], expected_plus)


def test_generate_ringdown_accepts_tensor_inputs(samples):
    tensor_samples = {
        key: torch.as_tensor(value) for key, value in samples.items()
    }

    cross, plus = generate_ringdown(
        tensor_samples,
        sample_rate=SAMPLE_RATE,
        waveform_duration=WAVEFORM_DURATION,
        right_pad=RIGHT_PAD,
    )

    assert cross.shape == plus.shape == (2, SAMPLE_RATE)


@pytest.mark.parametrize(
    ("parameter", "value", "error"),
    [
        ("frequency", np.array([0, 200]), "frequency must be positive"),
        (
            "frequency",
            np.array([1024, 200]),
            "frequency must be below the Nyquist frequency",
        ),
        ("quality", np.array([0, 10]), "quality must be positive"),
        ("epsilon", np.array([-0.1, 0.1]), "epsilon must be nonnegative"),
        ("distance", np.array([0, 100]), "distance must be positive"),
        ("phase", np.array([np.nan, 0]), "must contain finite values"),
    ],
)
def test_generate_ringdown_rejects_invalid_parameters(
    samples, parameter, value, error
):
    samples[parameter] = value

    with pytest.raises(ValueError, match=error):
        generate_ringdown(
            samples,
            sample_rate=SAMPLE_RATE,
            waveform_duration=WAVEFORM_DURATION,
            right_pad=RIGHT_PAD,
        )


def test_generate_ringdown_requires_all_parameters(samples):
    del samples["frequency"]

    with pytest.raises(KeyError, match="frequency"):
        generate_ringdown(
            samples,
            sample_rate=SAMPLE_RATE,
            waveform_duration=WAVEFORM_DURATION,
            right_pad=RIGHT_PAD,
        )


def test_generate_ringdown_requires_equal_batch_sizes(samples):
    samples["quality"] = np.array([10])

    with pytest.raises(ValueError, match="equal batch sizes"):
        generate_ringdown(
            samples,
            sample_rate=SAMPLE_RATE,
            waveform_duration=WAVEFORM_DURATION,
            right_pad=RIGHT_PAD,
        )


@pytest.mark.parametrize(
    ("sample_rate", "waveform_duration", "right_pad", "error"),
    [
        (0, 1, 0.5, "sample_rate"),
        (2048, 0, 0.5, "waveform_duration"),
        (2048, 1, 0, "right_pad"),
        (2048, 1, 2, "right_pad"),
        (1, 1, 0.5, "at least one sample"),
    ],
)
def test_generate_ringdown_rejects_invalid_generation_settings(
    samples, sample_rate, waveform_duration, right_pad, error
):
    with pytest.raises(ValueError, match=error):
        generate_ringdown(
            samples,
            sample_rate=sample_rate,
            waveform_duration=waveform_duration,
            right_pad=right_pad,
        )
