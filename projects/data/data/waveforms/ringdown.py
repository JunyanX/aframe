from collections.abc import Mapping

import numpy as np
import torch
from ml4gw.waveforms.adhoc import Ringdown


RINGDOWN_PARAMETERS = (
    "frequency",
    "quality",
    "epsilon",
    "phase",
    "inclination",
    "distance",
)

ParameterArray = np.ndarray | torch.Tensor


def _to_batch_tensor(name: str, value: ParameterArray) -> torch.Tensor:
    """Convert a ringdown parameter to a one-dimensional CPU tensor."""
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if tensor.ndim != 1:
        raise ValueError(
            f"Ringdown parameter '{name}' must be one-dimensional, "
            f"got shape {tuple(tensor.shape)}"
        )
    if not len(tensor):
        raise ValueError(f"Ringdown parameter '{name}' must not be empty")
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"Ringdown parameter '{name}' must contain finite values"
        )
    return tensor


def generate_ringdown(
    samples: Mapping[str, ParameterArray],
    sample_rate: float,
    waveform_duration: float,
    right_pad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Aframe-aligned ringdown polarizations.

    The ringdown begins ``right_pad`` seconds before the right edge of the
    returned waveform. Samples before the onset are zero.

    Args:
        samples:
            Batched ringdown parameters. The required keys are ``frequency``,
            ``quality``, ``epsilon``, ``phase``, ``inclination``, and
            ``distance``.
        sample_rate:
            Sampling rate in Hz.
        waveform_duration:
            Duration of the returned waveform in seconds.
        right_pad:
            Duration from the ringdown onset to the right edge in seconds.

    Returns:
        Cross and plus polarizations, each with shape
        ``(batch, int(sample_rate * waveform_duration))``.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if not np.isfinite(waveform_duration) or waveform_duration <= 0:
        raise ValueError("waveform_duration must be finite and positive")
    if (
        not np.isfinite(right_pad)
        or right_pad <= 0
        or right_pad > waveform_duration
    ):
        raise ValueError(
            "right_pad must be finite, positive, and no greater than "
            "waveform_duration"
        )

    waveform_size = int(sample_rate * waveform_duration)
    ringdown_size = int(sample_rate * right_pad)
    if waveform_size < 1:
        raise ValueError(
            "sample_rate * waveform_duration must produce at least one sample"
        )
    if ringdown_size < 1:
        raise ValueError(
            "sample_rate * right_pad must produce at least one sample"
        )

    missing = [name for name in RINGDOWN_PARAMETERS if name not in samples]
    if missing:
        raise KeyError(f"Missing ringdown parameters: {', '.join(missing)}")

    parameters = {
        name: _to_batch_tensor(name, samples[name])
        for name in RINGDOWN_PARAMETERS
    }
    batch_sizes = {len(value) for value in parameters.values()}
    if len(batch_sizes) != 1:
        sizes = {name: len(value) for name, value in parameters.items()}
        raise ValueError(
            f"Ringdown parameters must have equal batch sizes, got {sizes}"
        )

    if torch.any(parameters["frequency"] <= 0):
        raise ValueError("frequency must be positive")
    if torch.any(parameters["frequency"] >= sample_rate / 2):
        raise ValueError("frequency must be below the Nyquist frequency")
    if torch.any(parameters["quality"] <= 0):
        raise ValueError("quality must be positive")
    if torch.any(parameters["epsilon"] < 0):
        raise ValueError("epsilon must be nonnegative")
    if torch.any(parameters["distance"] <= 0):
        raise ValueError("distance must be positive")

    generator = Ringdown(sample_rate=sample_rate, duration=right_pad)
    with torch.no_grad():
        cross, plus = generator(**parameters)

    onset = waveform_size - ringdown_size
    cross = torch.nn.functional.pad(cross, (onset, 0))
    plus = torch.nn.functional.pad(plus, (onset, 0))
    return cross, plus
