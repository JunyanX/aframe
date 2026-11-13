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
    if (
        not np.isfinite(right_pad)
        or right_pad <= 0
        or right_pad > waveform_duration
    ):
        raise ValueError(
            "right_pad must be finite, positive, and no greater than "
            "waveform_duration"
        )

    parameters = {
        name: torch.as_tensor(samples[name], dtype=torch.float64, device="cpu")
        for name in RINGDOWN_PARAMETERS
    }
    # A negative frequency or distance still gives a finite waveform, so
    # the check on the output below can't catch either.
    if torch.any(parameters["frequency"] <= 0):
        raise ValueError("frequency must be positive")
    if torch.any(parameters["frequency"] >= sample_rate / 2):
        raise ValueError("frequency must be below the Nyquist frequency")
    if torch.any(parameters["distance"] <= 0):
        raise ValueError("distance must be positive")

    generator = Ringdown(sample_rate=sample_rate, duration=right_pad)
    with torch.no_grad():
        cross, plus = generator(**parameters)
    # ml4gw doesn't check its inputs. A quality below 1 or a negative
    # epsilon puts a negative number under a square root and comes back
    # as NaN, as does any parameter that isn't finite.
    if not (torch.isfinite(cross).all() and torch.isfinite(plus).all()):
        raise ValueError(
            "Ringdown waveforms are not finite: every parameter must be "
            "finite, quality above 1 and epsilon nonnegative"
        )

    waveform_size = int(sample_rate * waveform_duration)
    ringdown_size = int(sample_rate * right_pad)
    onset = waveform_size - ringdown_size
    cross = torch.nn.functional.pad(cross, (onset, 0))
    plus = torch.nn.functional.pad(plus, (onset, 0))
    return cross, plus
