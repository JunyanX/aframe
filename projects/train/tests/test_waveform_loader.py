import numpy as np
import pytest
from ledger.injections import (
    RingdownWaveformPolarizationSet,
    RingdownWaveformSet,
)
from train.data.waveforms import WaveformLoader


@pytest.fixture
def ringdown_waveform_file(tmp_path):
    size = 4
    waveform_size = 8
    values = np.arange(size, dtype=float)
    waveforms = np.arange(size * waveform_size, dtype=float).reshape(
        size, waveform_size
    )
    waveform_set = RingdownWaveformPolarizationSet(
        frequency=200 + values,
        quality=10 + values,
        epsilon=0.01 + values,
        phase=0.1 + values,
        inclination=0.2 + values,
        distance=100 + values,
        ra=0.3 + values,
        dec=0.4 + values,
        psi=0.5 + values,
        cross=waveforms,
        plus=waveforms + 1,
        sample_rate=8,
        duration=1,
        right_pad=0.25,
        num_injections=size,
    )
    path = tmp_path / "training_waveforms.hdf5"
    waveform_set.write(path, chunks=(2, waveform_size))
    return path


def test_waveform_loader_forwards_waveform_type_to_sampler(
    ringdown_waveform_file,
    ringdown_val_file,
):
    """`waveform_type` has to reach `WaveformSampler.__init__` as an argument.

    The base class reads the validation file before the subclass body runs,
    so setting the attribute after `super().__init__()` would be too late and
    the ringdown validation file would be read as a CBC one.
    """
    loader = WaveformLoader(
        training_waveform_files=[ringdown_waveform_file],
        waveform_type="ringdown",
        ifos=ringdown_val_file.ifos,
        sample_rate=ringdown_val_file.sample_rate,
        val_waveform_files=[ringdown_val_file.path],
    )

    assert loader.waveform_type == "ringdown"
    assert issubclass(loader.waveform_set_cls, RingdownWaveformSet)
    assert loader.get_val_waveforms(1, 0).shape == (
        ringdown_val_file.size,
        len(ringdown_val_file.ifos),
        ringdown_val_file.waveform_size,
    )
