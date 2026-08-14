from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ledger.injections import RingdownWaveformPolarizationSet
from train.data.waveforms import Hdf5WaveformLoader, WaveformLoader
from train.data.waveforms import loader as loader_module


@pytest.fixture
def bypass_waveform_sampler(monkeypatch):
    def init(self, *args, **kwargs):
        self.right_pad = 0.25

    monkeypatch.setattr(loader_module.WaveformSampler, "__init__", init)


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


@pytest.mark.parametrize("waveform_type", [None, "cbc"])
def test_waveform_loader_defaults_to_cbc(
    tmp_path,
    monkeypatch,
    bypass_waveform_sampler,
    waveform_type,
):
    cbc_read = Mock(return_value=SimpleNamespace(right_pad=0.25))
    ringdown_read = Mock()
    monkeypatch.setattr(
        loader_module.WaveformPolarizationSet,
        "read",
        cbc_read,
    )
    monkeypatch.setattr(
        loader_module.RingdownWaveformPolarizationSet,
        "read",
        ringdown_read,
    )
    path = tmp_path / "training_waveforms.hdf5"
    kwargs = {"training_waveform_path": path}
    if waveform_type is not None:
        kwargs["waveform_type"] = waveform_type

    WaveformLoader(**kwargs)

    cbc_read.assert_called_once_with(path)
    ringdown_read.assert_not_called()


def test_waveform_loader_reads_ringdown_file(
    ringdown_waveform_file,
    bypass_waveform_sampler,
):
    loader = WaveformLoader(
        training_waveform_path=ringdown_waveform_file,
        waveform_type="ringdown",
    )

    assert loader.training_waveform_files == [ringdown_waveform_file]


def test_waveform_loader_checks_ringdown_right_pad(
    ringdown_waveform_file,
    monkeypatch,
    bypass_waveform_sampler,
):
    monkeypatch.setattr(
        loader_module.WaveformSampler,
        "__init__",
        lambda self, *args, **kwargs: setattr(self, "right_pad", 0.5),
    )

    with pytest.raises(ValueError, match="same right pad"):
        WaveformLoader(
            training_waveform_path=ringdown_waveform_file,
            waveform_type="ringdown",
        )


def test_waveform_loader_rejects_unknown_waveform_type(
    tmp_path,
    bypass_waveform_sampler,
):
    with pytest.raises(ValueError, match="cbc.*ringdown.*burst"):
        WaveformLoader(
            training_waveform_path=tmp_path / "training_waveforms.hdf5",
            waveform_type="burst",
        )


def test_hdf5_waveform_loader_reads_ringdown_polarizations(
    ringdown_waveform_file,
):
    loader = Hdf5WaveformLoader(
        [ringdown_waveform_file],
        channels=["cross", "plus"],
        batch_size=2,
        batches_per_epoch=1,
        chunk_size=2,
        path="waveforms",
    )

    batch = next(iter(loader))

    assert batch.shape == (2, 2, 8)
    assert np.isfinite(batch.numpy()).all()
