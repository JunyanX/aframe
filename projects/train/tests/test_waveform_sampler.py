import numpy as np
import pytest

from ledger.injections import RingdownWaveformSet, WaveformSet
from train.data.waveforms import WaveformSampler


def make_sampler(val_file, **kwargs):
    return WaveformSampler(
        ifos=val_file.ifos,
        sample_rate=val_file.sample_rate,
        val_waveform_file=val_file.path,
        **kwargs,
    )


def test_sampler_defaults_to_cbc(cbc_val_file):
    sampler = make_sampler(cbc_val_file)

    assert sampler.waveform_type == "cbc"
    assert issubclass(sampler.waveform_set_cls, WaveformSet)
    assert sampler.num_val_waveforms == cbc_val_file.size
    assert sampler.right_pad == cbc_val_file.right_pad
    assert sampler.get_val_waveforms(1, 0).shape == (
        cbc_val_file.size,
        len(cbc_val_file.ifos),
        cbc_val_file.waveform_size,
    )


def test_sampler_reads_ringdown_validation_file(ringdown_val_file):
    sampler = make_sampler(ringdown_val_file, waveform_type="ringdown")

    cls = sampler.waveform_set_cls
    assert issubclass(cls, RingdownWaveformSet)
    # the ringdown set must not drag the CBC parameters along with it
    assert not issubclass(cls, WaveformSet)
    assert "mass_1" not in cls.__dataclass_fields__

    assert sampler.num_val_waveforms == ringdown_val_file.size
    assert sampler.right_pad == ringdown_val_file.right_pad

    waveforms = sampler.get_val_waveforms(1, 0)
    assert waveforms.shape == (
        ringdown_val_file.size,
        len(ringdown_val_file.ifos),
        ringdown_val_file.waveform_size,
    )
    expected = np.arange(ringdown_val_file.size, dtype=float)
    np.testing.assert_array_equal(waveforms[:, 0, 0].numpy(), expected)
    np.testing.assert_array_equal(waveforms[:, 1, 0].numpy(), 100 + expected)


def test_sampler_rejects_unknown_waveform_type(ringdown_val_file):
    with pytest.raises(ValueError, match="cbc.*ringdown.*burst"):
        make_sampler(ringdown_val_file, waveform_type="burst")


def test_ringdown_validation_file_is_not_readable_as_cbc(ringdown_val_file):
    """A mismatched file must fail loudly rather than load a partial set.

    Validation waveforms are added straight onto detector background, so a
    silent partial load would corrupt the metrics with no error.
    """
    with pytest.raises(ValueError, match="no dataset"):
        make_sampler(ringdown_val_file)
