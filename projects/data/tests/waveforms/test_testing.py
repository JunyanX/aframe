"""`testing_waveforms` writes a whole campaign in the requested family.

Only `load_psds` is patched. Injection-time arithmetic, rejection
sampling, response-set construction and both writes run for real.
"""

from unittest.mock import Mock

import data.waveforms.testing as data_testing
import data.waveforms.utils as data_utils
import h5py
import numpy as np
import torch
from ledger.injections import (
    InjectionParameterSet,
    InterferometerResponseSet,
    RingdownInjectionParameterSet,
    RingdownInterferometerResponseSet,
    waveform_class_factory,
)
from priors.priors import end_o3_ratesandpops, ringdown_prior

IFOS = ["H1", "L1"]
# ringdown_prior draws up to 1 kHz, so keep it below Nyquist. These
# values give 5 injection times, [9, 19, 29, 39, 49].
SAMPLE_RATE = 2048
DURATION = 2
RIGHT_PAD = 1
SPACING = 8
BUFFER = 8
SHIFTS = [0, 1]


def flat_psd(sample_rate, duration, num_ifos=2):
    """Flat PSD at roughly aLIGO design amplitude on the 1/duration grid."""
    num_freqs = int(sample_rate * duration) // 2 + 1
    return 1e-46 * torch.ones((num_ifos, num_freqs), dtype=torch.float64)


def rejected_columns(fname):
    """The parameter columns a rejected-parameters file holds.

    Read from the file: an object only ever shows its own class's fields,
    so it can't tell which family did the writing.
    """
    with h5py.File(fname, "r") as f:
        return set(f["parameters"].keys())


def parameter_fields(cls):
    return {
        name
        for name, attr in cls.__dataclass_fields__.items()
        if attr.metadata["kind"] == "parameter"
    }


def run_generator(tmp_path, monkeypatch, prior, waveform_type=None):
    monkeypatch.setattr(
        data_utils,
        "load_psds",
        Mock(return_value=flat_psd(SAMPLE_RATE, DURATION)),
    )
    kwargs = {} if waveform_type is None else {"waveform_type": waveform_type}
    return data_testing.testing_waveforms(
        start=0.0,
        end=60.0,
        ifos=IFOS,
        shifts=SHIFTS,
        spacing=SPACING,
        buffer=BUFFER,
        prior=prior,
        minimum_frequency=20,
        reference_frequency=50,
        sample_rate=SAMPLE_RATE,
        waveform_duration=DURATION,
        waveform_approximant="IMRPhenomPv2",
        right_pad=RIGHT_PAD,
        highpass=32,
        lowpass=None,
        snr_threshold=4,
        psd_file="unused: load_psds is patched",
        max_num_samples=32,
        waveforms_file=tmp_path / "out" / "waveforms.hdf5",
        rejected_file=tmp_path / "out" / "rejected_parameters.hdf5",
        jitter=0.1,
        seed=1122,
        **kwargs,
    )


def test_writes_a_ringdown_campaign(tmp_path, monkeypatch):
    waveform_fname, rejected_fname = run_generator(
        tmp_path, monkeypatch, ringdown_prior, waveform_type="ringdown"
    )

    cls = waveform_class_factory(
        IFOS, RingdownInterferometerResponseSet, "ResponseSet"
    )
    loaded = cls.read(waveform_fname)
    assert len(loaded) == 5
    assert loaded.waveforms.shape == (5, len(IFOS), SAMPLE_RATE * DURATION)
    np.testing.assert_allclose(
        loaded.injection_time, [9, 19, 29, 39, 49], atol=0.11
    )
    assert ((loaded.frequency >= 40) & (loaded.frequency <= 1000)).all()
    assert ((loaded.quality >= 2) & (loaded.quality <= 20)).all()
    assert "mass_1" not in cls.__dataclass_fields__

    # no count: the number of rejections varies run to run
    RingdownInjectionParameterSet.read(rejected_fname)
    assert rejected_columns(rejected_fname) == parameter_fields(
        RingdownInjectionParameterSet
    )


def test_still_writes_a_cbc_campaign(tmp_path, monkeypatch):
    """The default path is untouched. Slowest test here (pycbc)."""
    waveform_fname, rejected_fname = run_generator(
        tmp_path, monkeypatch, end_o3_ratesandpops
    )

    cls = waveform_class_factory(
        IFOS, InterferometerResponseSet, "ResponseSet"
    )
    loaded = cls.read(waveform_fname)
    assert len(loaded) == 5
    assert (loaded.mass_1 > 0).all()
    assert "frequency" not in cls.__dataclass_fields__

    InjectionParameterSet.read(rejected_fname)
    assert rejected_columns(rejected_fname) == parameter_fields(
        InjectionParameterSet
    )
