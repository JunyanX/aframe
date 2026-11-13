"""The read and the recovery must use the family that wrote the file."""

import h5py
import numpy as np
import pytest
from infer.data import ShiftBatchSequence
from ledger.events import (
    EventSet,
    RecoveredInjectionSet,
    RingdownRecoveredInjectionSet,
)
from ledger.injections import (
    InterferometerResponseSet,
    RingdownInterferometerResponseSet,
    waveform_class_factory,
)

IFOS = ["H1", "L1"]
SAMPLE_RATE = 128
T0 = 0.0
SEGMENT = 64.0
WAVEFORM_DURATION = 2
SIZE = SAMPLE_RATE * WAVEFORM_DURATION


@pytest.fixture
def background(tmp_path):
    fname = tmp_path / "background.hdf5"
    n = int(SEGMENT * SAMPLE_RATE)
    with h5py.File(fname, "w") as f:
        for ifo in IFOS:
            d = f.create_dataset(ifo, data=np.zeros(n))
            d.attrs["dx"] = 1 / SAMPLE_RATE
            d.attrs["x0"] = T0
    return fname


def _write_injections(path, base_cls, times):
    cls = waveform_class_factory(IFOS, base_cls, "ResponseSet")
    n = len(times)
    values = np.arange(n, dtype=float) + 1
    params = {}
    for name, attr in cls.__dataclass_fields__.items():
        if attr.metadata["kind"] != "parameter":
            continue
        params[name] = values.copy()
    params["ifo_snrs"] = np.repeat(values[:, None], len(IFOS), axis=1)
    params["injection_time"] = np.asarray(times, dtype=float)
    params["shift"] = np.zeros((n, len(IFOS)))
    for ifo in IFOS:
        params[ifo.lower()] = np.zeros((n, SIZE))
    obj = cls(
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        duration=WAVEFORM_DURATION,
        right_pad=1,
        num_injections=n,
        **params,
    )
    obj.write(path)
    return path


@pytest.fixture
def ringdown_injections(tmp_path):
    return _write_injections(
        tmp_path / "rd.hdf5",
        RingdownInterferometerResponseSet,
        [10.0, 20.0, 30.0],
    )


@pytest.fixture
def cbc_injections(tmp_path):
    return _write_injections(
        tmp_path / "cbc.hdf5",
        InterferometerResponseSet,
        [10.0, 20.0, 30.0],
    )


def _sequence(background, injections, waveform_type):
    """One shift of one background file, with its injection set."""
    return ShiftBatchSequence(
        background_fname=str(background),
        waveform_fnames=[str(injections)],
        ifos=IFOS,
        shifts=[[0, 0]],
        inference_sampling_rate=4,
        batch_size=8,
        waveform_type=waveform_type,
    )


def _event_set(times):
    times = np.asarray(times, dtype=float)
    return EventSet(
        detection_statistic=np.arange(len(times), dtype=float) + 5,
        detection_time=times,
        shift=np.zeros((len(times), len(IFOS))),
        Tb=100.0,
    )


def test_recovers_ringdown_parameters(background, ringdown_injections):
    seq = _sequence(background, ringdown_injections, "ringdown")
    recovered = seq.recover(_event_set([10.1, 20.1, 30.1]), 0)

    assert isinstance(recovered, RingdownRecoveredInjectionSet)
    assert len(recovered) == 3
    # this is the whole point: the parameters reach the recovered ledger
    assert np.all(recovered.frequency == np.array([1.0, 2.0, 3.0]))
    assert np.all(recovered.quality == np.array([1.0, 2.0, 3.0]))
    assert np.all(recovered.detection_time == np.array([10.1, 20.1, 30.1]))


def test_cbc_campaign_still_works(background, cbc_injections):
    """Regression: the default path must behave exactly as before."""
    seq = _sequence(background, cbc_injections, "cbc")
    recovered = seq.recover(_event_set([10.1, 20.1, 30.1]), 0)

    assert isinstance(recovered, RecoveredInjectionSet)
    assert not isinstance(recovered, RingdownRecoveredInjectionSet)
    assert np.all(recovered.mass_1 == np.array([1.0, 2.0, 3.0]))
