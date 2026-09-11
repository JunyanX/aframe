"""The read and the recovery must use the family that wrote the file."""

import h5py
import numpy as np
import pytest
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

from infer.data import Sequence

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
    return Sequence(
        background_fname=str(background),
        injection_set_fname=str(injections),
        ifos=IFOS,
        shifts=[0, 0],
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


def test_reads_a_ringdown_campaign(background, ringdown_injections):
    seq = _sequence(background, ringdown_injections, "ringdown")

    assert seq.injection_set is not None
    assert len(seq.injection_set) == 3
    # the intrinsic parameters survived the read
    assert np.all(seq.injection_set.frequency == np.array([1.0, 2.0, 3.0]))
    assert not hasattr(seq.injection_set, "mass_1")


def test_recovers_ringdown_parameters(background, ringdown_injections):
    seq = _sequence(background, ringdown_injections, "ringdown")
    recovered = seq.recover(_event_set([10.1, 20.1, 30.1]))

    assert isinstance(recovered, RingdownRecoveredInjectionSet)
    assert len(recovered) == 3
    # this is the whole point: the parameters reach the recovered ledger
    assert np.all(recovered.frequency == np.array([1.0, 2.0, 3.0]))
    assert np.all(recovered.quality == np.array([1.0, 2.0, 3.0]))
    assert np.all(recovered.detection_time == np.array([10.1, 20.1, 30.1]))


def test_cbc_campaign_still_works(background, cbc_injections):
    """Regression: the default path must behave exactly as before."""
    seq = _sequence(background, cbc_injections, "cbc")
    recovered = seq.recover(_event_set([10.1, 20.1, 30.1]))

    assert isinstance(recovered, RecoveredInjectionSet)
    assert not isinstance(recovered, RingdownRecoveredInjectionSet)
    assert np.all(recovered.mass_1 == np.array([1.0, 2.0, 3.0]))


def test_defaults_to_cbc(background, cbc_injections):
    """Omitting waveform_type entirely must read as CBC."""
    seq = Sequence(
        background_fname=str(background),
        injection_set_fname=str(cbc_injections),
        ifos=IFOS,
        shifts=[0, 0],
        inference_sampling_rate=4,
        batch_size=8,
    )
    assert seq.waveform_type == "cbc"
    assert np.all(seq.injection_set.mass_1 == np.array([1.0, 2.0, 3.0]))


def test_branch_with_no_injections_in_range(tmp_path, background):
    """A segment the campaign does not cover must yield no injections.

    `Infer.run` masks these out of the merge, so the empty case has to
    behave for ringdowns exactly as it does for CBC.
    """
    injections = _write_injections(
        tmp_path / "far.hdf5",
        RingdownInterferometerResponseSet,
        [10_000.0, 20_000.0],
    )
    seq = _sequence(background, injections, "ringdown")
    assert seq.injection_set is None


def test_reading_with_the_wrong_family_raises(background, ringdown_injections):
    """A missed dispatch site must crash rather than silently mislead."""
    with pytest.raises(ValueError):
        _sequence(background, ringdown_injections, "cbc")


def test_infer_task_declares_waveform_type():
    import luigi

    from aframe.tasks.infer.base import InferParameters

    parameter = dict(InferParameters.get_params())["waveform_type"]
    assert isinstance(parameter, luigi.ChoiceParameter)
    assert parameter._default == "cbc"
    assert parameter.parse("ringdown") == "ringdown"
    with pytest.raises(ValueError, match="not a valid choice"):
        parameter.parse("burst")


def test_merge_aggregates_with_the_ringdown_class(monkeypatch):
    """`Infer.run` must pick the recovered class from waveform_type.

    Driven with a stand-in `self` rather than a constructed luigi task:
    the method only touches the attributes set below, and this observes
    which class it actually calls rather than what its source says.
    """
    import ledger.events as le
    from aframe.tasks.infer.infer import Infer

    called = []

    def recorder(name):
        return classmethod(lambda cls, *a, **k: called.append(name))

    monkeypatch.setattr(le.EventSet, "aggregate", recorder("events"))
    monkeypatch.setattr(le.RecoveredInjectionSet, "aggregate", recorder("cbc"))
    monkeypatch.setattr(
        le.RingdownRecoveredInjectionSet, "aggregate", recorder("ringdown")
    )

    class FakeInfer:
        waveform_type = "ringdown"
        ifos = IFOS
        return_timeseries = False
        remove_tmpdir = False
        foreground_output = "fg.hdf5"
        background_output = "bg.hdf5"
        zero_lag_output = "0lag.hdf5"
        # every shift is zero-lag, so back_files is empty and the
        # read/sort/write branch at the end of run() is skipped
        background_files = np.array(["a.hdf5", "b.hdf5"])
        foreground_files = np.array(["a.hdf5", "b.hdf5"])

        def get_metadata(self):
            return (
                np.array([1, 1]),
                np.array([1, 1]),
                np.array([[0, 0], [0, 0]]),
            )

    Infer.run(FakeInfer())

    assert "ringdown" in called
    assert "cbc" not in called


def test_merge_aggregates_with_the_cbc_class(monkeypatch):
    """Regression guard for the same site."""
    import ledger.events as le
    from aframe.tasks.infer.infer import Infer

    called = []

    def recorder(name):
        return classmethod(lambda cls, *a, **k: called.append(name))

    monkeypatch.setattr(le.EventSet, "aggregate", recorder("events"))
    monkeypatch.setattr(le.RecoveredInjectionSet, "aggregate", recorder("cbc"))
    monkeypatch.setattr(
        le.RingdownRecoveredInjectionSet, "aggregate", recorder("ringdown")
    )

    class FakeInfer:
        waveform_type = "cbc"
        ifos = IFOS
        return_timeseries = False
        remove_tmpdir = False
        foreground_output = "fg.hdf5"
        background_output = "bg.hdf5"
        zero_lag_output = "0lag.hdf5"
        background_files = np.array(["a.hdf5", "b.hdf5"])
        foreground_files = np.array(["a.hdf5", "b.hdf5"])

        def get_metadata(self):
            return (
                np.array([1, 1]),
                np.array([1, 1]),
                np.array([[0, 0], [0, 0]]),
            )

    Infer.run(FakeInfer())

    assert "cbc" in called
    assert "ringdown" not in called


class _StopAfterSequence(Exception):
    """Raised by the Sequence stub so run() stops at the call we test."""


def test_deploy_forwards_waveform_type_to_sequence(monkeypatch, tmp_path):
    """`InferBase.run` must pass the family down to `Sequence`.

    Constructing `Sequence` directly, as every test above does, cannot
    catch a missing keyword at the one call site that matters.
    """
    import sys
    from types import ModuleType

    from aframe.tasks.infer import base as infer_base

    recorded = {}

    def fake_sequence(**kwargs):
        recorded.update(kwargs)
        raise _StopAfterSequence

    module = ModuleType("infer.data")
    module.Sequence = fake_sequence
    monkeypatch.setitem(sys.modules, "infer.data", module)

    class FakeDeploy:
        waveform_type = "ringdown"
        ifos = IFOS
        batch_size = 8
        inference_sampling_rate = 4
        rate_per_client = None
        injection_set_fname = "inj.hdf5"
        tmp_dir = tmp_path / "seq"
        branch_data = ("bg.hdf5", [0, 0])

    with pytest.raises(_StopAfterSequence):
        infer_base.InferBase.run(FakeDeploy())

    assert recorded.get("waveform_type") == "ringdown"
