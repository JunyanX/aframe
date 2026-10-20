"""The read and the recovery must use the family that wrote the file."""

import h5py
import numpy as np
import pytest
from infer.data import Hdf5Sequence
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
    return Hdf5Sequence(
        sequence_id=1,
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
    seq = Hdf5Sequence(
        sequence_id=1,
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

    `_merge_group` gets no foreground rows from these, so the empty case has to
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


def test_cli_passes_waveform_type_to_the_sequence(monkeypatch):
    """`_run_branch` is the one place a Hdf5Sequence is built for a job."""
    from types import SimpleNamespace

    from infer import cli

    recorded = {}

    class _Stop(Exception):
        pass

    def fake_sequence(*args, **kwargs):
        recorded.update(kwargs)
        raise _Stop

    monkeypatch.setattr(cli, "Hdf5Sequence", fake_sequence)
    cfg = SimpleNamespace(
        analysis_type="hdf5",
        ifos=IFOS,
        inference_sampling_rate=4,
        batch_size=8,
        waveform_type="ringdown",
    )
    branch = {"fname": "bg.hdf5", "waveforms": "inj.hdf5", "shifts": [0, 1]}
    with pytest.raises(_Stop):
        cli._run_branch(None, cfg, "0", branch, outdir=None)

    assert recorded["waveform_type"] == "ringdown"


@pytest.mark.parametrize("waveform_type", ["cbc", "ringdown"])
def test_merge_group_aggregates_with_the_family_class(
    monkeypatch, tmp_path, waveform_type
):
    from types import SimpleNamespace

    import ledger.events as le
    from infer import cli

    called = []

    def recorder(name):
        def aggregate(cls, files, fname, **kwargs):
            called.append(name)
            # _merge_group reads the row count back from the output
            with h5py.File(fname, "w") as f:
                f.attrs["length"] = 0

        return classmethod(aggregate)

    monkeypatch.setattr(le.EventSet, "aggregate", recorder("events"))
    monkeypatch.setattr(le.RecoveredInjectionSet, "aggregate", recorder("cbc"))
    monkeypatch.setattr(
        le.RingdownRecoveredInjectionSet, "aggregate", recorder("ringdown")
    )
    cfg = SimpleNamespace(
        analysis_type="hdf5",
        waveform_type=waveform_type,
        zero_lag_out=None,
        timeseries_out=None,
        background_out=str(tmp_path / "background.hdf5"),
        foreground_out=str(tmp_path / "foreground.hdf5"),
        metadata_out=str(tmp_path / "metadata.json"),
    )
    cli._merge_group(cfg, {"0": {"shifts": [0, 1]}}, tmp_path)

    other = {"cbc": "ringdown", "ringdown": "cbc"}[waveform_type]
    assert waveform_type in called
    assert other not in called
