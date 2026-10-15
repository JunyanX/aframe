"""aggregate_waveforms.py merges each output with its family's class."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ledger.injections import (
    InterferometerResponseSet,
    RingdownInjectionParameterSet,
    RingdownInterferometerResponseSet,
    waveform_class_factory,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_waveforms.py"
IFOS = ["H1", "L1"]
SAMPLE_RATE = 128
DURATION = 2


class _Named(dict):
    """Snakemake's named input/output: a dict also readable by attribute."""

    __getattr__ = dict.__getitem__


def _fill(cls, n, **extra):
    values = np.arange(n, dtype=float) + 1
    fields = {
        name: values.copy()
        for name, attr in cls.__dataclass_fields__.items()
        if attr.metadata["kind"] == "parameter"
    }
    fields["ifo_snrs"] = np.repeat(values[:, None], len(IFOS), axis=1)
    fields.update(extra)
    return fields


def _responses(path, base_cls, n):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cls = waveform_class_factory(IFOS, base_cls, "ResponseSet")
    fields = _fill(cls, n)
    fields["shift"] = np.zeros((n, len(IFOS)))
    for ifo in IFOS:
        fields[ifo.lower()] = np.zeros((n, SAMPLE_RATE * DURATION))
    cls(
        ifos=IFOS,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        right_pad=1,
        num_injections=n,
        **fields,
    ).write(path)
    return str(path)


def _run(monkeypatch, tmp_path, inputs, outputs, classes, **params):
    # the script points stdout/stderr at its log; get them back afterwards
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    snakemake = SimpleNamespace(
        log=[str(tmp_path / "aggregate.log")],
        params=SimpleNamespace(ifos=IFOS, classes=classes, **params),
        input=_Named(inputs),
        output=_Named(outputs),
    )
    runpy.run_path(str(SCRIPT), init_globals={"snakemake": snakemake})


def test_aggregates_a_ringdown_campaign(monkeypatch, tmp_path):
    branches = [
        _responses(
            tmp_path / f"b{i}" / "waveforms.hdf5",
            RingdownInterferometerResponseSet,
            2,
        )
        for i in range(2)
    ]
    out = tmp_path / "waveforms.hdf5"
    _run(
        monkeypatch,
        tmp_path,
        {"waveforms": branches},
        {"waveforms": str(out)},
        {"waveforms": "responses"},
        waveform_type="ringdown",
    )
    cls = waveform_class_factory(
        IFOS, RingdownInterferometerResponseSet, "ResponseSet"
    )
    assert len(cls.read(out)) == 4


def test_aggregate_with_the_wrong_family_raises(monkeypatch, tmp_path):
    """A CBC run must not quietly merge ringdown files, nor the reverse."""
    branch = _responses(
        tmp_path / "b0" / "waveforms.hdf5",
        RingdownInterferometerResponseSet,
        2,
    )
    with pytest.raises((KeyError, ValueError)):
        _run(
            monkeypatch,
            tmp_path,
            {"waveforms": [branch]},
            {"waveforms": str(tmp_path / "waveforms.hdf5")},
            {"waveforms": "responses"},
            waveform_type="cbc",
        )


def test_aggregate_keeps_a_branch_that_rejected_nothing(monkeypatch, tmp_path):
    populated = tmp_path / "b0" / "rejected_parameters.hdf5"
    populated.parent.mkdir()
    RingdownInjectionParameterSet(
        ifos=IFOS, **_fill(RingdownInjectionParameterSet, 3)
    ).write(populated)
    empty = tmp_path / "b1" / "rejected_parameters.hdf5"
    empty.parent.mkdir()
    # what rejection_sample writes when a branch rejects nothing
    RingdownInjectionParameterSet().write(empty)

    out = tmp_path / "rejected_parameters.hdf5"
    _run(
        monkeypatch,
        tmp_path,
        {"rejected_parameters": [str(populated), str(empty)]},
        {"rejected_parameters": str(out)},
        {"rejected_parameters": "parameters"},
        waveform_type="ringdown",
    )
    assert len(RingdownInjectionParameterSet.read(out)) == 3


def test_aggregates_a_cbc_campaign(monkeypatch, tmp_path):
    branch = _responses(
        tmp_path / "b0" / "waveforms.hdf5", InterferometerResponseSet, 2
    )
    out = tmp_path / "waveforms.hdf5"
    _run(
        monkeypatch,
        tmp_path,
        {"waveforms": [branch]},
        {"waveforms": str(out)},
        {"waveforms": "responses"},
        waveform_type="cbc",
    )
    cls = waveform_class_factory(
        IFOS, InterferometerResponseSet, "ResponseSet"
    )
    assert len(cls.read(out)) == 2
