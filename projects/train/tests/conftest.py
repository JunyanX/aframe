from types import SimpleNamespace

import numpy as np
import pytest

from ledger.injections import (
    RingdownWaveformSet,
    WaveformSet,
    waveform_class_factory,
)

# small enough to keep fixtures cheap; duration must equal
# waveform_size / sample_rate or InjectionMetadata rejects the set
VAL_SPEC = SimpleNamespace(
    ifos=["H1", "L1"],
    size=4,
    waveform_size=8,
    sample_rate=8,
    right_pad=0.25,
)


def _write_val_waveform_file(path, base_cls, spec):
    """Write a per-ifo validation file for whichever parameters the set has.

    Field names differ between the CBC and ringdown sets, so fill them from
    the dataclass rather than naming them.
    """
    cls = waveform_class_factory(spec.ifos, base_cls, "IfoWaveformSet")
    values = np.arange(spec.size, dtype=float)

    kwargs = {
        name: values.copy()
        for name, field in cls.__dataclass_fields__.items()
        if field.metadata["kind"] == "parameter"
    }
    # one column per interferometer, not one value per injection
    kwargs["ifo_snrs"] = np.zeros((spec.size, len(spec.ifos)))
    # give each detector a distinguishable value so channel order is testable
    for i, ifo in enumerate(spec.ifos):
        kwargs[ifo.lower()] = (
            np.full((spec.size, spec.waveform_size), 100.0 * i)
            + values[:, None]
        )

    kwargs["ifos"] = spec.ifos
    kwargs["sample_rate"] = spec.sample_rate
    kwargs["duration"] = spec.waveform_size / spec.sample_rate
    kwargs["right_pad"] = spec.right_pad
    kwargs["num_injections"] = spec.size

    cls(**kwargs).write(path)
    return SimpleNamespace(path=path, **vars(spec))


@pytest.fixture
def ringdown_val_file(tmp_path):
    return _write_val_waveform_file(
        tmp_path / "val_ringdown.hdf5", RingdownWaveformSet, VAL_SPEC
    )


@pytest.fixture
def cbc_val_file(tmp_path):
    return _write_val_waveform_file(
        tmp_path / "val_cbc.hdf5", WaveformSet, VAL_SPEC
    )
