import sys
from types import ModuleType
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from aframe.tasks.data.waveforms import validation as validation_tasks
from ledger.injections import (
    RingdownWaveformSet,
    WaveformSet,
    waveform_class_factory,
)

IFOS = ["H1", "L1"]
SAMPLE_RATE = 128
DURATION = 1
RIGHT_PAD = 0.5
WAVEFORM_SIZE = SAMPLE_RATE * DURATION


def make_task(task_cls, tmp_path, **kwargs):
    image = tmp_path / "data.sif"
    image.touch()
    parameters = {
        "workflow": "local",
        "image": str(image),
        "accounting_group": "",
        "accounting_group_user": "",
        "num_jobs": 1,
        "num_signals": 2,
        "sample_rate": SAMPLE_RATE,
        "waveform_duration": DURATION,
        "prior": "priors.priors.ringdown_prior",
        "right_pad": RIGHT_PAD,
        "ifos": IFOS,
        "snr_threshold": 4,
        "highpass": 32,
        "background_dir": tmp_path / "background",
        "condor_directory": tmp_path / "condor",
        "output_dir": tmp_path / "output",
        "tmp_dir": tmp_path / "tmp",
    }
    parameters.update(kwargs)
    return task_cls(**parameters)


def make_ringdown_parameters(size, offset=0):
    """The dict shape `rejection_sample` returns for a ringdown run."""
    values = np.arange(offset, offset + size, dtype=float)
    parameters = {
        "frequency": 200 + values,
        "quality": 10 + values,
        "epsilon": 0.01 + values,
        "phase": 0.1 + values,
        "inclination": 0.2 + values,
        "distance": 100 + values,
        "ra": 0.3 + values,
        "dec": 0.4 + values,
        "psi": 0.5 + values,
        "snr": 8 + values,
        "ifo_snrs": np.repeat((6 + values)[:, None], len(IFOS), axis=1),
        "ifos": IFOS,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "right_pad": RIGHT_PAD,
        "num_injections": size,
    }
    for i, ifo in enumerate(IFOS):
        parameters[ifo.lower()] = np.repeat(
            (values + i)[:, None], WAVEFORM_SIZE, axis=1
        )
    return parameters


def write_ringdown_waveforms(path, size, offset=0):
    cls = waveform_class_factory(IFOS, RingdownWaveformSet, "IfoWaveformSet")
    cls(**make_ringdown_parameters(size, offset)).write(path)


class FakePsdSegment:
    """Stands in for the law target `run` reads the PSD from."""

    def __init__(self, path):
        self.path = path

    def open(self, mode):
        return open(self.path, "rb")


def test_waveform_type_parameter_and_class_selection():
    parameter = dict(validation_tasks.DeployValidationWaveforms.get_params())[
        "waveform_type"
    ]

    assert parameter._default == "cbc"
    assert parameter.parse("ringdown") == "ringdown"
    with pytest.raises(ValueError, match="not a valid choice"):
        parameter.parse("burst")

    ringdown_cls = validation_tasks._get_waveform_set_cls(
        IFOS, "ringdown", "IfoWaveformSet"
    )
    cbc_cls = validation_tasks._get_waveform_set_cls(
        IFOS, "cbc", "IfoWaveformSet"
    )

    assert issubclass(ringdown_cls, RingdownWaveformSet)
    assert issubclass(cbc_cls, WaveformSet)
    # the ringdown set must not drag the CBC parameters along with it
    assert not issubclass(ringdown_cls, WaveformSet)
    assert "mass_1" not in ringdown_cls.__dataclass_fields__
    assert "frequency" in ringdown_cls.__dataclass_fields__

    # both get one pre-projected waveform field per interferometer
    for cls in (ringdown_cls, cbc_cls):
        fields = cls.__dataclass_fields__
        assert [
            name
            for name, f in fields.items()
            if f.metadata["kind"] == "waveform"
        ] == ["h1", "l1"]


def test_validation_requirement_forwards_waveform_type(tmp_path):
    task = make_task(
        validation_tasks.ValidationWaveforms,
        tmp_path,
        waveform_type="ringdown",
    )

    parameters = validation_tasks.DeployValidationWaveforms.req_params(task)

    assert parameters["waveform_type"] == "ringdown"


def test_deploy_forwards_ringdown_waveform_type(tmp_path, monkeypatch):
    # `run` reads the psd out of a law target as raw hdf5 bytes
    psd_file = tmp_path / "psd.hdf5"
    with h5py.File(psd_file, "w") as f:
        f.create_dataset("dummy", data=np.zeros(4))

    sample = Mock(return_value=(make_ringdown_parameters(2), Mock()))
    rejection = ModuleType("data.waveforms.rejection")
    rejection.rejection_sample = sample
    monkeypatch.setitem(sys.modules, "data.waveforms.rejection", rejection)

    utils = ModuleType("data.waveforms.utils")
    utils.load_psds = Mock(return_value="psd")
    monkeypatch.setitem(sys.modules, "data.waveforms.utils", utils)

    prior = Mock()
    monkeypatch.setattr(
        validation_tasks, "load_prior", Mock(return_value=prior)
    )

    task = make_task(
        validation_tasks.DeployValidationWaveforms,
        tmp_path,
        branch=0,
        waveform_type="ringdown",
    )
    monkeypatch.setattr(
        type(task),
        "branch_data",
        property(lambda _: (2, FakePsdSegment(psd_file))),
    )

    validation_tasks.DeployValidationWaveforms.run(task)

    assert sample.call_args.kwargs["waveform_type"] == "ringdown"
    # luigi.ListParameter deserializes to a tuple
    assert list(sample.call_args.kwargs["ifos"]) == IFOS
    assert sample.call_args.kwargs["prior"] is prior

    # the file on disk must be a ringdown set, not a CBC one
    cls = waveform_class_factory(IFOS, RingdownWaveformSet, "IfoWaveformSet")
    loaded = cls.read(task.output().path)
    assert loaded.waveforms.shape == (2, len(IFOS), WAVEFORM_SIZE)
    np.testing.assert_array_equal(loaded.frequency, [200, 201])


def test_deploy_defaults_to_cbc(tmp_path):
    task = make_task(validation_tasks.DeployValidationWaveforms, tmp_path)
    assert task.waveform_type == "cbc"


def test_validation_aggregates_ringdown_waveforms(tmp_path, monkeypatch):
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    files = [tmp_dir / "waveforms-0.hdf5", tmp_dir / "waveforms-1.hdf5"]
    write_ringdown_waveforms(files[0], 2)
    write_ringdown_waveforms(files[1], 1, offset=2)

    task = make_task(
        validation_tasks.ValidationWaveforms,
        tmp_path,
        num_jobs=2,
        num_signals=3,
        waveform_type="ringdown",
    )
    task.output_dir.mkdir()
    monkeypatch.setattr(
        validation_tasks.ValidationWaveforms,
        "waveform_files",
        property(lambda _: files),
    )

    validation_tasks.ValidationWaveforms.run(task)

    cls = waveform_class_factory(IFOS, RingdownWaveformSet, "IfoWaveformSet")
    loaded = cls.read(task.output().path)
    assert len(loaded) == loaded.num_injections == 3
    assert loaded.waveforms.shape == (3, len(IFOS), WAVEFORM_SIZE)
    np.testing.assert_array_equal(loaded.frequency, [200, 201, 202])
    assert loaded.sample_rate == SAMPLE_RATE
    assert loaded.duration == DURATION
    assert loaded.right_pad == RIGHT_PAD
    # metadata round-trips through hdf5 as an object-dtype array
    assert list(loaded.ifos) == IFOS
    assert not tmp_dir.exists()
