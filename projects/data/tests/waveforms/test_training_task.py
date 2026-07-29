import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from aframe.tasks.data.waveforms import training as training_tasks
from ledger.injections import (
    RingdownWaveformPolarizationSet,
    WaveformPolarizationSet,
)


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
        "sample_rate": 128,
        "waveform_duration": 1,
        "prior": "priors.priors.ringdown_prior",
        "right_pad": 0.5,
        "condor_directory": tmp_path / "condor",
        "output_dir": tmp_path / "output",
        "tmp_dir": tmp_path / "tmp",
    }
    parameters.update(kwargs)
    return task_cls(**parameters)


def make_ringdown_waveforms(size, offset=0):
    values = np.arange(offset, offset + size, dtype=float)
    waveform_size = 128
    return RingdownWaveformPolarizationSet(
        frequency=200 + values,
        quality=10 + values,
        epsilon=0.01 + values,
        phase=0.1 + values,
        inclination=0.2 + values,
        distance=100 + values,
        ra=0.3 + values,
        dec=0.4 + values,
        psi=0.5 + values,
        cross=np.repeat(values[:, None], waveform_size, axis=1),
        plus=np.repeat((values + 1)[:, None], waveform_size, axis=1),
        sample_rate=128,
        duration=1,
        right_pad=0.5,
        num_injections=size,
    )


def test_waveform_type_parameter_and_class_selection():
    parameter = dict(training_tasks.DeployTrainingWaveforms.get_params())[
        "waveform_type"
    ]

    assert parameter._default == "cbc"
    assert parameter.parse("ringdown") == "ringdown"
    with pytest.raises(ValueError, match="not a valid choice"):
        parameter.parse("burst")

    ringdown_cls = training_tasks._get_waveform_set_cls("ringdown")
    cbc_cls = training_tasks._get_waveform_set_cls("cbc")
    assert ringdown_cls is RingdownWaveformPolarizationSet
    assert issubclass(cbc_cls, WaveformPolarizationSet)


def test_training_requirement_forwards_waveform_type(tmp_path):
    task = make_task(
        training_tasks.TrainingWaveforms,
        tmp_path,
        waveform_type="ringdown",
    )

    parameters = training_tasks.DeployTrainingWaveforms.req_params(task)

    assert parameters["waveform_type"] == "ringdown"


def test_deploy_forwards_ringdown_waveform_type(tmp_path, monkeypatch):
    waveform_set = Mock()
    waveform_set.get_waveforms.return_value = np.zeros((2, 2, 128))
    generate = Mock(return_value=waveform_set)
    module = ModuleType("data.waveforms.training")
    module.training_waveforms = generate
    monkeypatch.setitem(sys.modules, "data.waveforms.training", module)

    prior = Mock()
    monkeypatch.setattr(training_tasks, "load_prior", Mock(return_value=prior))
    task = make_task(
        training_tasks.DeployTrainingWaveforms,
        tmp_path,
        branch=0,
        waveform_type="ringdown",
    )

    training_tasks.DeployTrainingWaveforms.run(task)

    generate.assert_called_once_with(
        num_signals=2,
        waveform_duration=1,
        sample_rate=128,
        prior=prior,
        minimum_frequency=20,
        reference_frequency=50,
        waveform_approximant="IMRPhenomXPHM",
        right_pad=0.5,
        waveform_type="ringdown",
    )
    waveform_set.write.assert_called_once()


def test_training_aggregates_ringdown_waveforms(tmp_path, monkeypatch):
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    files = [tmp_dir / "waveforms-0.hdf5", tmp_dir / "waveforms-1.hdf5"]
    make_ringdown_waveforms(2).write(files[0])
    make_ringdown_waveforms(1, offset=2).write(files[1])

    task = make_task(
        training_tasks.TrainingWaveforms,
        tmp_path,
        num_jobs=2,
        num_signals=3,
        waveform_type="ringdown",
    )
    task.output_dir.mkdir()
    monkeypatch.setattr(
        training_tasks.TrainingWaveforms,
        "waveform_files",
        property(lambda _: files),
    )

    training_tasks.TrainingWaveforms.run(task)

    loaded = RingdownWaveformPolarizationSet.read(task.output().path)
    assert len(loaded) == loaded.num_injections == 3
    assert loaded.get_waveforms().shape == (3, 2, 128)
    np.testing.assert_array_equal(loaded.frequency, [200, 201, 202])
    assert loaded.sample_rate == 128
    assert loaded.duration == 1
    assert loaded.right_pad == 0.5
    assert not tmp_dir.exists()
