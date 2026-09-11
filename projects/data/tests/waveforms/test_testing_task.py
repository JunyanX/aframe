import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import torch

# `sys` and `ModuleType` are for the module-stubbing mock below. The gwpy
# shim that makes `data.waveforms.*` importable at all lives in
# `tests/conftest.py` (Task 3) -- it has to run before collection, so it
# cannot live here.
import data.waveforms.testing as data_testing
import data.waveforms.utils as data_utils
from aframe.tasks.data.waveforms import testing as testing_tasks
from ledger.injections import (
    InjectionParameterSet,
    InterferometerResponseSet,
    RingdownInjectionParameterSet,
    RingdownInterferometerResponseSet,
    waveform_class_factory,
)
from priors.priors import end_o3_ratesandpops, ringdown_prior

IFOS = ["H1", "L1"]
SAMPLE_RATE = 128
DURATION = 1
RIGHT_PAD = 0.5
WAVEFORM_SIZE = SAMPLE_RATE * DURATION
NUM_IFOS = len(IFOS)

# The end-to-end tests need their own scale: ringdown_prior draws up to 1 kHz,
# so the sample rate has to keep the whole prior below Nyquist or
# generate_ringdown rejects every draw. `test_rejection.py` uses 2048 for the
# same reason. These values yield 5 injection times -- [9, 19, 29, 39, 49] --
# and the ringdown path completes in about 0.2 s.
E2E_SAMPLE_RATE = 2048
E2E_DURATION = 2
E2E_RIGHT_PAD = 1
E2E_SPACING = 8
E2E_BUFFER = 8
E2E_SHIFTS = [0, 1]


def flat_psd(sample_rate, duration, num_ifos=NUM_IFOS):
    """Flat PSD at roughly aLIGO design amplitude, on the 1/duration grid.

    Patched in place of `load_psds` so the tests do not need a background
    file. The magnitude matters: too loud and rejection sampling never
    reaches its quota, because `rejection_sample` loops until it does.
    """
    num_freqs = int(sample_rate * duration) // 2 + 1
    return 1e-46 * torch.ones((num_ifos, num_freqs), dtype=torch.float64)


def make_task(task_cls, tmp_path, **kwargs):
    image = tmp_path / "data.sif"
    image.touch()
    parameters = {
        "workflow": "local",
        "image": str(image),
        "accounting_group": "",
        "accounting_group_user": "",
        "num_signals": 2,
        "sample_rate": SAMPLE_RATE,
        "waveform_duration": DURATION,
        "prior": "priors.priors.ringdown_prior",
        "right_pad": RIGHT_PAD,
        "start": 0.0,
        "end": 1000.0,
        "ifos": IFOS,
        "shifts": [0, 1],
        "spacing": 16.0,
        "buffer": 16.0,
        "highpass": 32,
        "snr_threshold": 4,
        "psd_length": 8,
        "seed": 1122,
        "background_dir": tmp_path / "background",
        "condor_directory": tmp_path / "condor",
        "output_dir": tmp_path / "output",
    }
    parameters.update(kwargs)
    return task_cls(**parameters)


def test_waveform_type_parameter_and_class_selection():
    parameter = dict(testing_tasks.DeployTestingWaveforms.get_params())[
        "waveform_type"
    ]

    assert parameter._default == "cbc"
    assert parameter.parse("ringdown") == "ringdown"
    with pytest.raises(ValueError, match="not a valid choice"):
        parameter.parse("burst")

    ringdown_cls = testing_tasks._get_response_set_cls(
        IFOS, "ringdown", "ResponseSet"
    )
    cbc_cls = testing_tasks._get_response_set_cls(IFOS, "cbc", "ResponseSet")

    assert issubclass(ringdown_cls, RingdownInterferometerResponseSet)
    assert issubclass(cbc_cls, InterferometerResponseSet)
    # the ringdown campaign must not drag the CBC parameters along
    assert not issubclass(ringdown_cls, InterferometerResponseSet)
    assert "mass_1" not in ringdown_cls.__dataclass_fields__
    assert "frequency" in ringdown_cls.__dataclass_fields__
    # both keep the injection timing fields
    for cls in (ringdown_cls, cbc_cls):
        assert {"injection_time", "shift"} <= set(cls.__dataclass_fields__)


def test_deploy_defaults_to_cbc(tmp_path):
    task = make_task(testing_tasks.DeployTestingWaveforms, tmp_path)
    assert task.waveform_type == "cbc"


def test_testing_requirement_forwards_waveform_type(tmp_path):
    task = make_task(
        testing_tasks.TestingWaveforms, tmp_path, waveform_type="ringdown"
    )

    parameters = testing_tasks.DeployTestingWaveforms.req_params(task)

    assert parameters["waveform_type"] == "ringdown"


def test_deploy_forwards_waveform_type_to_generator(tmp_path, monkeypatch):
    generator = Mock(return_value=("waveforms.hdf5", "rejected.hdf5"))
    module = ModuleType("data.waveforms.testing")
    module.testing_waveforms = generator
    monkeypatch.setitem(sys.modules, "data.waveforms.testing", module)

    psd_file = tmp_path / "psd.hdf5"
    psd_file.write_bytes(b"")

    class FakePsdSegment:
        def open(self, mode):
            return open(psd_file, "rb")

    monkeypatch.setattr(
        testing_tasks, "load_prior", Mock(return_value="prior")
    )
    monkeypatch.setattr("h5py.File", Mock(return_value="psd"), raising=False)

    task = make_task(
        testing_tasks.DeployTestingWaveforms,
        tmp_path,
        branch=0,
        waveform_type="ringdown",
    )
    monkeypatch.setattr(
        type(task),
        "branch_data",
        property(lambda _: (0.0, 100.0, [0, 1], FakePsdSegment())),
    )

    testing_tasks.DeployTestingWaveforms.run(task)

    assert generator.call_args.kwargs["waveform_type"] == "ringdown"


def run_generator(tmp_path, monkeypatch, prior, waveform_type=None):
    """Drive the real `testing_waveforms`, patching only the PSD loader.

    Everything else runs for real: the injection-time arithmetic, rejection
    sampling, the response-set construction and both file writes. That is the
    point -- the mocked test above cannot catch `testing_waveforms` failing to
    pass `waveform_type` on to `rejection_sample`.
    """
    monkeypatch.setattr(
        data_utils,
        "load_psds",
        Mock(return_value=flat_psd(E2E_SAMPLE_RATE, E2E_DURATION)),
    )
    kwargs = {}
    if waveform_type is not None:
        kwargs["waveform_type"] = waveform_type
    return data_testing.testing_waveforms(
        start=0.0,
        end=60.0,
        ifos=IFOS,
        shifts=E2E_SHIFTS,
        spacing=E2E_SPACING,
        buffer=E2E_BUFFER,
        prior=prior,
        minimum_frequency=20,
        reference_frequency=50,
        sample_rate=E2E_SAMPLE_RATE,
        waveform_duration=E2E_DURATION,
        waveform_approximant="IMRPhenomPv2",
        right_pad=E2E_RIGHT_PAD,
        highpass=32,
        lowpass=None,
        snr_threshold=4,
        psd_file="unused: load_psds is patched",
        max_num_samples=32,
        output_dir=tmp_path / "out",
        jitter=0.1,
        seed=1122,
        **kwargs,
    )


def test_generator_end_to_end_writes_a_ringdown_campaign(
    tmp_path, monkeypatch
):
    waveform_fname, rejected_fname = run_generator(
        tmp_path, monkeypatch, ringdown_prior, waveform_type="ringdown"
    )

    cls = waveform_class_factory(
        IFOS, RingdownInterferometerResponseSet, "ResponseSet"
    )
    loaded = cls.read(waveform_fname)

    size = int(E2E_SAMPLE_RATE * E2E_DURATION)
    assert len(loaded) == 5
    assert loaded.waveforms.shape == (5, len(IFOS), size)
    # the real arithmetic, jittered by at most `jitter` seconds
    np.testing.assert_allclose(
        loaded.injection_time, [9, 19, 29, 39, 49], atol=0.11
    )
    # ringdown parameters, populated and inside the prior's support
    assert ((loaded.frequency >= 100) & (loaded.frequency <= 1000)).all()
    assert ((loaded.quality >= 8) & (loaded.quality <= 20)).all()
    assert "mass_1" not in cls.__dataclass_fields__

    # the rejected file is the ringdown class too. Do NOT assert a count:
    # the number of rejections varies run to run.
    rejected = RingdownInjectionParameterSet.read(rejected_fname)
    assert hasattr(rejected, "frequency")
    assert len(rejected) >= 0


def test_generator_end_to_end_still_writes_a_cbc_campaign(
    tmp_path, monkeypatch
):
    """The default path must be untouched.

    This one passes before Task 4 as well -- it characterises today's CBC
    behaviour so the ringdown work cannot quietly break it. It is the slowest
    test in the file (~6 s; pycbc waveform generation), which is why it uses
    the smaller campaign.
    """
    waveform_fname, rejected_fname = run_generator(
        tmp_path, monkeypatch, end_o3_ratesandpops
    )

    cls = waveform_class_factory(
        IFOS, InterferometerResponseSet, "ResponseSet"
    )
    loaded = cls.read(waveform_fname)

    size = int(E2E_SAMPLE_RATE * E2E_DURATION)
    assert len(loaded) == 5
    assert loaded.waveforms.shape == (5, len(IFOS), size)
    assert (loaded.mass_1 > 0).all()
    assert "frequency" not in cls.__dataclass_fields__

    rejected = InjectionParameterSet.read(rejected_fname)
    assert hasattr(rejected, "mass_1")
