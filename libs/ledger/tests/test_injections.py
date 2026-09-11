from unittest.mock import patch

import numpy as np
import pytest

from ledger.injections import (
    BilbyParameterSet,
    InjectionParameterSet,
    InterferometerResponseSet,
    RingdownInterferometerResponseSet,
    RingdownWaveformPolarizationSet,
    RingdownWaveformSet,
    WaveformSet,
    _WaveformGenerator,
    waveform_class_factory,
)


@pytest.fixture
def reference_frequency():
    return 20


@pytest.fixture
def bilby_param_set():
    params = {
        "mass_1": np.array([35]),
        "mass_2": np.array([20]),
        "a_1": np.array([0.5]),
        "a_2": np.array([0]),
        "tilt_1": np.array([np.pi / 2]),
        "tilt_2": np.array([np.pi / 2]),
        "phi_12": np.array([np.pi]),
        "phi_jl": np.array([np.pi]),
        "ra": np.array([np.pi / 4]),
        "dec": np.array([np.pi / 4]),
        "redshift": np.array([1]),
        "psi": np.array([np.pi / 8]),
        "theta_jn": np.array([np.pi / 8]),
        "phase": np.array([np.pi / 8]),
    }
    return BilbyParameterSet(**params)


def test_parameter_conversion(bilby_param_set, reference_frequency):
    lal_params = bilby_param_set.convert_to_lal_param_set(reference_frequency)
    new_params = lal_params.convert_to_bilby_param_set(reference_frequency)
    for key in new_params._get_params():
        np.allclose(
            getattr(bilby_param_set, key), getattr(new_params, key), rtol=1e-16
        )

    for _ in range(100):
        new_params = new_params.convert_to_lal_param_set(reference_frequency)
        new_params = new_params.convert_to_bilby_param_set(reference_frequency)

    for key in new_params._get_params():
        np.allclose(
            getattr(bilby_param_set, key), getattr(new_params, key), rtol=1e-16
        )


class TestRingdownWaveformPolarizationSet:
    @pytest.fixture
    def ringdown_waveforms(self):
        num_waveforms = 2
        sample_rate = 128
        duration = 1
        waveform_size = int(sample_rate * duration)
        return RingdownWaveformPolarizationSet(
            frequency=np.array([20, 30]),
            quality=np.array([10, 15]),
            epsilon=np.array([0.01, 0.02]),
            phase=np.array([0, np.pi / 4]),
            inclination=np.array([np.pi / 3, np.pi / 2]),
            distance=np.array([100, 200]),
            ra=np.array([0.1, 0.2]),
            dec=np.array([-0.1, 0.1]),
            psi=np.array([0.3, 0.4]),
            cross=np.ones((num_waveforms, waveform_size)),
            plus=2 * np.ones((num_waveforms, waveform_size)),
            sample_rate=sample_rate,
            duration=duration,
            right_pad=0.5,
            num_injections=num_waveforms,
        )

    def test_waveforms(self, ringdown_waveforms):
        waveforms = ringdown_waveforms.get_waveforms()

        assert len(ringdown_waveforms) == 2
        assert ringdown_waveforms.waveform_duration == 1
        assert waveforms.shape == (2, 2, 128)
        np.testing.assert_array_equal(waveforms[:, 0], 1)
        np.testing.assert_array_equal(waveforms[:, 1], 2)

    def test_hdf5_round_trip(self, ringdown_waveforms, tmp_path):
        fname = tmp_path / "ringdown-waveforms.hdf5"
        ringdown_waveforms.write(fname)
        loaded = RingdownWaveformPolarizationSet.read(fname)

        assert len(loaded) == len(ringdown_waveforms)
        for name, field in ringdown_waveforms.__dataclass_fields__.items():
            expected = getattr(ringdown_waveforms, name)
            actual = getattr(loaded, name)
            if field.metadata["kind"] == "metadata":
                assert actual == expected
            else:
                np.testing.assert_array_equal(actual, expected)

    def test_rejects_unequal_parameter_lengths(self, ringdown_waveforms):
        kwargs = {
            name: getattr(ringdown_waveforms, name)
            for name in ringdown_waveforms.__dataclass_fields__
        }
        kwargs["frequency"] = np.array([20])

        with pytest.raises(ValueError, match="entries, expected"):
            RingdownWaveformPolarizationSet(**kwargs)

    def test_rejects_inconsistent_waveform_duration(self, ringdown_waveforms):
        kwargs = {
            name: getattr(ringdown_waveforms, name)
            for name in ringdown_waveforms.__dataclass_fields__
        }
        kwargs["cross"] = np.ones((2, 64))

        with pytest.raises(ValueError, match="Specified waveform duration"):
            RingdownWaveformPolarizationSet(**kwargs)


class TestWaveformGenerator:
    @pytest.fixture
    def waveform_duration(self):
        return 8

    @pytest.fixture
    def sample_rate(self):
        return 2048

    @pytest.fixture(params=[0, 0.5, 1, 4])
    def right_pad(self, request):
        return request.param

    @pytest.fixture(params=[4, 10])
    def dummy_signal_duration(self, request):
        return request.param

    @pytest.fixture
    def dummy_signal(
        self, dummy_signal_duration, sample_rate, waveform_duration
    ):
        # Odd number of points to remove ambiguity of peak
        return np.bartlett(dummy_signal_duration * sample_rate - 1)[None]

    def test_align_waveforms(
        self,
        sample_rate,
        waveform_duration,
        right_pad,
        dummy_signal,
    ):
        if right_pad == 0:
            right_pad += 1 / sample_rate
        gen = _WaveformGenerator(
            waveform_approximant="",
            sample_rate=sample_rate,
            waveform_duration=waveform_duration,
            right_pad=right_pad,
            minimum_frequency=20,
            reference_frequency=20,
        )
        # If the peak is at t=0, the final time is half the signal length
        t_final = dummy_signal.shape[-1] // 2 / sample_rate
        waveforms = gen.align_waveforms(dummy_signal, t_final)
        assert waveforms.shape[-1] == int(sample_rate * waveform_duration)
        assert np.argmax(waveforms) == int(
            (waveform_duration - right_pad) * sample_rate
        )


class TestLigoResponseSet:
    @pytest.fixture
    def duration(self):
        return 4

    @pytest.fixture
    def sample_rate(self):
        return 128

    @pytest.fixture
    def N(self):
        return 10

    @pytest.fixture
    def ligo_response_set(self, response_set_cls, duration, sample_rate, N):
        size = int(duration * sample_rate)
        params = {}
        bad_waveforms = {}
        waveforms = {}
        fields = response_set_cls.__dataclass_fields__
        for name, attr in fields.items():
            if attr.metadata["kind"] == "parameter":
                params[name] = np.zeros((N,))
            elif attr.metadata["kind"] == "waveform":
                waveforms[name] = np.ones((N, size))
                bad_waveforms[name] = np.ones((N, size // 2))

        with pytest.raises(ValueError) as exc:
            kwargs = {}
            kwargs.update(params)
            kwargs.update(bad_waveforms)
            response_set_cls(
                sample_rate=sample_rate,
                duration=duration,
                num_injections=N,
                right_pad=duration / 2,
                **kwargs,
            )
        assert str(exc.value).startswith("Specified waveform duration")

        with pytest.raises(ValueError) as exc:
            kwargs = {}
            kwargs.update(params)
            kwargs.update(waveforms)
            response_set_cls(
                sample_rate=sample_rate,
                duration=duration,
                num_injections=N - 1,
                right_pad=duration / 2,
                **kwargs,
            )
        assert str(exc.value).startswith("LigoResponseSet")

        return response_set_cls(
            sample_rate=sample_rate,
            duration=duration,
            num_injections=N,
            right_pad=duration / 2,
            **kwargs,
        )

    def test_waveforms(self, ligo_response_set, sample_rate, duration, N):
        size = int(sample_rate * duration)
        expected_shape = (N, 2, size)

        assert ligo_response_set._waveforms is None
        assert ligo_response_set.waveforms.shape == expected_shape
        assert ligo_response_set._waveforms.shape == expected_shape
        with patch("numpy.stack") as mock:
            _ = ligo_response_set.waveforms
        mock.assert_not_called()

    def test_get_times(self, ligo_response_set, N):
        ligo_response_set.injection_time = np.arange(N)
        with pytest.raises(ValueError):
            ligo_response_set.get_times()

        obj = ligo_response_set.get_times(start=2)
        assert len(obj) == 8
        assert obj.num_injections == N

        obj = ligo_response_set.get_times(end=6)
        assert len(obj) == 6
        assert obj.num_injections == N

        obj = ligo_response_set.get_times(2, 6.5)
        assert len(obj) == 5
        assert obj.num_injections == N

    def test_read(self, response_set_cls, ligo_response_set, tmp_path, N):
        tmp_path.mkdir(exist_ok=True)
        fname = tmp_path / "obj.h5"

        ligo_response_set.injection_time = np.arange(N)
        ligo_response_set.write(fname)

        # TODO: generalize logic here to duration
        new = response_set_cls.read(fname, start=2.5)
        assert len(new) == N - 1
        assert (new.injection_time == np.arange(1, N)).all()

        new = response_set_cls.read(fname, end=6)
        assert len(new) == N - 1
        assert (new.injection_time == np.arange(N - 1)).all()

        new = response_set_cls.read(fname, start=3.5, end=5)
        assert len(new) == 6
        assert (new.injection_time == np.arange(2, 8)).all()

    def test_read_with_shifts(self, ligo_response_set, tmp_path, N):
        ligo_response_set.injection_time = np.arange(N)
        ligo_response_set.shift = np.zeros((N,))
        ligo_response_set.shift[N // 2 :] = 1

        tmp_path.mkdir(exist_ok=True)
        fname = tmp_path / "obj.h5"
        ligo_response_set.write(fname)

        # test 1D behavior with a single shift
        new = ligo_response_set.read(fname, shifts=[0])
        assert len(new) == N // 2
        assert (new.shift == 0).all()
        assert (new.injection_time == np.arange(N // 2)).all()

        # loading 1D with 2D indices fails
        with pytest.raises(ValueError):
            new = ligo_response_set.read(fname, shifts=[[0]])

        # try multiple shifts
        new = ligo_response_set.read(fname, shifts=[0, 1])
        assert len(new) == N
        assert (new.injection_time == np.arange(N)).all()

        # now write a response set with 2D shifts
        ligo_response_set.shift = np.zeros((N, 2))
        ligo_response_set.shift[N // 2 :, 1] = 1
        ligo_response_set.write(fname)

        # 1D read of a 2D shift
        new = ligo_response_set.read(fname, shifts=[0, 0])
        assert len(new) == N // 2
        assert (new.shift == 0).all()
        assert (new.injection_time == np.arange(N // 2)).all()

        # too many shifts raises error
        with pytest.raises(ValueError) as exc:
            ligo_response_set.read(fname, shifts=[0, 0, 0])
        assert str(exc.value).startswith("Specified 3 shifts")

        # single 2D read of a 2D shift
        new = ligo_response_set.read(fname, shifts=[[0, 0]])
        assert len(new) == N // 2
        assert (new.shift == 0).all()
        assert (new.injection_time == np.arange(N // 2)).all()

        # multiple 2D read of a 2D shift
        new = ligo_response_set.read(fname, shifts=[[0, 0], [0, 1]])
        assert len(new) == N
        assert (new.injection_time == np.arange(N)).all()

        # now try with times
        ligo_response_set.injection_time = np.arange(N) % (N // 2)
        ligo_response_set.duration = 2
        ligo_response_set.right_pad = ligo_response_set.duration / 2
        for ifo in "hl":
            key = f"{ifo}1"
            old = getattr(ligo_response_set, key)
            new = old[:, : old.shape[-1] // 2]
            setattr(ligo_response_set, key, new)
        ligo_response_set.write(fname)

        new = ligo_response_set.read(fname, shifts=[0, 1], start=2.5, end=3.5)
        assert len(new) == 3
        assert (new.injection_time == np.arange(2, 5)).all()

        new = ligo_response_set.read(
            fname, shifts=[[0, 0], [0, 1]], start=2.5, end=3.5
        )
        assert len(new) == 6

    def test_append(self, ligo_response_set, N):
        ligo_response_set.injection_time = np.arange(N)
        _ = ligo_response_set.waveforms

        new = ligo_response_set[:6]
        new.num_injections = 13
        new.injection_time = np.arange(N, N + 6)
        new.h1 *= 2
        new.l1 *= 2

        ligo_response_set.append(new)
        assert ligo_response_set.num_injections == N + 13
        assert (ligo_response_set.injection_time == np.arange(N + 6)).all()
        assert (ligo_response_set.h1[N:] == 2).all()
        assert (ligo_response_set.l1[N:] == 2).all()
        assert ligo_response_set._waveforms is None

    def test_inject(self, ligo_response_set, sample_rate, duration, N):
        ligo_response_set.injection_time = (duration + 1) * np.arange(10)
        ligo_response_set.h1 += np.arange(N)[:, None]
        ligo_response_set.l1 += np.arange(N)[:, None]
        ligo_response_set.l1 *= -1

        start = -1
        length = 27
        x = np.zeros((2, length * sample_rate))
        y = ligo_response_set.inject(x, start)
        assert (y[0, : (duration - 1) * sample_rate] == 1).all()
        assert (y[1, : (duration - 1) * sample_rate] == -1).all()

        offset = (duration - 1) * sample_rate
        for i in range(5):
            zero_start = offset + i * (duration + 1) * sample_rate
            zero_end = zero_start + sample_rate
            assert not y[:, zero_start:zero_end].any()

            wave_start = zero_end
            wave_end = wave_start + duration * sample_rate
            assert (y[0, wave_start:wave_end] == i + 2).all()
            assert (-y[1, wave_start:wave_end] == i + 2).all()

            if i == 4:
                assert wave_end == (x.shape[-1] + sample_rate)


# Field order is part of the on-disk contract for every archive written
# before the SnrParameterSet mixin was extracted, so pin it explicitly.
CBC_INJECTION_FIELDS = [
    "mass_1",
    "mass_2",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "ra",
    "dec",
    "redshift",
    "psi",
    "theta_jn",
    "phase",
    "snr",
    "ifo_snrs",
    "ifos",
]


def test_snr_mixin_preserves_cbc_field_order():
    fields = list(InjectionParameterSet.__dataclass_fields__)
    assert fields == CBC_INJECTION_FIELDS


CBC_RESPONSE_SET_FIELDS = [
    "mass_1",
    "mass_2",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "ra",
    "dec",
    "redshift",
    "psi",
    "theta_jn",
    "phase",
    "snr",
    "ifo_snrs",
    "ifos",
    "sample_rate",
    "duration",
    "right_pad",
    "num_injections",
    "injection_time",
    "shift",
]


def test_response_set_field_order_is_stable():
    """Splitting the timing behaviour out must not reorder the fields.

    Ledger reads and writes by name, but `RecoveredInjectionSet`
    (ledger/events.py:240) mixes this class with `EventSet`, and the
    dataclass MRO decides the field order that both of them see.
    """
    assert (
        list(InterferometerResponseSet.__dataclass_fields__)
        == CBC_RESPONSE_SET_FIELDS
    )


def test_response_set_factory_appends_ifo_fields(response_set_cls):
    assert list(response_set_cls.__dataclass_fields__) == (
        CBC_RESPONSE_SET_FIELDS + ["h1", "l1"]
    )


def test_mismatched_waveform_classes_fail_loudly(tmp_path):
    """Reading a file with the wrong ledger class must raise, not mis-load.

    The validation consumer adds waveform channels straight onto detector
    background, so a partial or mis-channelled load would corrupt training
    silently. Every declared field is looked up by name at read time, so a
    mismatch raises instead.
    """
    num, sample_rate, duration = 2, 128, 1
    size = int(sample_rate * duration)
    params = {
        "frequency": np.array([200.0, 300.0]),
        "quality": np.array([10.0, 15.0]),
        "epsilon": np.array([0.01, 0.02]),
        "phase": np.array([0.0, 0.5]),
        "inclination": np.array([0.3, 0.7]),
        "distance": np.array([100.0, 200.0]),
        "ra": np.array([0.1, 0.2]),
        "dec": np.array([-0.1, 0.1]),
        "psi": np.array([0.3, 0.4]),
    }
    meta = {
        "sample_rate": sample_rate,
        "duration": duration,
        "right_pad": 0.5,
        "num_injections": num,
    }

    polarizations = RingdownWaveformPolarizationSet(
        **params,
        cross=np.ones((num, size)),
        plus=2 * np.ones((num, size)),
        **meta,
    )
    polarization_file = tmp_path / "polarizations.hdf5"
    polarizations.write(polarization_file)

    ringdown_cls = waveform_class_factory(
        ["h1", "l1"], RingdownWaveformSet, "IfoRingdownWaveformSet"
    )
    per_ifo = ringdown_cls(
        **params,
        snr=np.array([8.0, 12.0]),
        ifo_snrs=np.array([[6.0, 5.0], [9.0, 8.0]]),
        ifos=["h1", "l1"],
        h1=np.ones((num, size)),
        l1=2 * np.ones((num, size)),
        **meta,
    )
    per_ifo_file = tmp_path / "per_ifo.hdf5"
    per_ifo.write(per_ifo_file)

    # a training-style cross/plus file is not a validation file
    with pytest.raises(ValueError, match="no dataset"):
        ringdown_cls.read(polarization_file)

    # and vice versa
    with pytest.raises(ValueError, match="no dataset"):
        RingdownWaveformPolarizationSet.read(per_ifo_file)

    # nor is a ringdown validation file readable as a CBC one
    cbc_cls = waveform_class_factory(
        ["h1", "l1"], WaveformSet, "IfoWaveformSet"
    )
    with pytest.raises(ValueError, match="no dataset"):
        cbc_cls.read(per_ifo_file)


def test_ringdown_set_carries_no_cbc_parameters():
    fields = set(RingdownWaveformSet.__dataclass_fields__)
    cbc_only = {
        "mass_1",
        "mass_2",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "redshift",
        "theta_jn",
    }
    assert not fields & cbc_only
    assert {"snr", "ifo_snrs", "ifos"} <= fields


class TestRingdownWaveformSet:
    @pytest.fixture
    def ifo_ringdown_cls(self):
        return waveform_class_factory(
            ["h1", "l1"],
            RingdownWaveformSet,
            cls_name="IfoRingdownWaveformSet",
        )

    @pytest.fixture
    def ringdown_set(self, ifo_ringdown_cls):
        num_waveforms = 2
        sample_rate = 128
        duration = 1
        waveform_size = int(sample_rate * duration)
        return ifo_ringdown_cls(
            frequency=np.array([20, 30]),
            quality=np.array([10, 15]),
            epsilon=np.array([0.01, 0.02]),
            phase=np.array([0, np.pi / 4]),
            inclination=np.array([np.pi / 3, np.pi / 2]),
            distance=np.array([100, 200]),
            ra=np.array([0.1, 0.2]),
            dec=np.array([-0.1, 0.1]),
            psi=np.array([0.3, 0.4]),
            snr=np.array([8.0, 12.0]),
            ifo_snrs=np.array([[6.0, 5.0], [9.0, 8.0]]),
            ifos=["h1", "l1"],
            h1=np.ones((num_waveforms, waveform_size)),
            l1=2 * np.ones((num_waveforms, waveform_size)),
            sample_rate=sample_rate,
            duration=duration,
            right_pad=0.5,
            num_injections=num_waveforms,
        )

    def test_waveforms_are_per_ifo_in_alphabetical_order(self, ringdown_set):
        waveforms = ringdown_set.waveforms

        assert len(ringdown_set) == 2
        assert ringdown_set.num_waveform_fields() == 2
        assert sorted(ringdown_set.waveform_fields) == ["h1", "l1"]
        assert waveforms.shape == (2, 2, 128)
        # Channels are stacked alphabetically by field name, while the
        # validation background is stacked in config `ifos` order and the two
        # are added element-wise. They only coincide when `ifos` is already
        # sorted, so pin the order the ledger side promises.
        np.testing.assert_array_equal(waveforms[:, 0], 1)
        np.testing.assert_array_equal(waveforms[:, 1], 2)

    def test_carries_snr_fields(self, ringdown_set):
        assert ringdown_set.ifos == ["h1", "l1"]
        np.testing.assert_array_equal(ringdown_set.snr, [8.0, 12.0])
        assert ringdown_set.ifo_snrs.shape == (2, 2)

    def test_hdf5_round_trip(self, ringdown_set, ifo_ringdown_cls, tmp_path):
        fname = tmp_path / "ringdown-val-waveforms.hdf5"
        ringdown_set.write(fname)
        loaded = ifo_ringdown_cls.read(fname)

        assert len(loaded) == len(ringdown_set)
        for name, field in ringdown_set.__dataclass_fields__.items():
            expected = getattr(ringdown_set, name)
            actual = getattr(loaded, name)
            if field.metadata["kind"] == "metadata":
                if name == "ifos":
                    assert list(actual) == list(expected)
                else:
                    assert actual == expected
            else:
                np.testing.assert_array_equal(actual, expected)

    def test_rejects_inconsistent_waveform_duration(self, ringdown_set):
        kwargs = {
            name: getattr(ringdown_set, name)
            for name in ringdown_set.__dataclass_fields__
        }
        kwargs["h1"] = np.ones((2, 64))

        with pytest.raises(ValueError, match="Specified waveform duration"):
            type(ringdown_set)(**kwargs)


RINGDOWN_RESPONSE_SET_FIELDS = [
    "frequency",
    "quality",
    "epsilon",
    "phase",
    "inclination",
    "distance",
    "ra",
    "dec",
    "psi",
    "snr",
    "ifo_snrs",
    "ifos",
    "sample_rate",
    "duration",
    "right_pad",
    "num_injections",
    "injection_time",
    "shift",
]


class TestRingdownResponseSet:
    """The ringdown counterpart of TestLigoResponseSet.

    Only covers what the split has to preserve: the parameter schema, and
    the four timing/injection methods now inherited from ResponseSetBase.
    """

    @pytest.fixture
    def duration(self):
        return 2

    @pytest.fixture
    def sample_rate(self):
        return 128

    @pytest.fixture
    def ringdown_response_cls(self):
        return waveform_class_factory(
            ["h1", "l1"],
            RingdownInterferometerResponseSet,
            cls_name="LigoRingdownResponseSet",
        )

    @staticmethod
    def build(cls, times, sample_rate, duration, shifts=None, frequency=None):
        """A response set with a distinct constant waveform per detector.

        h1 is filled with 1.0 and l1 with 2.0 so an injection test can tell
        the channels apart and assert exact sample placement.
        """
        n = len(times)
        size = int(duration * sample_rate)
        kwargs = {}
        for name, attr in cls.__dataclass_fields__.items():
            kind = attr.metadata["kind"]
            if kind == "parameter":
                kwargs[name] = np.zeros((n,))
            elif kind == "waveform":
                kwargs[name] = np.zeros((n, size))

        kwargs["injection_time"] = np.asarray(times, dtype=float)
        if shifts is None:
            shifts = np.repeat(np.array([[0.0, 1.0]]), n, axis=0)
        kwargs["shift"] = np.asarray(shifts, dtype=float)
        if frequency is not None:
            kwargs["frequency"] = np.asarray(frequency, dtype=float)
        kwargs["h1"] = np.ones((n, size))
        kwargs["l1"] = 2 * np.ones((n, size))

        return cls(
            sample_rate=sample_rate,
            duration=duration,
            num_injections=n,
            right_pad=duration / 2,
            **kwargs,
        )

    @pytest.fixture
    def response_set(self, ringdown_response_cls, duration, sample_rate):
        return self.build(
            ringdown_response_cls,
            [10.0, 20.0, 30.0, 40.0],
            sample_rate,
            duration,
            shifts=[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
            frequency=[200.0, 210.0, 220.0, 230.0],
        )

    def test_field_order(self, ringdown_response_cls):
        assert list(ringdown_response_cls.__dataclass_fields__) == (
            RINGDOWN_RESPONSE_SET_FIELDS + ["h1", "l1"]
        )

    def test_carries_no_cbc_parameters(self, ringdown_response_cls):
        fields = set(ringdown_response_cls.__dataclass_fields__)
        assert not fields & {"mass_1", "mass_2", "a_1", "theta_jn", "redshift"}

    def test_get_shift(self, response_set):
        shifted = response_set.get_shift(np.array([0.0, 1.0]))
        assert len(shifted) == 2
        np.testing.assert_array_equal(shifted.injection_time, [20.0, 40.0])
        np.testing.assert_array_equal(shifted.frequency, [210.0, 230.0])

    def test_get_times(self, response_set):
        within = response_set.get_times(start=15.0, end=35.0)
        np.testing.assert_array_equal(within.injection_time, [20.0, 30.0])

    def test_read_round_trip_with_shifts(
        self, response_set, ringdown_response_cls, tmp_path
    ):
        fname = tmp_path / "ringdown-response-set.hdf5"
        response_set.write(fname)

        loaded = ringdown_response_cls.read(fname)
        assert len(loaded) == 4
        np.testing.assert_array_equal(loaded.frequency, response_set.frequency)

        sliced = ringdown_response_cls.read(fname, shifts=[0.0, 1.0])
        assert len(sliced) == 2
        np.testing.assert_array_equal(sliced.frequency, [210.0, 230.0])

    def test_inject_writes_each_detector_at_the_expected_samples(
        self, response_set, sample_rate
    ):
        """Assert placement, not merely that something was written.

        A waveform spans `duration` seconds and is anchored so its first
        sample lands `duration - right_pad` before the injection time. With
        sample_rate=128, duration=2, right_pad=1 that puts an injection at
        t=10 s in samples [10*128 - 128, +256) = [1152, 1408). The spans
        below were confirmed by running `inject` against these inputs.
        """
        background = np.zeros((2, 60 * sample_rate))

        injected = response_set.inject(background, start=0.0)

        assert injected.shape == background.shape
        spans = [(1152, 1408), (2432, 2688), (3712, 3968), (4992, 5248)]
        for channel, value in ((0, 1.0), (1, 2.0)):
            marked = np.zeros(injected.shape[-1], dtype=bool)
            for lo, hi in spans:
                np.testing.assert_allclose(injected[channel, lo:hi], value)
                marked[lo:hi] = True
            # and nothing anywhere else
            np.testing.assert_allclose(injected[channel, ~marked], 0.0)

    def test_inject_trims_an_injection_that_precedes_the_chunk(
        self, ringdown_response_cls, sample_rate, duration
    ):
        """An injection whose window opens before the chunk start.

        `inject` pads the array, writes, then trims back. At t=0.5 s the
        window opens at -0.5 s, so the leading 64 samples fall outside the
        chunk and the remaining 192 land at the very start of the output.
        """
        response_set = self.build(
            ringdown_response_cls, [0.5], sample_rate, duration
        )
        background = np.zeros((2, 10 * sample_rate))

        injected = response_set.inject(background, start=0.0)

        assert injected.shape == background.shape
        for channel, value in ((0, 1.0), (1, 2.0)):
            np.testing.assert_allclose(injected[channel, :192], value)
            np.testing.assert_allclose(injected[channel, 192:], 0.0)

    def test_inject_trims_an_injection_that_overruns_the_chunk(
        self, ringdown_response_cls, sample_rate, duration
    ):
        """The mirror case: at t=9.5 s in a 10 s chunk the window closes at
        10.5 s, so the trailing 64 samples are cut and 192 land at the end.
        """
        response_set = self.build(
            ringdown_response_cls, [9.5], sample_rate, duration
        )
        background = np.zeros((2, 10 * sample_rate))

        injected = response_set.inject(background, start=0.0)

        assert injected.shape == background.shape
        for channel, value in ((0, 1.0), (1, 2.0)):
            np.testing.assert_allclose(injected[channel, 1088:], value)
            np.testing.assert_allclose(injected[channel, :1088], 0.0)
