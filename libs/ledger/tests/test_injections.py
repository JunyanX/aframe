from unittest.mock import patch

import numpy as np
import pytest
from ledger.injections import (
    BilbyParameterSet,
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
    def ringdown_response_cls(self):
        return waveform_class_factory(
            ["h1", "l1"],
            RingdownInterferometerResponseSet,
            cls_name="LigoRingdownResponseSet",
        )

    def test_field_order(self, ringdown_response_cls):
        assert list(ringdown_response_cls.__dataclass_fields__) == (
            RINGDOWN_RESPONSE_SET_FIELDS + ["h1", "l1"]
        )


class TestRingdownRemnantProperties:
    @pytest.fixture
    def params(self):
        from ledger.injections import RingdownParameterSet

        n = 3
        return RingdownParameterSet(
            frequency=np.array([150.0, 300.0, 600.0]),
            quality=np.array([10.0, 14.0, 18.0]),
            epsilon=np.full(n, 0.05),
            phase=np.zeros(n),
            inclination=np.zeros(n),
            distance=np.array([100.0, 500.0, 1000.0]),
            ra=np.zeros(n),
            dec=np.zeros(n),
            psi=np.zeros(n),
        )

    def test_remnant_mass_known_values(self, params):
        # ml4gw ringdown.py:79-84 inverted, converted to Msun
        np.testing.assert_allclose(
            params.remnant_mass,
            np.array([169.0, 89.16, 46.01]),
            rtol=1e-3,
        )

    def test_roundtrip_mass_spin_to_frequency_quality(self, params):
        from ledger.injections import MSUN, C, G

        m, chi = params.remnant_mass, params.remnant_spin
        q = 2 * (1 - chi) ** -0.45
        f = (
            (1 / (2 * np.pi))
            * (C**3 / (G * m * MSUN))
            * (1 - 0.63 * (1 - chi) ** 0.3)
        )
        np.testing.assert_allclose(q, params.quality, rtol=1e-10)
        np.testing.assert_allclose(f, params.frequency, rtol=1e-10)

    def test_redshift_across_the_prior_range(self):
        from astropy.cosmology import z_at_value
        from astropy.units import Mpc
        from ledger.injections import RingdownParameterSet
        from utils.cosmology import DEFAULT_COSMOLOGY

        # ringdown_prior draws distance from 100 Mpc to 20 Gpc
        distance = np.concatenate(
            [[100.0, 20000.0], np.random.default_rng(0).uniform(100, 2e4, 200)]
        )
        n = len(distance)
        params = RingdownParameterSet(
            frequency=np.full(n, 150.0),
            quality=np.full(n, 10.0),
            epsilon=np.full(n, 0.05),
            phase=np.zeros(n),
            inclination=np.zeros(n),
            distance=distance,
            ra=np.zeros(n),
            dec=np.zeros(n),
            psi=np.zeros(n),
        )
        expected = z_at_value(
            DEFAULT_COSMOLOGY.luminosity_distance, distance * Mpc
        ).value
        np.testing.assert_allclose(params.redshift, expected, rtol=1e-6)

    def test_properties_survive_append(self, params):
        from ledger.injections import RingdownParameterSet

        _ = params.remnant_mass_source  # force one evaluation first
        params.append(
            RingdownParameterSet(
                frequency=np.array([200.0]),
                quality=np.array([12.0]),
                epsilon=np.array([0.05]),
                phase=np.array([0.0]),
                inclination=np.array([0.0]),
                distance=np.array([300.0]),
                ra=np.array([0.0]),
                dec=np.array([0.0]),
                psi=np.array([0.0]),
            )
        )
        assert len(params.remnant_mass_source) == 4
        assert len(params.redshift) == 4

    def test_redshift_empty_ledger(self):
        from ledger.injections import RingdownParameterSet

        empty = RingdownParameterSet()
        redshift = empty.redshift
        assert redshift.shape == (0,)
        assert redshift.dtype == np.float64
