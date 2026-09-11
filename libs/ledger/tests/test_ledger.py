from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from ledger import ledger
from ledger.events import RecoveredInjectionSet
from ledger.injections import (
    InjectionParameterSet,
    RingdownInjectionParameterSet,
)


class TestInjectionSet:
    def _test_read_write(self, obj, tmp_path):
        fname = tmp_path / "obj.h5"

        # test normal write then read
        obj.write(fname)
        new = obj.__class__.read(fname)
        for key, field in obj.__dataclass_fields__.items():
            value = getattr(new, key)
            old = getattr(obj, key)
            truth = value == old
            if field.metadata["kind"] == "metadata":
                assert truth
            else:
                assert truth.all()

        # make sure we catch the error when we try
        # to sample without replacement
        with pytest.raises(ValueError):
            obj.__class__.sample_from_file(fname, len(obj) + 1, replace=False)

        # now test sampling with np.random.choice
        # patched so that we know what to expect.
        # Include duplicate and out-of-order indices
        idx = np.array([1, 0, 2, 1])
        with patch("numpy.random.choice", return_value=idx):
            new = obj.__class__.sample_from_file(fname, 3)
        assert len(new) == 4
        for key, field in obj.__dataclass_fields__.items():
            value = getattr(new, key)
            old = getattr(obj, key)
            kind = field.metadata["kind"]

            if kind == "metadata":
                assert value == old
                continue

            for i in range(4):
                truth = value[i] == old[idx[i]]
                if field.metadata["kind"] == "parameter":
                    assert truth
                else:
                    assert truth.all()

    def test_just_metadata(self):
        @dataclass
        class MetadataClass(ledger.Ledger):
            foo: str = ledger.metadata()
            bar: str = ledger.metadata()

        obj = MetadataClass("hey", "you")
        assert len(obj) == 0

        # TODO: what's the desired slicing
        # behavior on metadata-only objects
        assert obj[1].foo == "hey"

    @pytest.fixture
    def parameter_set(self):
        @dataclass
        class Dummy(ledger.Ledger):
            ids: np.ndarray = ledger.parameter()
            age: np.ndarray = ledger.parameter()

        return Dummy

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        return tmp_path

    def test_parameter_set(self, parameter_set, tmp_dir):
        ids = np.array([1001, 1002, 1003])
        age = np.array([31, 35, 39])
        with pytest.raises(ValueError):
            parameter_set(ids, age[:2])

        obj = parameter_set(ids, age)
        assert len(obj) == 3
        assert next(iter(obj)) == {"ids": 1001, "age": 31}

        subobj = obj[1]
        assert len(subobj) == 1
        assert subobj.ids[0] == 1002
        assert subobj.age[0] == 35

        subobj = obj[[0, 2]]
        assert len(subobj) == 2
        assert subobj.ids[-1] == 1003
        assert subobj.age[-1] == 39

        with pytest.raises(TypeError):
            obj.append([])

        obj.append(obj)
        assert len(obj) == 6
        assert obj.ids[3] == 1001
        assert obj.age[3] == 31

        self._test_read_write(obj, tmp_dir)

    def test_parameter_set_with_metadata(self, parameter_set, tmp_dir):
        @dataclass
        class DummyMetadata(parameter_set):
            foo: str = ledger.metadata()

        ids = np.array([1001, 1002, 1003])
        age = np.array([31, 35, 39])
        obj = DummyMetadata(ids, age, "test")
        assert obj.foo == "test"

        subobj = obj[:2]
        assert len(subobj) == 2
        assert subobj.foo == "test"

        obj2 = DummyMetadata(ids, age, "bar")
        with pytest.raises(ValueError):
            obj.append(obj2)

        @dataclass
        class DummyMetadataCompare(DummyMetadata):
            fuz: str = ledger.metadata()

            def compare_metadata(self, key, ours, theirs):
                if key == "foo":
                    return ours + theirs
                return super().compare_metadata(key, ours, theirs)

        obj = DummyMetadataCompare(ids, age, "test", "qux")
        obj2 = DummyMetadataCompare(ids + 3, age - 2, "bar", "quz")
        with pytest.raises(ValueError):
            obj.append(obj2)
        obj2.fuz = "qux"
        obj.append(obj2)
        assert len(obj) == 6

        self._test_read_write(obj, tmp_dir)

    def test_waveform_set(self, parameter_set, tmp_dir):
        @dataclass
        class DummyWaveform(parameter_set):
            waves: np.ndarray = ledger.waveform()

        ids = np.array([1001, 1002, 1003])
        age = np.array([31, 35, 39])
        waves = np.random.randn(3, 10)

        obj = DummyWaveform(ids, age, waves)
        assert len(obj) == 3

        waves2 = np.random.randn(3, 10)
        obj2 = DummyWaveform(ids + 3, age - 2, waves2)
        obj.append(obj2)
        assert len(obj) == 6

        all_waves = np.concatenate([waves, waves2])
        assert (obj.waves == all_waves).all()
        assert (obj[2:4].waves == all_waves[2:4]).all()

        self._test_read_write(obj, tmp_dir)

    def test_sort(self, parameter_set):
        ids = np.array([1, 2, 3])
        age = np.array([3, 2, 1])
        obj = parameter_set(ids, age)

        assert obj.is_sorted_by("ids")
        assert not obj.is_sorted_by("age")

        obj = obj.sort_by("age")

        assert (obj.age == np.array([1, 2, 3])).all()
        assert (obj.ids == np.array([3, 2, 1])).all()
        assert obj.is_sorted_by("age")
        assert not obj.is_sorted_by("ids")


IFOS = ["H1", "L1"]
EMPTY_SET_CLASSES = [InjectionParameterSet, RingdownInjectionParameterSet]


def _populated(cls, n):
    values = np.arange(n, dtype=float)
    kwargs = {
        name: values.copy()
        for name, attr in cls.__dataclass_fields__.items()
        if attr.metadata["kind"] == "parameter"
    }
    kwargs["ifo_snrs"] = np.repeat(values[:, None], len(IFOS), axis=1)
    kwargs["ifos"] = IFOS
    return cls(**kwargs)


def _empty_2d(cls):
    """An empty ledger whose ifo_snrs keeps its second dimension."""
    kwargs = {
        name: np.array([])
        for name, attr in cls.__dataclass_fields__.items()
        if attr.metadata["kind"] == "parameter"
    }
    kwargs["ifo_snrs"] = np.empty((0, len(IFOS)))
    kwargs["ifos"] = IFOS
    return cls(**kwargs)


@pytest.mark.parametrize("cls", EMPTY_SET_CLASSES)
def test_aggregate_of_only_empty_sources_stays_readable(cls, tmp_path):
    """Every input empty must still yield a valid, readable ledger.

    `rejection_sample` writes `rejected_cls()` for a branch that rejects
    nothing, so `TestingWaveforms` reaches this whenever no branch rejects.
    """
    files = []
    for i in range(2):
        fname = tmp_path / f"empty-{i}.hdf5"
        cls().write(fname)
        files.append(fname)

    merged_file = tmp_path / "merged.hdf5"
    cls.aggregate(files, merged_file, clean=False)

    merged = cls.read(merged_file)
    assert len(merged) == 0
    for name, attr in cls.__dataclass_fields__.items():
        if attr.metadata["kind"] == "parameter":
            assert len(getattr(merged, name)) == 0


@pytest.mark.parametrize("cls", EMPTY_SET_CLASSES)
def test_aggregate_of_only_empty_sources_keeps_dimensions(cls, tmp_path):
    """A 2-D parameter stays 2-D through an all-empty merge.

    `ifo_snrs` is (n_injections, n_ifos). A default `cls()` leaves it (0,),
    but a caller that builds it explicitly gets (0, n_ifos), and the merge
    must not flatten that.
    """
    files = []
    for i in range(2):
        fname = tmp_path / f"empty2d-{i}.hdf5"
        _empty_2d(cls).write(fname)
        files.append(fname)

    merged_file = tmp_path / "merged2d.hdf5"
    cls.aggregate(files, merged_file, clean=False)

    merged = cls.read(merged_file)
    assert len(merged) == 0
    assert merged.ifo_snrs.shape == (0, len(IFOS))
    assert list(merged.ifos) == IFOS


@pytest.mark.parametrize("cls", EMPTY_SET_CLASSES)
@pytest.mark.parametrize("empty_first", [True, False])
def test_aggregate_mixes_a_default_empty_with_populated(
    cls, empty_first, tmp_path
):
    """Regression guard: a default `cls()` beside real rows, either order.

    A default empty has `ifo_snrs` of shape (0,) while a populated set has
    (n, 2). If the empty one is allowed to create the dataset it fixes it at
    one dimension and the populated write fails to broadcast. Both orders
    work today and must keep working.
    """
    empty_file = tmp_path / "empty.hdf5"
    cls().write(empty_file)
    full_file = tmp_path / "full.hdf5"
    _populated(cls, 2).write(full_file)

    files = [empty_file, full_file] if empty_first else [full_file, empty_file]
    merged_file = tmp_path / "mixed.hdf5"
    cls.aggregate(files, merged_file, clean=False)

    merged = cls.read(merged_file)
    assert len(merged) == 2
    assert merged.ifo_snrs.shape == (2, len(IFOS))


def test_aggregate_of_empty_sets_missing_metadata(tmp_path):
    """A source may omit metadata whose value is None.

    `RecoveredInjectionSet()` writes no `sample_rate`, `duration` or
    `right_pad` attribute. Merging such a source must skip what is absent
    rather than raising `KeyError`.
    """
    files = []
    for i in range(2):
        fname = tmp_path / f"recovered-{i}.hdf5"
        RecoveredInjectionSet().write(fname)
        files.append(fname)

    merged_file = tmp_path / "recovered-merged.hdf5"
    RecoveredInjectionSet.aggregate(files, merged_file, clean=False)

    merged = RecoveredInjectionSet.read(merged_file)
    assert len(merged) == 0


def test_aggregate_raises_when_a_populated_source_omits_metadata(tmp_path):
    """Only an *empty* source is allowed to omit metadata.

    The skip above is gated on `source_length == 0` deliberately. Ungated,
    a populated source missing `duration` would inherit whatever the
    previous source declared, silently attributing one campaign's metadata
    to another's rows. Raising is the intended behaviour, so guard it.
    """
    values = np.arange(2, dtype=float)
    kwargs = {
        name: values.copy()
        for name, attr in RecoveredInjectionSet.__dataclass_fields__.items()
        if attr.metadata["kind"] == "parameter"
    }
    kwargs["ifo_snrs"] = np.repeat(values[:, None], len(IFOS), axis=1)
    kwargs["ifos"] = IFOS
    kwargs["num_injections"] = 2
    # __post_init__ validates sample_rate but not duration or right_pad,
    # so those stay None and are left off the file on write
    kwargs["sample_rate"] = 128

    files = []
    for i in range(2):
        fname = tmp_path / f"populated-{i}.hdf5"
        RecoveredInjectionSet(**kwargs).write(fname)
        files.append(fname)

    with pytest.raises(KeyError, match="duration"):
        RecoveredInjectionSet.aggregate(
            files, tmp_path / "populated-merged.hdf5", clean=False
        )
