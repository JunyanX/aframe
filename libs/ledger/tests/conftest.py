import pytest

from ledger.injections import (
    InterferometerResponseSet,
    RingdownInterferometerResponseSet,
    waveform_class_factory,
)


@pytest.fixture
def response_set_cls():
    return waveform_class_factory(
        ["h1", "l1"], InterferometerResponseSet, cls_name="LigoResponseSet"
    )


@pytest.fixture
def ringdown_response_set_cls():
    return waveform_class_factory(
        ["h1", "l1"],
        RingdownInterferometerResponseSet,
        cls_name="RingdownLigoResponseSet",
    )
