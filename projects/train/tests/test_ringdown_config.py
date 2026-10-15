"""configs/ringdown.yaml is train.yaml plus the ringdown changes, nothing else.

Anything else that differs is drift, and changes to train.yaml would
stop reaching ringdown runs.
"""

from pathlib import Path

import yaml

PROJECT = Path(__file__).parents[1]


def _load(name):
    with open(PROJECT / name) as f:
        return yaml.safe_load(f)


def test_ringdown_config_differs_from_train_yaml_only_where_intended():
    base = _load("train.yaml")
    ringdown = _load("configs/ringdown.yaml")

    init = ringdown["data"]["init_args"]
    sampler = init["waveform_sampler"]["init_args"]
    # the onset must stay at least right_pad from the kernel's right edge
    assert init["left_pad"] == 0.1
    assert init["right_pad"] == 1.0
    assert sampler["waveform_type"] == "ringdown"

    del sampler["waveform_type"]
    init["left_pad"] = base["data"]["init_args"]["left_pad"]
    init["right_pad"] = base["data"]["init_args"]["right_pad"]
    assert ringdown == base
