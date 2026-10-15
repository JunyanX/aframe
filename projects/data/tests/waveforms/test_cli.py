"""The generation CLIs carry the waveform family into the files they write.

These replace the Luigi task tests from the old branch, which checked that
each task forwarded `waveform_type`. Every Snakemake rule calls one of
these CLIs, so this is where the family has to arrive.
"""

import pytest
import torch
from data.waveforms import cli
from data.waveforms.utils import write_psds
from ledger.injections import (
    RingdownWaveformPolarizationSet,
    RingdownWaveformSet,
    WaveformPolarizationSet,
    waveform_class_factory,
)

IFOS = ["H1", "L1"]
# ringdown_prior reaches 1 kHz, so stay above twice that
SAMPLE_RATE = 2048
DURATION = 2
RIGHT_PAD = 1


def _cbc_args():
    return [
        "--minimum_frequency",
        "20",
        "--reference_frequency",
        "50",
        "--waveform_approximant",
        "IMRPhenomPv2",
    ]


@pytest.mark.parametrize(
    "parser", [cli.training_parser, cli.testing_parser, cli.validation_parser]
)
def test_every_generation_cli_defaults_to_cbc(parser):
    assert parser.get_defaults().waveform_type == "cbc"


def test_training_cli_writes_ringdown_polarizations(tmp_path):
    out = tmp_path / "training.hdf5"
    cli.training(
        [
            "--num_signals",
            "4",
            "--sample_rate",
            str(SAMPLE_RATE),
            "--waveform_duration",
            str(DURATION),
            "--prior",
            "priors.priors.ringdown_prior",
            "--right_pad",
            str(RIGHT_PAD),
            "--waveform_type",
            "ringdown",
            "--output_file",
            str(out),
            *_cbc_args(),
        ]
    )

    waveforms = RingdownWaveformPolarizationSet.read(out)
    assert len(waveforms) == 4
    assert waveforms.cross.shape == (4, SAMPLE_RATE * DURATION)
    # the wrong family must fail to read rather than mislabel
    with pytest.raises(ValueError):
        WaveformPolarizationSet.read(out)


def test_validation_cli_writes_a_ringdown_waveform_set(tmp_path):
    psd_file = tmp_path / "psd.hdf5"
    num_freqs = SAMPLE_RATE * DURATION // 2 + 1
    # flat, at roughly aLIGO design amplitude: louder and rejection
    # sampling never reaches its quota
    psds = 1e-46 * torch.ones((len(IFOS), num_freqs), dtype=torch.float64)
    write_psds(psds, psd_file, IFOS, 1 / DURATION)

    out = tmp_path / "val.hdf5"
    cli.validation(
        [
            "--num_signals",
            "3",
            "--prior",
            "priors.priors.ringdown_prior",
            "--ifos",
            "[H1,L1]",
            "--sample_rate",
            str(SAMPLE_RATE),
            "--waveform_duration",
            str(DURATION),
            "--right_pad",
            str(RIGHT_PAD),
            "--highpass",
            "32",
            "--lowpass",
            "null",
            "--snr_threshold",
            "4",
            "--psd",
            str(psd_file),
            "--max_num_samples",
            "32",
            "--seed",
            "1122",
            "--waveform_type",
            "ringdown",
            "--output_file",
            str(out),
            *_cbc_args(),
        ]
    )

    cls = waveform_class_factory(IFOS, RingdownWaveformSet, "IfoWaveformSet")
    waveforms = cls.read(out)
    assert len(waveforms) == 3
    assert ((waveforms.frequency >= 40) & (waveforms.frequency <= 1000)).all()
