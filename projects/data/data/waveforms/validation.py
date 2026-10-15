from data.waveforms.rejection import rejection_sample
from ledger.injections import (
    RingdownWaveformSet,
    WaveformSet,
    waveform_class_factory,
)


def main(args):
    args_dict = {k: v for k, v in args.as_dict().items() if k != "config"}
    output_file = args_dict.pop("output_file")

    base_cls = (
        RingdownWaveformSet
        if args_dict.get("waveform_type") == "ringdown"
        else WaveformSet
    )
    cls = waveform_class_factory(args_dict["ifos"], base_cls, "IfoWaveformSet")

    parameters, _ = rejection_sample(**args_dict)
    waveform_set = cls(**parameters)
    waveform_set.write(output_file)
