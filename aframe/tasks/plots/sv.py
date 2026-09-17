import math
import os

import law
import luigi

from aframe.base import AframeSingularityTask
from aframe.config import paths
from aframe.parameters import PathParameter, load_prior
from aframe.tasks.data.waveforms.testing import DeployTestingWaveforms
from aframe.tasks.infer import Infer


class SensitiveVolume(AframeSingularityTask):
    """
    Compute and plot the sensitive volume of an aframe analysis
    """

    ifos = luigi.ListParameter(
        description="List of interferometers used in analysis"
    )
    mass_combos = luigi.ListParameter(
        description="Mass combinations for which to calculate sensitive volume"
    )
    source_prior = luigi.Parameter(
        description="Python path to prior used for generating testing "
        "waveform injections"
    )
    dt = luigi.FloatParameter(
        default=math.inf,
        description="Time difference to enforce "
        "between injected and recovered events",
    )
    waveform_type = luigi.ChoiceParameter(
        default="cbc",
        choices=("cbc", "ringdown"),
        description="Waveform family of the testing campaign",
    )
    remnant_mass_combos = luigi.ListParameter(
        default=[],
        description="Median source-frame remnant masses in Msun, one per "
        "panel. Ringdown only. Medians, not arithmetic means.",
    )
    sigma = luigi.FloatParameter(
        default=0.1,
        description="Standard deviation of log mass for the target log "
        "normal. Dimensionless, not a width in Msun.",
    )
    max_far = luigi.FloatParameter(
        default=365,
        description="Maximum false alarm rate in yr^-1 to plot out to",
    )
    coverage_floor = luigi.FloatParameter(
        default=0.95,
        description="Support-coverage warning threshold. Reporting only: "
        "changing it never changes weights, curves or errors.",
    )
    ess_floor = luigi.FloatParameter(
        default=10,
        description="Minimum effective sample size per target. Rejects "
        "statistically useless targets that are numerically fine.",
    )
    output_dir = PathParameter(
        description="Path to the directory to save the output plots and data",
        default=paths().results_dir / "plots",
    )

    @property
    def default_image(self):
        return "plots.sif"

    def requires(self):
        reqs = {}
        reqs["ts"] = DeployTestingWaveforms.req(self)
        reqs["infer"] = Infer.req(self)
        return reqs

    def output(self):
        data = os.path.join(self.output_dir, "sensitive_volume.h5")
        plot = os.path.join(self.output_dir, "sensitive_volume.html")
        return [law.LocalFileTarget(data), law.LocalFileTarget(plot)]

    def run(self):
        from pathlib import Path

        foreground = self.input()["infer"]["foreground"]
        background = self.input()["infer"]["background"]
        rejected = self.input()["ts"][1].path
        source_prior = load_prior(self.source_prior)

        if self.waveform_type == "ringdown":
            from plots.legacy.ringdown import main as ringdown_main

            ringdown_main(
                Path(background.path),
                Path(foreground.path),
                Path(rejected),
                self.ifos,
                remnant_mass_combos=self.remnant_mass_combos,
                source_prior=source_prior,
                dt=self.dt,
                max_far=self.max_far,
                sigma=self.sigma,
                coverage_floor=self.coverage_floor,
                ess_floor=self.ess_floor,
                output_dir=Path(self.output_dir),
            )
            return

        from plots.legacy.main import main

        main(
            Path(background.path),
            Path(foreground.path),
            Path(rejected),
            self.ifos,
            mass_combos=self.mass_combos,
            source_prior=source_prior,
            dt=self.dt,
            max_far=self.max_far,
            sigma=self.sigma,
            output_dir=Path(self.output_dir),
        )
