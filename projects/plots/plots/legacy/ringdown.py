"""Ringdown sensitive volume, the sibling of `plots.legacy.main`.

Differs from the CBC path in five places: the weight construction, the
volume bounds (derived from the distance prior rather than read from a
redshift key), the variance (computed over every draw), the absence of a
GWTC-3 comparison set, and the absence of vetoes.

Vetoes are deliberately omitted. The CBC entry point carries optional O3
CBC veto categories, gate windows and catalog exclusions; the task never
passes them (`aframe/tasks/plots/sv.py` omits the argument), and veto
policy for this work lives in a separate performance-analysis repository.
Retaining the branch would also be a live numerical hazard: `apply_vetos`
deletes foreground rows while every rejected draw stays in the
normalization, so vetoing one of two recoveries against two rejected draws
gives 1/3 rather than 1/4 for a full-campaign estimand. If vetoes are ever
wanted here, they need an explicit decision about whether a vetoed
injection is a miss over the original exposure or defines a reduced one.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

import h5py
import numpy as np
from bokeh.io import save
from bokeh.layouts import column, gridplot
from bokeh.models import Div, Title

from ledger.events import EventSet, RingdownRecoveredInjectionSet
from ledger.injections import RingdownInjectionParameterSet
from plots.legacy import ringdown_stats as rs
from plots.legacy import tools
from priors.priors import log_normal_remnant_mass
from utils.cosmology import DEFAULT_COSMOLOGY, get_astrophysical_volume
from utils.logging import configure_logging


def _prior_description(prior, key):
    """Readable one-line description of a bilby prior, for provenance."""
    dist = prior[key]
    return f"{key} ~ {type(dist).__name__}({dist.minimum:g}, {dist.maximum:g})"


def main(
    background: Path,
    foreground: Path,
    rejected_params: Path,
    ifos: List[str],
    remnant_mass_combos: List[float],
    source_prior: Callable,
    output_dir: Path,
    log_file: Optional[Path] = None,
    dt: Optional[float] = None,
    max_far: float = 365,
    sigma: float = 0.1,
    coverage_floor: float = 0.95,
    ess_floor: float = 10,
    verbose: bool = False,
) -> None:
    """Compute and plot the sensitive volume of a ringdown analysis.

    Args:
        remnant_mass_combos:
            Median source-frame remnant masses in solar masses, one per
            panel. Medians, not arithmetic means.
        sigma:
            Standard deviation of log remnant mass. Dimensionless.
        coverage_floor:
            Warning threshold only. Changing it never changes weights,
            curves, errors or the population reference.
        ess_floor:
            Rejects targets whose effective sample size is too small for
            a useful estimate — a different failure from a weight vector
            that will not normalize.
    """
    configure_logging(log_file, verbose)
    background_set = EventSet.read(background)
    foreground_set = RingdownRecoveredInjectionSet.read(foreground)
    rejected = RingdownInjectionParameterSet.read(rejected_params)

    source, _ = source_prior(DEFAULT_COSMOLOGY)
    zmin, zmax = rs.redshift_bounds(source, DEFAULT_COSMOLOGY)
    c_z = rs.redshift_ratio_norm(zmin, zmax, DEFAULT_COSMOLOGY)
    try:
        dec_prior = source["dec"]
    except KeyError:
        dec_range = None
    else:
        dec_range = (dec_prior.minimum, dec_prior.maximum)
    v0 = (
        get_astrophysical_volume(zmin, zmax, DEFAULT_COSMOLOGY, dec_range)
        / 10**9
    )

    # Compute redshift ONCE per ledger and derive source mass from it.
    # The ledger properties are deliberately uncached (design Ruling 9),
    # so reading `.remnant_mass_source` and then `.redshift` would run the
    # z_at_value inversion twice over every draw.
    z_fore = foreground_set.redshift
    z_rej = rejected.redshift
    redshift = np.concatenate([z_fore, z_rej])
    m_src = np.concatenate(
        [
            foreground_set.remnant_mass / (1 + z_fore),
            rejected.remnant_mass / (1 + z_rej),
        ]
    )
    frequency = np.concatenate([foreground_set.frequency, rejected.frequency])

    n_fore = len(foreground_set)
    statistic = np.concatenate(
        [
            foreground_set.detection_statistic,
            np.full(len(rejected), -np.inf),
        ]
    )
    detected = np.zeros(len(statistic), dtype=bool)
    detected[:n_fore] = True
    if dt is not None:
        detected[:n_fore] = (
            np.abs(
                foreground_set.detection_time - foreground_set.injection_time
            )
            <= dt
        )

    fars, thresholds = rs.far_threshold_grid(
        background_set.detection_statistic, background_set.Tb, max_far
    )

    weights = np.empty((len(remnant_mass_combos), len(m_src)))
    coverages, ess_values, conditioning = [], [], []
    for i, m0 in enumerate(remnant_mass_combos):
        target, _ = log_normal_remnant_mass(m0, sigma=sigma)
        weights[i] = rs.normalize_log_weights(
            rs.log_importance_weights(
                m_src, frequency, redshift, source, target, c_z
            )
        )
        coverage = rs.support_coverage(m0, sigma, source, DEFAULT_COSMOLOGY)
        ess = rs.effective_sample_size(weights[i])
        coverages.append(coverage)
        ess_values.append(ess)
        if coverage < coverage_floor:
            logging.warning(
                "target M0=%s has support coverage %.6f below the floor "
                "%.3f; the reported volume is conditioned on the support",
                m0,
                coverage,
                coverage_floor,
            )
        # ESS is checked BEFORE conditioned_means. A target that the
        # statistical policy rejects must fail with the ESS error, not
        # with the means guard: at M0=1000, sigma=0.01 the interval
        # probabilities underflow and `conditioned_means` would raise
        # first, reporting an arithmetic problem for what is a
        # statistical one. Design Ruling 4 keeps those distinct.
        if ess < ess_floor:
            raise ValueError(
                f"target M0={m0} has effective sample size {ess:.4g}, "
                f"below ess_floor={ess_floor}; the campaign cannot "
                f"support this target"
            )
        conditioning.append(
            rs.conditioned_means(m0, sigma, source, DEFAULT_COSMOLOGY)
            if coverage < 1
            else None
        )

    mu, err = rs.sensitive_volume(statistic, detected, weights, thresholds)
    sv, sv_err = mu * v0, err * v0

    epsilon_desc = _prior_description(source, "epsilon")
    quality_desc = _prior_description(source, "quality")
    chi_lo = 1 - (2 / source["quality"].minimum) ** (20 / 9)
    chi_hi = 1 - (2 / source["quality"].maximum) ** (20 / 9)
    assumptions = (
        f"Volume conditional on the campaign priors, neither reweighted: "
        f"{epsilon_desc}; {quality_desc}, i.e. remnant spin in "
        f"[{chi_lo:.3f}, {chi_hi:.3f}]."
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    with h5py.File(output_dir / "sensitive_volume.h5", "w") as f:
        f.attrs["quantity"] = "sensitive_volume_Gpc3"
        f.attrs["sigma"] = sigma
        f.attrs["sigma_meaning"] = "standard deviation of log mass"
        f.attrs["mass_frame"] = "source"
        f.attrs["mass_units"] = "Msun"
        f.attrs["cosmology"] = DEFAULT_COSMOLOGY.name
        f.attrs["epsilon_prior"] = epsilon_desc
        f.attrs["quality_prior"] = quality_desc
        f.attrs["remnant_spin_range"] = [chi_lo, chi_hi]
        f.attrs["assumptions"] = assumptions
        f.attrs["redshift_measure"] = (
            "target redshift density proportional to dV_c/(1+z), matching "
            "utils.cosmology.get_astrophysical_volume"
        )
        f.attrs["uncertainty_method"] = (
            "self-normalized importance sampling over ALL draws; rejected "
            "injections and dt-window misses carry their weight with a "
            "zero detection indicator (design Ruling 6)"
        )
        f.attrs["coverage_floor"] = coverage_floor
        f.attrs["ess_floor"] = ess_floor
        f.create_dataset("thresholds", data=thresholds)
        f.create_dataset("fars", data=fars)
        for i, m0 in enumerate(remnant_mass_combos):
            g = f.create_group(str(m0))
            g.create_dataset("sv", data=sv[i])
            g.create_dataset("err", data=sv_err[i])
            g.attrs["remnant_mass_median"] = m0
            g.attrs["support_coverage"] = coverages[i]
            g.attrs["effective_sample_size"] = ess_values[i]
            g.attrs["support_conditioned"] = coverages[i] < 1
            if conditioning[i] is not None:
                g.attrs["conditioned_mean_quality"] = conditioning[i][
                    "quality"
                ]
                g.attrs["conditioned_mean_redshift"] = conditioning[i][
                    "redshift"
                ]

    figures = tools.make_grid(
        list(remnant_mass_combos),
        title_fn=lambda c: (
            r"$$M_f={}\,M_\odot\text{{ (source), }}\sigma_{{\ln M}}="
            + str(sigma)
            + r"$$"
        ).format(c),
    )
    for i, p in enumerate(figures):
        color = tools.palette[0]
        kwargs = {"legend_label": "aframe"} if i == 0 else {}
        p.line(fars, sv[i], line_width=1.5, line_color=color, **kwargs)
        tools.plot_err_bands(
            p,
            fars,
            sv[i],
            sv_err[i],
            line_color=color,
            line_width=0.8,
            fill_color=color,
            fill_alpha=0.4,
        )
        # A Title, not a Div. bokeh is pinned to 3.4.3
        # (projects/plots/uv.lock:448), where Plot.below is
        # List(Instance(Renderer)); Title is an Annotation and therefore a
        # Renderer, but a Div is a widget and is rejected. Plain text only
        # -- Title does not render HTML.
        note = f"coverage {coverages[i]:.4f} | ESS {ess_values[i]:.1f}"
        if coverages[i] < 1:
            note += " | conditioned on support"
        if coverages[i] < coverage_floor:
            note += " | BELOW COVERAGE FLOOR"
        p.add_layout(
            Title(
                text=note,
                text_font_size="9pt",
                text_font_style="normal",
            ),
            "below",
        )

    caption = Div(
        text=(
            f"<div style='font-size:10pt;max-width:760px'>{assumptions}"
            f" Uncertainties are self-normalized importance-sampling "
            f"errors over all draws, including rejected injections and "
            f"time-window misses.</div>"
        )
    )
    grid = gridplot(figures, ncols=2 if len(figures) > 1 else 1)
    save(
        column(grid, caption),
        filename=str(output_dir / "sensitive_volume.html"),
    )
