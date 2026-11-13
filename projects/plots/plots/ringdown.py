"""Ringdown sensitive volume, the sibling of `plots.main`.

Differs from the CBC path in four places: the weight construction, the
volume bounds (derived from the distance prior rather than read from a
redshift key), the absence of a GWTC-3 comparison set, and the absence of
vetoes.

Vetoes are deliberately omitted. The CBC entry point carries optional O3
CBC veto categories, gate windows and catalog exclusions; the ringdown
`sensitive_volume` rule never passes them (`projects/plots/plots.smk`
refuses vetos for ringdown runs), and veto policy for this work lives in a
separate performance-analysis repository.
Retaining the branch would also be a live numerical hazard: `apply_vetos`
deletes foreground rows while every rejected draw stays in the
normalization, so vetoing one of two recoveries against two rejected draws
gives 1/3 rather than 1/4 for a full-campaign estimand. If vetoes are ever
wanted here, they need an explicit decision about whether a vetoed
injection is a miss over the original exposure or defines a reduced one.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
from bokeh.io import save
from bokeh.layouts import column, gridplot
from bokeh.models import Div, Title
from ledger.events import EventSet, RingdownRecoveredInjectionSet
from ledger.injections import RingdownInjectionParameterSet
from priors.priors import log_normal_remnant_mass
from utils.cosmology import DEFAULT_COSMOLOGY, get_astrophysical_volume
from utils.logging import configure_logging

from plots.core import ringdown_stats as rs
from plots.core import style
from plots.core.compute import sensitive_volume


def _prior_description(prior, key):
    """Readable one-line description of a bilby prior, for provenance."""
    dist = prior[key]
    return f"{key} ~ {type(dist).__name__}({dist.minimum:g}, {dist.maximum:g})"


def main(
    background: Path,
    foreground: Path,
    rejected_params: Path,
    remnant_mass_combos: list[float],
    source_prior: Callable,
    output_file: Path,
    plot_file: Path,
    log_file: Path | None = None,
    dt: float | None = None,
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
    # style.make_grid draws a 1x1 or 2x2 grid. Say so before any work
    if len(remnant_mass_combos) not in (1, 4):
        raise ValueError(
            "Only support 2x2 or 1x1 grids, can't plot "
            f"{len(remnant_mass_combos)} targets"
        )
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
    # A recovery outside the dt window is a miss that keeps its weight:
    # -inf puts it below every threshold.
    statistic = foreground_set.detection_statistic
    if dt is not None:
        in_window = (
            np.abs(
                foreground_set.detection_time - foreground_set.injection_time
            )
            <= dt
        )
        statistic = np.where(in_window, statistic, -np.inf)

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
        )

    # Rejected draws can never be detected, so they enter through the
    # normalization of the weights and, in the error, through missed_w2.
    mu, err = sensitive_volume(
        statistic,
        weights[:, :n_fore],
        thresholds,
        missed_w2=(weights[:, n_fore:] ** 2).sum(axis=-1),
    )
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

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(output_file, "w") as f:
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
            # the same name whether the target came as 40 or 40.0
            g = f.create_group(str(float(m0)))
            g.create_dataset("sv", data=sv[i])
            g.create_dataset("err", data=sv_err[i])
            g.attrs["remnant_mass_median"] = m0
            g.attrs["support_coverage"] = coverages[i]
            g.attrs["effective_sample_size"] = ess_values[i]
            g.attrs["conditioned_mean_quality"] = conditioning[i]["quality"]
            g.attrs["conditioned_mean_redshift"] = conditioning[i]["redshift"]

    figures = style.make_grid(
        list(remnant_mass_combos),
        title_fn=lambda c: (
            r"$$M_f={}\,M_\odot\text{{ (source), }}\sigma_{{\ln M}}="
            + str(sigma)
            + r"$$"
        ).format(f"{c:g}"),
    )
    for i, p in enumerate(figures):
        color = style.palette[0]
        kwargs = {"legend_label": "aframe"} if i == 0 else {}
        p.line(fars, sv[i], line_width=1.5, line_color=color, **kwargs)
        style.plot_err_bands(
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
    plot_file.parent.mkdir(exist_ok=True, parents=True)
    save(
        column(grid, caption),
        filename=str(plot_file),
    )
