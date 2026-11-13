"""Snakemake rules for pipeline plots and metrics.

Rules:
  fetch_gwtc3_injections     copy the GWTC-3 injection set into the run dir
  fetch_veto_segments        query veto segments for the analyzed span
  compute_sensitive_volume   a population's sensitive volume vs. FAR, and its plot
  sensitive_volume           every active population's sensitive volume

The fetch rules are localrules, so compute_sensitive_volume reads only declared
inputs and needs no network access.
"""

import json

plots_dir = run_dir / "plots"
plots_log_dir = log_dir / "plots"

PLOTS_CONTAINER = container("plots")

VETOS = config["vetos"]


localrules:
    fetch_gwtc3_injections,
    fetch_veto_segments,


rule fetch_gwtc3_injections:
    """Copy the GWTC-3 sensitivity injection set from Zenodo (via cache)."""
    output:
        str(plots_dir / "gwtc3_injections.hdf5"),
    log:
        str(plots_log_dir / "fetch_gwtc3_injections.log"),
    container:
        PLOTS_CONTAINER
    script:
        "scripts/fetch_gwtc3_injections.py"


rule fetch_veto_segments:
    """Query DQSegDB for the veto segments covering the background."""
    input:
        background=str(infer_dir / "background.hdf5"),
    output:
        str(plots_dir / "veto_segments.json"),
    log:
        str(plots_log_dir / "fetch_veto_segments.log"),
    container:
        PLOTS_CONTAINER
    params:
        vetos=VETOS,
        ifos=config["ifos"],
        segment_server=config["segment_server"],
    script:
        "scripts/fetch_veto_segments.py"


if WAVEFORM_TYPE == "ringdown":
    if VETOS:
        raise WorkflowError(
            "vetos are not supported for ringdown runs: the ringdown "
            "sensitive volume keeps every rejected draw in its "
            "normalization, so dropping vetoed recoveries would bias it"
        )
    # The plot is a 1x1 or 2x2 grid. Say so before the run starts rather
    # than when its last rule fails.
    for _population in POPULATIONS:
        _num_targets = len(population_cfg(_population, "remnant_mass_combos"))
        if _num_targets not in (1, 4):
            raise WorkflowError(
                f"remnant_mass_combos of population {_population} has "
                f"{_num_targets} entries, but the ringdown sensitive volume "
                "plot holds 1 or 4"
            )

    rule compute_sensitive_volume:
        """Compute a population's ringdown sensitive volume vs. FAR for log
        normal remnant-mass targets, against the background shared by every
        population. No GWTC-3 comparison and no vetoes.
        """
        input:
            background=str(infer_dir / "background.hdf5"),
            foreground=str(infer_dir / "foreground_{population}.hdf5"),
            rejected_params=str(
                test_waveforms / "{population}" / "rejected_parameters.hdf5"
            ),
        output:
            sv_data=str(plots_dir / "{population}" / "sensitive_volume.hdf5"),
            sv_plot=str(plots_dir / "{population}" / "sensitive_volume.html"),
        log:
            str(plots_log_dir / "sensitive_volume-{population}.log"),
        container:
            PLOTS_CONTAINER
        resources:
            **rule_resources("sensitive_volume", "plots"),
        params:
            remnant_mass_combos=lambda wc: _fmt_list(
                population_cfg(wc.population, "remnant_mass_combos")
            ),
            sigma=lambda wc: population_cfg(wc.population, "remnant_mass_sigma"),
            source_prior=lambda wc: population_cfg(wc.population, "prior"),
            dt=config["dt"] or "null",
        shell:
            "ringdown-sensitive-volume"
            " --background {input.background}"
            " --foreground {input.foreground}"
            " --rejected_params {input.rejected_params}"
            " --remnant_mass_combos '{params.remnant_mass_combos}'"
            " --sigma {params.sigma}"
            " --source_prior {params.source_prior}"
            " --dt {params.dt}"
            " --output_file {output.sv_data}"
            " --plot_file {output.sv_plot}"
            " &> {log}"

else:

    rule compute_sensitive_volume:
        """Compute a population's sensitive volume vs. FAR, against the
        background shared by every population, and plot it against LVK pipelines.
        """
        input:
            **(
                {"veto_segments": str(plots_dir / "veto_segments.json")}
                if VETOS
                else {}
            ),
            background=str(infer_dir / "background.hdf5"),
            foreground=str(infer_dir / "foreground_{population}.hdf5"),
            rejected_params=str(
                test_waveforms / "{population}" / "rejected_parameters.hdf5"
            ),
            injection_file=str(plots_dir / "gwtc3_injections.hdf5"),
        output:
            sv_data=str(plots_dir / "{population}" / "sensitive_volume.hdf5"),
            sv_plot=str(plots_dir / "{population}" / "sensitive_volume.html"),
            gwtc3_sv=str(plots_dir / "{population}" / "gwtc-3_pipeline_sv.hdf5"),
        log:
            str(plots_log_dir / "sensitive_volume-{population}.log"),
        container:
            PLOTS_CONTAINER
        resources:
            **rule_resources("sensitive_volume", "plots"),
        params:
            ifos=_fmt_list(config["ifos"]),
            mass_combos=lambda wc: json.dumps(
                population_cfg(wc.population, "mass_combos")
            ),
            source_prior=lambda wc: population_cfg(wc.population, "prior"),
            dt=config["dt"] or "null",
            vetos=_fmt_list(VETOS or []),
        shell:
            "sensitive-volume"
            " --background {input.background}"
            " --foreground {input.foreground}"
            " --rejected_params {input.rejected_params}"
            " --injection_file {input.injection_file}"
            " --ifos '{params.ifos}'"
            " --mass_combos '{params.mass_combos}'"
            " --source_prior {params.source_prior}"
            " --output_file {output.sv_data}"
            " --plot_file {output.sv_plot}"
            " --gwtc3_file {output.gwtc3_sv}"
            " --dt {params.dt}"
            # Omitted entirely when unset. If we pass an empty list,
            # we still do a query.
            + (
                " --vetos '{params.vetos}' --veto_segments {input.veto_segments}"
                if VETOS
                else ""
            )
            + " &> {log}"


rule sensitive_volume:
    """Target rule: every active population's sensitive volume.

    Runs no command. compute_sensitive_volume has a {population} wildcard,
    so it can't be run by name; this asks for all of its outputs instead.
    R&P runs have none, since they have no testing injections.
    """
    input:
        (
            expand(rules.compute_sensitive_volume.output, population=POPULATIONS)
            if ANALYSIS_TYPE == "hdf5"
            else []
        ),
    localrule: True
