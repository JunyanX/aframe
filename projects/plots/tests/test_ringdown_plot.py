import numpy as np
import pytest


def test_make_grid_cbc_default_unchanged():
    """Regression: the CBC FOUR-combo call is unchanged.

    The one-combo call is not: Step 3 deliberately gives it the y-axis
    label it currently lacks. That is a visible CBC display change, and
    `test_make_grid_one_panel_has_a_y_axis_label` pins the new behaviour.
    """
    from plots.legacy.tools import make_grid

    combos = [[35, 35], [35, 20], [20, 20], [20, 10]]
    plots = make_grid(combos)
    assert len(plots) == 4
    assert "m_1=35, m_2=35" in plots[0].title.text
    assert "Sensitive Volume" in plots[0].yaxis[0].axis_label


def test_make_grid_accepts_scalar_combos_with_formatter():
    from plots.legacy.tools import make_grid

    combos = [40, 80, 120, 150]
    plots = make_grid(
        combos,
        title_fn=lambda c: r"$$\text{{Log Normal }}M_f={}$$".format(c),
    )
    assert len(plots) == 4
    assert "M_f=40" in plots[0].title.text


def test_make_grid_still_rejects_three_combos():
    from plots.legacy.tools import make_grid

    with pytest.raises(ValueError, match="2x2 or 1x1"):
        make_grid([40, 80, 120])


def test_seconds_per_year_matches_tools():
    from plots.legacy.ringdown_stats import SECONDS_PER_YEAR
    from plots.legacy.tools import SECONDS_PER_YEAR as TOOLS_SPY

    assert SECONDS_PER_YEAR == TOOLS_SPY


def test_make_grid_one_panel_has_a_y_axis_label():
    from plots.legacy.tools import make_grid

    plots = make_grid([[35, 35]])
    assert "Sensitive Volume" in plots[0].yaxis[0].axis_label


@pytest.fixture
def campaign(tmp_path):
    """A small, deterministic campaign written under tmp_path.

    The committed tests must NOT read
    /home/junyan.xu/aframe/ringdown-e2e-260911: that path does not exist
    on another checkout or on the GitHub runner, which mounts the repo at
    /opt/aframe and runs the project suite
    (.github/workflows/project-build-test.yaml). The real campaign stays
    available as the local smoke command in Step 6.

    Sized so the assertions stay meaningful: enough background ranks for
    a FAR grid, and enough accepted and rejected draws for the default
    targets to clear ess_floor.
    """
    import h5py

    from ledger.events import EventSet, RingdownRecoveredInjectionSet
    from ledger.injections import RingdownInjectionParameterSet

    rng = np.random.default_rng(17)
    n_fore, n_rej, n_back = 4000, 4000, 5000

    def draw(n):
        return {
            "frequency": np.exp(rng.uniform(np.log(100), np.log(1000), n)),
            "quality": rng.uniform(8, 20, n),
            "epsilon": rng.uniform(0, 0.1, n),
            "phase": rng.uniform(0, 2 * np.pi, n),
            "inclination": np.arccos(rng.uniform(-1, 1, n)),
            "distance": rng.uniform(100, 1000, n),
            "ra": rng.uniform(0, 2 * np.pi, n),
            "dec": np.arcsin(rng.uniform(-1, 1, n)),
            "psi": rng.uniform(0, np.pi, n),
            "snr": rng.uniform(4, 40, n),
            "ifo_snrs": rng.uniform(3, 30, (n, 2)),
        }

    # One year of background livetime, so far_threshold_grid has ranks to
    # work with at the default max_far.
    tb = 60 * 60 * 24 * 365.25
    # EventSet declares exactly detection_statistic, detection_time,
    # shift and the Tb metadata (events.py:42-56). It takes NO `ifos`;
    # passing one raises TypeError. Verified against the tree.
    background = EventSet(
        detection_statistic=rng.normal(0, 1, n_back),
        detection_time=rng.uniform(0, tb, n_back),
        shift=np.zeros((n_back, 2)),
        Tb=tb,
    )
    back_path = tmp_path / "background.hdf5"
    background.write(back_path)

    fore = draw(n_fore)
    foreground = RingdownRecoveredInjectionSet(
        detection_statistic=rng.normal(2, 1, n_fore),
        detection_time=rng.uniform(0, tb, n_fore),
        injection_time=rng.uniform(0, tb, n_fore),
        shift=np.zeros((n_fore, 2)),
        Tb=tb,
        ifos=["H1", "L1"],
        sample_rate=2048.0,
        duration=4.0,
        right_pad=2.0,
        num_injections=n_fore,
        **fore,
    )
    fore_path = tmp_path / "foreground.hdf5"
    foreground.write(fore_path)

    rej = draw(n_rej)
    rejected = RingdownInjectionParameterSet(ifos=["H1", "L1"], **rej)
    rej_path = tmp_path / "rejected-parameters.hdf5"
    rejected.write(rej_path)

    # The foreground and rejected constructors carry more fields than
    # EventSet; if one does not match, read
    # libs/ledger/ledger/injections.py and adjust. Do NOT fall back to
    # the private campaign path.
    with h5py.File(fore_path) as f:
        assert "frequency" in (f["parameters"] if "parameters" in f else f)

    return back_path, fore_path, rej_path


def test_main_writes_assumptions_and_diagnostics(campaign, tmp_path):
    """Integration test: call main() and inspect what it wrote.

    The other tests here only exercise make_grid and a constant. The
    design requires the epsilon and Q assumptions on every figure and
    coverage information in the annotations; nothing else checks that.
    """
    import h5py

    from plots.legacy.ringdown import main
    from priors.priors import ringdown_prior

    back, fore, rej = campaign
    out = tmp_path / "out"
    main(
        back,
        fore,
        rej,
        ["H1", "L1"],
        remnant_mass_combos=[40, 80, 120, 150],
        source_prior=ringdown_prior,
        output_dir=out,
    )

    with h5py.File(out / "sensitive_volume.h5") as f:
        assert f.attrs["quantity"] == "sensitive_volume_Gpc3"
        assert f.attrs["mass_frame"] == "source"
        assert "epsilon ~ Uniform(0, 0.1)" in f.attrs["epsilon_prior"]
        assert "quality ~ Uniform(8, 20)" in f.attrs["quality_prior"]
        assert "dV_c/(1+z)" in f.attrs["redshift_measure"]
        assert "ALL draws" in f.attrs["uncertainty_method"]
        groups = [k for k in f if k not in ("fars", "thresholds")]
        assert sorted(groups) == ["120", "150", "40", "80"]
        n = f["fars"].shape[0]
        assert n > 0 and f["thresholds"].shape[0] == n
        for key in groups:
            g = f[key]
            assert g["sv"].shape[0] == n and g["err"].shape[0] == n
            assert g.attrs["support_coverage"] > 0.9999
            assert np.isfinite(g.attrs["effective_sample_size"])

    html = (out / "sensitive_volume.html").read_text()
    assert "epsilon ~ Uniform" in html
    assert "quality ~ Uniform" in html


def test_main_rejects_a_target_below_the_ess_floor(campaign, tmp_path):
    """The ESS policy must fire, not the conditioned-means guard."""
    from plots.legacy.ringdown import main
    from priors.priors import ringdown_prior

    back, fore, rej = campaign
    with pytest.raises(ValueError, match="effective sample size"):
        main(
            back,
            fore,
            rej,
            ["H1", "L1"],
            remnant_mass_combos=[1000],
            source_prior=ringdown_prior,
            output_dir=tmp_path / "rejected",
            sigma=0.01,
        )


def test_main_refuses_a_non_finite_background_before_writing(
    campaign, tmp_path
):
    """The raise must come BEFORE any output exists.

    With one NaN in the background the committed code exited 0 and wrote
    both files with `fars=[1.0], thresholds=[NaN], sv=[0.0], err=[0.0]`.
    A successfully written wrong figure is the failure mode this whole
    stage is built to avoid, so the test asserts on the absence of the
    files, not only on the exception.
    """
    from ledger.events import EventSet
    from plots.legacy.ringdown import main
    from priors.priors import ringdown_prior

    back, fore, rej = campaign
    poisoned = EventSet.read(back)
    poisoned.detection_statistic[0] = np.nan
    bad_back = tmp_path / "background-nan.hdf5"
    poisoned.write(bad_back)

    out = tmp_path / "nan-out"
    with pytest.raises(ValueError, match="non-finite"):
        main(
            bad_back,
            fore,
            rej,
            ["H1", "L1"],
            remnant_mass_combos=[40, 80, 120, 150],
            source_prior=ringdown_prior,
            output_dir=out,
        )
    assert not (out / "sensitive_volume.h5").exists()
    assert not (out / "sensitive_volume.html").exists()
