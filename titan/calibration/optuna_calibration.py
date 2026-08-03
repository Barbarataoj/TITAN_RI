# titan/calibration/optuna_calibration.py
#
# Multi-objective Optuna calibration for TITAN_RI.
# Objectives (NSGA-II Pareto front):
#   1. oud_rmse              — RMSE of (oud_active, oud_remission, oud_none) at final timestep
#   2. hiv_traj_rmse         — RMSE of model annual PWID HIV incidence vs APCD 2012–2022 series
#   3. hiv_prev_err          — squared error of final PWID HIV prevalence vs target
#   4. od_err                — squared error of fatal + nonfatal annual OD rates vs targets
#   5. moud_err              — squared error of MOUD uptake vs target
#
# Reads basicReport.txt as TSV (one row per timestep × demographic stratum).

import optuna
import yaml
import numpy as np
import pandas as pd
import subprocess
import copy
import os
import sys
from pathlib import Path

# =====================================================
# Paths (absolute, Slurm-safe)
# =====================================================
REPO_ROOT = Path(__file__).resolve().parents[2]

CALIB_YAML = REPO_ROOT / "titan/calibration/calibration_targets.yml"
BASE_SCENARIO = REPO_ROOT / "calibrated_scenario_multi_5obj.yml"

SCRATCH = Path(os.environ.get("SCRATCH", f"/oscar/scratch/{os.environ['USER']}"))
OUTPUT_DIR = SCRATCH / "titan_calibration" / "optuna_multi_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum pooled HIV infections (10 reps x 11 yrs) a Pareto trial must show to be
# eligible for the canonical pick. Below this, hiv_traj_rmse can pass by luck (a
# single random infection event) rather than genuine calibrated transmission -- the
# 2026-07-21 rerun (job 4134855) found 30/61 Pareto trials had <=1 pooled infection
# and scored fine on hiv_traj_rmse anyway; the naive Pareto[0] pick that run had
# exactly 1 infection and ranked 27th/61 by composite score. See
# docs/calibration_log/phase_4_ridoh_idu_validation_2026-07-21.md.
#
# Absolute floor only. The screen that actually binds is derived from the targets
# by min_transmission_events() below: 2 events was ~2% of what the target series
# implies, so it screened out only the completely dead trials and passed anything
# with a couple of stray infections.
MIN_HIV_TRANSMISSION_EVENTS = 2

# Fraction of the target-implied pooled infection count a trial must reach.
# Poisson: at 25% of expectation the one-sided p of a correctly-calibrated trial
# being screened out is <1e-9 for the ~90 events the targets imply, so this costs
# nothing in false rejections while excluding trials running an order of magnitude
# cold. Added 2026-07-30.
MIN_HIV_TRANSMISSION_FRACTION = 0.25

# =====================================================
# Utilities
# =====================================================
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def set_yaml_value(scenario, path, val, is_int=False):
    keys = path.split(".")
    ref = scenario
    for k in keys[:-1]:
        if k not in ref or not isinstance(ref[k], dict):
            raise KeyError(f"Invalid YAML path: {path}")
        ref = ref[k]
    ref[keys[-1]] = int(val) if is_int else float(val)

def set_nested(block, path, val, is_int=False):
    """
    Set a dotted path inside `block`, CREATING intermediate dicts as needed.

    Unlike set_yaml_value, which requires every intermediate key to already
    exist. Used for the engine `calibration` block, whose sub-dicts (e.g.
    calibration.sex) are absent from the base scenario -- see the block in
    objective() for what that cost.
    """
    keys = path.split(".")
    ref = block
    for k in keys[:-1]:
        nxt = ref.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            ref[k] = nxt
        ref = nxt
    ref[keys[-1]] = int(val) if is_int else float(val)

MSM_SEX_TYPES = ("MSM", "MSMW")

def set_noninj_hiv_init(scenario, het, msm, dx_init=0.9):
    """
    Seed hiv.init/hiv.dx.init on every NonInj stratum, leaving Inj untouched.

    Not routed through set_yaml_value/parameters_to_optimize because the quantity
    lives at 24 separate dotted paths (6 races x 4 sex_types) that must move
    together, and set_yaml_value takes one exact path with no wildcard.

    See the block in objective() for why this is fitted rather than fixed and why
    MSM/MSMW are split from HF/HM. Returns strata touched, for smoke tests.
    """
    n = 0
    for race in scenario.get("demographics", {}).values():
        for sx_name, sx in (race or {}).get("sex_type", {}).items():
            nd = (sx or {}).get("drug_type", {}).get("NonInj")
            if nd is None:
                continue
            init = msm if sx_name in MSM_SEX_TYPES else het
            nd["hiv"] = {"init": float(init), "dx": {"init": float(dx_init)}}
            n += 1
    return n

def run_titan(params_file, output_dir, n_reps=None):
    # Monte-Carlo reps are driven ONLY by the -n/--nMC CLI flag; params.model.num_reps
    # is dead config (nothing in the engine reads it). Default 10 per CALIBRATION_PLAN;
    # override with TITAN_CALIB_REPS for smoke tests. Reps append to one basicReport.txt
    # and are pooled by parse_basic_report's groupby("t").sum() (numerator + denominator).
    if n_reps is None:
        n_reps = int(os.environ.get("TITAN_CALIB_REPS", 10))
    # sys.executable, not "python": the subprocess must run under the SAME
    # interpreter as the driver. Resolving "python" off PATH silently picks up
    # whatever the shell has when the venv is not activated -- which fails on the
    # first import inside the child and surfaces only as CalledProcessError(1)
    # from every trial.
    cmd = [
        sys.executable, "-m", "titan.run_titan",
        "-p", str(params_file),
        "-o", str(output_dir),
        "-n", str(n_reps),
    ]
    subprocess.run(cmd, check=True)

    report = output_dir / "basicReport.txt"
    if not report.exists():
        raise RuntimeError(f"TITAN failed – missing basicReport.txt in {output_dir}")

    return parse_basic_report(report)

# =====================================================
# basicReport.txt parser (TSV, per-timestep × stratum)
# =====================================================
def parse_basic_report(path, steps_per_year=12):
    """
    Parse basicReport.txt (TSV, one row per timestep × demographic stratum) into
    an aggregated stats dict.

    Returns (subset):
      oud_active, oud_remission, oud_none        final-timestep fractions
      hiv_prevalence_pwid                        final-timestep HIV prev among PWID
      hiv_incidence_series_pwid                  np.array, annual HIV inc per PWID-yr
      moud_uptake_series                         np.array, annual MOUD coverage among active OUD
      fatal_od_rate                              annualized fatal OD rate, final sim year
      nonfatal_od_rate_inj                       annualized nonfatal OD per injection-active-OUD-yr
      nonfatal_od_rate_noninj                    annualized nonfatal OD per non-inj-active-OUD-yr
      hiv_new_pooled                             int, total simulated PWID HIV infections,
                                                   pooled over all reps and the post-burn window
                                                   (same population/window as
                                                   hiv_incidence_series_pwid) -- used to screen out
                                                   trials whose HIV objectives pass on 0-1 random
                                                   events rather than real transmission
    """
    df = pd.read_csv(path, sep="\t")
    grp = df.groupby("t").sum(numeric_only=True)
    last_t = grp.index.max()
    last = grp.loc[last_t]
    total = last["agents"]

    stats = {}
    if total > 0:
        stats["oud_active"]    = last.get("oud_active", 0) / total
        stats["oud_remission"] = last.get("oud_remission", 0) / total
        stats["oud_none"]      = last.get("oud_none", 0) / total

    # Injection share among ACTIVE OUD -- the denominator the injecting_pct target
    # is defined over ("Injecting behavior among OUD-active agents"). The report is
    # already stratified by drug_type, so summing oud_active within the Inj stratum
    # over all agents gives the person-time-weighted share directly.
    #
    # Time-averaged over the post-burn window rather than read off the final
    # timestep: injecting_pct is a steady-state target, and at the calibration
    # population of size 500 a single-timestep read is dominated by Monte Carlo
    # noise. Until 2026-07-29 this quantity was not measured or scored at all,
    # while the injection share drifted from 0.49 to 0.18 over a run
    # (EPIDEMIC_DYNAMICS_MAP_2026-07-29.md finding P2).
    post_burn = df[df["t"] >= 1]
    active_all = post_burn["oud_active"].sum()
    if active_all > 0:
        active_inj = post_burn[post_burn["drug_type"] == "Inj"]["oud_active"].sum()
        stats["inj_share_active_oud"] = float(active_inj / active_all)

    # ---------- validation quantities (NOT scored; see validation_targets) ----------
    # Data-anchored inputs rather than fitted parameters, measured here so that a
    # parameterisation drifting away from RIDOC / CDC-PWID observations is visible
    # in validate_candidates.py. Time-averaged over the post-burn window for the
    # same Monte Carlo reason as inj_share_active_oud above.
    pb_agents = post_burn["agents"].sum()
    if pb_agents > 0:
        stats["incar_share"] = float(post_burn.get("incar", 0).sum() / pb_agents)

    pb_dx = post_burn["hiv_dx"].sum()
    if pb_dx > 0:
        stats["art_among_dx"] = float(post_burn["haart"].sum() / pb_dx)
        # adherent == virally suppressed; this is the quantity that actually
        # carries the transmission benefit via haart_scaling
        stats["suppressed_among_dx"] = float(
            post_burn.get("haart_adherent", 0).sum() / pb_dx
        )
    pb_hiv = post_burn["hiv"].sum()
    if pb_hiv > 0:
        stats["dx_among_hiv"] = float(pb_dx / pb_hiv)

    # Helper to build an annual series from per-timestep aggregates
    def annual_series(per_t, numerator_col, denom_col):
        """Per-year rate = sum(numerator) / mean person-count*steps_per_year."""
        series = []
        first_post_burn = max(1, per_t.index.min())
        n_years = int((last_t - first_post_burn + 1) // steps_per_year)
        for y in range(n_years):
            start = first_post_burn + y * steps_per_year
            end   = start + steps_per_year - 1
            win = per_t.loc[(per_t.index >= start) & (per_t.index <= end)]
            person_yr = win[denom_col].sum() / steps_per_year if denom_col else 1.0
            if person_yr > 0:
                series.append(float(win[numerator_col].sum()) / person_yr)
            else:
                series.append(0.0)
        return np.array(series)

    # ---------- PWID HIV ----------
    df_last = df[df["t"] == last_t]
    pwid_last = df_last[df_last["drug_type"] == "Inj"]
    pwid_agents = pwid_last["agents"].sum()
    if pwid_agents > 0:
        stats["hiv_prevalence_pwid"] = float(pwid_last["hiv"].sum() / pwid_agents)

    # Same prevalence over the look-back PWID denominator that matches how the
    # APCD target was constructed (see OUDState.set_stats). Reported for
    # validate_candidates, NOT scored -- swapping the objective's denominator is a
    # target-definition decision, and it trades an easier prevalence fit for a
    # harder incidence fit.
    if "pwid_recent" in last and last.get("pwid_recent", 0) > 0:
        stats["hiv_prevalence_pwid_recent"] = float(
            last["pwid_recent_hiv"] / last["pwid_recent"]
        )

    df_pwid = df[df["drug_type"] == "Inj"].groupby("t").sum(numeric_only=True)
    stats["hiv_incidence_series_pwid"] = annual_series(df_pwid, "hiv_new", "agents")

    # Diagnosed-incidence counterpart. The target series is an APCD claims proxy,
    # i.e. DIAGNOSED incidence, while hiv_new counts true conversions; the two
    # coincide only once the undiagnosed pool is at equilibrium (dx_among_hiv is
    # ~0.97 post-rebuild, so the gap is small but it is not zero and it is
    # systematic). Reported so the discrepancy is visible rather than absorbed
    # silently into hiv_traj_rmse.
    stats["hiv_dx_incidence_series_pwid"] = annual_series(df_pwid, "hiv_dx_new", "agents")

    first_post_burn_pwid = max(1, df_pwid.index.min()) if len(df_pwid) else 1
    stats["hiv_new_pooled"] = int(df_pwid.loc[
        (df_pwid.index >= first_post_burn_pwid) & (df_pwid.index <= last_t), "hiv_new"
    ].sum())

    # ---------- MOUD trajectory ----------
    # MOUD coverage denominator = ACTIVE OUD only (oud_active), matching the APCD
    # target's denominator ("F11-diagnosed OUD in year Y" ~= currently-active OUD;
    # remission agents would not carry a current-year F11 dx). Numerator = moud_active
    # (MOUD among active OUD), so numerator and denominator share the same population.
    # Was (moud) / (oud_active + oud_remission), which mismatched the target denominator
    # and inflated calibrated MOUD ~2x. Changed 2026-07-20.
    moud_traj = []
    first_post_burn = max(1, grp.index.min())
    n_years = int((last_t - first_post_burn + 1) // steps_per_year)
    for y in range(n_years):
        start = first_post_burn + y * steps_per_year
        end   = start + steps_per_year - 1
        win = grp.loc[(grp.index >= start) & (grp.index <= end)]
        denom = win["oud_active"].sum()
        if denom > 0:
            moud_traj.append(float(win.get("moud_active", pd.Series([0])).sum()) / denom)
        else:
            moud_traj.append(0.0)
    stats["moud_uptake_series"] = np.array(moud_traj)
    # Final-year point too (back-compat)
    if len(moud_traj) > 0:
        stats["moud_uptake"] = float(moud_traj[-1])

    # ---------- Overdose, stratified by injection ----------
    final_year_t = grp.index[grp.index > last_t - steps_per_year]
    fy_inj    = df[(df["drug_type"] == "Inj")    & (df["t"].isin(final_year_t))]
    fy_noninj = df[(df["drug_type"] == "NonInj") & (df["t"].isin(final_year_t))]
    fy_all    = grp.loc[final_year_t]

    # Denominators must be ACTIVE-OUD person-years within each stratum, not all
    # agent person-years. Overdose events are gated on has_oud
    # (titan/features/overdose.py), remission agents are forced to NonInj, and
    # non-OUD recreational injectors keep drug_type == "Inj" -- so an "agents"
    # denominator includes large sub-populations that can never produce an
    # event. That deflated nonfatal_od_rate_noninj by ~2.5x and pushed the
    # optimizer toward a base_prob_noninj beyond its own search bound. It also
    # left the nonfatal objectives on a different denominator from fatal_od_rate.
    # The report is already stratified by drug_type, so oud_active gives the
    # correct per-stratum denominator directly.
    py_inj    = fy_inj["oud_active"].sum()    / steps_per_year
    py_noninj = fy_noninj["oud_active"].sum() / steps_per_year
    py_active = fy_all["oud_active"].sum() / steps_per_year

    fatal_total = fy_all.get("overdose_deaths", pd.Series([0])).sum()
    if py_active > 0:
        stats["fatal_od_rate"] = float(fatal_total) / py_active

    # Targets are nonfatal-only, but overdose_events counts every overdose and
    # the fatality roll happens afterwards, so fatal events are a strict subset.
    # Subtract them, apportioning the aggregate death count across strata by
    # each stratum's share of events (deaths are not reported per drug_type).
    events_inj    = float(fy_inj.get("overdose_events", pd.Series([0])).sum())
    events_noninj = float(fy_noninj.get("overdose_events", pd.Series([0])).sum())
    events_total  = events_inj + events_noninj
    fatal_f = float(fatal_total)

    if events_total > 0:
        fatal_inj    = fatal_f * (events_inj / events_total)
        fatal_noninj = fatal_f * (events_noninj / events_total)
    else:
        fatal_inj = fatal_noninj = 0.0

    if py_inj > 0:
        nonfatal_inj = max(events_inj - fatal_inj, 0.0)
        stats["nonfatal_od_rate_inj"] = nonfatal_inj / py_inj
    if py_noninj > 0:
        nonfatal_noninj = max(events_noninj - fatal_noninj, 0.0)
        stats["nonfatal_od_rate_noninj"] = nonfatal_noninj / py_noninj

    return stats

# =====================================================
# Multi-objective error
# =====================================================
def min_transmission_events(calib):
    """
    Pooled-infection floor for canonical-pick eligibility, derived from the targets
    rather than hard-coded.

    Expected pooled PWID infections over the scored window =
        size x (injecting_pct x oud_active) x n_runs x years x mean(incidence series)

    At the current config (size 500, 10 reps, 11 yrs) and the targets as written
    (0.45 x 0.55 population injection share, mean incidence 0.0066/PWID-yr) that is
    ~90 events; MIN_HIV_TRANSMISSION_FRACTION of it is ~22, against the flat
    MIN_HIV_TRANSMISSION_EVENTS = 2 that was in force through 2026-07-29.

    Deriving it matters because the floor has to track `size`, `n_runs` and the
    injection share: the post-rebuild model runs at an injection share of 0.11
    rather than the target 0.45, which alone shrinks the PWID denominator ~4x.
    """
    c = calib["calibration"]
    size = c["population"]["size"]
    steps = c["population"]["steps"]
    n_runs = c["n_runs"]
    inj_pop_share = c["injecting"]["injecting_pct"] * c["oud_baseline"]["oud_active"]
    mean_inc = float(np.mean(c["hiv"]["incidence_series_per_pwid_yr"]))

    expected = size * inj_pop_share * n_runs * (steps / 12.0) * mean_inc
    return max(MIN_HIV_TRANSMISSION_EVENTS, int(MIN_HIV_TRANSMISSION_FRACTION * expected))

def _traj_rmse(model_series, target_series):
    """RMSE between two trajectories, trimming model_series to length of target."""
    target = np.array(target_series)
    if model_series is None or len(model_series) == 0:
        return float(np.sqrt(np.mean(target ** 2)))  # max-penalty
    n = min(len(target), len(model_series))
    diff = model_series[-n:] - target[-n:]
    return float(np.sqrt(np.mean(diff ** 2)))

def null_traj_rmse(target_series):
    """RMSE hiv_traj_rmse takes when a trial's simulated series is flat zero --
    i.e. no PWID HIV transmission happened at all. Same formula as _traj_rmse's
    max-penalty branch; used post-hoc to flag trials hitting this floor."""
    return float(np.sqrt(np.mean(np.array(target_series) ** 2)))


def _period_groups(years, periods):
    """Index groups into an annual series, one per [start_year, end_year] window."""
    yrs = np.asarray(years)
    return [np.where((yrs >= a) & (yrs <= b))[0] for a, b in periods]


def _period_rmse(model_series, target_series, years, periods):
    """
    RMSE between period MEANS of a model series and a target series.

    Replaces the 11-point annual RMSE for HIV incidence as of 2026-08-03. See
    calibration_targets.yml `incidence_periods` for the measurement that forced
    it: no Pareto member beat a flat-at-mean predictor on the annual series, so
    the year-to-year shape was contributing error rather than signal. Collapsing
    to three period means scores the level and the coarse trend, which the model
    does reproduce, and drops the wiggles it does not.

    Alignment follows _traj_rmse: the model series is right-aligned against the
    target, since the model's first scored year is its first post-burn year and
    the two series end together. Periods that end up with no model coverage
    contribute the target mean itself (i.e. full penalty for that period) rather
    than being silently skipped, so a short series cannot score better than a
    complete one.
    """
    target = np.asarray(target_series, dtype=float)
    groups = _period_groups(years, periods)
    tgt_p = np.array([target[g].mean() for g in groups])

    if model_series is None or len(model_series) == 0:
        return float(np.sqrt(np.mean(tgt_p ** 2)))  # max-penalty, same as _traj_rmse

    n = min(len(target), len(model_series))
    aligned = np.full(len(target), np.nan)
    aligned[-n:] = np.asarray(model_series, dtype=float)[-n:]

    mod_p = np.array([
        np.nanmean(aligned[g]) if not np.all(np.isnan(aligned[g])) else 0.0
        for g in groups
    ])
    return float(np.sqrt(np.mean((mod_p - tgt_p) ** 2)))


def null_period_rmse(target_series, years, periods):
    """Value _period_rmse takes on a flat-zero model series (no transmission)."""
    groups = _period_groups(years, periods)
    target = np.asarray(target_series, dtype=float)
    return float(np.sqrt(np.mean(np.array([target[g].mean() for g in groups]) ** 2)))

def multi_error(stats, targets):
    """
    Return 6 scalar errors for NSGA-II:
      (oud_rmse, hiv_traj_rmse, hiv_prev_err, od_err, moud_traj_rmse,
       inj_share_err)

    OD error now folds in stratified targets (inj / non-inj nonfatal + fatal).
    MOUD error is now a trajectory RMSE against mofan's APCD #16/#43 series.

    inj_share_err was added 2026-07-29. `calibration.injecting.injecting_pct` had
    been defined in calibration_targets.yml since the OUD redesign but was never
    read here, so the injection share -- the denominator of the HIV incidence
    objective and the population the whole intervention analysis is about -- was
    an unscored consequence of five fitted transition rates. New objectives are
    appended, never inserted, because select_canonical and the null-trial screen
    index into `trial.values` positionally.
    """
    oud_rmse = float(np.sqrt(
        sum(
            (stats.get(k, 0) - targets["oud"][k]) ** 2
            for k in ["oud_active", "oud_remission", "oud_none"]
        ) / 3
    ))

    # Period means since 2026-08-03, annual points before that. Objective slot 1
    # is reused rather than appended: the quantity being scored is the same one,
    # measured at a coarser resolution, and select_canonical / the null screen
    # index into trial.values positionally. Studies persisted before this date
    # cannot be resumed against it -- the objective changed meaning, not position.
    hiv_traj_rmse = _period_rmse(
        stats.get("hiv_incidence_series_pwid"),
        targets["hiv_inc_series"],
        targets["hiv_inc_years"],
        targets["hiv_inc_periods"],
    )

    hiv_prev_err = float((stats.get("hiv_prevalence_pwid", 0) - targets["hiv_prev"]) ** 2)

    od_err = float(
        (stats.get("fatal_od_rate", 0)             - targets["od_fatal"])    ** 2
        + (stats.get("nonfatal_od_rate_inj", 0)    - targets["od_nonfatal_inj"]) ** 2
        + (stats.get("nonfatal_od_rate_noninj", 0) - targets["od_nonfatal_noninj"]) ** 2
    )

    moud_traj_rmse = _traj_rmse(
        stats.get("moud_uptake_series"),
        targets["moud_series"],
    )

    inj_share_err = float(
        (stats.get("inj_share_active_oud", 0) - targets["inj_share"]) ** 2
    )

    return (
        oud_rmse,
        hiv_traj_rmse,
        hiv_prev_err,
        od_err,
        moud_traj_rmse,
        inj_share_err,
    )

# =====================================================
# Optuna objective
# =====================================================
def objective(trial):

    calib = load_yaml(CALIB_YAML)
    base = load_yaml(BASE_SCENARIO)
    scenario = copy.deepcopy(base)

    targets = {
        "oud":              calib["calibration"]["oud_baseline"],
        "hiv_prev":         calib["calibration"]["hiv"]["baseline_prevalence_pct"],
        "hiv_inc_series":   calib["calibration"]["hiv"]["incidence_series_per_pwid_yr"],
        "hiv_inc_years":    calib["calibration"]["hiv"]["incidence_series_years"],
        "hiv_inc_periods":  calib["calibration"]["hiv"]["incidence_periods"],
        "od_fatal":         calib["calibration"]["overdose"]["fatal_rate_annual"],
        "od_nonfatal_inj":  calib["calibration"]["overdose"]["nonfatal_rate_inj_annual"],
        "od_nonfatal_noninj": calib["calibration"]["overdose"]["nonfatal_rate_noninj_annual"],
        "moud_series":      calib["calibration"]["moud"]["uptake_series_per_yr"],
        "inj_share":        calib["calibration"]["injecting"]["injecting_pct"],
    }

    # YAML-backed parameters.
    #
    # calibration.* paths are collected rather than applied here. They belong to
    # the engine `calibration` block, which is rebuilt wholesale below -- writing
    # them into `scenario` at this point would work and then be silently thrown
    # away by that rebuild. See the block below.
    calib_block_params = {}
    for p in calib["calibration"]["parameters_to_optimize"]:
        is_int = p.get("type") == "int"
        if is_int:
            val = trial.suggest_int(p["path"], int(p["lower"]), int(p["upper"]))
        else:
            val = trial.suggest_float(p["path"], p["lower"], p["upper"])
        if p["path"].startswith("calibration."):
            calib_block_params[p["path"]] = (int(val) if is_int else float(val))
            continue
        try:
            set_yaml_value(scenario, p["path"], val, is_int=is_int)
        except KeyError:
            # New optimize entry whose YAML key doesn't exist yet in the base scenario
            # — store it in a side-channel block so the engine can pick it up if the
            # corresponding code path reads from there. Skip rather than fail.
            #
            # NOTE this is a dead end, not a fallback: the engine's own parameter
            # check reports "[._calibration_overrides] is unused" on every run.
            # Anything landing here is NOT being fitted. set_yaml_value only
            # raises when an INTERMEDIATE key is missing (it creates the leaf
            # freely), so this fires far less often than it looks -- which is
            # exactly why calibration.sex.act sat here unnoticed.
            scenario.setdefault("_calibration_overrides", {})[p["path"]] = (
                int(val) if is_int else float(val)
            )

    # Engine-level HIV scaling.
    #
    # calibration.test_frequency is NOT optimized: its only consumer is the
    # legacy per-step diagnosis roll in titan/exposures/hiv.py, which is gated
    # off whenever features.hiv_testing is true (as it is in every scenario
    # here). Suggesting it added a search dimension that could not change any
    # objective, which degrades the NSGA-II front by making distinct parameter
    # vectors score identically. HIV testing intensity is calibrated through
    # hiv_testing.base_prob_annual instead.
    acq_scale = trial.suggest_float("calibration.acquisition", 0.1, 3.0)

    # BUG FIXED 2026-08-03: this block used to be assigned as
    #     scenario["calibration"] = {"acquisition": acq_scale}
    # which discarded everything else the base scenario carried under
    # `calibration` and everything the sampler had suggested for it. The
    # casualty was calibration.sex.act, read by titan/interactions/sex.py to
    # scale the act count on every sexual partnership:
    #
    #     mean_sex_acts = rel.get_number_of_sex_acts(...) * model.calibration.sex.act
    #
    # It was doubly lost. The base scenario has a `calibration` block but no
    # `calibration.sex` sub-dict, so set_yaml_value raised on the intermediate
    # key and dumped the value into _calibration_overrides, which the engine
    # reports unused; and had it landed, this assignment would have wiped it.
    # So sexual transmission ran at the engine default for the whole of the
    # calibration while the sampler burned a search dimension on a value that
    # never reached the model -- the same class of defect as the
    # calibration.test_frequency dimension removed below, but worse, because
    # this one was meant to be live (see the 2026-07-29 P1-P8 sexual-route fix).
    #
    # Rebuild from the base block and layer the suggestions on top instead.
    calib_block = copy.deepcopy(base.get("calibration") or {})
    for path, val in calib_block_params.items():
        # strip the leading "calibration." -- the rest is the path within the block
        set_nested(calib_block, path.split(".", 1)[1], val)
    calib_block["acquisition"] = acq_scale
    scenario["calibration"] = calib_block

    # NonInj HIV seed. Added 2026-08-03.
    #
    # hiv.init = 0.075 was set on drug_type Inj ONLY, across all 24 race x
    # sex_type strata. Every NonInj agent -- 75% of each stratum, including all
    # MSM -- started HIV-negative, so the model grew that reservoir from zero
    # across the scored window. That is a RISING force of infection through
    # exactly the window where the data falls, and it held the modelled
    # trajectory flat: job 4576015 (20 Pareto members, 5 populations x 10 reps)
    # measured model T1->T2 of +1% against the target's -36%, with no candidate
    # beating a flat-at-mean predictor. Aggregating the objective to period means
    # did not help (discrimination 1.17 -> 1.19), because the error was a
    # near-constant level offset -- averaging shrank signal and noise alike.
    #
    # Probe 4578413 (2 trials x 6 seed values x 3 populations x 10 reps) measured
    # the effect PAIRED within population seed. Pairing is not optional here: the
    # effect (~0.002) is smaller than between-population variance (SD 0.0022), so
    # unpaired it reads as noise. Change in T2-T1 relative to a zero seed:
    #
    #   seed 0.010   -0.00191   t = -2.21
    #   seed 0.020   -0.00201   t = -3.50
    #   seed 0.030   -0.00297   t = -2.74      (target slope -0.00295)
    #
    # FITTED rather than fixed, because it trades directly against hiv_prev_err:
    # over the same sweep PWID prevalence rose 0.066 -> 0.117 against a 0.062
    # target, while T1 and T3 both rose with the seed (corr +0.46, +0.54). The
    # seed lifts the whole curve as well as tilting it, so any hardcoded value
    # forces the fit to trade shape against level with no freedom to compensate.
    # Letting transmission absorb the level is the whole point.
    #
    # MSM/MSMW are split from HF/HM. They are ~5% of the population, so a single
    # uniform seed cannot place a realistic MSM reservoir without swamping
    # aggregate prevalence with heterosexual seeding. Bounds are wide priors, NOT
    # data anchors -- the fitted values still need checking against APCD (RI HIV
    # prevalence among non-injecting people with OUD, and separately among MSM),
    # and validate_candidates.py reports them unscored for that purpose.
    #
    # dx.init is seeded at the same 0.9 the Inj strata carry. Seeding prevalence
    # without diagnosis would place the entire new reservoir outside the ART
    # cascade and inflate transmission for a reason unrelated to the mechanism
    # being fitted.
    #
    # NOTE: this adds two search dimensions. Studies persisted before this date
    # cannot be resumed against it.
    noninj_hiv_het = trial.suggest_float("calibration.noninj_hiv_init_het", 0.0, 0.04)
    noninj_hiv_msm = trial.suggest_float("calibration.noninj_hiv_init_msm", 0.0, 0.20)
    set_noninj_hiv_init(scenario, het=noninj_hiv_het, msm=noninj_hiv_msm)

    # Run TITAN
    trial_yaml = OUTPUT_DIR / f"trial_{trial.number}.yml"
    write_yaml(trial_yaml, scenario)
    outdir = OUTPUT_DIR / f"trial_{trial.number}"
    outdir.mkdir(exist_ok=True)

    stats = run_titan(trial_yaml, outdir)
    trial.set_user_attr("hiv_new_pooled", stats.get("hiv_new_pooled", 0))
    return multi_error(stats, targets)

# =====================================================
# Canonical-pick selection
# =====================================================
def select_canonical(pareto, min_events=MIN_HIV_TRANSMISSION_EVENTS):
    """
    Pick the canonical Pareto-front member to save as calibrated_scenario_best.yml.

    NOT pareto[0] -- optuna's study.best_trials order is arbitrary, not ranked by
    quality. Ranks by composite min-max-normalized error across all objectives,
    restricted to trials with >= min_events pooled HIV infections (see
    min_transmission_events for how that floor is derived). Falls back to the full
    front, with a warning, if no trial clears the transmission-activity bar.

    Returns (best_trial, composite_score, used_transmission_filter).
    """
    n_obj = len(pareto[0].values)
    lo = [min(t.values[i] for t in pareto) for i in range(n_obj)]
    hi = [max(t.values[i] for t in pareto) for i in range(n_obj)]

    def composite(t):
        return sum(
            (t.values[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0.0
            for i in range(n_obj)
        )

    candidates = [
        t for t in pareto
        if t.user_attrs.get("hiv_new_pooled", 0) >= min_events
    ]
    if not candidates:
        print(
            f"WARNING: no Pareto trial reached {min_events}+ pooled HIV "
            "infections -- falling back to best composite score across the whole front. "
            "The run likely needs revisiting; see docs/calibration_log/"
            "phase_4_ridoh_idu_validation_2026-07-21.md."
        )
        pool, used_filter = pareto, False
    else:
        pool, used_filter = candidates, True

    best = min(pool, key=composite)
    return best, composite(best), used_filter

# =====================================================
# Main
# =====================================================
def main():
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    n_trials = int(os.environ.get("TITAN_CALIB_TRIALS", 120))
    print(f"Running Optuna NSGA-II with {n_workers} workers, {n_trials} trials")

    # Seed the sampler. The winning trial of this study becomes
    # calibrated_scenario_best.yml and every downstream scenario result, so the
    # search that produced it has to be reproducible; an unseeded NSGAIISampler
    # makes the canonical pick unrepeatable even from the same code and targets.
    # Model-level stochasticity is separate and still varies across the n_runs
    # reps within each trial. Override with TITAN_CALIB_SEED.
    sampler_seed = int(os.environ.get("TITAN_CALIB_SEED", 20260730))
    print(f"NSGA-II sampler seed: {sampler_seed}")

    # Persist the study. Added 2026-08-03 after a study was created in-memory and
    # the 640-member Pareto front it produced existed ONLY in the driver process:
    # recovering it meant regex-parsing "Trial N finished with values: [...]" back
    # out of the SLURM stdout (rank_pareto_from_log.py), which cannot recover
    # user_attrs at all -- hiv_new_pooled had to be re-derived by re-parsing every
    # candidate's basicReport.txt. A 6 h run should not be one lost log file away
    # from being unrepeatable.
    storage = os.environ.get("TITAN_CALIB_STORAGE")
    if storage is None:
        job = os.environ.get("SLURM_JOB_ID", "local")
        storage = f"sqlite:///{OUTPUT_DIR}/study_{job}.db"
    print(f"Study storage: {storage}")

    study = optuna.create_study(
        # 6 objectives since 2026-07-29 (inj_share_err appended). Studies persisted
        # from the 5-objective era cannot be resumed against this. Nor can any
        # study persisted before 2026-08-03: objective slot 1 was redefined from
        # an 11-point annual RMSE to a 3-period one, and two search dimensions
        # were added. Same slots, different meaning.
        directions=["minimize"] * 6,
        storage=storage,
        study_name=os.environ.get("TITAN_CALIB_STUDY", "titan_ri_multi"),
        load_if_exists=True,
        # population_size defaults to 50. With 21 search dimensions and 6
        # objectives, non-dominated sorting stops discriminating on a front that
        # small -- most of the population ends up mutually non-dominated and
        # selection pressure collapses. 100 keeps front diversity at the cost of
        # halving the generation count for a given trial budget.
        sampler=optuna.samplers.NSGAIISampler(
            seed=sampler_seed,
            population_size=int(os.environ.get("TITAN_CALIB_POP", 100)),
        ),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_workers,
        show_progress_bar=True,
    )

    # Persist the Pareto front
    pareto = study.best_trials
    print(f"Pareto front: {len(pareto)} parameter sets")

    # Select and write the canonical calibrated scenario (see select_canonical)
    calib = load_yaml(CALIB_YAML)
    int_params = {
        p["path"] for p in calib["calibration"]["parameters_to_optimize"]
        if p.get("type") == "int"
    }
    min_events = min_transmission_events(calib)
    print(f"Transmission-activity screen: >= {min_events} pooled PWID infections")
    best, best_composite, used_transmission_filter = select_canonical(pareto, min_events)
    print(
        f"Canonical pick: trial {best.number} (composite={best_composite:.4f} across "
        f"{len(pareto[0].values)} min-max-normalized objectives, pooled hiv_new="
        f"{best.user_attrs.get('hiv_new_pooled', 'n/a')}, "
        f"transmission filter {'applied' if used_transmission_filter else 'had no eligible trials'})"
    )

    # ---- HIV null-transmission diagnostic ----
    # hiv_traj_rmse == null_traj_rmse(target) exactly when a trial's simulated PWID
    # HIV incidence is flat zero -- see docs/calibration_log/
    # phase_4_oud_addendum_2026-07-20.md (the RMSE floor a null model scores by
    # construction) and phase_4_ridoh_idu_validation_2026-07-21.md (why this check
    # exists, and why it alone isn't sufficient -- see select_canonical above).
    null_value = null_period_rmse(
        calib["calibration"]["hiv"]["incidence_series_per_pwid_yr"],
        calib["calibration"]["hiv"]["incidence_series_years"],
        calib["calibration"]["hiv"]["incidence_periods"],
    )
    null_trials = [t for t in pareto if abs(t.values[1] - null_value) < 1e-6]
    print(
        f"HIV null-transmission check: {len(null_trials)}/{len(pareto)} Pareto trials "
        f"have hiv_traj_rmse == {null_value:.6f} (zero simulated PWID HIV incidence)."
    )
    ridoh = calib["calibration"]["hiv"].get("ridoh_idu_validation")
    if ridoh:
        lo, hi = ridoh["annual_count_range"]
        print(
            f"RIDOH IDU-attributed reference: ~{lo}-{hi} new HIV diagnoses/yr statewide "
            "(3-6% of all diagnoses, 2009-2023) -- sanity-check only, not a per-PWID-year "
            "rate (no RI PWID population denominator exists). See calibration_targets.yml "
            "hiv.ridoh_idu_validation and docs/calibration_log/"
            "phase_4_ridoh_idu_validation_2026-07-21.md."
        )

    calibrated = load_yaml(BASE_SCENARIO)
    calib_block_best = {}
    for k, v in best.params.items():
        if k.startswith("calibration."):
            # Collected, not dropped. This used to `continue` outright and then
            # rebuild the block as {"acquisition": ...}, so the canonical
            # scenario shipped WITHOUT the winning trial's calibration.sex.act --
            # i.e. calibrated_scenario_best.yml did not reproduce the run that
            # selected it. Same defect as in objective(); fixed together
            # 2026-08-03.
            # ...except the NonInj seeds, which are not engine `calibration`
            # keys at all -- they are applied to `demographics` below, exactly as
            # objective() applies them. Leaving them here would put keys in the
            # canonical scenario that no trial YAML carries and the engine does
            # not read.
            if not k.startswith("calibration.noninj_hiv_init"):
                calib_block_best[k] = v
            continue
        is_int = k in int_params
        try:
            set_yaml_value(calibrated, k, v, is_int=is_int)
        except KeyError:
            calibrated.setdefault("_calibration_overrides", {})[k] = (
                int(v) if is_int else float(v)
            )
    calib_block = copy.deepcopy(load_yaml(BASE_SCENARIO).get("calibration") or {})
    for path, val in calib_block_best.items():
        set_nested(calib_block, path.split(".", 1)[1], val)
    calib_block["acquisition"] = best.params["calibration.acquisition"]
    calibrated["calibration"] = calib_block

    # The NonInj HIV seed lives in `demographics`, not in a suggest path, so it
    # has to be re-applied here the same way objective() applied it. Without
    # this the canonical scenario would run with the reservoir back at zero --
    # a different model from the one that won the fit.
    set_noninj_hiv_init(
        calibrated,
        het=best.params["calibration.noninj_hiv_init_het"],
        msm=best.params["calibration.noninj_hiv_init_msm"],
    )

    # Write the canonical pick to a SEPARATE file so the base scenario stays intact
    outpath = REPO_ROOT / "calibrated_scenario_best.yml"
    try:
        write_yaml(outpath, calibrated)
        print(f"Saved canonical Pareto pick to {outpath}")
        print(f"(Base scenario {BASE_SCENARIO} left untouched.)")
    except PermissionError:
        fallback = OUTPUT_DIR / "calibrated_scenario_best.yml"
        write_yaml(fallback, calibrated)
        print(f"No write access to {outpath}; saved canonical Pareto pick to {fallback} instead.")

    # Dump full Pareto front to CSV for review
    pareto_csv = OUTPUT_DIR / "pareto_front.csv"
    rows = []
    for t in pareto:
        row = {"trial": t.number}
        row.update({f"err_{i}": v for i, v in enumerate(t.values)})
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(pareto_csv, index=False)
    print(f"Saved Pareto front CSV: {pareto_csv}")

if __name__ == "__main__":
    main()
