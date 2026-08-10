# titan/calibration/optuna_calibration.py
#
# Multi-objective Optuna calibration for TITAN_RI.
# Objectives (NSGA-II Pareto front), in the order multi_error() returns them:
#   1. oud_rmse              — RMSE of (oud_active, oud_remission, oud_none) at final timestep
#   2. ridoh_hiv_err         — RMSE, statewide cases/yr, of HIV diagnoses among PWID vs
#                              RIDOH's pooled windows: injection-attributed against the IDU
#                              category, and all-route against IDU + MSM/IDU
#   3. od_err                — mean squared PROPORTIONAL error of fatal + nonfatal annual
#                              OD rates vs targets (relative since 2026-08-04)
#   4. moud_traj_rmse        — RMSE of model MOUD uptake series vs APCD 2012–2022
#   5. inj_share_err         — squared error of injection share among active OUD vs target
#
# Note the mixed units: slots 1/2/4 are RMSEs (in the units of the quantity), slots
# 3/5 are SQUARED errors. NSGA-II does not care -- it only ever compares an
# objective against itself -- but any cross-objective aggregation does. See
# SQUARED_OBJECTIVES and select_canonical().
#
# The HIV objective changed on 2026-08-04 from an APCD per-PWID-year incidence
# series to RIDOH IDU-attributed diagnosis counts, and HIV prevalence was dropped
# as an objective at the same time. See multi_error() for why, and
# calibration_targets.yml hiv.ridoh_idu_dx for the target derivation.
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

# Where main() writes the canonical pick. Overridable because it is the one
# output that is NOT under SCRATCH: setting SCRATCH to a temp dir isolates the
# trial directories, the study journal and the front CSV, but a short smoke run
# would still land on the repo's real calibrated_scenario_best.yml and destroy
# the last completed calibration's output. That happened on 2026-08-04; a 2-trial
# 1-rep test overwrote job 4586902's pick, which had to be reconstructed from
# trial_1165.yml. Point TITAN_CALIB_BEST_OUT somewhere disposable when smoke
# testing.
BEST_SCENARIO_OUT = Path(
    os.environ.get("TITAN_CALIB_BEST_OUT", REPO_ROOT / "calibrated_scenario_best.yml")
)

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

# Timesteps per calendar year. Was a parse_basic_report default only; promoted to
# a module constant 2026-08-04 so the null-transmission screen can work out how
# many model years the scored window covers without restating the number.
STEPS_PER_YEAR = 12

# Which slots of multi_error()'s return are SQUARED errors rather than RMSEs.
# Indices are positional and must be updated whenever an objective is added,
# removed or reordered. Used only by select_canonical, to square-root those slots
# before the composite so all objectives are compared in the units of the
# quantity they measure.
#
# Reindexed 2026-08-04 when hiv_prev_err was dropped and the vector went 6 -> 5.
SQUARED_OBJECTIVES = (2, 4)   # od_err, inj_share_err

# Upper anchor for the composite's per-objective normalization, as a quantile of
# the candidate pool rather than its max. Scores above the anchor clip to 1.0.
#
# WHY, measured on job 4586902's 637-member front: the objectives are heavy-tailed
# and each one's own tail sets its normalizer, so a plain min-max hands the widest
# objective the least influence. hiv_prev_err ranged [0, 0.347] on that front (a
# 65% PWID prevalence at the top end), which normalized the canonical pick's own
# 0.00926 -- a 9.6-percentage-point prevalence miss, 15.8% against a 6.2% target --
# down to a 0.027 contribution, less than the 0.081 it took from od_err. 272 of the
# 637 front members sat within 2 points of the prevalence target and none of them
# was picked. Clipping at the 90th percentile drops that trial from 1st to 36th of
# the 507 members clearing the transmission floor. Added 2026-08-04.
COMPOSITE_CLIP_QUANTILE = 0.90

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

# MSM/MSMW NonInj HIV seed as a multiple of the HF/HM one. Module scope, not a
# local, because objective() applies it per trial and main() re-applies it when
# writing the canonical scenario -- two call sites that must not drift. A
# duplicated targets dict of exactly this kind broke validate_candidates for five
# days in July (see build_targets).
#
# An ASSUMPTION, not a derivation; see the block in objective() for what supports
# it and what does not.
NONINJ_HIV_MSM_TO_HET_RATIO = 3.4

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
def parse_basic_report(path, steps_per_year=STEPS_PER_YEAR):
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

    # Injection-attributed diagnoses per model-year, as a COUNT rather than a
    # rate: this is what the RIDOH IDU target is scored against, and counts are
    # the whole point of the swap -- RIDOH's IDU category has no PWID
    # denominator, so a rate cannot be built from it (see calibration_targets.yml
    # hiv.ridoh_idu_dx). Divided by n_runs because reps append into one
    # basicReport.txt and are pooled by the groupby above.
    #
    # `hiv_dx_new_inj`, not `hiv_new_inj`: RIDOH counts diagnoses, not true
    # conversions. And not `hiv_dx_new`, which includes the sexual and background
    # routes -- only ~31% of new infections among PWID are injection-acquired, so
    # the all-route count overstates the target quantity roughly threefold.
    #
    # Reps are counted off run_id rather than passed in, so the divisor cannot
    # drift out of sync with a report written at a different -n than the caller
    # assumes.
    n_reps = int(df["run_id"].nunique()) if "run_id" in df.columns else 1
    stats["n_reps"] = n_reps
    stats["idu_dx_counts_per_year"] = (
        annual_series(df_pwid, "hiv_dx_new_inj", None) / n_reps
    )

    # ALL diagnoses among PWID, whatever the route, as a count on the same
    # footing. Scored against RIDOH's IDU + MSM/IDU categories combined, which is
    # what constrains the sexual and background routes; without it the objective
    # sees only the injection route while calibration.acquisition scales all
    # three. See ridoh_hiv_rmse and calibration_targets.yml
    # hiv.ridoh_pwid_dx_all_routes.
    stats["pwid_dx_counts_per_year"] = (
        annual_series(df_pwid, "hiv_dx_new", None) / n_reps
    )

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


# _period_rmse, null_period_rmse and null_traj_rmse are off the scored path as of
# 2026-08-04, when the HIV objective moved to RIDOH counts. They are retained so
# the APCD comparison stays computable as a diagnostic -- being able to say how a
# candidate scores against the old target is useful when explaining the swap --
# and because parse_basic_report still emits the series they consume.
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

def _window_errors(counts_per_year, first_year, pop_scale, windows):
    """
    Signed error, in statewide cases per year, of one model series against one
    set of RIDOH pooled windows.

    `counts_per_year` is per model-year per model population; multiplying by
    pop_scale puts it on RIDOH's statewide footing. Each RIDOH window is compared
    against the mean of the model years it spans, because RIDOH publishes 5-year
    pooled shares and suppresses single years below 5.

    Returns [] when the series is empty or spans none of the windows; the caller
    decides what that means.
    """
    if counts_per_year is None or len(counts_per_year) == 0:
        return []

    series = np.asarray(counts_per_year, dtype=float) * pop_scale
    years = first_year + np.arange(len(series))

    errs = []
    for w in windows:
        lo, hi = w["years"]
        sel = (years >= lo) & (years <= hi)
        if sel.any():
            errs.append(float(series[sel].mean()) - float(w["cases_per_year"]))
    return errs


def ridoh_hiv_rmse(idu_counts_per_year, pwid_counts_per_year, first_year, targets):
    """
    RMSE, in statewide cases per year, of the model's HIV diagnoses among people
    who inject drugs against RIDOH's pooled exposure-category windows.

    TWO series are scored together, not one:

      * `idu_counts_per_year` -- diagnoses the model attributes to the injection
        route, against RIDOH's `IDU` category (1.8 and 2.5 per year).
      * `pwid_counts_per_year` -- ALL diagnoses among PWID whatever the route,
        against RIDOH's `IDU` + `MSM/IDU` categories (5.4 and 8.7 per year).

    Both are statewide cases per year, so they share units and combine into a
    single RMSE with no normalization step. They are deliberately one objective
    rather than two: NSGA-II loses selection pressure as the objective count
    grows -- at six, nearly every candidate is non-dominated -- and these two
    quantities are nested measurements of the same thing.

    Scoring only the injection series leaves the sexual route unconstrained. The
    optimizer's cheapest route to a good IDU score is then to crush
    partnership.injection.transmission.base while calibration.sex.act stays high,
    which inerts the one route the interventions act on. Scoring both series
    together closes that. See calibration_targets.yml
    hiv.ridoh_pwid_dx_all_routes.

    (Until 2026-08-06 the mechanism named here was `calibration.acquisition`,
    a global scalar over both routes. It has since left the search as a redundant
    dimension -- see the block in objective() -- but the argument is unchanged:
    two route-specific levers still need two constraints between them.)

    The pre-2015 bound is one-sided: PMC6055286 bounds IDU-attributed diagnoses
    at <6/yr over 2009-2014, which is a ceiling and not an estimate, so it costs
    nothing until the model exceeds it.

    Returns the max-penalty value when the model produced no usable series, so a
    failed parse cannot look like a good fit.
    """
    idu_windows = targets["ridoh_windows"]
    pwid_windows = targets["ridoh_pwid_windows"]
    scale = targets["pop_scale"]

    def max_penalty():
        cases = [w["cases_per_year"] for w in idu_windows + pwid_windows]
        return float(np.sqrt(np.mean(np.square(cases))))

    errs = _window_errors(idu_counts_per_year, first_year, scale, idu_windows)
    errs += _window_errors(pwid_counts_per_year, first_year, scale, pwid_windows)

    bound = targets.get("ridoh_upper_bound")
    if bound is not None and idu_counts_per_year is not None and len(idu_counts_per_year):
        series = np.asarray(idu_counts_per_year, dtype=float) * scale
        years = first_year + np.arange(len(series))
        lo, hi = bound["years"]
        sel = (years >= lo) & (years <= hi)
        if sel.any():
            excess = float(series[sel].mean()) - float(bound["max_cases_per_year"])
            errs.append(max(0.0, excess))

    if not errs:
        return max_penalty()

    return float(np.sqrt(np.mean(np.square(errs))))

def build_targets(calib):
    """
    Build the targets dict multi_error() reads, from the parsed
    calibration_targets.yml `calibration` block.

    Single definition on purpose. objective() and validate_candidates both need
    this and used to construct it separately; when inj_share_err was added as an
    objective on 2026-07-29 only one of the two was updated, and every
    validate_candidates run from then until 2026-08-03 died with
    KeyError: 'inj_share' on its first trial.

    args:
        calib: the "calibration" mapping from calibration_targets.yml
    """
    hiv = calib["hiv"]
    ridoh = hiv["ridoh_idu_dx"]

    # Model counts are per model population; RIDOH counts are statewide. See
    # calibration_targets.yml hiv.population_scale for why the anchor is APCD's
    # F11-dx OUD count against oud_active rather than PROFOUND's wider at-risk
    # figure.
    active_frac = calib["oud_baseline"]["oud_active"]
    pop_size = calib["population"]["size"]
    pop_scale = hiv["population_scale"]["ri_active_oud"] / (active_frac * pop_size)

    return {
        "oud": calib["oud_baseline"],
        "od_fatal": calib["overdose"]["fatal_rate_annual"],
        "od_nonfatal_inj": calib["overdose"]["nonfatal_rate_inj_annual"],
        "od_nonfatal_noninj": calib["overdose"]["nonfatal_rate_noninj_annual"],
        "moud_series": calib["moud"]["uptake_series_per_yr"],
        "inj_share": calib["injecting"]["injecting_pct"],
        "ridoh_windows": ridoh["windows"],
        "ridoh_pwid_windows": hiv["ridoh_pwid_dx_all_routes"]["windows"],
        "ridoh_upper_bound": ridoh.get("upper_bound_pre_2015"),
        "pop_scale": pop_scale,
        # First calendar year of the scored window. Still taken from the APCD
        # series' year list, which is retained for provenance even though its
        # values are no longer scored -- it is the only place the window's
        # calendar anchor is written down.
        "first_year": hiv["incidence_series_years"][0],
        # Reported but no longer scored; kept so validate_candidates can display
        # PWID HIV prevalence against the old APCD figure as a diagnostic.
        "hiv_prev": hiv["baseline_prevalence_pct"],
    }

def multi_error(stats, targets):
    """
    Return 5 scalar errors for NSGA-II:
      (oud_rmse, ridoh_hiv_err, od_err, moud_traj_rmse, inj_share_err)

    OD error folds in stratified targets (inj / non-inj nonfatal + fatal).
    MOUD error is a trajectory RMSE against mofan's APCD #16/#43 series.

    inj_share_err was added 2026-07-29. `calibration.injecting.injecting_pct` had
    been defined in calibration_targets.yml since the OUD redesign but was never
    read here, so the injection share -- the population the whole intervention
    analysis is about -- was an unscored consequence of five fitted transition
    rates.

    CHANGED 2026-08-04, and this is a reindexing, not an append. Two slots moved:

      * `hiv_traj_rmse` (APCD per-PWID-year incidence series) is replaced by
        `ridoh_idu_err`, RMSE in statewide cases per year against RIDOH's
        IDU-attributed diagnosis counts. The APCD series measures diagnoses in a
        claims-identified OUD population on any transmission route; fitting it
        drove the model to 4-6x RIDOH's observed IDU counts. See
        calibration_targets.yml hiv.ridoh_idu_dx for the derivation.
      * `hiv_prev_err` is REMOVED. Its target (APCD #25) has the same
        any-transmission-route problem, RIDOH publishes no PLWH breakdown by
        transmission category to replace it with (checked across the 2019, 2022
        and 2024 surveillance reports), and paired with the incidence target it
        was jointly infeasible -- the two implied a 9.4-year HIV+ dwell time in
        the PWID compartment against the model's 22.9. PWID HIV prevalence is
        still reported by parse_basic_report and checked by validate_candidates;
        it is simply no longer optimized against.

    Everything that indexes `trial.values` positionally -- SQUARED_OBJECTIVES,
    select_canonical, the null-transmission screen, validate_candidates.OBJECTIVES
    -- was updated with this change. No study persisted before 2026-08-04 can be
    resumed against it.
    """
    oud_rmse = float(np.sqrt(
        sum(
            (stats.get(k, 0) - targets["oud"][k]) ** 2
            for k in ["oud_active", "oud_remission", "oud_none"]
        ) / 3
    ))

    ridoh_hiv_err = ridoh_hiv_rmse(
        stats.get("idu_dx_counts_per_year"),
        stats.get("pwid_dx_counts_per_year"),
        targets["first_year"],
        targets,
    )

    # RELATIVE, changed 2026-08-04. This was a sum of ABSOLUTE squared errors,
    # which weighted each component by the square of its own scale and so let the
    # smallest one drift unpenalised. The fatal rate's target is 0.015 against
    # 0.091 for nonfatal-injection -- a 36x ratio once squared -- so on the
    # canonical trial a fatal rate 52% BELOW target and a non-injection nonfatal
    # rate 44% below target still summed to 0.000598 against a 0.005 gate, an
    # apparent 0.12x pass. Every trial on the 2026-08-04 front was 2-5x low on
    # fatal overdose and every one of them passed this objective.
    #
    # Scaling each residual by its own target makes the three commensurate: the
    # value is now the RMS proportional miss, so 0.10 means "within 10% on all
    # three". Splitting fatal into its own objective would have been the other
    # fix and was rejected -- six objectives collapses NSGA-II's selection
    # pressure (see ridoh_hiv_rmse for the same reasoning).
    #
    # This changes the objective's SCALE by ~3 orders of magnitude. The
    # validate_candidates gate moved with it; no earlier study is comparable.
    od_err = float(np.mean([
        ((stats.get("fatal_od_rate", 0) - targets["od_fatal"])
         / targets["od_fatal"]) ** 2,
        ((stats.get("nonfatal_od_rate_inj", 0) - targets["od_nonfatal_inj"])
         / targets["od_nonfatal_inj"]) ** 2,
        ((stats.get("nonfatal_od_rate_noninj", 0) - targets["od_nonfatal_noninj"])
         / targets["od_nonfatal_noninj"]) ** 2,
    ]))

    moud_traj_rmse = _traj_rmse(
        stats.get("moud_uptake_series"),
        targets["moud_series"],
    )

    inj_share_err = float(
        (stats.get("inj_share_active_oud", 0) - targets["inj_share"]) ** 2
    )

    return (
        oud_rmse,
        ridoh_hiv_err,
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

    targets = build_targets(calib["calibration"])

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
    #
    # calibration.acquisition is NOT optimized either, as of 2026-08-06. It was
    # `trial.suggest_float("calibration.acquisition", 0.1, 3.0)` and it is now
    # pinned at the base scenario's 1.0, for two reasons.
    #
    # (1) IT WAS REDUNDANT. It multiplies the per-act probability of BOTH routes
    # at titan/exposures/hiv.py, after the route branch closes, on top of the
    # route-specific bases those routes already have:
    #
    #     injection level  ~  partnership.injection.transmission.base x acquisition
    #     sexual level     ~  partnership.sex.acquisition[type][role]
    #                             x calibration.sex.act x acquisition
    #
    # `calibration.injection.act` is not fitted, so `base x acquisition` is
    # EXACTLY degenerate -- only the product is identifiable -- and
    # `sex.act x acquisition` is near-degenerate wherever 1-(1-p)^n ~= np, which
    # is the whole operating range here. Three parameters spanned two effective
    # degrees of freedom, so NSGA-II was searching a ridge rather than a peak.
    # Dropping it costs no reachable model: the target region measured by
    # probe_transmission_feasibility and probe_pwid_share (acquisition ~0.5 with
    # base 0.0076 and sex.act 1.75) is reproduced at acquisition 1.0 by base
    # 0.0038 and sex.act 0.87, both comfortably inside their existing bounds.
    #
    # (2) IT MADE THE EXOGENOUS ROUTE IMPLICITLY FITTED. hiv.py also applies it
    # to the background force of infection, whose own inputs
    # (hiv.background_sex.rate_nonoud_per_timestep, .partner_hiv_prevalence_nonoud)
    # are data-anchored and deliberately NOT in parameters_to_optimize -- see the
    # F9 fix, which removed that route's free lever precisely because an
    # exogenous route no intervention can reach biases every scenario contrast
    # toward the null. A fitted global scalar on top of fixed inputs re-created
    # the same defect at one remove. At 1.0 the background route finally runs at
    # exactly its documented intensity.
    #
    # CONSEQUENCE TO WATCH: relative to the acquisition ~0.5 region the probes
    # identified, the background route is now 2x hotter, because it no longer
    # rides down with the scalar. It carried 1-6% of infections across the
    # 2026-08-05 sweep, so the effect is bounded, but its share is worth checking
    # on the refit's canonical pick. If it has grown materially, the fix is to
    # re-derive rate_nonoud_per_timestep, NOT to restore this parameter.

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
    # acquisition is no longer overridden here -- it stays at the base scenario's
    # 1.0. See the block above for why it left the search.
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
    # MSM/MSMW are split from HF/HM. They are ~5% of the population, so a single
    # uniform seed cannot place a realistic MSM reservoir without swamping
    # aggregate prevalence with heterosexual seeding.
    #
    # dx.init is seeded at the same 0.9 the Inj strata carry. Seeding prevalence
    # without diagnosis would place the entire new reservoir outside the ART
    # cascade and inflate transmission for a reason unrelated to the mechanism
    # being fitted.
    #
    # COLLAPSED TO ONE DIMENSION 2026-08-04. These were two independently fitted
    # parameters. The stated justification was that they trade against
    # hiv_prev_err -- "the seed lifts the whole curve as well as tilting it, so
    # any hardcoded value forces the fit to trade shape against level" -- and
    # hiv_prev_err was dropped as an objective on the same day. That left two free
    # parameters with no objective constraining them.
    #
    # Restoring a prevalence objective was not an option: RIDOH publishes no PLWH
    # breakdown by transmission category (checked across the 2019, 2022 and 2024
    # reports). Rebuilding the burn-in so the reservoir grows endogenously was
    # measured and refuted -- probe_burnin_length showed HIV prevalence still
    # climbing after 360 burn steps with no plateau, so there is no equilibrium to
    # burn in to. What remains is to fit the LEVEL and fix the RATIO.
    #
    # RATIO = 3.4, and it is an ASSUMPTION, not a derivation. Two independent
    # numbers land within 5% of it:
    #   * the pre-refit fitted ratio on job 4586902's canonical trial, 0.0953 /
    #     0.0283 = 3.37;
    #   * RIDOH's 2020-2024 exposure shares, GBMSM 65.5% of known-mode diagnoses
    #     against all-heterosexual 18.5%, a ratio of 3.54.
    # Those are different quantities -- one is a seeded prevalence, the other a
    # diagnosis share, and converting between them would need the MSM population
    # share (0.05 here) and would give a far larger number. Their agreement is
    # support, not proof. Revisit if a prevalence-side anchor ever appears.
    # LEFT THE SEARCH 2026-08-07, and the reason is job 4779604. Everything above
    # this line is the history of a parameter that no longer needs to exist.
    #
    # The argument for fitting it rested on one measured claim: "rebuilding the
    # burn-in so the reservoir grows endogenously was measured and refuted --
    # probe_burnin_length showed HIV prevalence still climbing after 360 burn
    # steps with no plateau, so there is no equilibrium to burn in to." That was
    # measured at a transmission level roughly 3x too hot, where nothing
    # equilibrates at any burn length.
    #
    # Extending the sweep to 1440 burn steps across three transmission levels
    # shows the equilibrium exists and is reached by 720. At the RIDOH-consistent
    # level the NonInj reservoir builds itself from trial 1267's floor seed of
    # 0.000398 up to 0.0367 and holds flat to 1440, while the Inj compartment
    # relaxes DOWN from its 0.075 seed to 0.0431. Two compartments starting two
    # orders of magnitude apart converge -- the seed sets nothing but how long
    # the burn-in takes to get there.
    #
    # So it is seeded at zero and the burn-in does the work. This also removes
    # the degeneracy that let NSGA-II fix the allele at its search floor on job
    # 4743939 while leaving PWID prevalence unconstrained. The ratio constant
    # NONINJ_HIV_MSM_TO_HET_RATIO is retained for the MSM reservoir work but is
    # not exercised here.
    #
    # Requires burn_steps >= 720. At the pinned 60 this zeroes a reservoir that
    # has no time to rebuild. See
    # docs/calibration_log/phase_6_burnin_equilibrium_2026-08-07.md.
    set_noninj_hiv_init(scenario, het=0.0, msm=0.0)

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
def _quantile(values, q):
    """Linear-interpolated quantile of a list, without pulling in numpy for one call."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    f = int(k)
    if f + 1 >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[f + 1] - s[f])

def select_canonical(pareto, min_events=MIN_HIV_TRANSMISSION_EVENTS):
    """
    Pick the canonical Pareto-front member to save as calibrated_scenario_best.yml.

    NOT pareto[0] -- optuna's study.best_trials order is arbitrary, not ranked by
    quality. Ranks by a composite of normalized errors across all objectives,
    restricted to trials with >= min_events pooled HIV infections (see
    min_transmission_events for how that floor is derived). Falls back to the full
    front, with a warning, if no trial clears the transmission-activity bar.

    The composite normalizes each objective onto [0, 1] and sums, with two
    corrections added 2026-08-04 after job 4586902 picked a trial whose PWID HIV
    prevalence was 2.5x its target:

      1. The SQUARED_OBJECTIVES slots are square-rooted first, so all six enter on
         the scale of the quantity rather than its square. Mixing the two is what
         let a large error in a squared objective read as a small normalized one.
      2. The upper anchor is COMPOSITE_CLIP_QUANTILE of the pool, not its max, with
         scores above it clipped to 1.0. A single runaway trial can otherwise set an
         objective's normalizer and mute that objective for every other trial.

    Normalization is over the filtered pool, not the whole front: trials excluded
    for insufficient transmission are not eligible to be picked, so they should not
    be setting the scale the eligible ones are judged on either.

    Composite scores are therefore NOT comparable to those printed by runs before
    2026-08-04, and their magnitude is only meaningful relative to the same pool.

    Returns (best_trial, composite_score, used_transmission_filter).
    """
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

    n_obj = len(pareto[0].values)

    def scaled(t):
        return [
            t.values[i] ** 0.5 if i in SQUARED_OBJECTIVES else t.values[i]
            for i in range(n_obj)
        ]

    cols = list(zip(*[scaled(t) for t in pool]))
    lo = [min(c) for c in cols]
    hi = [_quantile(list(c), COMPOSITE_CLIP_QUANTILE) for c in cols]

    def composite(t):
        v = scaled(t)
        return sum(
            min(1.0, (v[i] - lo[i]) / (hi[i] - lo[i])) if hi[i] > lo[i] else 0.0
            for i in range(n_obj)
        )

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
    # JournalStorage, not SQLite. Job 4586379 died four minutes in with
    #
    #     sqlite3.OperationalError: database is locked
    #
    # and took three of the sixteen workers with it, unwinding study.optimize
    # after 14 trials. The job still exited 0, so SLURM recorded it COMPLETED.
    #
    # The cause is n_jobs=16 threads committing to one SQLite file on
    # /oscar/scratch. SQLite serialises writers behind a single file lock and
    # gives up after a 5 s busy timeout, and its locking is advisory-file-based,
    # which is exactly what a network filesystem does not implement reliably.
    # Every earlier calibration ran in memory, so this path had never carried a
    # parallel write load.
    #
    # JournalStorage is Optuna's supported backend for concurrent workers on a
    # shared filesystem: append-only, one lock per append, no long-lived
    # transaction to contend over. TITAN_CALIB_STORAGE still overrides with an
    # RDB URL if a real database is ever available.
    storage_url = os.environ.get("TITAN_CALIB_STORAGE")
    job = os.environ.get("SLURM_JOB_ID", "local")
    if storage_url is not None:
        storage = storage_url
        print(f"Study storage: {storage} (from TITAN_CALIB_STORAGE)")
    else:
        journal_path = f"{OUTPUT_DIR}/study_{job}.journal"
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(journal_path)
        )
        print(f"Study storage: JournalStorage at {journal_path}")

    study = optuna.create_study(
        # 5 objectives since 2026-08-04, when hiv_prev_err was dropped and slot 1
        # became ridoh_idu_err. No earlier study can be resumed against this: the
        # vector changed length AND meaning. History of the same problem -- 6
        # objectives from 2026-07-29 (inj_share_err appended), slot 1 redefined
        # from an 11-point annual RMSE to a 3-period one on 2026-08-03.
        directions=["minimize"] * 5,
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
        f"{len(pareto[0].values)} objectives normalized to the eligible pool's "
        f"[min, p{int(COMPOSITE_CLIP_QUANTILE * 100)}], pooled hiv_new="
        f"{best.user_attrs.get('hiv_new_pooled', 'n/a')}, "
        f"transmission filter {'applied' if used_transmission_filter else 'had no eligible trials'})"
    )
    print(
        "Composite definition changed 2026-08-04 (squared objectives square-rooted, "
        "upper anchor clipped to a pool quantile); scores are not comparable to "
        "earlier runs. See select_canonical."
    )

    # ---- HIV null-transmission diagnostic ----
    # A trial simulating zero injection-attributed transmission scores exactly the
    # RMSE of the RIDOH window targets about zero -- the value a null model gets
    # for free. See docs/calibration_log/phase_4_oud_addendum_2026-07-20.md for
    # the original version of this floor and
    # phase_4_ridoh_idu_validation_2026-07-21.md for why the check exists and why
    # it is not sufficient alone (see select_canonical above).
    #
    # Rewritten 2026-08-04 with the objective: the floor is now in statewide cases
    # per year against the RIDOH windows, not the APCD per-PWID-year series, and
    # it reads objective slot 1, which is ridoh_hiv_err.
    #
    # BOTH series are zeroed, because slot 1 now scores injection-attributed AND
    # all-route PWID diagnoses together. Zeroing only the injection one would
    # define a "null" that still carries the all-route error and never matches
    # any real trial.
    #
    # Derived by calling the objective on all-zero series rather than by
    # restating its formula here. Restating it gets the wrong answer: the
    # one-sided pre-2015 bound contributes a zero term to the RMSE that a
    # windows-only expression misses, giving 2.178 where the objective actually
    # returned 1.779 under the IDU-only version, and the screen would then never
    # fire.
    targets_for_null = build_targets(calib["calibration"])
    n_model_years = calib["calibration"]["population"]["steps"] // STEPS_PER_YEAR
    null_value = ridoh_hiv_rmse(
        np.zeros(n_model_years),
        np.zeros(n_model_years),
        targets_for_null["first_year"],
        targets_for_null,
    )
    null_trials = [t for t in pareto if abs(t.values[1] - null_value) < 1e-6]
    print(
        f"HIV null-transmission check: {len(null_trials)}/{len(pareto)} Pareto trials "
        f"have ridoh_hiv_err == {null_value:.6f} (zero HIV diagnoses among PWID)."
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
    # No acquisition override: it is not a suggested parameter as of 2026-08-06,
    # so best.params has no such key and the base scenario's 1.0 stands. Reading
    # it here would KeyError on every run of a post-2026-08-06 study.
    calibrated["calibration"] = calib_block

    # The NonInj HIV seed lives in `demographics`, not in a suggest path, so it
    # has to be re-applied here the same way objective() applied it. Without
    # this the canonical scenario would run with a reservoir the fit never saw --
    # a different model from the one that won.
    #
    # Zero since 2026-08-07: the seed left the search and the burn-in builds the
    # reservoir. Reading best.params for it would KeyError on any study run after
    # that date, the same failure mode `calibration.acquisition` caused above.
    set_noninj_hiv_init(calibrated, het=0.0, msm=0.0)

    # Write the canonical pick to a SEPARATE file so the base scenario stays intact
    outpath = BEST_SCENARIO_OUT
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
