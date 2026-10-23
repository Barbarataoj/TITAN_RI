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
#
# Note the mixed units: slots 1/2/4 are RMSEs (in the units of the quantity), slot
# 3 is a SQUARED error. NSGA-II does not care -- it only ever compares an
# objective against itself -- but any cross-objective aggregation does. See
# SQUARED_OBJECTIVES and select_canonical().
#
# inj_share_err was the 5th objective from 2026-07-29 to 2026-08-17 and is now
# reported, not scored -- validate_candidates.VALIDATION carries
# inj_share_active_oud against 0.32. See multi_error() for the reasoning.
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

from titan.calibration.lethality import apply_lethality, projection_from_calib

# =====================================================
# Paths (absolute, Slurm-safe)
# =====================================================
REPO_ROOT = Path(__file__).resolve().parents[2]

CALIB_YAML = REPO_ROOT / "titan/calibration/calibration_targets.yml"
BASE_SCENARIO = REPO_ROOT / "calibrated_scenario_multi_5obj.yml"

# Designed starting points enqueued ahead of the search (Stage D4). Absent file
# is not an error -- the search simply starts cold -- but a malformed one is; see
# warm_start_trials().
WARM_START_YAML = REPO_ROOT / "titan/calibration/warm_start_trials.yml"

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
# Reindexed 2026-08-04 when hiv_prev_err was dropped and the vector went 6 -> 5,
# and again 2026-08-17 when inj_share_err was demoted and it went 5 -> 4.
SQUARED_OBJECTIVES = (2,)   # od_err

# Names for multi_error()'s slots, in order. Used to write the objective
# components into the Optuna journal under stable names, so that reading a study
# does not require knowing which positional order was in force on the day it ran
# -- that order has changed four times since 2026-07-29.
OBJECTIVE_NAMES = ("oud_rmse", "ridoh_hiv_err", "od_err", "moud_traj_rmse")

# The pass/fail gate for each slot of OBJECTIVE_NAMES, in the same order.
#
# STATED HERE BECAUSE THE SEARCH NOW READS THEM. Until 2026-09-01 the gates lived
# only in validate_candidates.OBJECTIVES and were applied after a run had
# finished; the search itself never saw them, so NSGA-II selected on Pareto
# domination alone and a trial that failed every gate could dominate one that
# passed three (completion plan section 2.2). D2 passes these to the sampler as
# constraints, which makes feasibility rather than domination drive selection.
#
# validate_candidates.OBJECTIVES imports this tuple rather than restating the
# numbers, so the gate the search targets and the gate a candidate is validated
# against cannot drift apart. The DERIVATION of each number stays where it was
# written, in the comments of that list -- this is the value, not the argument
# for it.
#
#   oud_rmse       0.05   OUD state distribution
#   ridoh_hiv_err  1.0    statewide PWID diagnoses/yr, RMSE over the scored RIDOH
#                         windows plus the one-sided pre-2015 bound (normally 0).
#                         Was 1.1 until 2026-10-08. The RMSE averages one zero
#                         term with the windows, so the per-window tolerance the
#                         gate buys depends on how many windows are scored: with
#                         four (to 2026-10-07) 1.1 allowed an RMS window error of
#                         1.1 x sqrt(5/4) = 1.23/yr; with two (since 2026-10-07,
#                         hiv.ridoh_scored_window_starts) 1.1 would allow
#                         1.1 x sqrt(3/2) = 1.35/yr, a silent loosening found in
#                         review. 1.0 x sqrt(3/2) = 1.22/yr restores the old
#                         tolerance: +/-49% on the 2.5/yr injection window, +/-19%
#                         on the 6.59/yr all-route window, or up to 1.73/yr on one
#                         window if the other is exact. No 2026-10-07 member
#                         changes (largest member score 0.957 of 1.0).
#   od_err         0.01   mean squared proportional error, i.e. within ~10% on
#                         both nonfatal overdose components. Confirmed reachable
#                         at 0.00327 by job 5053123 and kept 2026-08-24 after B1
#                         proposed relaxing it to 0.08. The fatal component left
#                         the objective 2026-09-29 when overdose.fatal_frac was
#                         derived from the targets -- see od_error().
#   moud_traj_rmse 0.05   MOUD uptake trajectory, 2021-2022 weighted x3. Was 0.10
#                         until 2026-10-05: a flat line at 0.67 scored 0.080 and
#                         passed while Rhode Island rose to 0.81 (close-out of the
#                         care-seeking refit, section 3.2); the best flat line now
#                         scores ~0.06 and fails.
OBJECTIVE_GATES = (0.05, 1.0, 0.01, 0.05)

# Model years the overdose rates are measured over, counted back from the end of
# the run. The three overdose targets are APCD period-3 (2021-2022) averages, so
# the model is scored on the same two years. Until 2026-09-29 it was the final
# year only, which matched neither the target's window nor the precision the gate
# was priced at: a single model year at 40 replicates holds ~320 nonfatal events
# per stratum, and the Monte Carlo noise on those counts is what od_error()
# subtracts.
OD_SCORING_YEARS = 2

# What constraints_func reports for a trial that has no recorded constraint
# vector. Trials from a study created before 2026-09-01 have none, and neither
# does a trial whose objective() raised before it could record one.
#
# Positive means infeasible, so an unknown trial is treated as failing rather
# than passing. The alternative -- defaulting to 0.0 -- would mark every
# pre-existing trial feasible on no evidence and let them crowd out the
# measured-feasible ones in exactly the selection step this change exists to fix.
UNKNOWN_CONSTRAINT = 1e6

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
# A DATA SLOT, set 2026-09-29 to put the NonInj MSM seed at ~0.07: RIDOH 2022
# puts about 6-7% of Rhode Island GBMSM living with HIV (GBMSM denominator
# ~26,500 from the published per-100k new-diagnosis rates). 2.8 x
# NONINJ_HIV_INIT_HET 0.025 = 0.070. Was 3.4 (x 0.020 = 0.068) as an assumption.
#
# NOT "where the model settles", and the reason matters: hiv.init is also the
# HIV prevalence of every agent that ENTERS the cohort (titan/exposures/hiv.py
# init_agent, called by Population.create_agent on replacement), so the seed
# props up each compartment's long-run level and a settle-point derivation is
# circular. Setting it to 8.7 from grid 4's window mean (earlier the same day)
# lifted the MSM compartment to 0.37-0.50 at every act_msm tried
# (analysis/probe_act_msm.py --burn 600, job 6832264), above NHBS-PWID's 25%
# for MSM who INJECT. Upstream TITAN treats hiv.init as data (2026-08-12).
NONINJ_HIV_MSM_TO_HET_RATIO = 2.8

# NonInj HIV initial condition, restored 2026-08-12 after being zeroed on
# 2026-08-07. An ANCHORED INITIAL CONDITION, not a fitted parameter: stated once,
# never varied by a trial. Set where the model settles. 0.020 until 2026-09-29;
# re-derived from grid 4's best point (last-year level 0.0254).
NONINJ_HIV_INIT_HET = 0.025

# Inj HIV initial condition. Was 0.075, described as "series-anchored", but every
# arm of probe_seed_burn relaxed downward from it toward 0.042-0.047 regardless of
# burn length or seed -- the model has been disagreeing with that number since the
# refit. Set where it settles: 0.045 until 2026-09-29, then grid 4's best point
# (last-year level 0.046, window mean 0.049).
INJ_HIV_INIT = 0.046

# Share of people with HIV who know their status, at t = -burn_steps.
# RIDOH 2023 Surveillance Report, 2022 HIV care continuum: 2,796 of an estimated
# ~3,000 people living with HIV in Rhode Island, i.e. 93.2%. Replaces a round 0.9
# that had no source. Closes citations GAP #29.
HIV_DX_INIT = 0.932

# HIV prevalence among the non-OUD adults the background sexual route draws
# partners from. ~3,000 people living with HIV against ~885,000 Rhode Island
# adults. Replaces 0.01, which was ~3x too high and unsourced. Closes GAP #27.
BACKGROUND_PARTNER_HIV_PREV = 0.0034

# The same two background inputs for MSM and MSMW agents, added 2026-09-29
# after the HIV grid (job 6812107) showed the model reaching RIDOH's all-route
# PWID level only by driving its own non-injecting MSM pool to 56-64%
# prevalence. MSM agents' non-OUD partners are the MSM community, not all
# adults, and they have more of them.
#
# PREVALENCE 0.10. RIDOH's 2022 surveillance report gives GBMSM new-diagnosis
# rates of 105-167 per 100,000 GBMSM (2018-2022) against 38-44 GBMSM cases a
# year, which implies RIDOH's GBMSM denominator is about 26,500. GBMSM are
# 64-76% of new diagnoses and a smaller share of the 2,749 prevalent diagnosed
# cases (2021; older cohorts carry more IDU), so roughly 1,500-1,700 GBMSM
# living with diagnosed HIV, about 6% of GBMSM, 7% with the 13% undiagnosed.
# Partners of MSM who use drugs skew toward the higher-risk sexual network:
# among GBMSM with infectious syphilis, 26-54% were HIV-positive (RIDOH 2018-
# 2022, Figure 31). 0.10 sits between the population and the network figure.
# Sensitivity 0.065-0.25.
#
# RATE 1.0 acts per timestep (12/yr with non-OUD partners). A JUDGMENT, not a
# measurement: the all-population 0.5/step (6/yr) is itself unsourced (GAP), and
# for MSM whose partners are mostly outside the OUD cohort it is too low for
# the ~3%/yr incidence RIDOH's MSM/IDU counts imply for MSM who inject (6.2
# diagnoses/yr statewide over roughly 200-250 MSM PWID). At 12 acts/yr x 0.10 x
# 0.00745 per act the background route gives about 0.9%/yr for non-injecting
# MSM and 1.3%/yr for MSM who inject (inj_multiplier 1.5). Set at 2.0 first;
# the 2026-09-29 grid re-run (job 6817937) showed that alone pushing the
# non-injecting MSM compartment of this closed cohort to 0.42 prevalence at the
# coldest network setting, so it was halved. Sensitivity 0.5-2.0; the first
# axis of the sensitivity analysis on the HIV side.
#
# The exogenous-share diagnostic (validation_targets.background_foi,
# max_exogenous_incidence_share 0.25) was written when this route was a free
# lever applied uniformly. For MSM who inject it is now the mechanism RIDOH
# describes, and the PrEP and testing arms still act on it through the
# acquisition-side multipliers, so a higher share within that subgroup is
# expected; the diagnostic should be read for the heterosexual and injection
# routes, not the MSM one.
#
# RATE RESET TO THE ALL-POPULATION 0.5, later on 2026-09-29. At 1.0 the route
# still fed the non-injecting MSM compartment of this closed cohort toward
# 0.3-0.4 prevalence (grid 3, job 6824974). The MSM-specific background input
# is now the data-backed partner prevalence alone; the extra outside exposure
# of MSM who INJECT is the fitted calibration.background.msm_inj_mult.
BACKGROUND_MSM_PARTNER_HIV_PREV = 0.10
BACKGROUND_MSM_RATE_PER_TIMESTEP = 0.5

# Injection initiation and cessation, moved out of the search 2026-08-13 (step 2
# of the 2026-08-12 plan). Neither rate is something any of the five scenarios
# changes -- every scenario file moves MOUD, HIV testing or syringe-service risk
# -- so under the design rule adopted from upstream TITAN they are inputs, not
# free parameters, and their uncertainty belongs in a sensitivity analysis.
#
# INITIATION, 0.039/yr: 3.9 per 100 person-years among people already using
# illicit drugs (PubMed 27780105). Closes citations GAP #4. The previous search
# box was [0.010, 0.0926] and the last fitted value 0.0400, so this barely moves
# the model -- the change is in status, not level.
#
# CESSATION, 0.10/yr: A JUDGMENT PICK inside a wide cohort spread, not a
# published central estimate. The literature runs 4.1 to 32.6 per 100 py
# (PMC5889739; Tijuana 7.3, Iran 26.1), which is a factor of eight. 0.10 sits in
# the lower half because the RI cohort is fentanyl-era and severe. The last
# fitted value was 0.153, so this one DOES move the model: lower cessation means
# a higher injection share, and probe.py measures the direction and size.
# Closes GAP #5 only in the sense that a sourced range now exists; treat the
# point value as the sensitivity analysis's first axis.
#
# relapse_to_inj_prob is deliberately NOT here. It is GAP #6, no clean source was
# found (the Iranian cohort reports a relapse-to-injection RATE among people who
# had stopped, which is a different quantity), and it stays in the search as the
# one remaining free lever on the injection share.
INJECT_START_RATE = 0.039
INJECT_STOP_RATE = 0.10

# Sensitivity range for the above, for the analysis that has to accompany them.
# Not read by the calibration.
INJECT_START_RANGE = (0.010, 0.093)
INJECT_STOP_RANGE = (0.041, 0.326)

# Per-act sexual transmission, anchored 2026-08-13 (step 3 of the 2026-08-12
# plan). Patel et al., as published by CDC: receptive anal 0.0138, insertive anal
# 0.0011, receptive vaginal 0.0008, insertive vaginal 0.0004. `versatile` is the
# mean of the two figures the agent alternates between. Closes citations GAP #32.
#
# THE TABLE IS INDEXED BY THE HIV-NEGATIVE AGENT'S sex_type AND ROLE
# (titan/exposures/hiv.py:429), and a versatile agent takes the opposite role to
# its partner. With `sex_role.init` a point mass everywhere and HM sleeping only
# with HF, just five of the twelve cells are reachable:
#
#   HF   receptive    0.0008    from HM, and from an MSMW resolved to insertive
#   HM   insertive    0.0004    from HF
#   MSM  versatile    0.00745   from MSM and MSMW, both versatile
#   MSMW versatile    0.00745   from MSM and MSMW
#   MSMW insertive    0.0004    ONLY from HF -- so this cell is VAGINAL insertive
#
# Only one reachable value was wrong: MSM versatile was 0.010 against 0.00745,
# 34% hot, in the group that carries most sexual transmission.
#
# A NOTE ON MSMW, because the 2026-08-12 analysis got this wrong and the wrong
# version is more intuitive: MSMW versatile 0.00745 was read as "assumes bisexual
# men only ever have male partners" and a blended ~0.0039 was proposed. That is
# incorrect. Female partners are HF, who are always receptive, which forces the
# MSMW into the insertive slot -- where the value is already the vaginal 0.0004.
# The versatile cell is only ever reached in male-male partnerships, where the
# anal mean is right. Blending would have made the male-male route ~2x cold.
#
# The unreachable cells are set to their published values too. They are dead
# config today and stop being dead the moment sex_role.init is no longer
# degenerate -- and a wrong value there is invisible until then, which is exactly
# how MSM insertive sat at 0.005 against a published 0.0011. See
# tests/sex_acquisition_test.py, which pins the reachable set.
SEX_ACQUISITION = {
    "HF": {"insertive": 0.0004, "receptive": 0.0008, "versatile": 0.0006},
    "HM": {"insertive": 0.0004, "receptive": 0.0008, "versatile": 0.0006},
    "MSM": {"insertive": 0.0011, "receptive": 0.0138, "versatile": 0.00745},
    "MSMW": {"insertive": 0.0004, "receptive": 0.0138, "versatile": 0.00745},
}

# Per-act injection HIV transmission, anchored 2026-08-13 (step 4). Patel et al.
# / CDC: 63 per 10,000 exposures for sharing injection equipment. Closes
# citations GAP #31, which has read "DEFAULT -- use canonical 0.0063" since the
# register was written while the search box ran [0.000581, 0.00466] -- i.e. the
# published value sat ABOVE the entire range every trial could reach.
#
# THE UNITS WERE CHECKED BEFORE PINNING, because 0.0063 is per SHARING event and
# a model applying it per injection act would need it scaled down. TITAN_RI
# applies it per sharing event, so no adjustment:
#
#   titan/interactions/injection.py:47-89  draws `share_acts` from a Poisson on
#       num_acts, then filters each act through `p_unsafe_injection` -- SSP
#       enrolled risk 0.16, otherwise demographics unsafe_prob 0.38 -- so only
#       unsafe (sharing) acts survive to be exposed.
#   titan/exposures/hiv.py:455  combines with utils.total_probability, 1-(1-p)^n.
#
# The population mean unsafe probability at 71 SSP slots is about 0.25, against
# NHBS-PWID 2018's 26% receptive sharing, so the act filter is itself anchored.
#
# PAIRED WITH THE HIV OBJECTIVE, and it only works as a pair. ridoh_hiv_rmse
# scores two series: injection-attributed diagnoses against RIDOH's IDU category,
# and all-route PWID diagnoses against IDU + MSM/IDU. With this parameter free,
# the cheapest way to hit the IDU series is to crush it and let the sexual route
# carry the epidemic, which inerts the one route every planned intervention acts
# on. Fixing it from the literature removes that escape hatch; `calibration.sex.act`
# is left as the fitted lever on the sexual side, so the route MIX is now set by
# evidence and only the sexual LEVEL is fitted.
INJECTION_TRANSMISSION_BASE = 0.0063

def set_per_act_transmission(scenario):
    """
    Write the published per-act sexual acquisition table into the scenario.

    Same contract as the other anchors: stated once, applied by both objective()
    and main(), never fitted. `calibration.sex.act` remains the fitted lever on
    sexual transmission intensity -- this sets what it multiplies.

    Returns the number of cells written, for smoke tests.
    """
    partnership = scenario.setdefault("partnership", {})
    acq = partnership.setdefault("sex", {}).setdefault("acquisition", {})
    n = 0
    for sx_type, roles in SEX_ACQUISITION.items():
        cell = acq.setdefault(sx_type, {})
        for role, value in roles.items():
            cell[role] = float(value)
            n += 1

    injection = partnership.setdefault("injection", {}).setdefault("transmission", {})
    injection["base"] = float(INJECTION_TRANSMISSION_BASE)
    n += 1
    return n

def set_noninj_hiv_init(scenario, het, msm, dx_init=HIV_DX_INIT):
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

def set_hiv_initial_conditions(scenario):
    """
    Apply the full data-anchored HIV initial condition: both compartments'
    prevalence, the diagnosed share, and the background route's partner
    prevalence.

    One function because these four numbers are one statement about the state of
    the epidemic at t = -burn_steps, and splitting them across call sites is how
    the NonInj seed and burn_steps drifted apart in the first place. objective()
    and main() both call this; neither should set any of these directly.

    Every value is sourced -- see the module constants and the block in
    objective(). None of them is fitted, and none appears in
    parameters_to_optimize.
    """
    n = set_noninj_hiv_init(
        scenario,
        het=NONINJ_HIV_INIT_HET,
        msm=NONINJ_HIV_INIT_HET * NONINJ_HIV_MSM_TO_HET_RATIO,
        dx_init=HIV_DX_INIT,
    )

    for race in scenario.get("demographics", {}).values():
        for sx in (race or {}).get("sex_type", {}).values():
            inj = (sx or {}).get("drug_type", {}).get("Inj")
            if inj is None:
                continue
            # dx.init moves on Inj as well as NonInj. Leaving Inj at the old 0.9
            # while NonInj carried RIDOH's 0.932 would make the population-wide
            # diagnosed share a function of the injection share, which is itself
            # a fitted quantity -- a data statement contaminated by a free
            # parameter.
            hiv = inj.setdefault("hiv", {})
            hiv["init"] = float(INJ_HIV_INIT)
            hiv.setdefault("dx", {})["init"] = float(HIV_DX_INIT)
            n += 1

    bg = scenario.setdefault("hiv", {}).setdefault("background_sex", {})
    bg["partner_hiv_prevalence_nonoud"] = float(BACKGROUND_PARTNER_HIV_PREV)
    # MSM/MSMW draw their outside partners from the MSM community (2026-09-29).
    # HF/HM keep the scalars above: the schema's zero default falls back to them.
    by_sex = bg.setdefault("by_sex_type", {})
    for sex_type in MSM_SEX_TYPES:
        by_sex[sex_type] = {
            "partner_hiv_prevalence": float(BACKGROUND_MSM_PARTNER_HIV_PREV),
            "rate_per_timestep": float(BACKGROUND_MSM_RATE_PER_TIMESTEP),
        }
    return n

# Paths this module asserts rather than fits. Every one of them is written by
# set_oud_initial_conditions() AFTER objective()'s suggest loop has run, so a
# path that appears in both places would be silently overwritten -- the trial
# would record a value it never simulated. assert_not_fitted() makes that a
# startup error instead of a fitting artifact nobody notices.
ANCHORED_PATHS = (
    # 2026-10-01: a seed for init_at_equilibrium only once care_seeking is on;
    # the MOUD level lever became moud.acceptance (calibration_targets.yml).
    "moud.reinit_prob",
    "oud_state.init_active",
    "oud_state.init_remission",
    "oud_state.init_none",
    "oud_state.transitions.inject_start_rate_per_year",
    "oud_state.transitions.inject_stop_rate_per_year",
    "partnership.injection.transmission.base",
    # A population seed is a sampling decision, not a parameter. Fitting it would
    # let the search pick the network realization that flatters a vector, which is
    # the opposite of what varying it is for.
    "model.seed.ppl",
    # Fixed from data on 2026-09-29 (docs/calibration_log/phase_6_parameter_audit
    # _2026-09-01.md tiers 1-2, applied). Values live in calibration_targets.yml
    # `anchored_parameters` and are written by set_anchored_parameters(); the
    # fatal fraction is derived there from the overdose targets.
    "hiv_testing.base_prob_annual",
    # Non-clinical testing, pinned 2026-10-07: unidentified by the objective
    # (the diagnosed share saturates), so it left the search. See the
    # anchored_parameters entry in calibration_targets.yml.
    "hiv_testing.nonclinical_annual",
    "moud.post_overdose_rr",
    "moud.discontinue_per_month",
    "moud.overdose_scale.active",
    "overdose.high_risk_mult",
    "overdose.fatal_frac",
    "oud_state.transitions.relapse_to_inj_prob",
    # The MSM background-route inputs (2026-09-29), written by
    # set_hiv_initial_conditions() from the module constants above.
    "hiv.background_sex.by_sex_type.MSM.partner_hiv_prevalence",
    "hiv.background_sex.by_sex_type.MSM.rate_per_timestep",
    "hiv.background_sex.by_sex_type.MSMW.partner_hiv_prevalence",
    "hiv.background_sex.by_sex_type.MSMW.rate_per_timestep",
    # Both sexual act scalars are data slots in the base scenario since
    # 2026-09-29: `act` at the NSFG-anchored prior 1.0 (grid 3, job 6824974,
    # showed it barely moves the all-route PWID series once male-male
    # partnerships are decoupled), `act_msm` at 0.47, derived so the
    # non-injecting MSM compartment sits at 0.20 HIV prevalence
    # (analysis/probe_act_msm.py, job 6826397). The HIV sexual route is fitted
    # through calibration.background.msm_inj_mult instead.
    "calibration.sex.act",
    "calibration.sex.act_msm",
)

def assert_not_fitted(calib):
    """Fail loudly if an anchored path has been added back to the search."""
    fitted = {p["path"] for p in calib["calibration"]["parameters_to_optimize"]}
    # The YAML-stated pins by path only: the derived fatal fraction is already a
    # literal in ANCHORED_PATHS, and computing it here would make the guard
    # depend on the overdose targets, which a minimal calib need not carry.
    yaml_anchored = {e["path"] for e in calib["calibration"].get("anchored_parameters") or []}
    anchored = set(ANCHORED_PATHS) | yaml_anchored
    clash = sorted(fitted & anchored)
    if clash:
        raise RuntimeError(
            "These paths are data-anchored in optuna_calibration.py but also appear "
            "in parameters_to_optimize, where the anchor would silently overwrite the "
            "suggested value: " + ", ".join(clash)
        )


def derived_fatal_frac(calib):
    """The fatal share of overdose events the three overdose targets imply.

    titan/features/overdose.py rolls `overdose.fatal_frac` as a flat coin flip on
    every event, so in expectation fatal / (fatal + nonfatal) IS the parameter,
    whatever the base probabilities and multipliers do. Stating it as a fitted
    parameter alongside a fatal-rate target therefore stated one number twice:
    the 2026-08-11 box [0.1755, 0.2102] and the 2026-09-01 audit's proposal to
    pin it at RIDOH's EMS-based 0.15-0.18 both sat below the ~0.20 the targets
    require, and either would have made od_err unpassable by construction.

    The nonfatal side is weighted by the injection share among active OUD, the
    same `injecting_pct` the initial condition is derived from.
    """
    c = calib["calibration"]
    od = c["overdose"]
    share = float(c["injecting"]["injecting_pct"])          # injecting share among people in active use
    if cohort_footing() == "all":
        # the targets are per cohort member-year; injectors are the active share x the injecting share
        share = share * float(c["oud_baseline"]["oud_active"])
    fatal = float(od["fatal_rate_annual"])
    nonfatal = (share * float(od["nonfatal_rate_inj_annual"])
                + (1.0 - share) * float(od["nonfatal_rate_noninj_annual"]))
    return fatal / (fatal + nonfatal)


def anchored_parameters(calib):
    """{path: value} of every parameter the targets file fixes from data.

    Read from calibration_targets.yml `anchored_parameters`, plus the derived
    fatal fraction. One owner: the numbers are stated there with their sources,
    applied here, and guarded by assert_not_fitted() -- the same contract as the
    module-level anchors above, but with the values in YAML so that a change is
    a data edit and not a code edit.
    """
    entries = calib["calibration"].get("anchored_parameters") or []
    out = {}
    for entry in entries:
        path = entry["path"]
        if path in out:
            raise RuntimeError(f"anchored_parameters states {path} twice")
        value = entry["value"]
        out[path] = int(value) if entry.get("type") == "int" else float(value)
    out["overdose.fatal_frac"] = derived_fatal_frac(calib)
    return out


def set_anchored_parameters(scenario, calib):
    """Write the data-anchored parameters into a scenario.

    Called AFTER objective()'s suggest loop, like the other set_* anchors, and at
    every other site that materialises a runnable scenario from the base -- the
    canonical write in main() and the probes. Returns the values written.
    """
    values = anchored_parameters(calib)
    for path, value in values.items():
        set_yaml_value(scenario, path, value, is_int=isinstance(value, int))
    return values

def set_population_window(scenario, calib):
    """
    Write the calibration's population size and time window into the scenario.

    FOUND MISSING 2026-08-13, and it had been silently wrong since 2026-08-12.
    `calibration.population` in calibration_targets.yml was read only by the
    SCORING path -- build_targets uses `size` for pop_scale and
    min_transmission_events uses `size` and `steps` for the expected event count
    -- while the ENGINE takes the same three quantities from `model.num_pop` and
    `model.time.*` in the base scenario. Nothing connected them.

    `size` (500) and `steps` (132) happened to agree, so the gap was invisible.
    `burn_steps` did not: step 1 of the 2026-08-12 plan set it to 120 in the
    targets file and every trial kept running at the base scenario's 720. The
    headline "a trial now runs 252 steps instead of 852" was never true, the
    2026-08-12 seed and burn decision was applied to the arithmetic but not to
    the model, and the step 4 probe measured a burn-720 model while reporting
    itself as burn 120.

    The values now move together or the run stops, which is the same contract as
    the other anchors: stated once in the targets file, applied here, checked.
    """
    pop = calib["calibration"]["population"]
    size, steps, burn = int(pop["size"]), int(pop["steps"]), int(pop["burn_steps"])

    model = scenario.setdefault("model", {})
    model["num_pop"] = size
    time = model.setdefault("time", {})
    time["num_steps"] = steps
    time["burn_steps"] = burn

    spy = time.get("steps_per_year", STEPS_PER_YEAR)
    if spy != STEPS_PER_YEAR:
        raise RuntimeError(
            f"model.time.steps_per_year is {spy}, but the calibration's targets, "
            f"parser and pop_scale all assume {STEPS_PER_YEAR}"
        )
    return size, steps, burn

def population_seeds(calib):
    """The network realizations every trial is scored across.

    ADDED 2026-08-17, and it closes the same class of gap set_population_window()
    closed on 2026-08-13: a quantity that governed every run while living only as
    a literal in the base scenario, with nothing in the driver reading or writing
    it.

    `model.seed.ppl` was 12345 in `calibrated_scenario_multi_5obj.yml` and was
    never overridden, so every trial of jobs 4743939, 4830724 and 4874631 was
    scored against one network realization. Replicates vary `model.seed.run` only
    and therefore cannot touch between-population variance -- within a study that
    variance is a fixed offset shared by every trial, not noise. See
    calibration_targets.yml `calibration.population.population_seeds` for the
    consequences and phase_6_completion_plan_2026-08-14.md section 2.8.

    Returns a list, always. A single-entry list is today's behaviour stated
    explicitly rather than a scalar that a caller might forget can be plural.
    """
    pop = calib["calibration"]["population"]
    seeds = pop.get("population_seeds")
    if seeds is None:
        raise RuntimeError(
            "calibration.population.population_seeds is missing from "
            "calibration_targets.yml. It is required: leaving the population seed "
            "to the base scenario is what conditioned four refits on one network "
            "realization."
        )
    if not isinstance(seeds, list) or not seeds:
        raise RuntimeError(
            f"calibration.population.population_seeds must be a non-empty list, "
            f"got {seeds!r}"
        )
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            f"calibration.population.population_seeds has duplicates: {seeds!r}. "
            "A repeated seed re-draws the same population and buys no variance "
            "reduction, while costing a full run."
        )
    return [int(s) for s in seeds]

def reps_per_seed(calib):
    """Replicates per population seed, as recorded in the targets file.

    The engine takes the replicate count from run_titan's `-n` flag;
    `params.model.num_reps` is dead config that nothing reads. This is the value
    that gets passed there.
    """
    return int(calib["calibration"]["population"].get("reps_per_seed", 40))

def trial_constraints(errs):
    """The constraint vector Optuna's NSGA-II reads, one entry per objective.

    Optuna's convention: <= 0 is satisfied, > 0 is violated, and the magnitude is
    used to rank infeasible trials against each other. So the natural expression
    of "this objective is inside its gate" is (error - gate), which is negative by
    exactly the margin the trial has in hand and positive by exactly the amount it
    misses by.

    ADDED 2026-09-01 (D2). The gates existed before this only as a post-hoc filter
    in validate_candidates, so the search optimised four objectives with no notion
    of a target value: on job 4874631 a trial missing every gate could dominate one
    passing three, because domination compares objectives pairwise and knows
    nothing about where the acceptable region is. Feeding these to
    NSGAIISampler(constraints_func=...) makes feasible trials sort ahead of
    infeasible ones regardless of domination, which is the selection pressure
    section 2.2 of the completion plan found missing.
    """
    return [float(err) - float(gate) for err, gate in zip(errs, OBJECTIVE_GATES)]


def constraints_func(trial):
    """Hand a completed trial's constraint vector back to the sampler.

    Optuna calls this once per trial, after objective() returns, and requires the
    value to be reachable from the trial object alone -- which is why objective()
    records it as a user_attr rather than returning it.
    """
    recorded = trial.user_attrs.get("constraints")
    if recorded is None:
        return [UNKNOWN_CONSTRAINT] * len(OBJECTIVE_GATES)
    return [float(c) for c in recorded]


def warm_start_trials(calib, path=WARM_START_YAML):
    """Designed starting points to enqueue before the search samples anything.

    Returns a list of (label, params) with params keyed by the same dotted paths
    parameters_to_optimize declares, ready for study.enqueue_trial(). An absent
    file returns [] -- a cold start is a legitimate configuration and must not
    need a file to express.

    VALIDATED AT STARTUP, NOT AT USE. Every vector in the file was lifted from a
    trial of an earlier study, so it can disagree with today's search in three
    ways: a parameter that has since left parameters_to_optimize, one that has
    since joined it, or a value outside a bound that has since moved. All three
    are silent at enqueue time and expensive later -- Optuna would either drop the
    unknown key or clamp the out-of-range value and run a trial that is not the
    point anyone intended. Each is a startup error here instead, for the same
    reason assert_not_fitted() is.
    """
    if not Path(path).exists():
        return []

    spec = {p["path"]: p for p in calib["calibration"]["parameters_to_optimize"]}
    doc = load_yaml(path) or {}
    entries = doc.get("warm_start") or []

    out = []
    for i, entry in enumerate(entries):
        label = entry.get("label", f"warm_start_{i}")
        params = entry.get("params") or {}

        missing = sorted(set(spec) - set(params))
        extra = sorted(set(params) - set(spec))
        if missing or extra:
            raise RuntimeError(
                f"{path} entry '{label}' does not match parameters_to_optimize: "
                f"missing {missing}, unknown {extra}. A warm start has to be a "
                f"complete point in today's search space -- Optuna silently "
                f"ignores keys it was not asked to suggest."
            )

        cleaned = {}
        for name, value in params.items():
            lower, upper = spec[name]["lower"], spec[name]["upper"]
            if not (lower <= value <= upper):
                raise RuntimeError(
                    f"{path} entry '{label}': {name}={value} is outside its box "
                    f"[{lower}, {upper}]. Re-anchor the vector or move the bound; "
                    f"do not leave Optuna to clamp it."
                )
            cleaned[name] = int(value) if spec[name].get("type") == "int" else float(value)
        out.append((label, cleaned))
    return out


def set_population_seed(scenario, seed):
    """Write one network realization into the scenario.

    Paired with population_seeds(): the caller loops the list and calls this per
    run. `model.seed.run` is deliberately left at 0 (random per replicate) -- the
    point is to vary the POPULATION across runs while replicates still vary the
    run seed within each one.
    """
    model = scenario.setdefault("model", {})
    model.setdefault("seed", {})["ppl"] = int(seed)
    model["seed"]["run"] = 0
    return int(seed)

def set_oud_initial_conditions(scenario, calib):
    """
    Apply the OUD-side data anchors: the baseline state split, the injection
    share it implies, and the two injection transition rates.

    Same contract as set_hiv_initial_conditions -- one function, both call sites,
    nothing here fitted. Split from it only because the sources are different
    (APCD and the OUD cohort literature, not RIDOH surveillance).

    THE INJECTION SHARE IS DERIVED, NOT COPIED. `demographics.*.drug_type.Inj.ppl`
    is a POPULATION share, not a share of active OUD: population.create_agent
    draws the OUD state first and assigns every remission agent NonInj, then
    conditional_drug_weights rescales the Inj weight by 1 / (1 - init_remission)
    to restore the marginal (titan/population.py:374-411). So the share of ACTIVE
    OUD agents who inject comes out at Inj.ppl / oud_active, and the value that
    puts the run on its own `injecting_pct` target is oud_active x injecting_pct.

    That pairing was previously left to a hardcoded 0.25 in the base scenario
    against the 0.55 x 0.45 = 0.2475 the targets imply -- a run that starts at a
    0.4545 injection share while being scored against 0.45. Small, but it is the
    seed-and-burn failure mode again: two numbers that must move together, stored
    in two files, with nothing checking them.

    Returns (strata touched, inj_ppl) for smoke tests.
    """
    cfg = calib["calibration"]
    baseline = cfg["oud_baseline"]
    active = float(baseline["oud_active"])
    remission = float(baseline["oud_remission"])
    none = float(baseline.get("oud_none", 0.0))
    inj_ppl = active * float(cfg["injecting"]["injecting_pct"])

    oud = scenario.setdefault("oud_state", {})
    oud["init_active"] = active
    oud["init_remission"] = remission
    oud["init_none"] = none

    transitions = oud.setdefault("transitions", {})
    transitions["inject_start_rate_per_year"] = float(INJECT_START_RATE)
    transitions["inject_stop_rate_per_year"] = float(INJECT_STOP_RATE)

    n = 0
    for race in scenario.get("demographics", {}).values():
        for sx in (race or {}).get("sex_type", {}).values():
            dt = (sx or {}).get("drug_type")
            if not dt or "Inj" not in dt or "NonInj" not in dt:
                continue
            dt["Inj"]["ppl"] = float(inj_ppl)
            dt["NonInj"]["ppl"] = float(1.0 - inj_ppl)
            n += 1
    return n, inj_ppl

def run_titan(params_file, output_dir, n_reps=None, score_t_max=None, run_seed_base=None):
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
    ]
    if run_seed_base is None:
        cmd += ["-n", str(n_reps)]
    else:
        # Common random numbers (2026-10-09): replicate i runs with
        # model.seed.run = run_seed_base + i, so two scenarios validated with
        # the same base share every stream and differ only where their
        # parameters make them differ. A sweep over the seed, one replicate
        # per value (drange's stop is exclusive).
        cmd += ["-n", "1", "-w", f"model.seed.run:{int(run_seed_base)}:{int(run_seed_base) + int(n_reps)}:1"]
    subprocess.run(cmd, check=True)

    report = output_dir / "basicReport.txt"
    if not report.exists():
        raise RuntimeError(f"TITAN failed – missing basicReport.txt in {output_dir}")

    return parse_basic_report(report, score_t_max=score_t_max)

# =====================================================
# basicReport.txt parser (TSV, per-timestep × stratum)
# =====================================================
def concat_reports(paths, dest, score_t_max=None):
    """Join several basicReport.txt files into one, keeping the replicates distinct.

    `run_id` is what parse_basic_report counts replicates with
    (`df["run_id"].nunique()`), and it divides every annual COUNT series by that
    number -- so an id repeated across two files would make 40 replicates look
    like 5 and inflate the counts eightfold. TITAN writes a random 8-character id
    per run, which is already unique across invocations, but the guarantee is
    incidental rather than contractual: an earlier build wrote a per-invocation
    counter, and nothing stops one doing so again. Prefixing each file's ids with
    its block index makes the uniqueness structural and costs nothing.

    Everything else pools by construction -- the parser groups by timestep and
    sums numerators and denominators, which is the same arithmetic whether the
    rows came from one population or eight.

    Returns the parsed stats for the joined file, and leaves the file on disk so a
    trial's pooled report can be re-read without re-running anything.
    """
    frames = []
    for block, path in enumerate(paths):
        frame = pd.read_csv(path, sep="\t")
        if "run_id" in frame.columns:
            frame["run_id"] = f"b{block}_" + frame["run_id"].astype(str)
        frames.append(frame)

    joined = pd.concat(frames, ignore_index=True)
    joined.to_csv(dest, sep="\t", index=False)
    return parse_basic_report(dest, score_t_max=score_t_max)


def scoring_t_max(calib):
    """Last model step the calibration scores: `population.steps` (132 = 2022-12).

    The OUD shares are read at the final step and the overdose rates over the
    last OD_SCORING_YEARS, both anchored to the END of the report. On a
    calibration run that end is 2022; on a 240-step Phase 7 run it is 2031, so
    the gates were silently read over 2030-2031 (found 2026-10-07 on the
    OUD-split sensitivity, jobs 7067373-75). Callers that score a report longer
    than the calibration pass this as parse_basic_report's `score_t_max`.
    """
    return int(calib["calibration"]["population"]["steps"])


def parse_basic_report(path, steps_per_year=STEPS_PER_YEAR, score_t_max=None):
    """
    Parse basicReport.txt (TSV, one row per timestep × demographic stratum) into
    an aggregated stats dict.

    `score_t_max` drops every row after that model step before anything is
    computed, so a report from a longer run scores exactly as the calibration
    run it extends would (see scoring_t_max). None keeps the whole report.

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
    if score_t_max is not None:
        df = df[df["t"] <= int(score_t_max)]
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

    # ---------- NonInj MSM HIV prevalence (validation) ----------
    # Mean over the scored window. Added 2026-09-29 after the HIV grid found the
    # model reaching RIDOH's all-route PWID level by driving this compartment to
    # 0.4-0.8, against ~0.07 among RI GBMSM (RIDOH 2022) and perhaps 0.15-0.25
    # among MSM with OUD. Checked in validate_candidates, never scored.
    if "sex_type" in df.columns:
        msm_rows = post_burn[(post_burn["drug_type"] == "NonInj")
                             & post_burn["sex_type"].isin(("MSM", "MSMW"))]
        msm_agents = msm_rows["agents"].sum()
        if msm_agents > 0:
            stats["hiv_prevalence_noninj_msm"] = float(msm_rows["hiv"].sum() / msm_agents)

    # ---------- Stationarity of the scored window ----------
    #
    # Fractional change in HIV prevalence from the first scored year to the last,
    # per compartment. Reported, never scored -- but a fit measured across a
    # window the epidemic is still moving through is only valid for that exact
    # window, and every scenario contrast run on a longer horizon inherits the
    # drift as if it were an intervention effect.
    #
    # This exists because the drift was invisible for months. It was found by
    # hand on 2026-08-12 (NonInj growing 10-30% at trial 290's parameters), was
    # wrongly attributed to burn-in length, then to transmission level, and was
    # finally traced to the MSM per-act value running 34% hot -- fixed in step 3,
    # after which both compartments came in flat (-0.5% and -0.4% at the anchored
    # values, job 4944499). Nothing in the pipeline would have caught any of that.
    #
    # A gate rather than a fix, deliberately: drift depends on the fitted
    # transmission level, so it cannot be pinned by configuration. What can be
    # done is to refuse to certify a trial that has it. See
    # validate_candidates.VALIDATION for the tolerance.
    # MEASURED AS AN OLS SLOPE SINCE 2026-08-17. It was a ratio of a 12-month
    # opening window to a 12-month closing window, which threw away ~80% of the
    # scored steps and then squared the noise in what remained into a ratio of two
    # small numbers. Over a PWID stratum of ~88 agents at 2-13% prevalence, each
    # window is single-digit HIV-positive counts.
    #
    # Job 4963250 measured what that cost: replicating the same configuration
    # under a different population seed moved hiv_prev_drift_pwid by a median of
    # 0.12 and a maximum of 0.79 against a +/-0.10 tolerance -- so the metric could
    # not distinguish drift from the population draw. hiv_prev_drift_noninj was
    # better behaved (median ~0.06) because its denominator is 82% of the
    # population, which confirms the mechanism is compartment size rather than the
    # arithmetic.
    #
    # A least-squares slope over all 132 steps uses every observation instead of
    # 24 of them. Reported as fractional change per decade so the number stays
    # readable and comparable to the old ratio: prevalence is regressed on time,
    # and the slope is scaled by ten years and divided by mean prevalence.
    #
    # The +/-0.10 tolerance is INHERITED and provisional. It was a guess attributed
    # to the run-to-run spread at 40 reps and is contradicted by measurement;
    # Stage A1 re-derives it from the measured between-population spread. Until
    # then it is reported, never scored -- penalising drift inside the objective
    # would let the optimiser trade it against HIV fit rather than surface it.
    scored = df[df["t"] >= 1]
    if not scored.empty:
        for drug_type, key in (("Inj", "pwid"), ("NonInj", "noninj")):
            sub = scored[scored["drug_type"] == drug_type]
            per_step = sub.groupby("t")[["hiv", "agents"]].sum()
            per_step = per_step[per_step["agents"] > 0]
            if len(per_step) < 2:
                continue

            prev = (per_step["hiv"] / per_step["agents"]).to_numpy(dtype=float)
            t = per_step.index.to_numpy(dtype=float)
            mean_prev = float(prev.mean())
            if not np.isfinite(mean_prev) or mean_prev <= 0:
                continue

            slope = float(np.polyfit(t, prev, 1)[0])
            decade = slope * steps_per_year * 10.0
            stats[f"hiv_prev_drift_{key}"] = decade / mean_prev
            # Kept alongside the slope because the objective's units are
            # prevalence, and a fractional slope on a near-zero base is unstable
            # in a way the absolute one is not.
            stats[f"hiv_prev_slope_{key}"] = decade
            stats[f"hiv_prev_mean_{key}"] = mean_prev

    # Injection share, measured as a slope over the same scored window and by the
    # same method as the prevalence drift above.
    #
    # WHY THIS EXISTS. `inj_share_active_oud` is a post-burn time AVERAGE, and an
    # average cannot distinguish a quantity sitting on its target from one passing
    # through it. Three anchored quantities imply three different injection shares
    # -- 0.32 from the initial condition, 0.281 from INJECT_START_RATE /
    # (INJECT_START_RATE + INJECT_STOP_RATE), and a measured 0.262 at burn 120
    # rising to 0.337 at burn 720 (completion plan section 2.5). The model starts
    # at 0.32, falls below the two-state steady state, crosses back near burn 240
    # and is still climbing at 720, so it is not a simple two-state process and
    # the average alone cannot say whether any burn length is long enough.
    #
    # Reported, never scored: `inj_share_err` was demoted to a validation target
    # on 2026-08-17 and this is a diagnostic on that target, not a second one.
    if not scored.empty:
        per_step = scored.groupby(["t", "drug_type"])["oud_active"].sum().unstack(
            fill_value=0.0
        )
        if "Inj" in per_step:
            active_all = per_step.sum(axis=1)
            per_step = per_step[active_all > 0]
            active_all = active_all[active_all > 0]
            if len(per_step) >= 2:
                share = (per_step["Inj"] / active_all).to_numpy(dtype=float)
                t = per_step.index.to_numpy(dtype=float)
                mean_share = float(share.mean())
                if np.isfinite(mean_share) and mean_share > 0:
                    slope = float(np.polyfit(t, share, 1)[0])
                    decade = slope * steps_per_year * 10.0
                    stats["inj_share_drift"] = decade / mean_share
                    stats["inj_share_slope"] = decade

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

    # The same two series per replicate, for ridoh_hiv_rmse's noise correction.
    # Added 2026-09-29: at pop_scale ~50 the 2015-2019 injection window is about
    # seven simulated diagnoses across all 40 runs, and a squared error on a
    # count that small carries its own sampling variance as a bias -- a PERFECT
    # model scored uncorrected at 40 runs sits at the 1.1 gate, not below it.
    pwid_rows = df[df["drug_type"] == "Inj"]
    first_post_burn_pwid = max(1, df_pwid.index.min()) if len(df_pwid) else 1
    n_years_pwid = int((last_t - first_post_burn_pwid + 1) // steps_per_year)
    stats["idu_dx_counts_by_run"] = _annual_counts_by_run(
        pwid_rows, "hiv_dx_new_inj", first_post_burn_pwid, n_years_pwid, steps_per_year)
    stats["pwid_dx_counts_by_run"] = _annual_counts_by_run(
        pwid_rows, "hiv_dx_new", first_post_burn_pwid, n_years_pwid, steps_per_year)

    # HIV tests per PWID person-year over the scored window, for the NHBS-PWID
    # past-year testing check (validation_targets.hiv_testing). A rate, not the
    # share tested: with lambda tests per person-year the share tested at least
    # once is about 1 - exp(-lambda), which is how the target is converted.
    if "hiv_tests" in df.columns:
        pb_pwid = post_burn[post_burn["drug_type"] == "Inj"]
        pwid_py = pb_pwid["agents"].sum() / steps_per_year
        if pwid_py > 0:
            stats["hiv_tests_per_pwid_yr"] = float(pb_pwid["hiv_tests"].sum() / pwid_py)

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
        if cohort_footing() == "all":
            # whole-cohort footing: everyone on MOUD over every agent (remission agents included)
            denom = win["oud_active"].sum() + win.get("oud_remission", pd.Series([0])).sum()
            num = float(win.get("moud", pd.Series([0])).sum())
        else:
            denom = win["oud_active"].sum()
            num = float(win.get("moud_active", pd.Series([0])).sum())
        if denom > 0:
            moud_traj.append(num / denom)
        else:
            moud_traj.append(0.0)
    stats["moud_uptake_series"] = np.array(moud_traj)
    # Final-year point too (back-compat)
    if len(moud_traj) > 0:
        stats["moud_uptake"] = float(moud_traj[-1])

    # ---------- Overdose, stratified by injection ----------
    # Over the last OD_SCORING_YEARS model years, the APCD T3 window the targets
    # average over. Final-year-only until 2026-09-29.
    final_year_t = grp.index[grp.index > last_t - OD_SCORING_YEARS * steps_per_year]
    fy_inj    = df[(df["drug_type"] == "Inj")    & (df["t"].isin(final_year_t))]
    fy_noninj = df[(df["drug_type"] == "NonInj") & (df["t"].isin(final_year_t))]
    fy_all    = grp.loc[final_year_t]

    # Per-replicate rates over the same window, for od_error()'s noise
    # correction. Only meaningful with two or more replicates.
    if "run_id" in df.columns and df["run_id"].nunique() >= 2:
        stats.update(_od_rates_by_run(df[df["t"].isin(final_year_t)], steps_per_year))

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
    py_col = "agents" if cohort_footing() == "all" else "oud_active"   # whole-cohort footing: per agent-year
    py_inj    = fy_inj[py_col].sum()    / steps_per_year
    py_noninj = fy_noninj[py_col].sum() / steps_per_year
    py_active = fy_all[py_col].sum() / steps_per_year

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

    # Post-overdose linkage (2026-10-01). Survivors split by MOUD status at the
    # event, and restarts within one step of it, pooled over the scored window.
    # `survivor_any_moud_30d_share` is the model's counterpart of APCD job 2b's
    # "any MOUD within 30 days of the index overdose" (13-20% in RI, 2018-2022),
    # which includes survivors already on MOUD; the other two decompose it.
    surv_on = float(fy_all.get("od_survivors_on_moud", pd.Series([0])).sum())
    surv_off = float(fy_all.get("od_survivors_off_moud", pd.Series([0])).sum())
    starts_1 = float(fy_all.get("moud_start_postod_1step", pd.Series([0])).sum())
    if surv_on + surv_off > 0:
        stats["survivor_moud_at_index_share"] = surv_on / (surv_on + surv_off)
        stats["survivor_any_moud_30d_share"] = (surv_on + starts_1) / (surv_on + surv_off)
    if surv_off > 0:
        stats["survivor_restart_1step_share"] = starts_1 / surv_off

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


def _annual_counts_by_run(df_sub, col, first_post_burn, n_years, steps_per_year):
    """(replicates x model years) array of `col` summed per model year per run.

    The mean over axis 0 is the same series `annual_series(..., None) / n_reps`
    produces; the rows are what the objective's noise correction needs and the
    pooled series discards.
    """
    if "run_id" not in df_sub.columns or col not in df_sub.columns or df_sub.empty:
        return None
    per = df_sub.groupby(["run_id", "t"])[col].sum().unstack(fill_value=0)
    if per.shape[0] < 2:
        return None
    t_index = per.columns.to_numpy(dtype=float)
    out = np.zeros((per.shape[0], n_years))
    for y in range(n_years):
        start = first_post_burn + y * steps_per_year
        sel = (t_index >= start) & (t_index <= start + steps_per_year - 1)
        if sel.any():
            out[:, y] = per.loc[:, sel].sum(axis=1).to_numpy(dtype=float)
    return out


def _od_rates_by_run(win, steps_per_year):
    """Per-replicate overdose rates over the scoring window.

    Same arithmetic as the pooled rates in parse_basic_report -- active-OUD
    person-years per stratum, deaths apportioned across strata by event share --
    applied run by run. Returned as arrays keyed `<stat>_by_run`; a run with no
    person-time in a stratum contributes NaN, which od_error() drops.
    """
    py_col = "agents" if cohort_footing() == "all" else "oud_active"
    needed = {"run_id", "drug_type", py_col, "overdose_events"}
    if not needed.issubset(win.columns) or win.empty:
        return {}
    by = win.groupby(["run_id", "drug_type"])
    events = by["overdose_events"].sum().unstack(fill_value=0.0)
    active = by[py_col].sum().unstack(fill_value=0.0)
    runs = active.index

    def column(frame, name):
        if name in frame.columns:
            return frame[name].reindex(runs).fillna(0.0).to_numpy(dtype=float)
        return np.zeros(len(runs))

    e_inj, e_non = column(events, "Inj"), column(events, "NonInj")
    py_inj = column(active, "Inj") / steps_per_year
    py_non = column(active, "NonInj") / steps_per_year
    py_all = active.sum(axis=1).reindex(runs).fillna(0.0).to_numpy(dtype=float) / steps_per_year
    if "overdose_deaths" in win.columns:
        deaths = win.groupby("run_id")["overdose_deaths"].sum()
        deaths = deaths.reindex(runs).fillna(0.0).to_numpy(dtype=float)
    else:
        deaths = np.zeros(len(runs))

    total = e_inj + e_non
    safe_total = np.where(total > 0, total, 1.0)
    f_inj = np.where(total > 0, deaths * e_inj / safe_total, 0.0)
    f_non = np.where(total > 0, deaths * e_non / safe_total, 0.0)

    def rate(events_, fatal_, py):
        safe = np.where(py > 0, py, 1.0)
        return np.where(py > 0, np.maximum(events_ - fatal_, 0.0) / safe, np.nan)

    return {
        "nonfatal_od_rate_inj_by_run": rate(e_inj, f_inj, py_inj),
        "nonfatal_od_rate_noninj_by_run": rate(e_non, f_non, py_non),
        "fatal_od_rate_by_run": np.where(py_all > 0, deaths / np.where(py_all > 0, py_all, 1.0), np.nan),
    }

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

    DO NOT "FIX" THE REP COUNT HERE WITHOUT RE-ANCHORING THE TARGET. There is a
    real bug in this function -- it reads calibration.n_runs (10) while the job
    actually runs at TITAN_CALIB_REPS (40 on job 4830724), so the floor comes out
    22 where this formula at the true rep count gives 90. Correcting the rep count
    alone makes the screen WORSE, and measurably so: on job 4830724's front the
    trials that satisfy the live HIV objective (ridoh_hiv_err <= 1.1) carry 41-91
    pooled infections, median 61, and a floor of 90 excludes 35 of those 36 trials.
    The bug has been accidentally protective.

    The root cause is that `expected` is derived from
    hiv.incidence_series_per_pwid_yr -- the APCD per-PWID-year series that was
    RETIRED as an objective on 2026-08-04 when the HIV objective became RIDOH
    diagnosis counts. That series implies ~359 pooled infections at 40 reps; the
    RIDOH target that is actually scored is best satisfied at ~61. The two HIV
    data sources disagree by roughly 6x, and this function still encodes the one
    that is no longer scored.

    The screen is also largely redundant now. It was introduced when the HIV
    objective was an incidence RATE, where a trial could score well on almost no
    events. Counts penalise too-cold directly: a zero-transmission trial scores
    exactly hiv_null_value() and select_canonical's null screen removes it.

    Left as-is deliberately, 2026-08-11. Re-anchoring the floor to the RIDOH
    target means inverting the scaling chain (pop_scale, diagnosed fraction, route
    share) and is a modelling decision, not a code fix.
    """
    c = calib["calibration"]
    size = c["population"]["size"]
    steps = c["population"]["steps"]
    n_runs = c["n_runs"]
    inj_pop_share = c["injecting"]["injecting_pct"] * c["oud_baseline"]["oud_active"]
    mean_inc = float(np.mean(c["hiv"]["incidence_series_per_pwid_yr"]))

    expected = size * inj_pop_share * n_runs * (steps / 12.0) * mean_inc
    return max(MIN_HIV_TRANSMISSION_EVENTS, int(MIN_HIV_TRANSMISSION_FRACTION * expected))


def hiv_null_value(calib):
    """
    The value ridoh_hiv_err takes for a trial that simulates zero HIV diagnoses
    among PWID -- the score a null model gets for free.

    Derived by calling the objective on all-zero series rather than by restating
    its formula. Restating it gets the wrong answer: the one-sided pre-2015 bound
    contributes a zero term to the RMSE that a windows-only expression misses.

    Both series are zeroed, because the objective scores injection-attributed and
    all-route PWID diagnoses together. Zeroing only the injection one would define
    a "null" that still carries the all-route error and matches no real trial.

    Factored out of main() 2026-08-11 so select_canonical can use it as an
    eligibility bar as well as main() using it as a diagnostic.
    """
    targets = build_targets(calib["calibration"])
    n_model_years = calib["calibration"]["population"]["steps"] // STEPS_PER_YEAR
    return ridoh_hiv_rmse(
        np.zeros(n_model_years),
        np.zeros(n_model_years),
        targets["first_year"],
        targets,
    )

def _traj_rmse(model_series, target_series, weights=None):
    """RMSE between two annual trajectories that both start in the same calendar year.

    Both series begin at the first post-burn year (2012 for the APCD MOUD
    target), so a model series longer than the target is trimmed from the END:
    model[:n] against target[:n]. Until 2026-10-01 this took model[-n:], the
    tail, which is the same thing for a 132-step calibration run (11 model
    years, 11 target years) but scores a 240-step Phase 7 run on 2021-2031
    against the 2012-2022 target -- every ensemble member's moud_traj_rmse rose
    by 0.015 at 240 steps for that reason alone. A model series shorter than
    the target is compared on the years it has, as before.
    """
    target = np.array(target_series)
    if model_series is None or len(model_series) == 0:
        return float(np.sqrt(np.mean(target ** 2)))  # max-penalty
    n = min(len(target), len(model_series))
    diff = np.asarray(model_series)[:n] - target[:n]
    if weights is None:
        return float(np.sqrt(np.mean(diff ** 2)))
    # weighted (2026-10-05): the MOUD series weights 2021-2022 x3 so the end-of-
    # calibration level, which every Phase 7 projection starts from, is scored
    w = np.asarray(weights, dtype=float)[:n]
    return float(np.sqrt(np.sum(w * diff ** 2) / np.sum(w)))

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


def _window_variances(by_run, first_year, pop_scale, windows):
    """Sampling variance of each window mean, aligned with _window_errors.

    `by_run` is (replicates x model years). The window mean the objective scores
    is the mean over replicates of each replicate's own window mean, so its
    variance is the across-replicate variance of those means divided by the
    replicate count. Replicates are treated as exchangeable; with populations
    blocked 8 x 5 that understates the between-population share by about
    (1/8 - 1/40) of it, which A1's decomposition puts at a few percent of the
    total for this objective.

    Returns [] when there is nothing to estimate from, so the caller can fall
    back to the uncorrected score rather than a half-corrected one.
    """
    if by_run is None:
        return []
    arr = np.asarray(by_run, dtype=float) * pop_scale
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] == 0:
        return []
    years = first_year + np.arange(arr.shape[1])

    out = []
    for w in windows:
        lo, hi = w["years"]
        sel = (years >= lo) & (years <= hi)
        if sel.any():
            means = arr[:, sel].mean(axis=1)
            out.append(float(means.var(ddof=1)) / len(means))
    return out


def all_route_windows(block):
    """The all-route PWID windows the objective scores, derived from their parts.

    Since 2026-09-29 the all-route target is IDU + (1 - s) x MSM/IDU, with s the
    stimulant (methamphetamine) share of MSM/IDU injection -- the model is an
    opioid use disorder cohort and RIDOH's MSM/IDU category counts any
    injection. See calibration_targets.yml hiv.ridoh_pwid_dx_all_routes. A block
    without `parts` scores its `windows` unchanged.
    """
    parts = block.get("parts")
    if not parts:
        return block["windows"]
    share = float(block.get("stimulant_share_of_msm_idu", 0.0))
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"stimulant_share_of_msm_idu {share} outside [0, 1]")
    return [
        {"years": list(p["years"]),
         "cases_per_year": float(p["idu"]) + (1.0 - share) * float(p["msm_idu"])}
        for p in parts
    ]


def split_scored_windows(windows, scored_starts):
    """(scored, unscored) RIDOH windows by first year.

    SINCE 2026-10-07 only the 2018-2022 windows are scored
    (calibration_targets.yml hiv.ridoh_scored_window_starts). Every fixed input
    is of 2022 vintage -- the injecting share (RIHRSS), the care-setting table
    (2022 APCD receipt), testing -- and the model has no time-varying HIV
    driver, so its diagnosis series is flat (slope -0.11 to +0.21 cases/yr on
    all 20 care_refit3 trials) while RIDOH rises 4.18 -> 6.59. Scored
    together, the two windows pull every trial to the same floor (mean 0.85 of
    the gate, true between-trial SD 0.08 against a draw SD 0.23), so pass/fail
    was Monte Carlo noise. Scored on the window the Phase 7 projections start
    from, trials differ (true SD 0.38). The 2015-2019 windows are reported by
    validate_candidates as validation rows; the model's failure to reproduce
    the rise is a stated limitation. Decision record:
    docs/calibration_log/hiv_objective_window_decision_2026-10-07.md.

    `scored_starts` None or empty means every window is scored (the pre-2026-10-07
    objective, and what a minimal calib without the key gets).
    """
    if not scored_starts:
        return list(windows), []
    starts = {int(s) for s in scored_starts}
    scored = [w for w in windows if int(w["years"][0]) in starts]
    unscored = [w for w in windows if int(w["years"][0]) not in starts]
    if not scored:
        raise ValueError(f"ridoh_scored_window_starts {sorted(starts)} matches no window")
    return scored, unscored


def ridoh_hiv_rmse(idu_counts_per_year, pwid_counts_per_year, first_year, targets,
                   idu_by_run=None, pwid_by_run=None):
    """
    RMSE, in statewide cases per year, of the model's HIV diagnoses among people
    who inject drugs against RIDOH's pooled exposure-category windows.

    `idu_by_run` / `pwid_by_run` are the same two series per replicate
    (replicates x years); when both are given the squared errors are corrected
    for their sampling variance -- see the block below.

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

    # NOISE-CORRECTED SINCE 2026-09-29. E[(x + e)^2] = x^2 + var(e): a squared
    # error on a noisy window mean carries that mean's sampling variance as a
    # bias, exactly the mechanism probe_split_bias measured on od_err. Here the
    # counts are small enough that the bias is the whole gate -- a Poisson
    # simulation of a PERFECT model at 40 runs scores 0.87 on average and clears
    # 1.1 only 76% of the time. Subtracting each window's estimated variance
    # makes the expected score the true squared error, so the constrained search
    # ranks feasibility on the model rather than on the draw. Without replicate
    # detail (single-run reports, the all-zero null in hiv_null_value) nothing
    # is subtracted and the score is what it always was.
    variances = [0.0] * len(errs)
    if idu_by_run is not None and pwid_by_run is not None:
        est = _window_variances(idu_by_run, first_year, scale, idu_windows)
        est += _window_variances(pwid_by_run, first_year, scale, pwid_windows)
        if len(est) == len(errs):
            variances = est

    bound = targets.get("ridoh_upper_bound")
    if bound is not None and idu_counts_per_year is not None and len(idu_counts_per_year):
        series = np.asarray(idu_counts_per_year, dtype=float) * scale
        years = first_year + np.arange(len(series))
        lo, hi = bound["years"]
        sel = (years >= lo) & (years <= hi)
        if sel.any():
            excess = float(series[sel].mean()) - float(bound["max_cases_per_year"])
            errs.append(max(0.0, excess))
            variances.append(0.0)   # one-sided: no correction

    if not errs:
        return max_penalty()

    corrected = np.square(np.asarray(errs, dtype=float)) - np.asarray(variances, dtype=float)
    return float(np.sqrt(max(0.0, float(np.mean(corrected)))))


def hiv_err_all_windows(stats, targets, bias_correct=True):
    """The HIV objective scored over EVERY RIDOH window, scored and unscored.

    Since 2026-10-07 the objective scores the 2018-2022 windows only
    (split_scored_windows). This is the pre-change score, computed from the
    same stats, so both can be reported side by side for the canonical pick
    and every ensemble member -- the change was made after the trials were
    seen, and reporting both is the safeguard. Not used for pass/fail.
    """
    t = dict(targets)
    t["ridoh_windows"] = list(targets["ridoh_windows"]) + list(targets.get("ridoh_windows_unscored", []))
    t["ridoh_pwid_windows"] = (list(targets["ridoh_pwid_windows"])
                               + list(targets.get("ridoh_pwid_windows_unscored", [])))
    return ridoh_hiv_rmse(
        stats.get("idu_dx_counts_per_year"),
        stats.get("pwid_dx_counts_per_year"),
        t["first_year"],
        t,
        idu_by_run=stats.get("idu_dx_counts_by_run") if bias_correct else None,
        pwid_by_run=stats.get("pwid_dx_counts_by_run") if bias_correct else None,
    )


def od_error(stats, targets, bias_correct=True):
    """Mean squared proportional error of the two nonfatal overdose rates.

    TWO COMPONENTS SINCE 2026-09-29, not three. overdose.fatal_frac is a flat
    per-event coin flip, so once it is derived from the targets
    (derived_fatal_frac) the fatal rate is fatal_frac / (1 - fatal_frac) times
    the nonfatal rate by construction, and a fatal term measured only what the
    nonfatal terms already measure plus its own Monte Carlo noise -- 81.5% of
    trial 561's od_err on 2026-08-17 was that one term. fatal_od_rate is still
    reported and checked in validate_candidates.

    NOISE-CORRECTED. probe_split_bias (job 5569467) measured the objective at
    0.0201 / 0.0149 / 0.0061 / 0.0037 across 5 / 10 / 20 / 40 replicates on a
    fixed configuration: the squared error carries the rate estimate's sampling
    variance as a bias. Each component subtracts that variance, estimated from
    the per-replicate rates in `<stat>_by_run`, so the expectation is the true
    squared error whatever the replicate count. `bias_correct=False` gives the
    raw value, which objective() also records for comparison.
    """
    comps = []
    for key, tkey in (
        ("nonfatal_od_rate_inj", "od_nonfatal_inj"),
        ("nonfatal_od_rate_noninj", "od_nonfatal_noninj"),
    ):
        target = float(targets[tkey])
        sq = ((float(stats.get(key, 0.0)) - target) / target) ** 2
        by_run = stats.get(f"{key}_by_run") if bias_correct else None
        if by_run is not None:
            v = np.asarray(by_run, dtype=float)
            v = v[np.isfinite(v)]
            if len(v) >= 2:
                sq -= float(v.var(ddof=1)) / len(v) / target ** 2
        comps.append(sq)
    return float(max(0.0, float(np.mean(comps))))

def cohort_footing():
    """Which agents the 13,704 F11-diagnosed cohort maps to: 'all' (the whole simulated cohort, active
    use and remission; the definition since 2026-10-23, work record §19.31, one owner:
    calibration_targets.yml `calibration.cohort_footing`) or 'active' (the footing of every fit before
    that date, kept so the archived rounds can be rescored). TITAN_COHORT_FOOTING overrides the file."""
    v = os.environ.get("TITAN_COHORT_FOOTING", "").strip().lower()
    if not v:
        try:
            v = str(load_yaml(CALIB_YAML)["calibration"].get("cohort_footing", "all")).strip().lower()
        except Exception:
            v = "all"
    if v not in ("active", "all"):
        raise ValueError(f"cohort footing must be 'active' or 'all', got {v!r}")
    return v


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
    # COHORT FOOTING (2026-10-23, work record §19.31). "active" (default, the footing every fit to
    # date used): the 13,704 F11-diagnosed members are the ACTIVE agents, scale = 13,704 / (0.55 N).
    # "all" (Jun's definition under test): the 13,704 are the WHOLE cohort, scale = 13,704 / N; the
    # MOUD and overdose denominators switch to all agents in parse_basic_report. Set
    # TITAN_COHORT_FOOTING=all to score on the new footing; nothing else changes.
    if cohort_footing() == "all":
        pop_scale = hiv["population_scale"]["ri_active_oud"] / pop_size
    else:
        pop_scale = hiv["population_scale"]["ri_active_oud"] / (active_frac * pop_size)

    scored_starts = hiv.get("ridoh_scored_window_starts")
    idu_scored, idu_unscored = split_scored_windows(ridoh["windows"], scored_starts)
    pwid_scored, pwid_unscored = split_scored_windows(
        all_route_windows(hiv["ridoh_pwid_dx_all_routes"]), scored_starts)

    return {
        "oud": calib["oud_baseline"],
        # The fatal check is read over the scoring window (2021-2022). With the
        # lethality index on, the model's deaths there sit at RIDOH's 2021-2022
        # level, not at the ~2019 level fatal_rate_annual states (2026-10-06).
        "od_fatal": (calib["overdose"]["fatal_rate_annual_t3"]
                     if str(calib["overdose"].get("lethality_index", "none")) != "none"
                     else calib["overdose"]["fatal_rate_annual"]),
        "od_nonfatal_inj": calib["overdose"]["nonfatal_rate_inj_annual"],
        "od_nonfatal_noninj": calib["overdose"]["nonfatal_rate_noninj_annual"],
        "moud_series": calib["moud"]["uptake_series_per_yr"],
        "moud_weights": calib["moud"].get("uptake_series_weights"),
        "inj_share": calib["injecting"]["injecting_pct"],
        # Scored and unscored RIDOH windows, split by first year (2026-10-07):
        # see split_scored_windows. The unscored ones are carried so
        # validate_candidates can report them as validation rows.
        "ridoh_windows": idu_scored,
        "ridoh_pwid_windows": pwid_scored,
        "ridoh_windows_unscored": idu_unscored,
        "ridoh_pwid_windows_unscored": pwid_unscored,
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

def multi_error(stats, targets, bias_correct=True):
    """
    Return 4 scalar errors for NSGA-II:
      (oud_rmse, ridoh_hiv_err, od_err, moud_traj_rmse)

    `bias_correct` subtracts the Monte Carlo sampling variance from the two
    squared-error objectives that are dominated by it (ridoh_hiv_err, od_err);
    see od_error() and ridoh_hiv_rmse(). False returns the raw scores, which
    objective() records alongside the corrected ones under `raw_<name>`.

    OD error folds in stratified targets (inj / non-inj nonfatal + fatal).
    MOUD error is a trajectory RMSE against mofan's APCD #16/#43 series.

    inj_share_err was added 2026-07-29 and DEMOTED 2026-08-17. It is now reported
    by validate_candidates.VALIDATION, not scored. Four reasons, in the order they
    were established:

      * It fails at the configured burn. Job 4963250 measured
        `inj_share_active_oud` at 0.262 against the 0.32 target, so the objective
        scores 0.00336 against a 0.0025 gate. The 0.363 recorded in step 5 as
        "already passing" was a burn-720 figure taken before
        set_population_window() existed.
      * It is scoring a transient, not a level. The share starts at 0.32, falls
        to 0.262 by burn 120, crosses the transition rates' implied steady state
        near burn 240, and is still climbing at 0.337 by 720. Which value the
        objective sees is a function of burn length and almost nothing else.
      * Step 2 removed its levers. Anchoring INJECT_START/INJECT_STOP from
        literature took out the two strongest parameters acting on it (-0.68 and
        +0.32 on job 4874631), leaving it at risk of becoming a second
        uncontrolled objective alongside od_err.
      * No Phase 7 intervention moves the injection share directly, so under the
        2026-08-12 rule adopted from upstream -- anything with data is asserted
        as an initial condition, only transitions an intervention moves stay
        free -- it belongs on the validation side.

    The target itself is unaffected: injecting_pct 0.32 rests on RIHRSS
    (PMC10872563), not on any simulator run, and remains the value
    demographics.*.drug_type.Inj.ppl is derived from.

    Reducing the objective count is also the point, not a side effect. With five
    objectives NSGA-II's front absorbs nearly everything and selection goes slack;
    see docs/calibration_log/phase_6_completion_plan_2026-08-14.md section 2.2 for
    the yield arithmetic. Everything that indexes trial.values positionally was
    updated with this change, and no study persisted before 2026-08-17 can be
    resumed against it.

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
        idu_by_run=stats.get("idu_dx_counts_by_run") if bias_correct else None,
        pwid_by_run=stats.get("pwid_dx_counts_by_run") if bias_correct else None,
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
    # Scaling each residual by its own target makes the components commensurate:
    # the value is the RMS proportional miss, so 0.10 means "within 10% on all
    # of them". Splitting fatal into its own objective would have been the other
    # fix and was rejected -- six objectives collapses NSGA-II's selection
    # pressure (see ridoh_hiv_rmse for the same reasoning).
    #
    # This changes the objective's SCALE by ~3 orders of magnitude. The
    # validate_candidates gate moved with it; no earlier study is comparable.
    #
    # Two components and noise-corrected since 2026-09-29 -- see od_error().
    od_err = od_error(stats, targets, bias_correct=bias_correct)

    moud_traj_rmse = _traj_rmse(
        stats.get("moud_uptake_series"),
        targets["moud_series"],
        targets.get("moud_weights"),
    )

    return (
        oud_rmse,
        ridoh_hiv_err,
        od_err,
        moud_traj_rmse,
    )

# =====================================================
# Optuna objective
# =====================================================
# ---------------------------------------------------------------------------
# Time trends fitted as ratios (2026-10-05). Rhode Island's MOUD coverage among
# people with active OUD rose 0.69 -> 0.81 over 2012-2022, and nothing in the
# model changed MOUD access over time, so every member of the 2026-10-02 refit
# was flat at ~0.68 while still passing the old 0.10 trajectory gate. A probe
# on trial 931 (job 7025851) found that acceptance alone caps coverage at
# ~0.77 and that reaching 0.81 needs retention to improve as well. Each
# `trend.<name>_ratio` searched parameter is the value at the end of the ramp
# divided by the value before it; the ramp is applied as one timeline_scaling
# entry per year from Jan 2016 to Jan 2022 (geometric steps), then held, so a
# Phase 7 run carries the 2022 level forward.
# ---------------------------------------------------------------------------
TREND_TARGETS = {
    "trend.moud_acceptance_ratio": "moud|acceptance",
    "trend.moud_discontinue_ratio": "moud|discontinue_per_month",
}
TREND_STEPS = (49, 61, 73, 85, 97, 109, 121)   # Jan 2016 ... Jan 2022 (t = 1 is Jan 2012)


def apply_trends(scenario, trends):
    """
    Write the timeline_scaling entries for fitted trend ratios into a scenario.

    args:
        scenario: the scenario dict (modified in place and returned)
        trends: {"trend.<name>_ratio": ratio}; a ratio of 1 adds nothing
    """
    for path, ratio in trends.items():
        target = TREND_TARGETS[path]
        if abs(float(ratio) - 1.0) < 1e-12:
            continue
        step = float(ratio) ** (1.0 / len(TREND_STEPS))
        tl = scenario.setdefault("timeline_scaling", {}).setdefault("timeline", {})
        name = path.split(".", 1)[1]
        for i, t in enumerate(TREND_STEPS):
            tl[f"{name}_{i}"] = {"parameter": target, "start_time": int(t),
                                 "stop_time": 10000, "scalar": round(step, 8)}
        scenario.setdefault("features", {})["timeline_scaling"] = True
    return scenario


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
    trend_params = {}  # trend.* ratios -> timeline entries via apply_trends()
    for p in calib["calibration"]["parameters_to_optimize"]:
        is_int = p.get("type") == "int"
        if is_int:
            val = trial.suggest_int(p["path"], int(p["lower"]), int(p["upper"]))
        else:
            val = trial.suggest_float(p["path"], p["lower"], p["upper"])
        if p["path"].startswith("calibration."):
            calib_block_params[p["path"]] = (int(val) if is_int else float(val))
            continue
        if p["path"] in TREND_TARGETS:
            trend_params[p["path"]] = float(val)
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
    # (As of 2026-08-13 the injection pair has swapped roles: `base` is pinned at
    # the published 0.0063 and `calibration.injection.act` is fitted instead. The
    # product is still the only identifiable thing, so this argument is unchanged
    # -- what changed is which half carries the measurement. See the block in
    # calibration_targets.yml.)
    #
    # `calibration.injection.act` was not fitted, so `base x acquisition` was
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
    apply_trends(scenario, trend_params)

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
    # RESTORED 2026-08-12, and burn_steps shortened 720 -> 120 alongside it.
    #
    # Everything above this line reasons from a premise upstream TITAN does not
    # share: that `hiv.init` is a modelling degree of freedom. It is a data slot
    # -- "Probability that an agent in this class initially has HIV" -- and its
    # sibling `hiv.dx.init` is the documented place to state how many of those
    # agents know their status. Zeroing the first and leaving the second at a
    # round 0.9 turned a two-number data statement into 720 burn steps, which is
    # 84.5% of all compute, and left burn length as an unfitted parameter with
    # more leverage on the HIV objective than most of the fitted ones.
    #
    # probe_seed_burn (2026-08-12) measured seed against burn on trial 290's
    # vector, 20 runs per arm. Drift is close/open across the scored window:
    #
    #   burn 120, seed 0                    PWID -21.8%   NonInj  +1.0%
    #   burn 120, seed 0.0034/0.115 RIDOH   PWID  +7.0%   NonInj +12.3%
    #   burn 120, seed 0.020/0.068          PWID  -1.9%   NonInj  -1.9%
    #   burn 240, seed 0.0034/0.115 RIDOH   PWID +12.2%   NonInj +13.4%
    #
    # The values below are the third row. They are NOT the RIDOH-derived seed,
    # and that is a deliberate reversal: deriving a seed by applying RIDOH's
    # 2019-2023 exposure shares (GBMSM 78% of sexual transmissions) to the
    # PREVALENT pool implies an MSM:het ratio near 34, and it performed worse --
    # concentrating the reservoir in the ~5% of agents who transmit most
    # efficiently drove the epidemic upward rather than holding it. The caution
    # already recorded above ("RIDOH publishes no PLWH breakdown by transmission
    # category ... would give a far larger number") was correct, and the
    # prevalent pool accumulated across four decades over which the mix shifted
    # sharply -- 37% of new AIDS diagnoses were injection-attributed in 2001
    # against 9% of HIV diagnoses in 2023.
    #
    # So NONINJ_HIV_MSM_TO_HET_RATIO = 3.4 is retained and the LEVEL is set where
    # the model's own long burn converges. This is an anchored initial condition,
    # not a fitted one: it is stated once here and no trial varies it.
    #
    # dx_init is RIDOH's measured 93.2% (2022 care continuum), replacing the
    # round 0.9. See the hiv_testing.base_prob_annual block in
    # calibration_targets.yml for why the cascade saturated regardless of seed.
    #
    # CAVEAT, stated because the next person will want it: the scored window is
    # not fully stationary at these values. The NonInj reservoir still grows
    # 10-30% across the window at trial 290's transmission level. That is a
    # transmission problem -- the injection route runs at 5-14% of incidence
    # against a literature per-act value (0.0063, citations #31) that sits ABOVE
    # the entire fitted box -- and it is what step 4 of the 2026-08-12 plan
    # addresses. Do not try to fix it by lengthening the burn; burn 240 drifted
    # more than burn 120, not less.
    set_hiv_initial_conditions(scenario)

    # OUD-side anchors: the baseline state split, the injection share it implies,
    # and the two injection transition rates (step 2 of the 2026-08-12 plan).
    #
    # AFTER the suggest loop on purpose -- these paths must not be fitted, and
    # writing them last means an accidental re-entry into parameters_to_optimize
    # would be overwritten rather than half-applied. assert_not_fitted turns that
    # case into a startup error so it cannot pass silently.
    set_oud_initial_conditions(scenario, calib)

    # Per-act sexual transmission, published values (step 3).
    set_per_act_transmission(scenario)

    # Population size and time window. Must come from the targets file, not the
    # base scenario -- see set_population_window for the burn_steps divergence
    # this closes.
    set_population_window(scenario, calib)

    # Parameters fixed from data (2026-09-29): stated in the targets file, not
    # suggested. After the suggest loop for the same reason the OUD anchors are.
    set_anchored_parameters(scenario, calib)
    # Drug-supply lethality index (2026-10-06, titan/calibration/lethality.py):
    # RIDOH's 2020-2022 rise in deaths without a rise in nonfatal overdoses.
    # After the anchors, because its scalars are computed from the anchored
    # (derived) fatal fraction.
    apply_lethality(scenario, projection_from_calib(calib))

    # Run TITAN once per population seed, and score the trial on the mean.
    #
    # ADDED 2026-09-01 (E1). Every trial of jobs 4743939, 4830724 and 4874631 was
    # run on the single population `model.seed.ppl: 12345` carried by the base
    # scenario, with the replicates varying `model.seed.run` only. Between-
    # population variance is not something replicates average away -- within a
    # study it is a fixed offset shared by every trial -- so the search stayed
    # internally consistent while the fitted values it produced were conditioned
    # on one network realization. That is why validate_candidates, which has
    # varied the population since Phase 4, has repeatedly been the step where a
    # candidate lost its margin.
    #
    # THE SEED LIST IS THE SAME FOR EVERY TRIAL, deliberately. The point is not to
    # average the offset away trial by trial; it is to keep the comparison BETWEEN
    # trials paired, so that a candidate whose margin exists only on population
    # 12345 cannot win the selection step. A per-trial random population would
    # remove the bias and replace it with noise of the same size.
    #
    # Cost is unchanged from the single-population design: A1 measured that only
    # the number of populations `p` and the total simulation count `B` matter, not
    # how the budget splits between them, so the 8 x 5 read from the targets file
    # buys 8 populations at the same 40 runs per trial that 1 x 40 spent on one.
    # See docs/calibration_log/phase_6_noise_floor_2026-08-17.md section 3.
    seeds = population_seeds(calib)
    n_reps = reps_per_seed(calib)

    # `trial_<n>.yml` is still written, with the first population applied, because
    # six downstream readers address a trial by that exact name -- validate_candidates
    # (F1), probe_discrimination, probe_noninj_seed, probe_noise_floor and
    # probe_od_reachability all reconstruct a trial's fitted vector from it. They
    # read the fitted values and re-anchor and re-seed the scenario themselves, so
    # which population it carries does not matter to them; that the file exists
    # does. The per-population copies below differ from it only in `model.seed.ppl`.
    set_population_seed(scenario, seeds[0])
    write_yaml(OUTPUT_DIR / f"trial_{trial.number}.yml", scenario)

    per_seed_errs = []
    report_paths = []
    for seed in seeds:
        set_population_seed(scenario, seed)

        # One scenario file and one output directory per population, so a failed
        # trial can be reproduced exactly and the per-seed basicReport.txt files
        # are not overwritten by the next seed in the loop.
        tag = f"trial_{trial.number}_pop{seed}"
        trial_yaml = OUTPUT_DIR / f"{tag}.yml"
        write_yaml(trial_yaml, scenario)
        outdir = OUTPUT_DIR / tag
        outdir.mkdir(exist_ok=True)

        stats = run_titan(trial_yaml, outdir, n_reps=n_reps)
        record_trial_diagnostics(trial, stats, targets, prefix=f"seed{seed}_")
        per_seed_errs.append(multi_error(stats, targets))
        report_paths.append(outdir / "basicReport.txt")

    # THE TRIAL IS SCORED ON THE POOLED REPORT, NOT ON THE MEAN OF THE PER-
    # POPULATION SCORES. Measured 2026-09-01 by analysis/probe_split_bias.py, at
    # `t290_s0.8` on these eight populations:
    #
    #   replicates per population      5       10       20       40
    #   od_err (gate 0.01)        0.0201   0.0149   0.0061   0.0037
    #   oud_rmse (gate 0.05)      0.0192   0.0198   0.0207   0.0204
    #
    # `od_err` is a mean of SQUARED proportional errors, and E[(x+e)^2] = x^2 +
    # var(e): sampling noise does not average out of a squared objective, it adds.
    # Scoring each population separately and averaging the scores averages that
    # bias IN, so a configuration truly at ~0.003 reads 0.020 at five replicates
    # and fails a gate it comfortably satisfies. The two well-measured RMSE
    # objectives move by 0.94x and 1.00x over the same sweep, which is what makes
    # the mechanism identifiable rather than a story about the configuration.
    #
    # Pooling first restores the precision the gates were priced at -- every rate
    # rests on all 40 runs, exactly as it did under the 1 x 40 design -- while the
    # trial is still scored across eight networks. The per-population scores stay
    # in the journal as diagnostics; they are no longer what the search selects on.
    pooled_path = OUTPUT_DIR / f"trial_{trial.number}_pooled_basicReport.txt"
    pooled = concat_reports(report_paths, pooled_path)
    errs = tuple(float(e) for e in multi_error(pooled, targets))

    record_trial_diagnostics(trial, pooled, targets, errs=errs)

    # The uncorrected scores alongside the corrected ones, so the size of the
    # noise correction is recoverable per trial. A corrected score far below its
    # raw one is a trial whose pass rests on the correction, and the 8 x 40
    # validation stage is where that gets settled.
    raw = multi_error(pooled, targets, bias_correct=False)
    for name, value in zip(OBJECTIVE_NAMES, raw):
        trial.set_user_attr(f"raw_{name}", float(value))

    # The spread the pooled score hides. Recorded per objective so that a trial
    # whose populations disagree is visible without re-reading eight reports.
    for i, name in enumerate(OBJECTIVE_NAMES):
        column = [e[i] for e in per_seed_errs]
        trial.set_user_attr(f"spread_{name}", float(np.std(column, ddof=1)))
    trial.set_user_attr("population_seeds", [int(s) for s in seeds])
    trial.set_user_attr("reps_per_seed", n_reps)

    # Feasibility for D2's constrained sampler. Recorded rather than returned:
    # Optuna reads it back through constraints_func() after the trial completes.
    trial.set_user_attr("constraints", trial_constraints(errs))
    return errs

# Everything worth having in the journal when a run is being diagnosed months
# later, as (stat key, user_attr name). Objective VALUES are stored by optuna
# already; what was missing is everything that explains them.
#
# Kept as a table rather than inline so that adding a diagnostic is one line and
# so the set is greppable from the analysis side.
TRIAL_DIAGNOSTICS = (
    # The three rates od_err averages over. od_err is the mean of three squared
    # PROPORTIONAL errors, so the aggregate cannot distinguish "one component
    # badly off" from "all three modestly off" -- and those have different fixes.
    ("fatal_od_rate", "fatal_od_rate"),
    ("nonfatal_od_rate_inj", "nonfatal_od_rate_inj"),
    ("nonfatal_od_rate_noninj", "nonfatal_od_rate_noninj"),
    # Demoted from the objective vector 2026-08-17; still the quantity the whole
    # intervention analysis is about.
    ("inj_share_active_oud", "inj_share_active_oud"),
    # Stationarity of the scored window, as an OLS slope since 2026-08-17.
    ("hiv_prev_drift_pwid", "hiv_prev_drift_pwid"),
    ("hiv_prev_drift_noninj", "hiv_prev_drift_noninj"),
    ("hiv_prev_slope_pwid", "hiv_prev_slope_pwid"),
    ("hiv_prev_slope_noninj", "hiv_prev_slope_noninj"),
    # Level, not just movement: a flat trajectory at the wrong level and a flat
    # trajectory at the right one are indistinguishable without this.
    ("hiv_prevalence_pwid", "hiv_prevalence_pwid"),
    ("hiv_prev_mean_pwid", "hiv_prev_mean_pwid"),
    # The null-transmission screen reads this, and a trial can score well on the
    # HIV trajectory with 0-1 stochastic infections.
    ("hiv_new_pooled", "hiv_new_pooled"),
    # Cascade position, which decides how much of the transmission level is
    # actually infectious.
    ("dx_among_hiv", "dx_among_hiv"),
    ("art_among_dx", "art_among_dx"),
    ("suppressed_among_dx", "suppressed_among_dx"),
    ("incar_share", "incar_share"),
    # OUD compartment sizes, so an oud_rmse miss can be attributed without a rerun.
    ("oud_active", "oud_active"),
    ("oud_remission", "oud_remission"),
    ("survivor_any_moud_30d_share", "survivor_any_moud_30d_share"),
    ("survivor_moud_at_index_share", "survivor_moud_at_index_share"),
    ("survivor_restart_1step_share", "survivor_restart_1step_share"),
    ("oud_none", "oud_none"),
    ("moud_uptake", "moud_uptake"),
    # HIV tests per PWID person-year, checked against NHBS-PWID's past-year
    # testing share (validation_targets.hiv_testing). Added 2026-09-29 with the
    # pinning of hiv_testing.base_prob_annual, so the pin is checkable.
    ("hiv_tests_per_pwid_yr", "hiv_tests_per_pwid_yr"),
    # Non-injecting MSM HIV prevalence, the compartment the 2026-09-29 grids
    # found carrying the all-route target at 0.4-0.8. Checked against 0.15 +/-
    # 0.10; added to the journal 2026-09-30 after the pilot could not report it.
    ("hiv_prevalence_noninj_msm", "hiv_prevalence_noninj_msm"),
    # Sampling design in force for this trial, so a journal read months later does
    # not have to reconstruct it from the scenario file.
    ("n_reps", "n_reps"),
)


def record_trial_diagnostics(trial, stats, targets, prefix="", errs=None):
    """Write the per-trial diagnostics into the Optuna journal.

    ADDED 2026-08-17. Until now the only user_attr stored was `hiv_new_pooled`, so
    nothing about WHY a trial scored as it did survived the run. That has already
    cost a diagnosis: on 2026-08-11 the decisive question about od_err -- which of
    its three components was missing, and by how much -- could not be answered from
    job 4874631's journal at all, and the analysis was abandoned rather than
    re-run. The objective components are the difference between the next
    post-mortem taking an afternoon and taking another probe job.

    `prefix` namespaces a per-population-seed record (e.g. "seed12345_"), so that
    when a trial is scored across several network realizations the spread is
    recoverable and not just the mean. That spread is the quantity Stage A1
    measures and the one the gates have to be justified against.

    `errs` supplies the objective vector instead of recomputing it from `stats`.
    The trial-level record passes the vector the search was actually given, so a
    journal value can never disagree with the trial's own optimisation history --
    the sort of quiet inconsistency the journal exists to rule out.
    """
    for stat_key, attr in TRIAL_DIAGNOSTICS:
        value = stats.get(stat_key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        trial.set_user_attr(f"{prefix}{attr}", numeric)

    # The objective components themselves, named, so that a reader does not have
    # to know the positional order the vector had on the day the study ran. That
    # order has changed four times.
    if errs is None:
        errs = multi_error(stats, targets)
    for name, value in zip(OBJECTIVE_NAMES, errs):
        trial.set_user_attr(f"{prefix}err_{name}", float(value))

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

def constraint_violation(trial):
    """Total amount by which a trial misses its gates, 0.0 if it misses none."""
    return sum(max(0.0, c) for c in constraints_func(trial))


def unconstrained_pareto_front(trials):
    """The non-dominated set of `trials`, ignoring feasibility.

    ADDED 2026-09-01, and it exists for one case: `study.best_trials` returns only
    FEASIBLE non-dominated trials once a constraints_func is attached, so a run in
    which nothing clears all four gates returns an EMPTY front. Every consumer
    downstream -- select_canonical, the null-transmission diagnostic, the front CSV
    -- indexes into that list, and the first smoke run of the constrained sampler
    died on `pareto[0]` after a clean two-trial search.

    An all-infeasible run is not an error. It is the most likely outcome of the
    first constrained refit and it is exactly the result that has to be reported
    rather than crashed on: the front computed without the feasibility filter is
    what the study would have returned before D2, so falling back to it keeps the
    canonical pick and the CSV intact and leaves the constrained front as the thing
    that was empty.

    O(n^2) in the number of completed trials, which at the 1000-trial budget is a
    few seconds once, after the search.
    """
    front = []
    for candidate in trials:
        dominated = False
        for other in trials:
            if other is candidate:
                continue
            if all(o <= c for o, c in zip(other.values, candidate.values)) and any(
                o < c for o, c in zip(other.values, candidate.values)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def select_canonical(pareto, min_events=MIN_HIV_TRANSMISSION_EVENTS, hiv_null=None):
    """
    Pick the canonical Pareto-front member to save as calibrated_scenario_best.yml.

    NOT pareto[0] -- optuna's study.best_trials order is arbitrary, not ranked by
    quality. Ranks by a composite of normalized errors across all objectives,
    restricted to trials with >= min_events pooled HIV infections (see
    min_transmission_events for how that floor is derived). Falls back to the full
    front, with a warning, if no trial clears the transmission-activity bar.

    ELIGIBILITY IS TWO-SIDED as of 2026-08-11. min_events screens out trials
    running cold; `hiv_null` screens out trials running hot. A trial whose
    ridoh_hiv_err is worse than the zero-transmission null (hiv_null_value) scores
    worse than simulating no HIV at all, so it is not a calibration candidate on
    any reading, and -- more importantly -- it must not set the normalizer.

    Why the second screen was needed. On job 4830724 the front's ridoh_hiv_err
    spanned 0.36 to 204, so the COMPOSITE_CLIP_QUANTILE anchor landed at 46.0 and
    the HIV span became 45.7. Every trial in the feasible region then differed by
    less than 0.08 of normalized HIV error while od_err alone contributed 0.25-0.41
    -- HIV was effectively muted, and the composite picked a trial at 4.02, which
    is 84% of the way to the null value of 4.78 and 11x worse than the best member
    available. Filtering to better-than-null first collapses the span to 2.8 and
    restores HIV to 0.53 of the winner's composite. This is the same failure the
    2026-08-04 clip-quantile change was meant to prevent; the quantile alone is not
    enough when an objective's front distribution is this heavy-tailed.

    The bar is derived, not chosen: "better than a null model" needs no threshold
    invented for it. It is deliberately weak -- it removes trials that cannot be
    right rather than selecting trials that are.

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

    # Hot-side screen. Applied after the cold-side one and reported separately, so
    # a run that loses most of its front here is visible rather than silent.
    if hiv_null is not None:
        better_than_null = [t for t in pool if t.values[1] < hiv_null]
        if not better_than_null:
            print(
                f"WARNING: no eligible Pareto trial beat the zero-transmission null "
                f"(ridoh_hiv_err < {hiv_null:.4f}) -- keeping the pool unfiltered. The "
                "canonical pick is not a calibrated model; the run needs revisiting."
            )
        else:
            print(
                f"Null-transmission screen: {len(better_than_null)}/{len(pool)} eligible "
                f"trials beat ridoh_hiv_err = {hiv_null:.4f}."
            )
            pool = better_than_null

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

    # Before any compute: a data-anchored path that has drifted back into the
    # search would waste the whole run, since the anchor overwrites it per trial.
    calib = load_yaml(CALIB_YAML)
    assert_not_fitted(calib)

    # The sampling design, printed because it is now plural and because a journal
    # read months later should not have to infer it from the trial directories.
    seeds = population_seeds(calib)
    print(
        f"Sampling design: {len(seeds)} populations x {reps_per_seed(calib)} "
        f"replicates = {len(seeds) * reps_per_seed(calib)} runs per trial; "
        f"population seeds {seeds}"
    )

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
        # TITAN_CALIB_JOURNAL (2026-10-01): a fixed journal path so a refit can be
        # split into several jobs under a 24 h wall. Each chunk opens the same
        # journal with load_if_exists=True, enqueues the warm starts with
        # skip_if_exists=True, and runs TITAN_CALIB_TRIALS NEW trials; NSGA-II
        # reads its generations back from the journal, so the search continues
        # rather than restarts. scripts/submit_refit_chunks.sh chains the jobs.
        # ONE STUDY AT A TIME against OUTPUT_DIR: trial directories are named
        # trial_<number>_pop<seed> and every study numbers from 0, while
        # run_titan clears its output directory on start -- two studies running
        # concurrently delete each other's reports mid-run (pilot 6937917 lost
        # its four warm starts to a 2-trial resume test on 2026-10-01).
        journal_path = os.environ.get("TITAN_CALIB_JOURNAL") or f"{OUTPUT_DIR}/study_{job}.journal"
        resumed = bool(os.environ.get("TITAN_CALIB_JOURNAL")) and os.path.exists(journal_path)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(journal_path)
        )
        print(f"Study storage: JournalStorage at {journal_path}" + (" (resuming an existing journal)" if resumed else " (new journal)"))

    study = optuna.create_study(
        # 4 objectives since 2026-08-17, when inj_share_err was demoted to a
        # validation target. No earlier study can be resumed against this: the
        # vector has changed length or meaning four times now -- 6 objectives from
        # 2026-07-29 (inj_share_err appended), slot 1 redefined from an 11-point
        # annual RMSE to a 3-period one on 2026-08-03, 5 objectives from
        # 2026-08-04 (hiv_prev_err dropped, slot 1 became ridoh_idu_err), 4 today.
        #
        # The count is itself a lever. Non-dominated sorting discriminates less as
        # the objective count rises, and at five the front was absorbing nearly
        # every trial; see phase_6_completion_plan_2026-08-14.md section 2.2.
        directions=["minimize"] * 4,
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
            # D2, 2026-09-01. Without this the sampler ranks trials by Pareto
            # domination alone and has no notion of the gates the run is trying to
            # clear; with it, a feasible trial always sorts ahead of an infeasible
            # one and infeasible trials are ordered by total violation. See
            # trial_constraints() for the sign convention and completion plan
            # section 2.2 for the yield arithmetic this is meant to fix.
            constraints_func=constraints_func,
        ),
    )

    # D4's designed starting points, ahead of anything the sampler draws.
    #
    # skip_if_exists keeps a relaunch against the same journal from re-running
    # them: JournalStorage makes a timed-out study resumable, and a warm start
    # that re-enqueued itself on every resume would spend the budget on points
    # already evaluated.
    for label, params in warm_start_trials(calib):
        study.enqueue_trial(
            params, user_attrs={"warm_start": label}, skip_if_exists=True
        )
        print(f"Enqueued warm start: {label}")

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_workers,
        show_progress_bar=True,
    )

    # Persist the Pareto front
    #
    # With D2's constraints attached, study.best_trials is the front among trials
    # that satisfy every gate. Report the feasibility count alongside it: it is the
    # number the constrained search exists to move, and it is not recoverable from
    # the front's size alone.
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]
    feasible = [t for t in completed if constraint_violation(t) <= 0.0]
    print(
        f"Feasible trials (all {len(OBJECTIVE_GATES)} gates): {len(feasible)}/"
        f"{len(completed)} completed"
    )

    pareto = study.best_trials
    if not pareto:
        # Not an error -- see unconstrained_pareto_front(). The pick that follows is
        # the best available compromise, not a calibrated model, and every screen in
        # select_canonical still applies to it.
        pareto = unconstrained_pareto_front(completed)
        if pareto:
            closest = min(constraint_violation(t) for t in pareto)
            print(
                "WARNING: no trial satisfied all gates, so the constrained Pareto "
                f"front is empty. Falling back to the unconstrained front "
                f"({len(pareto)} trials); the smallest total gate violation on it "
                f"is {closest:.4f}. The canonical pick below is the best available "
                "compromise, not a calibrated model."
            )
        else:
            print("WARNING: no completed trials to select from.")
    print(f"Pareto front: {len(pareto)} parameter sets")

    if not pareto:
        print("No completed trials; nothing to select or write.")
        return

    # Select and write the canonical calibrated scenario (see select_canonical)
    int_params = {
        p["path"] for p in calib["calibration"]["parameters_to_optimize"]
        if p.get("type") == "int"
    }
    min_events = min_transmission_events(calib)
    null_value = hiv_null_value(calib)
    print(f"Transmission-activity screen: >= {min_events} pooled PWID infections")
    best, best_composite, used_transmission_filter = select_canonical(
        pareto, min_events, hiv_null=null_value
    )
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
    # The value itself now comes from hiv_null_value(), computed above and shared
    # with select_canonical's hot-side eligibility screen. It is reported here as
    # an exact-match count -- how many front members transmit nothing at all --
    # which is a stricter question than the screen's "worse than null" and stays
    # worth printing on its own.
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
    trends_best = {}
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
        if k in TREND_TARGETS:
            trends_best[k] = float(v)
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
    apply_trends(calibrated, trends_best)

    # The NonInj HIV seed lives in `demographics`, not in a suggest path, so it
    # has to be re-applied here the same way objective() applied it. Without
    # this the canonical scenario would run with a reservoir the fit never saw --
    # a different model from the one that won.
    #
    # Zero since 2026-08-07: the seed left the search and the burn-in builds the
    # reservoir. Reading best.params for it would KeyError on any study run after
    # that date, the same failure mode `calibration.acquisition` caused above.
    set_hiv_initial_conditions(calibrated)

    # Same reason for the OUD anchors: they are not suggest paths, so the
    # canonical scenario has to have them re-applied or it is not the model the
    # winning trial ran.
    set_oud_initial_conditions(calibrated, calib)
    set_per_act_transmission(calibrated)
    set_population_window(calibrated, calib)
    set_anchored_parameters(calibrated, calib)
    apply_lethality(calibrated, projection_from_calib(calib))

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
