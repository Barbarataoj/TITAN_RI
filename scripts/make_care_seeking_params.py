#!/usr/bin/env python
"""
Derive the care-seeking layer's parameters and write them into the scenario source.

WHY THIS EXISTS
---------------
`titan/params/care_seeking.yml` is a schema: every setting value defaults to 0 and
only `primary_care` is declared. The Phase 7 interventions act through this layer
(one paper per setting), so it has to be parameterized before the refit that
turns `features.care_seeking` on. The values come from three tiers, in the order
Jun set on 2026-10-01: Rhode Island data first (the APCD delivery under
`docs/apcd_incoming/ABM_2/`, Rhode Island studies), other literature second,
and a stated assumption only where neither exists. Every number this script
writes carries a `source` tag with that tier, and the arithmetic is printed so a
reviewer can follow each derivation from the CSV to the YAML.

WHAT IT DERIVES
---------------
Utilization and provider structure (tier 1, APCD 2022 unless stated):
  visit_rate_annual          visit-days / OUD cohort (12,774), job1_visit_counts
  visit_heterogeneity_shape  negative-binomial shape, registry section 7.2
  visit_mult_inj, _moud      registry section 7.2 (job1_strata)
  visit_mult_hiv_positive    HIV+ / HIV- visit rate among users, job1_strata 2022
  n_providers_ri             job4_q1, 2022
  panel_skew                 registry section 7.2 (job4_q2)
  attrition_annual           registry section 7.2 (job4_q4; an upper bound)

Willingness (`p_willing_*`): the share of providers with 50 or more OUD patients
who ever delivered the service (job4_q5, the 50+ bin). That bin is where the
patients are, and the model draws a provider's willingness independently of
panel size, so the patient-weighted share is the one that reproduces the
observed product. ED buprenorphine is the exception: the claims share (0.99)
counts every NPI tied to the visit, and Rhode Island measured 7% of ED
clinicians prescribing at discharge in 2019 (Myles 2023, PMID 38165180), so the
Rhode Island study wins (tier 1 study over tier 1 claims with a known
attribution problem; registry 7.2 note 4).

Intensity (`intensity_*`, the per-visit offer rate among willing providers):
  hiv_test, prep   solved so that the model reproduces the 2022 receipt share
                   among the setting's HIV-negative users (job3a), given the
                   willingness above, the 2022 visit distribution among users
                   (job1_visit_distribution) and the patient acceptance below:
                       P = W * sum_n p(n) * [1 - (1 - I * a)^n]
                   HIV-test receipt outside the ED is first corrected for the
                   reference-laboratory undercount (next paragraph).
  moud             ED 0.15 (Rhode Island, registry 7.6); SUD treatment 1.0
                   (MOUD is what the visit is for); primary care and mental
                   health 0.75 (an assumption inside registry 7.6's 0.47-1.0,
                   varied in sensitivity; 0.47 capped MOUD coverage at 0.73 in
                   the 2026-10-01 smoke run)
  counseling       0 everywhere. Nothing in the engine consumes counseling.

HIV-test receipt correction (2026-10-02). job3a counts a test only when it is
billed on the same claim as the visit. The ED bills its own laboratory work and
its per-visit rate matches the published figure (0.75% vs 0.8%), but the other
three settings send specimens to reference laboratories that bill separately,
so their same-claim shares are a floor (REVIEW_2026-09-29.md section 2; the
2026-10-01 derivation used the floor as-is and the fitted `nonclinical_annual`
absorbed the difference, which left every clinical-testing arm acting on a base
10-100x too small). The correction keeps the ED share and scales the three
other settings' receipt shares by one factor so that the cohort's expected
number of people tested equals the whole-extract per-member rate of the
screening codes job3a counts (job3e: CPT 87389, 87806, 86703 and the G-code
family, all Rhode Island APCD members of each payer in 2022) weighted by the
cohort's insurance mix (job1 strata, primary-care users as the cohort proxy),
minus the ED's same-claim count. Missing tests are allocated across the three
settings in proportion to their same-claim counts (an assumption; the
primary-care result, 1.3% per visit, sits near the published 1.09%, a sanity
bound rather than a validation). The direction of the remaining bias is not
known: job3e's counting unit (members vs claim lines) is unconfirmed and
repeat tests would push the factor up, while children and partial-year members
in the whole extract push it down. Sensitivity: halve and double it. The
arm-relevant quantity, clinical test events per injecting person-year off
MOUD, rises from 0.078 (ED alone, in effect) to about 0.19; the fitted
`nonclinical_annual` still supplies most of the NHBS 0.43-0.80 band.

Visit composition (2026-10-02 fix). `visit_rate_annual` is the REFERENCE
group's rate (off MOUD, not injecting, HIV-negative): cohort visit-days per
person divided by E[multiplier] over the cohort's composition. The
multipliers are cohort-level ratios (registry 7.2), so the expectation must be
taken over cohort shares (job3b: on MOUD 0.772, injecting 0.254 in 2022), not
over the setting's users as the 2026-10-01 version did (users are enriched for
both, which understated the ED and SUD bases by about 10%, roughly half of the
smoke run's undershoot). Two approximations stay: the three strata are treated
as independent (no cross-tab exists; the engine stacks the multipliers the same
way), and `visit_mult_hiv_positive` is a ratio among users applied with a
cohort share (effect under 2%). The HIV+ cohort share has no cohort-level
count; the largest single-setting HIV+ user count over the cohort (primary
care, 0.020) is used as a lower bound.

Emergency visit after a nonfatal overdose (`visit_prob_after_overdose`, 2026-10-02):
the nonfatal overdose targets are APCD claims (T3), so every modelled nonfatal
overdose is by construction an ED- or hospital-treated event. The ED setting
therefore gets one extra visit in the step after each nonfatal overdose with
probability 1.0 (other settings 0). Before this the engine had no link between
an overdose and an ED encounter, so post-overdose linkage to MOUD was
unmodelled with the care layer on.

Mental health = opioid treatment programs. This APCD extract codes methadone
clinics as mental health (registry 7.2 note 3; REVIEW_2026-09-29.md section 2),
which is why that setting carries 13.0 reference-group visits per person-year
(11.6 cohort mean) and a buprenorphine willingness near 1. A "mental health" intervention arm acts on
OTP dosing encounters unless the setting is split, which the claims cannot do.

Patient acceptance (scalars outside the setting table):
  hiv_testing.acceptance  0.55: Rhode Island ED patients who use drugs accepted
                          an offered rapid HIV/HCV test 64.5% (Merchant 2014,
                          PMID 25125271) and 45% (Merchant 2015, PMID 26036465)
  prep.acceptance         0.60: prescribed once discussed (registry 7.6)
  moud.acceptance         FITTED. Replaces `moud.reinit_prob` in the search:
                          with care-seeking on, MOUD starts only through
                          encounters, so this is the level lever.
  hiv_testing.nonclinical_annual  PINNED 0.38 (2026-10-07; fitted 10-01 to 10-07).
                          Testing outside clinical settings (syringe services,
                          community), an annual probability applied through a
                          hazard, so it supplies -ln(1 - 0.38) = 0.48 tests per
                          PWID-year; clinic encounters add about 0.14, total
                          0.62 against NHBS-PWID's 0.60 (B1 lever run 7098444:
                          0.61-0.62). Nothing scored identified it (the
                          diagnosed share saturates), hence the pin.

USAGE
-----
    python scripts/make_care_seeking_params.py            # print the arithmetic
    python scripts/make_care_seeking_params.py --write    # also write the source
    python scripts/make_care_seeking_params.py --check    # fail if the source is stale

After --write, regenerate the scenario: python scripts/make_scenario.py
"""

import argparse
import io
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
APCD = REPO / "docs" / "apcd_incoming" / "ABM_2"
SOURCE = REPO / "titan" / "calibration" / "scenario_source.yml"

BEGIN = "  # >>> care_seeking: generated by scripts/make_care_seeking_params.py, do not edit by hand >>>"
END = "  # <<< care_seeking <<<"

COHORT_2022 = 12774  # people with OUD in the 2022 APCD cohort (REVIEW_2026-09-29)
YEAR = 2022

# Setting names as the model will know them (also the APCD names, except ED's
# key in job4_q5/q5b).
SETTINGS = ["ED", "primary_care", "sud_treatment", "mental_health"]
Q5_KEY = {"ED": "ed", "primary_care": "pcp", "sud_treatment": "sud", "mental_health": "mh"}

# Registry section 7.2 constants derived by Mofan / the 2026-09-29 review from
# the same CSVs (negative-binomial fits, gamma shapes, stratified rates with the
# user share folded in). Restated here rather than re-derived so that the YAML
# and the registry cannot disagree.
REGISTRY_7_2 = {
    "visit_heterogeneity_shape": {"ED": 0.33, "primary_care": 0.62, "sud_treatment": 0.045, "mental_health": 0.20},
    "visit_mult_inj": {"ED": 2.33, "primary_care": 1.23, "sud_treatment": 1.85, "mental_health": 1.70},
    "visit_mult_moud": {"ED": 0.55, "primary_care": 0.90, "sud_treatment": 1.81, "mental_health": 0.68},
    "panel_skew": {"ED": 0.16, "primary_care": 0.47, "sud_treatment": 0.15, "mental_health": 0.19},
    "attrition_annual": {"ED": 0.50, "primary_care": 0.30, "sud_treatment": 0.60, "mental_health": 0.39},
}

# Tier 1 (Rhode Island study) and tier 2 (literature) constants, registry 7.6.
ED_WILLING_MOUD_RI = 0.07          # Myles 2023, PMID 38165180: ED clinicians prescribing at discharge, RI 2019
INTENSITY_MOUD = {"ED": 0.15,      # of OUD encounters where willing, RI (registry 7.6)
                  "sud_treatment": 1.0,   # assumption: MOUD is the purpose of the visit
                  "primary_care": 0.75,   # assumption: between registry 7.6's 0.47 (half of waivered physicians treat anyone) and 1.0; smoke run 2026-10-01 showed 0.47 caps coverage at 0.73 even at acceptance 1
                  "mental_health": 0.75}  # same assumption as primary care
INTENSITY_MOUD_SOURCE = {"ED": "RI literature (registry 7.6, PMID 38165180)",
                         "sud_treatment": "assumption: MOUD is the purpose of the visit",
                         "primary_care": "assumption 0.75 (range 0.47-1.0; registry 7.6 gives 0.47 of waivered physicians treating anyone; 0.47 caps MOUD coverage at 0.73 in the smoke run)",
                         "mental_health": "assumption, same as primary care"}
ED_VISIT_AFTER_OD = 1.0            # assumption: APCD-defined nonfatal overdoses are ED/hospital-treated by construction (docstring)
TEST_CODES = ["cpt_87389", "cpt_87806", "cpt_86703", "G0475", "G0432", "G0433", "G0435"]  # the screening codes job3a's lab_test counts (naive set, docs/apcd_hiv_code_reference_2026-08-17.md) as they appear in job3e; REVIEW_2026-09-29 section 2 quoted 87389 alone (factor 10.8 instead of 11.8)
INSURANCE_PROXY_SETTING = "primary_care"  # the most inclusive setting (9,663 of 12,774 cohort members in 2022) stands in for the cohort's insurance mix
ACCEPT_HIV_TEST = 0.55   # RI ED drug-using patients: 64.5% (PMID 25125271) and 45% (PMID 26036465) accepted an offered test
ACCEPT_PREP = 0.40       # starts once discussed. Clinical review 2026-10-07 (Phil, RI clinician: "25-50%, a lot decline"); was 0.60 (registry 7.6); sensitivity 0.25-0.60
# Share of each setting's visits made by agents with a PrEP indication under
# prep.target_model [pwid, cdc_msm, cdc_women] (clinical review 2026-10-07).
# With eligibility restricted, an offer to someone without an indication starts
# nothing, so the per-visit rate is solved against 2022 receipt divided by this
# share. MEASURED, not assumed: trial 26 (calibrated_scenario_best.yml), job
# 7165864, 2 seeds x 10 reps, 2022 (steps 121-132): prep_indicated / HIV-negative
# agents per race x sex x drug stratum, weighted by the setting's encounters.
# Cohort share 0.357 (all who inject; 26 % of those who do not, through a
# partner who injects, is MSM or is diagnosed). Re-measure if the rule changes.
PREP_INDICATED_SHARE = {"ED": 0.473, "primary_care": 0.384, "sud_treatment": 0.375, "mental_health": 0.430}
# Test-and-offer bundle (2026-10-09, Jun / clinical review): PrEP raised at the
# visit where an HIV test comes back negative, for a patient with an indication.
#   PREP_BUNDLE_PROB_WILLING   P(PrEP raised | negative test, provider prescribes PrEP).
#       No direct US primary-care measure; bounds: 1.1 % of eligible patients
#       receive PrEP in a year (PMID 37795848), 8 % at one academic centre
#       (PMC9580717), 15-25 % of at-risk patients have it discussed (registry
#       7.6). 0.5 (0.3-0.8) among prescribers.
#   PREP_SAME_DAY              share prescribed at the testing visit; the rest deferred
#       to a result visit. Same-day programmes feasible (PMID 33128874, 31341933);
#       no measured share among primary-care prescribers: 0.5 (0.3-0.8),
#       ASSUMPTION consistent with the clinical review ("some clinicians prescribe
#       right away and only call if positive"); ED 0.2.
#   PREP_DEFERRED_RETURN       P(returns and starts | deferred): 43 % attended a first
#       PrEP visit after a co-located referral (81 % warm handoff, 36 % EHR message,
#       27 % card; PMID 34448603), 45 % after partner-services referral (PMID
#       35797587), 51 % returned for results at rural syringe programmes (PMID
#       39886694). 0.35 (0.25-0.45); a navigator's warm handoff 0.8.
#   PREP_REFERRAL_PROB         P(referral to PrEP care raised | negative test, indicated,
#       provider does not prescribe) (2026-10-12). Among academic general internists
#       34.9 % had ever prescribed or referred (Blackstock 2017, PMID 27778215) while
#       17 % of primary-care providers had ever prescribed (Petroll 2017, PMID
#       27885552): ever-referred-only is roughly the size of ever-prescribed, i.e. about
#       one fifth of non-prescribers; if they refer at the prescribers' bundle rate
#       0.5, the non-prescriber average is 0.5 x 0.2 = 0.10; 15 % of internists prefer
#       referring out (Edelman 2019, PMID 31890474). 0.10 (0.05-0.30), DERIVED from
#       those surveys (Jun 2026-10-12: use the evidence-supported value); the EHR arm's
#       prompt lifts it to 0.15 (CDS x1.5, within the x1.3-2.6 of PMID 38412047).
#   PREP_REFERRAL_RETURN       P(attends elsewhere and starts | agreed to a referral):
#       attendance 27 % card / 36 % EHR message / 81 % warm handoff, 43 % filled after
#       a co-located referral (PMID 34448603), fill given attendance ~0.72; off-site
#       0.27-0.36 x 0.72 = 0.19-0.26 -> 0.25. ED 0.20 (0.13-0.59): 59 % attended when
#       booked in the ED (n = 22), 2 of 15 in Baltimore, 21 % any follow-up within 90
#       days (Jackson 2024, PMID 38127296). The ED's deferred start is also a referral
#       (no result visit), so its deferred return equals its referral return.
#   PREP_TEST_LINKED_PER_INDEPENDENT  B / I0 per setting: test-linked (bundled +
#       referred) 2022 starts per unit of the starts the independent per-visit offer
#       makes at its full (un-reduced) rate. The independent rate is solved against
#       p_prep x (1 - B/I0) so 2022 receipt matches the APCD. MEASURED per setting
#       (bundle only, referral off): 11 members x 8 seeds x 40 reps at the full
#       independent rate with the bundle on (chain_fix_members_2026-10-12, jobs
#       7326917-20, read-out 7326939; per-setting start stats of the 2026-10-12
#       engine): ED 0.121 (SE 0.013, members 0.056-0.212), primary care 0.231 (0.013,
#       0.190-0.295), SUD treatment 0.401 (0.065, 0.083-0.778), mental health 0.025
#       (0.005, 0.006-0.075); statewide 0.144 (the earlier statewide 0.158, job
#       7240112, was measured at the reduced rate). Attribution closes (independent
#       by setting 1.477 = prep_new - bundled); the pool effect is -0.8 % (1.477 vs
#       1.489 with the bundle off). RE-MEASURE when the referral path is switched on
#       (referral starts are test-linked too) or any bundle input changes.
#       Was 0.226 = 0.292/1.292 from a single-member 8 x 40 run (SE 0.108), the
#       bundle's share of the inflated total rather than B/I0; that left 2022 starts
#       8.3 % below receipt (work record 2026-10-09 section 10.5).
PREP_BUNDLE_PROB_WILLING = 0.5
PREP_SAME_DAY = {"ED": 0.2, "primary_care": 0.5, "sud_treatment": 0.5, "mental_health": 0.5}
PREP_REFERRAL_PROB = 0.10                # derived (Blackstock / Petroll / Edelman), range 0.05-0.30
PREP_REFERRAL_RETURN = {"ED": 0.20, "primary_care": 0.25, "sud_treatment": 0.25, "mental_health": 0.25}
PREP_DEFERRED_RETURN = {"ED": PREP_REFERRAL_RETURN["ED"], "primary_care": 0.35, "sud_treatment": 0.35, "mental_health": 0.35}
PREP_TEST_LINKED_PER_INDEPENDENT = {"ED": 0.121, "primary_care": 0.231, "sud_treatment": 0.401, "mental_health": 0.025}
TEST_LINKED_OFF = {s: 0.0 for s in PREP_TEST_LINKED_PER_INDEPENDENT}   # the reference: full independent rate
PREP_LATE_AFTER = 6                     # steps (months) on PrEP before the late-phase hazard applies
PREP_LATE_MULT = 0.4                    # late-phase discontinuation hazard / early-phase (0.3-0.5)
PREP_MONITORING_INTERVAL = 3            # steps (months) between PrEP monitoring visits (CDC guideline: HIV test every 3 months)
PREP_MONITORING_ATTENDANCE = 0.60       # unsupported attendance (0.55-0.70; registry/CHIA note 7.4)
MOUD_ACCEPTANCE_INITIAL = 0.75          # placeholder; the search fits it (smoke run 2026-10-01: od_err passes above ~0.75)
NONCLINICAL_TESTING_NONINJ = 0.20        # 2026-10-12 (Jun): past-year testing ~27 % with the clinic route; Ahrens 2023 PMID 36594172 (Medicaid MOUD initiators, 11 states: HIV test within 90 days 20 % 2016 -> 25 % 2019), Meade 2009 PMID 19444732 (33 % non-injecting vs 71 % injecting); range 0.10-0.30; -1 = PWID rate
NONCLINICAL_TESTING_INITIAL = 0.38      # PINNED 2026-10-07 (calibration_targets.yml anchored_parameters; set_anchored_parameters overwrites this at every materialisation): annual probability, -ln(1-0.38) = 0.48 tests/PWID-yr from this route + ~0.14 from clinics = 0.62 vs NHBS 0.60 (B1 job 7098444). Was a fitted placeholder 0.4 from 2026-10-01


def read_csv(name):
    """Read an APCD CSV, skipping Mofan's '#' comment lines."""
    import pandas as pd
    lines = [l for l in open(APCD / name, encoding="utf-8", errors="replace") if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)))
    return df[[c for c in df.columns if not str(c).startswith("Unnamed")]]


def visit_distribution(dist, counts, setting):
    """Return [(n, p)] over users for one setting, 2022, with the 10+ bucket at its tail mean."""
    d = dist[(dist.year == YEAR) & (dist.setting == setting)]
    c = counts[(counts.year == YEAR) & (counts.setting == setting)].iloc[0]
    mean_users = c.visit_days / c.members_with_1plus_visit
    rows = [(int(r.visit_days), r.pct_members / 100.0) for r in d.itertuples()]
    head = [(n, p) for n, p in rows if n < 10]
    p_tail = sum(p for n, p in rows if n >= 10)
    head_mean = sum(n * p for n, p in head)
    tail_mean = (mean_users - head_mean) / p_tail if p_tail > 0 else 10.0
    return head + [(tail_mean, p_tail)], mean_users, tail_mean


def receipt_given(w, intensity, acc, dist):
    return w * sum(p * (1 - (1 - intensity * acc) ** n) for n, p in dist)


def solve_intensity(w, acc, target, dist):
    """Bisection for I in [0, 1] with W * E_n[1 - (1 - I a)^n] = target; None if unreachable."""
    if w <= 0 or target <= 0:
        return 0.0, True
    if receipt_given(w, 1.0, acc, dist) < target:
        return 1.0, False
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if receipt_given(w, mid, acc, dist) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, True


def derive(verbose=True, test_linked=None):
    """test_linked: B/I0 per setting for the PrEP re-solve (default
    PREP_TEST_LINKED_PER_INDEPENDENT; TEST_LINKED_OFF gives the reference scenario
    with the full independent rate, used to measure B/I0)."""
    test_linked = PREP_TEST_LINKED_PER_INDEPENDENT if test_linked is None else test_linked
    import pandas as pd
    out = {}
    log = []

    def say(s):
        log.append(s)
        if verbose:
            print(s)

    strata = read_csv("job1_strata.csv")
    counts = read_csv("job1_visit_counts.csv")
    dist = read_csv("job1_visit_distribution.csv")
    q1 = read_csv("job4_q1_providers_per_setting_fix.csv")
    q5 = read_csv("job4_q5_provider_service_crosstab_fix.csv")
    j3 = read_csv("job3a_hiv_services_by_setting_fix.csv")
    j3b = read_csv("job3b_hiv_test_per_person_fix.csv")
    j3e = read_csv("job3e_payer_code_share.csv")

    # Cohort composition for E[multiplier] (job3b partitions the whole 2022 cohort).
    b22 = j3b[j3b.year == YEAR]

    n_cohort_3b = int(b22[b22.stratum_type == "overall"].n_cohort.iloc[0])
    assert n_cohort_3b == COHORT_2022, f"job3b {YEAR} cohort {n_cohort_3b} != COHORT_2022 {COHORT_2022}"

    def cohort_share(stype, value):
        n = int(b22[(b22.stratum_type == stype) & (b22.stratum_value == value)].n_cohort.iloc[0])
        return n / n_cohort_3b

    cohort_comp = {
        "moud_status": cohort_share("moud_status", "on_moud"),
        "injection_proxy": cohort_share("injection_proxy", "injection"),
    }
    hiv_pos_members = max(
        int(strata[(strata.year == YEAR) & (strata.setting == s) & (strata.stratum_type == "hiv_status")
                   & (strata.stratum_value == "HIV+")].members.iloc[0])
        for s in SETTINGS
    )
    cohort_comp["hiv_status"] = hiv_pos_members / COHORT_2022

    # Reference-laboratory undercount factor for HIV-test receipt outside the ED.
    ins = strata[(strata.year == YEAR) & (strata.setting == INSURANCE_PROXY_SETTING) & (strata.stratum_type == "insurance")]
    mix = {str(r.stratum_value).lower(): float(r.members) for r in ins.itertuples()}
    mix_total = sum(mix.values())
    e22 = j3e[j3e.year == YEAR]
    rate_by_payer = {
        str(r.payer_type).lower(): sum(float(getattr(r, c)) for c in TEST_CODES) / float(r.n_members)
        for r in e22.itertuples()
    }
    rate_whole_extract = sum(mix[p] / mix_total * rate_by_payer[p] for p in mix)
    expected_tested = rate_whole_extract * COHORT_2022
    same_claim_any = int(b22[b22.stratum_type == "overall"].n_with_lab_test_any_setting.iloc[0])
    j3_22 = j3[j3.year == YEAR]
    ed_tested = int(j3_22[j3_22.setting == "ED"].lab_test.iloc[0])
    non_ed_tested = int(j3_22[j3_22.setting != "ED"].lab_test.sum())
    test_factor = (expected_tested - ed_tested) / non_ed_tested

    q5_50 = q5[q5.k_bin == "50+"].groupby(["setting", "service", "ever_delivered"]).n_npi.sum().unstack(fill_value=0)
    q5_50["share"] = q5_50[1] / (q5_50[0] + q5_50[1])

    say(f"Care-seeking parameters, APCD {YEAR}, cohort {COHORT_2022}\n")
    say("cohort composition for E[multiplier] (job3b, whole cohort): on MOUD %.3f, injecting %.3f, HIV+ %.3f (lower bound: largest single-setting HIV+ user count %d / cohort)"
        % (cohort_comp["moud_status"], cohort_comp["injection_proxy"], cohort_comp["hiv_status"], hiv_pos_members))
    say("HIV-test receipt correction outside the ED:")
    say("  whole-extract rate of %s per member, %d: %s" % ("+".join(TEST_CODES), YEAR,
        ", ".join(f"{p} {r:.4f}" for p, r in sorted(rate_by_payer.items()))))
    say("  cohort insurance mix (%s users): %s" % (INSURANCE_PROXY_SETTING,
        ", ".join(f"{p} {v / mix_total:.3f}" for p, v in sorted(mix.items()))))
    say(f"  weighted rate {rate_whole_extract:.4f} x cohort {COHORT_2022} = {expected_tested:.0f} people expected tested;"
        f" same-claim any-setting {same_claim_any} (job3b); ED same-claim {ed_tested} kept (bills its own labs)")
    say(f"  factor on primary_care / sud_treatment / mental_health receipt = ({expected_tested:.0f} - {ed_tested}) / {non_ed_tested} = {test_factor:.2f}"
        "  [assumption: missing tests allocated in proportion to same-claim counts; job3e counting unit unconfirmed; sensitivity x0.5 / x2]\n")
    for s in SETTINGS:
        say(f"=== {s} ===")
        c = counts[(counts.year == YEAR) & (counts.setting == s)].iloc[0]
        cohort_rate = c.visit_days / COHORT_2022
        # The engine applies visit_mult_moud / _inj / _hiv_positive to the agents
        # in those groups ON TOP of visit_rate_annual, so visit_rate_annual must be
        # the REFERENCE group's rate (off MOUD, not injecting, HIV-negative), not
        # the cohort mean: cohort mean = base x E[multiplier] over the cohort's
        # composition. Composition from job1_strata (users of the setting, 2022);
        # the smoke run of 2026-10-01 overshot SUD visits 1.7x with the cohort
        # mean used as the base.
        comp = {}
        for stype, flag in (("moud_status", "MOUD"), ("injection_proxy", "flagged"), ("hiv_status", "HIV+")):
            st2 = strata[(strata.year == YEAR) & (strata.setting == s) & (strata.stratum_type == stype)]
            comp[stype] = float(st2[st2.stratum_value == flag].members.iloc[0] / st2.members.sum())
        # The multipliers are COHORT-level ratios (registry 7.2), so the
        # expectation runs over the cohort's shares (job3b), not the users'.
        # The 2026-10-01 version used the users' shares, which are enriched for
        # MOUD and injection, and understated the ED and SUD bases by ~10%.
        e_mult = ((cohort_comp["moud_status"] * REGISTRY_7_2["visit_mult_moud"][s] + (1 - cohort_comp["moud_status"]))
                  * (cohort_comp["injection_proxy"] * REGISTRY_7_2["visit_mult_inj"][s] + (1 - cohort_comp["injection_proxy"])))
        rate = cohort_rate / e_mult  # the HIV+ term is folded in below once mult_hiv is known
        say(f"cohort visit rate = {int(c.visit_days)} visit-days / {COHORT_2022} = {cohort_rate:.3f}  [APCD job1_visit_counts]")
        say(f"composition among this setting's users (reported, not used): on MOUD {comp['moud_status']:.3f}, injecting {comp['injection_proxy']:.3f}, HIV+ {comp['hiv_status']:.3f}  [APCD job1_strata]")

        st = strata[(strata.year == YEAR) & (strata.setting == s) & (strata.stratum_type == "hiv_status")]
        r_pos = float(st[st.stratum_value == "HIV+"].visit_days.iloc[0] / st[st.stratum_value == "HIV+"].person_months.iloc[0])
        r_neg = float(st[st.stratum_value == "HIV-"].visit_days.iloc[0] / st[st.stratum_value == "HIV-"].person_months.iloc[0])
        mult_hiv = r_pos / r_neg
        say(f"visit_mult_hiv_positive = {r_pos * 12:.2f} / {r_neg * 12:.2f} per user-year = {mult_hiv:.2f}  [APCD job1_strata, among users]")
        e_mult *= cohort_comp["hiv_status"] * mult_hiv + (1 - cohort_comp["hiv_status"])
        rate = cohort_rate / e_mult
        say(f"visit_rate_annual (reference group) = {cohort_rate:.3f} / E[multiplier over the cohort] {e_mult:.3f} = {rate:.3f}")

        n_prov = int(q1[(q1.year == YEAR) & (q1.setting == s)].n_providers.iloc[0])
        say(f"n_providers_ri = {n_prov}  [APCD job4_q1]")

        k = Q5_KEY[s]
        w_test = float(q5_50.loc[(k, "hiv_test"), "share"])
        w_prep = float(q5_50.loc[(k, "prep"), "share"])
        w_bup = float(q5_50.loc[(k, "moud_bup"), "share"])
        say(f"p_willing_hiv_test = {w_test:.3f}, p_willing_prep = {w_prep:.3f}  [APCD job4_q5, providers with 50+ OUD patients, ever delivered]")
        if s == "ED":
            w_moud, w_moud_src = ED_WILLING_MOUD_RI, "RI literature PMID 38165180 (claims 50+ bin says %.2f: NPI attribution, registry 7.2 note 4)" % w_bup
        else:
            w_moud, w_moud_src = w_bup, "APCD job4_q5, 50+ bin, buprenorphine"
        say(f"p_willing_moud = {w_moud:.3f}  [{w_moud_src}]")

        d, mean_users, tail = visit_distribution(dist, counts, s)
        say(f"visits among users {YEAR}: mean {mean_users:.2f}, 10+ bucket at {tail:.1f}  [APCD job1_visit_distribution]")
        row = j3[(j3.year == YEAR) & (j3.setting == s)].iloc[0]
        p_test_claims = row.lab_test / row.n_denominator_hiv_negative
        p_test = p_test_claims if s == "ED" else p_test_claims * test_factor
        p_prep = row.prep_or_pep / row.n_denominator_hiv_negative
        i_test, ok_t = solve_intensity(w_test, ACCEPT_HIV_TEST, p_test, d)
        p_prep_residual = p_prep * (1.0 - test_linked[s])       # the bundle and referral supply the rest (B/I0 per setting)
        i_prep, ok_p = solve_intensity(w_prep, ACCEPT_PREP, p_prep_residual / PREP_INDICATED_SHARE[s], d)
        say(f"receipt {YEAR}: HIV test {int(row.lab_test)}/{int(row.n_denominator_hiv_negative)} = {p_test_claims:.4f} same-claim"
            + ("" if s == "ED" else f" x {test_factor:.2f} = {p_test:.4f} corrected")
            + f", PrEP/PEP {int(row.prep_or_pep)} = {p_prep:.4f}  [APCD job3a]")
        say(f"intensity_hiv_test: solve {w_test:.3f} * E[1-(1-I*{ACCEPT_HIV_TEST})^n] = {p_test:.4f} -> I = {i_test:.4f}{'' if ok_t else '  UNREACHABLE, capped at 1'}")
        say(f"intensity_prep:     solve {w_prep:.3f} * E[1-(1-I*{ACCEPT_PREP})^n] = {p_prep:.4f} x (1 - test-linked B/I0 {test_linked[s]}) / indicated share {PREP_INDICATED_SHARE[s]:.3f} = {p_prep_residual / PREP_INDICATED_SHARE[s]:.4f} -> I = {i_prep:.4f}{'' if ok_p else '  UNREACHABLE, capped at 1'}")
        say(f"test-and-offer bundle: P(PrEP raised | negative test) {PREP_BUNDLE_PROB_WILLING} at prescribing providers, same-day {PREP_SAME_DAY[s]}, deferred return {PREP_DEFERRED_RETURN[s]}; referral at non-prescribers {PREP_REFERRAL_PROB}, return {PREP_REFERRAL_RETURN[s]}")
        say(f"intensity_moud = {INTENSITY_MOUD[s]}  [{INTENSITY_MOUD_SOURCE[s]}]")

        out[s] = {
            "visit_rate_annual": round(rate, 4),
            "visit_heterogeneity_shape": REGISTRY_7_2["visit_heterogeneity_shape"][s],
            "visit_mult_inj": REGISTRY_7_2["visit_mult_inj"][s],
            "visit_mult_moud": REGISTRY_7_2["visit_mult_moud"][s],
            "visit_mult_hiv_positive": round(mult_hiv, 3),
            "n_providers_ri": n_prov,
            "panel_skew": REGISTRY_7_2["panel_skew"][s],
            "attrition_annual": REGISTRY_7_2["attrition_annual"][s],
            "p_willing_hiv_test": round(w_test, 4),
            "p_willing_prep": round(w_prep, 4),
            "p_willing_moud": round(w_moud, 4),
            "p_willing_counseling": 0.0,
            "intensity_hiv_test": round(i_test, 5),
            "intensity_prep": round(i_prep, 5),
            "intensity_moud": INTENSITY_MOUD[s],
            "intensity_counseling": 0.0,
            "visit_prob_after_overdose": ED_VISIT_AFTER_OD if s == "ED" else 0.0,
            "prep_bundle_prob": PREP_BUNDLE_PROB_WILLING,
            "prep_same_day_prob": PREP_SAME_DAY[s],
            "prep_deferred_return_prob": PREP_DEFERRED_RETURN[s],
            "prep_referral_prob": PREP_REFERRAL_PROB,
            "prep_referral_return_prob": PREP_REFERRAL_RETURN[s],
        }
        say(f"visit_prob_after_overdose = {out[s]['visit_prob_after_overdose']}  "
            + ("[assumption: claims-defined nonfatal overdoses are ED-treated by construction]" if s == "ED" else "[none]"))
        say("")
    return out, log


def render_block(settings):
    """The YAML text inserted into scenario_source.yml under `scenario:` (2-space indent)."""
    lines = [BEGIN,
             "  # Derived from docs/apcd_incoming/ABM_2 (Rhode Island APCD 2022), registry",
             "  # sections 7.2 and 7.6, and Rhode Island studies; the derivation prints with",
             "  # `python scripts/make_care_seeking_params.py`. See",
             "  # docs/calibration_log/phase_7_care_seeking_parameters_2026-10-01.md",
             "  # (section 8: the 2026-10-02 corrections). mental_health is the setting",
             "  # this APCD extract codes opioid treatment programs under, so it is",
             "  # OTP-dominated; HIV-test intensities outside the ED are corrected for",
             "  # the reference-laboratory undercount; ED gets one visit after each",
             "  # nonfatal overdose (visit_prob_after_overdose).",
             "  care_seeking:",
             "    shared_trait_prob: 0.0",
             "    setting:"]
    for s in SETTINGS:
        lines.append(f"      {s}:")
        for k, v in settings[s].items():
            lines.append(f"        {k}: {v}")
    lines.append(END)
    return "\n".join(lines) + "\n"


SCALARS = [
    # (top-level key under scenario:, field, value, comment)
    ("features", "care_seeking", "true", "the care-setting layer is on; MOUD, testing and PrEP run through encounters"),
    ("hiv_testing", "acceptance", ACCEPT_HIV_TEST, "RI ED drug-using patients accepting an offered test: 0.645 / 0.45 (PMIDs 25125271, 26036465)"),
    ("hiv_testing", "nonclinical_annual", NONCLINICAL_TESTING_INITIAL, "PINNED (anchored_parameters, 2026-10-07): annual probability; -ln(1-0.38) = 0.48 tests/PWID-yr + clinic ~0.14 = 0.62 vs NHBS 0.60 (B1 job 7098444); unidentified by the objective, was fitted 2026-10-01 to 10-07"),
    ("hiv_testing", "nonclinical_annual_noninj", NONCLINICAL_TESTING_NONINJ, "people with OUD who do not inject (2026-10-12): 0.20 (0.10-0.30) gives ~0.32 tests/yr with the clinic route (~27 % tested in a year): Medicaid MOUD initiators tested for HIV within 90 days 20 % (2016) -> 25 % (2019), 11 states (Ahrens 2023, PMID 36594172); past-year tested 33 % oxycodone vs 71 % heroin (Meade 2009, PMID 19444732); OTP clients 17-41 % (NDATSS); replaced by the APCD any-claim table when delivered"),
    ("moud", "acceptance", MOUD_ACCEPTANCE_INITIAL, "FITTED (parameters_to_optimize), replaces reinit_prob as the level lever; placeholder"),
    ("prep", "acceptance", ACCEPT_PREP, "starts once discussed: clinical review 2026-10-07 (25-50 %), was 0.60 (registry 7.6); sensitivity 0.25-0.60"),
    ("prep", "monitoring_interval_steps", PREP_MONITORING_INTERVAL, "guideline PrEP monitoring every 3 months: a primary-care visit with an HIV test (2026-10-02)"),
    ("prep", "monitoring_attendance", PREP_MONITORING_ATTENDANCE, "share of due monitoring visits attended, unsupported: RI PrEP clinic 53-68% retained at 6 months (PMID 27302837), chatbot non-users 54% (41792418)"),
    ("prep", "monitoring_setting", "primary_care", "where PrEP monitoring happens"),
    ("prep", "discontinue_late_after_steps", PREP_LATE_AFTER, "months on PrEP after which the discontinuation hazard drops (persistence curves are front-loaded; CHIA note 7.4)"),
    ("prep", "discontinue_late_mult", PREP_LATE_MULT, "late-phase hazard as a share of the early one: general 0.08 -> 0.03/month (PMID 33527117), PWID 0.30 -> 0.14 (40222706, 31436610) -> 0.4 (0.3-0.5)"),
]
SCALAR_TAG = "# care_seeking layer (scripts/make_care_seeking_params.py)"


def apply_to_source(text, block):
    """Insert or replace the generated block and the scalar lines in the source text."""
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", block, text, flags=re.S)
    else:
        # place it just before `  classes:` under `scenario:` (alphabetical home)
        m = re.search(r"^  classes:\n", text, flags=re.M)
        if not m:
            raise RuntimeError("could not find `  classes:` under scenario: in the source")
        text = text[: m.start()] + block + text[m.start():]

    for top, field, value, comment in SCALARS:
        line = f"    {field}: {value}  {SCALAR_TAG}: {comment}\n"
        # Work inside the `  <top>:` block only: `acceptance` exists under both
        # hiv_testing and prep, so an unscoped match would rewrite the wrong one.
        m = re.search(rf"^  {top}:\n", text, flags=re.M)
        if not m:
            raise RuntimeError(f"could not find `  {top}:` under scenario: in the source")
        start = m.end()
        nxt = re.search(r"^  \S", text[start:], flags=re.M)
        end = start + nxt.start() if nxt else len(text)
        block_text = text[start:end]
        pat = re.compile(rf"^    {field}: .*{re.escape(SCALAR_TAG)}.*\n", flags=re.M)
        if pat.search(block_text):
            block_text = pat.sub(line, block_text)
        else:
            block_text = line + block_text
        text = text[:start] + block_text + text[end:]
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the block and scalars into the scenario source")
    ap.add_argument("--check", action="store_true", help="exit 1 if the source's block differs from a fresh derivation")
    ap.add_argument("--source", type=Path, default=SOURCE)
    args = ap.parse_args()

    settings, _ = derive(verbose=not args.check)
    block = render_block(settings)

    if args.check:
        text = args.source.read_text()
        m = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END), text, flags=re.S)
        if not m:
            print("STALE: no generated care_seeking block in the source")
            return 1
        current = yaml.safe_load(m.group(1))
        fresh = yaml.safe_load(block.replace(BEGIN, "").replace(END, ""))
        if current != fresh:
            print("STALE: the source's care_seeking block differs from a fresh derivation; run --write")
            return 1
        src = yaml.safe_load(text)["scenario"]
        for top, field, value, _ in SCALARS:
            have = src.get(top, {}).get(field)
            want = yaml.safe_load(str(value))
            if have != want:
                print(f"STALE: scenario.{top}.{field} is {have!r}, derivation says {want!r}")
                return 1
        print("care_seeking block and scalars are current")
        return 0

    print(block)
    if args.write:
        text = args.source.read_text()
        new = apply_to_source(text, block)
        args.source.write_text(new)
        print(f"written into {args.source}; now run: python scripts/make_scenario.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
