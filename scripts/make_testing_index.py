#!/usr/bin/env python
"""
Clinical HIV-testing index, 2012-2022: timeline_scaling entries that move the
care layer's HIV-test receipt with Rhode Island's testing over time, instead of
holding it at the 2022 level the care-seeking parameters were solved for
(2026-10-06).

STATUS (audit, 2026-10-06 afternoon): SENSITIVITY BOUND ONLY, NOT FOR THE
BASELINE. Two defects found after the first check was queued (job 7061726):

1. The whole-extract table (job3e) carries only the screening codes 87389, 87806,
   86703 and the G-code family. It has no column for 86701 / 86702, which job3d
   shows were the only HIV test code for 42-54 % of the cohort's tested members
   in 2012-2017, 27-31 % in 2018-2019 and 13-19 % in 2020-2022 (most likely
   front-line antibody tests, not confirmatory ones: REVIEW notes on job3d). The
   rise of the screening codes through 2019 is therefore at least partly
   laboratories moving from 86701/86702 to the fourth-generation 87389, not more
   testing; corrected crudely for that share, 2014-2019 would sit ABOVE 2022
   (1.2-1.5), the opposite of the index below. The pre-2020 trend outside the ED
   is not identified by the current pull. What would settle it: per year, on any
   claim, distinct members with any HIV test code INCLUDING 86701/86702 (and
   87390/87391), for the OUD cohort and for the whole extract by payer.
2. The 2012 level started at step 1, so the burn-in ran at the 2022 level and
   clinical testing dropped 41 % at January 2012. Fixed 2026-10-06: the first
   level now starts at the first burn step. When the any-claim table arrives,
   `--anyclaim <job7a csv>` derives the index from it directly.

What is robust: the ED series (all test codes, own laboratory billing): 0.67 of
the 2022 level in 2012, 0.71-0.85 in 2013-2017, 0.95-1.02 from 2018, no COVID
dip. And a one-year fall in 2020 outside the ED, by 16 % (screening codes, any
claim) to 65 % (cohort same-claim, artifact-prone). Clinical offers are 20-28 %
of the model's HIV tests (trial 26), so a one-year 2020 dip moves total testing
by 3-5 % for one year.

WHAT THE APCD SHOWS, AND WHICH SERIES CAN BE USED
-------------------------------------------------
Three per-year series exist in the APCD pull (docs/apcd_incoming/ABM_2):

* job3a / job3b / job3d (OUD cohort): count a test only when it is billed on the
  same claim as a visit to a care setting. In primary care the count falls 65 %
  in 2020 (2.3 % -> 0.8 % of HIV-negative users) and stays there through 2022,
  while primary-care visits rose. Specimens sent to reference laboratories bill
  on their own claims and are invisible to this measure (REVIEW_2026-09-29.md
  section 2), and a step change to a new stable level at 2020 is what a shift to
  patient-service-centre draws would produce. **Not usable for a time trend
  outside the ED.** The ED bills its own laboratory work, so its same-claim
  series is valid: no COVID dip (2019 2.96 %, 2020 3.02 %, 2021 2.79 %, 2022
  3.12 %), and a rise from about 2.1-2.6 % in 2012-2015.
* job3e (whole RI APCD extract, every member, any claim): HIV screening codes
  (CPT 87389, 87806, 86703 and the G-code family) per member, by payer. Not tied
  to claim linkage. Rising through 2019 (Medicaid 0.055 -> 0.099, commercial
  0.031 -> 0.064 per member), a one-year dip in 2020 (-17 to -19 %), back to the
  2019 level by 2021-2022.

THE INDEX
---------
T(y) = w_ED * ED(y) + (1 - w_ED) * X(y), each term relative to 2022, where
  ED(y) = the cohort's ED same-claim receipt share (job3a),
  X(y)  = the whole-extract screening rate per member, weighted by the cohort's
          2022 insurance mix (job1_strata, primary-care users: the proxy the
          care-seeking generator uses), standing in for the three settings whose
          tests go to reference laboratories,
  w_ED  = the ED's share of the cohort's corrected 2022 clinical tests
          (make_care_seeking_params: ED same-claim count kept, the other settings
          scaled by the reference-laboratory factor).
Applied as a multiplier on `hiv_testing.acceptance`, which the engine reads on
every clinical offer and only there (titan/features/hiv_testing.py,
encounter_prob); receipt is proportional to it. Provider intensities cannot carry
the index: they are frozen into the provider records when drawn.

T = 1 from 2022 on, so the Phase 7 arms (dated 2023) act on the 2022 level as
before. Nonclinical testing (`hiv_testing.nonclinical_annual`, fitted) is left
flat: nothing in the APCD measures it.

Assumptions to state: the cohort's non-ED testing follows the whole-extract
trend; the 2016 drop in commercial membership (self-insured plans leaving after
Gobeille) changes the commercial composition; job3e's counting unit (members vs
claim lines) is unconfirmed but constant across years.

Usage:
    python scripts/make_testing_index.py --print
    python scripts/make_testing_index.py --in-dir <trial ymls> --out-dir <dir>
"""
import argparse
import io
from pathlib import Path

import pandas as pd
import yaml

APCD = Path(__file__).resolve().parents[1] / "docs" / "apcd_incoming" / "ABM_2"
STEPS_PER_YEAR = 12
FIRST_YEAR = 2012
ANCHOR = 2022
TEST_CODES = ["cpt_87389", "cpt_87806", "cpt_86703", "G0475", "G0432", "G0433", "G0435"]
PREFIX = "testing_"
PARAM = "hiv_testing|acceptance"


def read_csv(name):
    lines = [l for l in open(APCD / name, encoding="utf-8", errors="replace") if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)))
    return df[[c for c in df.columns if not str(c).startswith("Unnamed")]]


def derive():
    j3a = read_csv("job3a_hiv_services_by_setting_fix.csv")
    j3b = read_csv("job3b_hiv_test_per_person_fix.csv")
    j3e = read_csv("job3e_payer_code_share.csv")
    strata = read_csv("job1_strata.csv")

    ed = j3a[j3a.setting == "ED"].set_index("year")
    ed_rate = ed.lab_test / ed.n_denominator_hiv_negative

    ins = strata[(strata.year == ANCHOR) & (strata.setting == "primary_care") & (strata.stratum_type == "insurance")]
    mix = {str(r.stratum_value).lower(): float(r.members) for r in ins.itertuples()}
    tot = sum(mix.values())
    mix = {p: v / tot for p, v in mix.items()}
    j3e = j3e.copy()
    j3e["payer_type"] = j3e.payer_type.str.lower()
    j3e["rate"] = j3e[TEST_CODES].sum(axis=1) / j3e.n_members
    by_payer = j3e.pivot(index="year", columns="payer_type", values="rate")
    x_rate = sum(mix[p] * by_payer[p] for p in mix)

    # The ED's share of the cohort's corrected 2022 clinical tests, as the
    # care-seeking generator computes it.
    e22 = j3e[j3e.year == ANCHOR]
    rate_by_payer = {r.payer_type: r.rate for r in e22.itertuples()}
    cohort = int(j3b[(j3b.year == ANCHOR) & (j3b.stratum_type == "overall")].n_cohort.iloc[0]) \
        if "overall" in set(j3b.stratum_type) else 12774
    expected = sum(mix[p] * rate_by_payer[p] for p in mix) * cohort
    ed_22 = float(ed.loc[ANCHOR, "lab_test"])
    w_ed = ed_22 / expected

    t = pd.DataFrame({
        "ed_rate": ed_rate,
        "ed_rel": ed_rate / ed_rate[ANCHOR],
        "x_rate": x_rate,
        "x_rel": x_rate / x_rate[ANCHOR],
    })
    t["index"] = w_ed * t.ed_rel + (1 - w_ed) * t.x_rel
    return t, w_ed, mix, expected


def derive_from_anyclaim(path):
    """
    The index from the any-claim cohort table requested on 2026-10-06
    (docs/apcd_request_hiv_testing_anyclaim_2026-10-06.md, job7a): people tested
    on any claim over the cohort, overall rows, relative to 2022. It covers every
    setting at once, so neither the ED blend nor the whole-extract proxy is
    needed. Suppressed cells are not expected in the overall rows.
    """
    d = pd.read_csv(io.StringIO("".join(l for l in open(path) if not l.startswith("#"))))
    d = d[d["group"] == "overall"].set_index("year")
    rate = d["n_tested_any_claim"].astype(float) / d["n_cohort"].astype(float)
    t = pd.DataFrame({"anyclaim_rate": rate, "index": rate / rate[ANCHOR]})
    t["same_claim_rate"] = d["n_tested_same_claim"].astype(float) / d["n_cohort"].astype(float)
    return t


def testing_entries(index, burn_steps):
    """
    One entry per year before ANCHOR. The first year's level also covers the
    burn-in (it starts at the first burn step, 1 - burn_steps), so the model does
    not run the burn-in at the 2022 level and then drop at January 2012 -- the
    defect of the first version.
    """
    entries = {}
    years = [y for y in sorted(index.index) if FIRST_YEAR <= y < ANCHOR]
    for i, y in enumerate(years):
        s = float(index[y])
        start = 1 + (y - FIRST_YEAR) * STEPS_PER_YEAR
        if i == 0:
            start = 1 - int(burn_steps)
        entries[f"{PREFIX}{y}"] = {
            "parameter": PARAM,
            "start_time": start,
            "stop_time": 1 + (y + 1 - FIRST_YEAR) * STEPS_PER_YEAR,
            "scalar": round(s, 6),
        }
    return entries


def apply_testing_index(scenario, index):
    tl = scenario.setdefault("timeline_scaling", {}).setdefault("timeline", {})
    for key in [k for k in tl if k.startswith(PREFIX)]:
        del tl[key]
    tl.update(testing_entries(index, scenario["model"]["time"]["burn_steps"]))
    scenario.setdefault("features", {})["timeline_scaling"] = True
    return scenario


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--glob", default="trial_*.yml")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--anyclaim", default=None,
                    help="job7a CSV (any-claim cohort table); derive the index from it instead of the "
                         "ED + whole-extract blend")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    if args.anyclaim:
        t = derive_from_anyclaim(args.anyclaim)
        print(f"index from the any-claim cohort table {args.anyclaim}")
    else:
        t, w_ed, mix, expected = derive()
        print("index from the ED + whole-extract blend (SENSITIVITY ONLY; see the docstring)")
        print(f"cohort insurance mix (2022 primary-care users): " + ", ".join(f"{p} {v:.3f}" for p, v in mix.items()))
        print(f"expected tested 2022 (whole-extract rate x cohort) {expected:.0f}; ED share w_ED = {w_ed:.3f}")
    print(t.round(4).to_string())
    if args.in_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(Path(args.in_dir).glob(args.glob)):
            scen = yaml.safe_load(p.read_text())
            apply_testing_index(scen, t["index"])
            (out / p.name).write_text(yaml.safe_dump(scen, sort_keys=False))
            n += 1
        print(f"wrote {n} scenarios to {out}")


if __name__ == "__main__":
    main()
