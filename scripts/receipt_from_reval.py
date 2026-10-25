#!/usr/bin/env python
"""
Receipts from a re-validation's status-quo runs (work record §19.41, §19.47): one row per member from counts summed
over its draws and populations, then medians and ranges over members, and the same ratios on the counts pooled over
every member. Nothing here is fitted.

  clinic_tested_insured   clinic-visible people tested in 2022 on the insured rows   job 7a 0.117
  tests_per_pwid_yr       HIV tests per injecting person-year                        NHBS 0.60
  insured_frac, dx_among_hiv                                                         0.949 / RIDOH all PWH 0.932 (CDC PWID national 0.96)
  housing_share, housing_share_pwid   agent-months with the housing component        TEDS-A RI 15.5 % inj / 6.9 % non-inj at admission (upper), SUDORS-derived 2-4 %
  od_deaths_homeless_share   overdose deaths of agents homeless at the time          RI SUDORS 2020-22: 5.4 % of opioid decedents
  hiv_acq_ratio_pwid      new infections per person-month, homeless vs housed PWID   Arum 2021 aRR 1.39 (crude 1.55)
  adherent_or_housing     odds ratio of ART adherence, homeless vs housed            viral suppression OR 0.38 (US meta-analysis)
  sex_work_share, hardship_mean

Usage: python scripts/receipt_from_reval.py <reval_dir> [members_file]   (parallel over report files; compute node)
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

OUT = sys.argv[1]
MEMBERS = set(open(sys.argv[2]).read().split()) if len(sys.argv) > 2 else None
WANT = ["t", "drug_type", "insured_state", "agents", "hiv_tests", "hiv_tested_this_year_clinical", "hiv_tested_this_year",
        "hiv", "hiv_dx", "hiv_new", "hiv_new_housing", "overdose_deaths", "overdose_deaths_housing",
        "haart", "haart_adherent", "haart_housing", "haart_adherent_housing",
        "hardship_housing", "hardship_1", "hardship_2", "hardship_3", "sex_work"]


def one(d):
    name = os.path.basename(d); member, seed = name.replace("trial", "", 1).rsplit("_seed", 1)
    cols = pd.read_csv(f"{d}/basicReport.txt", sep="\t", nrows=0).columns
    df = pd.read_csv(f"{d}/basicReport.txt", sep="\t", usecols=[c for c in WANT if c in cols])
    for c in WANT:
        if c not in df.columns:
            df[c] = 0
    y = df[(df.t >= 121) & (df.t <= 132)]; dec = y[(y.insured_state == "insured") & (y.t == 132)]
    pb = df[df.t >= 1]; inj = pb[pb.drug_type == "Inj"]
    s = {"member": member, "seed": seed,
         "tested_clinic_dec": dec["hiv_tested_this_year_clinical"].sum(), "tested_any_dec": dec["hiv_tested_this_year"].sum(), "agents_dec": dec["agents"].sum(),
         "tests_inj_2022": y[y.drug_type == "Inj"]["hiv_tests"].sum(), "agents_inj_2022": y[y.drug_type == "Inj"]["agents"].sum(),
         "agents": pb["agents"].sum(), "agents_insured": pb[pb.insured_state == "insured"]["agents"].sum(),
         "hiv": pb["hiv"].sum(), "hiv_dx": pb["hiv_dx"].sum(),
         "housing": pb["hardship_housing"].sum(), "housing_inj": inj["hardship_housing"].sum(), "agents_inj": inj["agents"].sum(),
         "hiv_new_inj": inj["hiv_new"].sum(), "hiv_new_housing_inj": inj["hiv_new_housing"].sum(),
         "od_deaths": pb["overdose_deaths"].sum(), "od_deaths_housing": pb["overdose_deaths_housing"].sum(),
         "haart": pb["haart"].sum(), "haart_adh": pb["haart_adherent"].sum(), "haart_h": pb["haart_housing"].sum(), "haart_adh_h": pb["haart_adherent_housing"].sum(),
         "sex_work": pb["sex_work"].sum(), "hardship_points": (pb["hardship_1"] + 2 * pb["hardship_2"] + 3 * pb["hardship_3"]).sum()}
    return s


def ratios(c):
    def sd(a, b):
        return float(a) / float(b) if float(b) > 0 else np.nan
    a1 = sd(c["haart_adh_h"], c["haart_h"]); a0 = sd(c["haart_adh"] - c["haart_adh_h"], c["haart"] - c["haart_h"])
    return {
        "clinic_tested_insured": sd(c["tested_clinic_dec"], c["agents_dec"]), "any_tested_insured": sd(c["tested_any_dec"], c["agents_dec"]),
        "tests_per_pwid_yr": sd(c["tests_inj_2022"], c["agents_inj_2022"] / 12.0), "insured_frac": sd(c["agents_insured"], c["agents"]),
        "dx_among_hiv": sd(c["hiv_dx"], c["hiv"]), "housing_share": sd(c["housing"], c["agents"]), "housing_share_pwid": sd(c["housing_inj"], c["agents_inj"]),
        "od_deaths_homeless_share": sd(c["od_deaths_housing"], c["od_deaths"]),
        "hiv_acq_ratio_pwid": sd(sd(c["hiv_new_housing_inj"], c["housing_inj"]), sd(c["hiv_new_inj"] - c["hiv_new_housing_inj"], c["agents_inj"] - c["housing_inj"])),
        "adherent_or_housing": sd(a1 / (1 - a1) if 0 < a1 < 1 else np.nan, a0 / (1 - a0) if 0 < a0 < 1 else np.nan),
        "sex_work_share": sd(c["sex_work"], c["agents"]), "hardship_mean": sd(c["hardship_points"], c["agents"]),
    }


if __name__ == "__main__":
    dirs = sorted(x for x in glob.glob(f"{OUT}/draw*_out_*/trial*_seed*") if os.path.isdir(x))
    if MEMBERS:
        dirs = [d for d in dirs if os.path.basename(d).replace("trial", "", 1).rsplit("_seed", 1)[0] in MEMBERS]
    print(f"{len(dirs)} report files, {os.cpu_count()} cpus", flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
        rows = p.map(one, dirs, chunksize=4)
    r = pd.DataFrame(rows)
    counts = r.drop(columns=["seed"]).groupby("member").sum(numeric_only=True)
    per = pd.DataFrame({m: ratios(c) for m, c in counts.iterrows()}).T
    pooled = pd.Series(ratios(counts.sum()))
    pd.set_option("display.width", 220)
    print(f"\nper member ({len(per)} members; counts summed over draws and populations): median [min, max]; pooled over members")
    for c in per.columns:
        print(f"  {c:26s} {per[c].median():.4f} [{per[c].min():.4f}, {per[c].max():.4f}]   pooled {pooled[c]:.4f}")
    print("targets: clinic_tested_insured 0.117; tests_per_pwid_yr 0.60; insured 0.949; dx_among_hiv 0.932 (RIDOH, all PWH; CDC PWID national 0.96); od_deaths_homeless_share 0.054 (SUDORS);"
          " hiv_acq_ratio_pwid 1.39 (Arum aRR; crude 1.55); adherent_or_housing ~0.38 (suppression OR); housing_share_pwid 0.02-0.155")
    print("\nSE over member ratios (sd/sqrt(n)):"); print((per.std(ddof=1) / np.sqrt(len(per))).round(4).to_string())
    per.to_csv(f"{OUT}/receipt_by_member.csv"); counts.to_csv(f"{OUT}/receipt_counts_by_member.csv"); r.to_csv(f"{OUT}/receipt_per_run.csv", index=False)
    print("written", f"{OUT}/receipt_by_member.csv")
