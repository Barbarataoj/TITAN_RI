"""
Drug-supply lethality index (2026-10-06): timeline_scaling entries that raise
the share of overdoses that are fatal in the years Rhode Island's opioid deaths
rose without a matching rise in nonfatal overdoses.

WHY. The model rolls a fixed fatal fraction per overdose (overdose.fatal_frac,
derived from the targets), so deaths per active-OUD person-year move only with the
event rate. Against Rhode Island (analysis/od_lethality_drift.py, ratio to 2019):

    year   RIDOH opioid-involved deaths   model deaths (no index)   nonfatal, APCD   model nonfatal
    2016   1.13                           1.07                      0.94             1.08
    2019   1.00                           1.00                      1.00             1.00
    2020   1.26                           0.99                      0.96             0.98
    2021   1.46                           0.95                      0.97             0.96
    2022   1.40                           0.93                      0.87             0.93

Through 2019 the model follows the data (both fall as MOUD rises); from 2020
deaths rose 26-46 % while claims-identified nonfatal overdoses did not: case
fatality rose (fentanyl). RIDOH's opioid-involved deaths then fell, 359 (2022) ->
345 (2023) -> 228 (2024).

DATA. Deaths: RIDOH / Office of State Medical Examiners, opioid-involved
accidental overdose deaths (Governor's Overdose Task Force decks 2023-06-14 and
2025-06-11; read off stacked bars, about +/- 5): 256 (2019), 323, 375, 359, 345,
228 (2024). Death COUNTS relative to 2019, no denominator: the model's population
is a fixed size with a near-constant active-OUD share, and the three APCD
vintages disagree on the yearly OUD count (2022: 14,766 workbook, 13,562
verification document, 12,774 current extract), which moves a per-person ratio
by up to 9 % in 2022 (k 1.48-1.61) and by 5 % in 2020-2021. Nonfatal: the
"did not rise" statement holds in every APCD series available:
RI_parameter_verification_APCD.md rows 17-18 (the vintage the overdose targets
use; shown above), job2b index overdoses per cohort member (0.91 / 0.93 / 0.97
of 2019 in 2020-2022) and the superseded workbook. docs/apcd_incoming/ABM/
"R01 parameter.xlsx" is stale (README section 3) and is not used; the first
version of this index (2026-10-06 morning, check jobs 7061213/7061215) took its
denominators from it: k 1.30 / 1.48 / 1.48 / 1.40 / 0.96, within 4 % of the
values below.

THE INDEX. k(y) = data ratio / model ratio, i.e. the death multiplier the model is
missing in year y, with nonfatal overdoses held where they are:

    fatal_frac      x  k / (1 - f0 + k f0)
    base_prob_inj   x  (1 - f0 + k f0)        (more events, so that
    base_prob_noninj x (1 - f0 + k f0)         nonfatal = events x (1 - f) is unchanged)

where f0 is the scenario's own fatal_frac. Deaths then scale by k and nonfatal
overdoses by 1, to first order (measured on the 11 members at 8 x 40: deaths
1.28 / 1.45 / 1.38 of 2019 against 1.29 / 1.41 / 1.37 with the first version's k;
nonfatal rates within 1 % of the no-index runs). The model ratio for 2023 onward
is the 2022 one (0.93): the MOUD ramp is held after 2022 and the status-quo arm
is flat (four baseline members, 2023-2031).

2024 CAVEAT. Opioid-involved deaths fell 34 % in 2024 while non-opioid deaths
rose (about 59 -> 101), so the opioid-involved share of all overdose deaths went
from about 85 % to 69 % in one year. On all accidental overdose deaths (329 vs
308 in 2019) k(2024) would be 1.15, not 0.96. Sensitivity for the projection.

PROJECTION (2025 onward), `overdose.lethality_index` in calibration_targets.yml
or the script's --projection:
    hold2025      (default since 2026-10-19) k follows RIDOH through 2025 (0.67,
                  preliminary) and stays there;
    hold2024      k stays at its 2024 value (0.96), about the pre-2020
                  level, on opioid-involved deaths;
    hold2024_all  the same history, but 2024 on all accidental overdose deaths
                  (k 1.15; see the 2024 caveat);
    peak          k = 1.52 (the 2021-2022 level) from 2023 on;
    pre2020       k = 1 from 2023 on (the index covers 2020-2022 only);
    none          no index.
For calibration-length runs (132 steps, through 2022) every projection except
"none" is identical.

USED BY optuna_calibration (objective and canonical writer, after the anchored
fatal fraction is set), scripts/make_phase7_baseline.build (so every Phase 7
baseline and arm carries it) and scripts/make_lethality_index.py (to add it to
existing trial files).
"""

STEPS_PER_YEAR = 12
FIRST_YEAR = 2012

# RIDOH opioid-involved deaths relative to 2019 (323, 375, 359, 345, 228 over 256)
# divided by the model's own no-index ratio (0.991, 0.950, 0.929; 0.93 from 2023).
K_DATA = {2020: 1.27, 2021: 1.54, 2022: 1.51, 2023: 1.45, 2024: 0.96, 2025: 0.67}
# 2025 (added 2026-10-19, PRELIMINARY): RIDOH reported 219 accidental overdose deaths in 2025
# (governor's release, January 2026; 329 in 2024), 73 % involving opioids -> about 160
# opioid-involved deaths, 160 / 256 = 0.625 over the model's 0.93 = 0.67. The January 2026
# task-force slides (data to June 2025) show opioid-involved deaths down 29 % on the first half
# of 2024, which gives the same figure. Revise when RIDOH publishes the full-year count.
# All accidental overdose deaths, 2024: 329 / 308 = 1.068, over 0.93.
K_2024_ALL = 1.15
K_PEAK = 1.52
PREFIX = "lethality_"
PROJECTIONS = ("hold2025", "hold2024", "hold2024_all", "peak", "pre2020", "none")


def start_step(year):
    """Model step of January of `year` (t = 1 is January 2012)."""
    return 1 + (year - FIRST_YEAR) * STEPS_PER_YEAR


def k_path(projection):
    """Year -> k for each year with an index entry; the last entry is held."""
    if projection == "none":
        return {}
    path = {y: K_DATA[y] for y in (2020, 2021, 2022)}
    if projection == "hold2025":
        path.update({2023: K_DATA[2023], 2024: K_DATA[2024], 2025: K_DATA[2025]})
    elif projection == "hold2024":
        path.update({2023: K_DATA[2023], 2024: K_DATA[2024]})
    elif projection == "hold2024_all":
        path.update({2023: K_DATA[2023], 2024: K_2024_ALL})
    elif projection == "peak":
        path[2023] = K_PEAK
    elif projection == "pre2020":
        path[2023] = 1.0
    else:
        raise ValueError(f"unknown lethality projection {projection!r}; one of {PROJECTIONS}")
    return path


def lethality_entries(f0, projection="hold2025"):
    """
    The timeline_scaling entries for one projection.

    args:
        f0: the scenario's overdose.fatal_frac before any index
        projection: one of PROJECTIONS

    returns:
        {entry name: timeline_scaling definition}
    """
    path = k_path(projection)
    years = sorted(path)
    entries = {}
    for i, y in enumerate(years):
        k = path[y]
        if abs(k - 1.0) < 1e-12:
            continue
        stop = start_step(years[i + 1]) if i + 1 < len(years) else 10000
        events = 1.0 - f0 + k * f0
        for name, param, scalar in (
            ("ff", "overdose|fatal_frac", k / events),
            ("bpi", "overdose|base_prob_inj", events),
            ("bpn", "overdose|base_prob_noninj", events),
        ):
            entries[f"{PREFIX}{y}_{name}"] = {
                "parameter": param,
                "start_time": start_step(y),
                "stop_time": stop,
                "scalar": round(scalar, 8),
            }
    return entries


def projection_from_calib(calib):
    """The projection named in calibration_targets.yml (`none` when absent)."""
    od = (calib.get("calibration") or {}).get("overdose") or {}
    projection = str(od.get("lethality_index", "none"))
    if projection not in PROJECTIONS:
        raise ValueError(f"calibration.overdose.lethality_index = {projection!r}; one of {PROJECTIONS}")
    return projection


def apply_lethality(scenario, projection="hold2025"):
    """
    Replace any lethality entries in a scenario with those of `projection`.

    Idempotent: earlier `lethality_*` entries are removed first, so applying it
    to a file that already carries an index (or a different projection) leaves
    exactly one. Reads f0 from the scenario, so call it after the anchored
    fatal fraction has been written.

    args:
        scenario: the scenario dict (modified in place and returned)
        projection: one of PROJECTIONS
    """
    tl = scenario.setdefault("timeline_scaling", {}).setdefault("timeline", {})
    for key in [k for k in tl if k.startswith(PREFIX)]:
        del tl[key]
    entries = lethality_entries(float(scenario["overdose"]["fatal_frac"]), projection)
    if entries:
        tl.update(entries)
        scenario.setdefault("features", {})["timeline_scaling"] = True
    return scenario
