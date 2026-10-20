#!/usr/bin/env python
"""
Add the drug-supply lethality index to existing trial or scenario files.

The index, its data and its derivation live in titan/calibration/lethality.py.
New calibration trials and every Phase 7 baseline get it automatically (from
`overdose.lethality_index` in calibration_targets.yml); this script is for files
written before that, e.g. a pass list being re-scored.

Usage:
    python scripts/make_lethality_index.py --in-dir <trial ymls> --out-dir <dir>
        [--projection hold2025|hold2024|hold2024_all|peak|pre2020|none] [--glob 'trial_*.yml']
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.calibration.lethality import PROJECTIONS, apply_lethality, lethality_entries  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--projection", default="hold2025", choices=PROJECTIONS)
    ap.add_argument("--glob", default="trial_*.yml")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n, f0 = 0, None
    for p in sorted(Path(args.in_dir).glob(args.glob)):
        scen = yaml.safe_load(p.read_text())
        apply_lethality(scen, args.projection)
        f0 = float(scen["overdose"]["fatal_frac"])
        (out / p.name).write_text(yaml.safe_dump(scen, sort_keys=False))
        n += 1
    print(f"wrote {n} scenarios to {out} (projection {args.projection})")
    if n:
        for name, entry in lethality_entries(f0, args.projection).items():
            print(f"  {name}: {entry}")


if __name__ == "__main__":
    main()
