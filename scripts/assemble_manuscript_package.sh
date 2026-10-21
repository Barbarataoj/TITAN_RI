#!/bin/bash
# Assemble the manuscript package for the writing team into one folder (and a zip).
# Usage: scripts/assemble_manuscript_package.sh [out_dir]   (from the repo, titan_env active)
# Nothing from docs/apcd_incoming/ is copied (unsuppressed small cells); the package carries only
# archived, aggregated results.
set -euo pipefail
cd /oscar/data/jtao9/TITAN_RI
OUT=${1:-docs/manuscript/package_2026-10-21}; rm -rf "$OUT"; mkdir -p "$OUT"/{01_manuscript,02_figures,03_tables,04_supplement,05_references,06_notes}
M=docs/manuscript
# 1. manuscript: numbered text + reference list
python analysis/build_references.py $M/draft_2026-10-21_v3.md "$OUT/01_manuscript/manuscript_numbered.md" "$OUT/01_manuscript/references.md"
cp $M/draft_2026-10-21_v3.md "$OUT/01_manuscript/manuscript_with_citation_keys.md"
cp $M/tables/table1_main.md "$OUT/01_manuscript/table1.md"; cp $M/tables/table1_main.csv "$OUT/01_manuscript/table1.csv"
cp $M/figures/CAPTIONS.md "$OUT/01_manuscript/figure_captions.md"
# 2. figures (main: 1, 2, 3, 6->4; supplement: 4, 5, 7, 8)
for f in fig1_model_schematic fig2_status_quo fig3_effects fig4_chain fig5_sensitivity fig6_attribution fig7_cashflow fig8_forward_check; do
  for ext in png pdf; do [ -f $M/figures/$f.$ext ] && cp $M/figures/$f.$ext "$OUT/02_figures/"; done; done
# 3. tables (csv + md)
cp $M/tables/table_*.csv $M/tables/table_*.md "$OUT/03_tables/"
# 4. supplement
cp $M/supplement_validation_odd_2026-10-21.md "$OUT/04_supplement/" 2>/dev/null || cp $M/supplement_validation_odd_2026-10-20.md "$OUT/04_supplement/"
cp docs/MODEL_VALIDATION.md "$OUT/04_supplement/model_validation_notes.md"
cp docs/phase7_unit_costs.yml "$OUT/04_supplement/unit_costs.yml"
mkdir -p "$OUT/04_supplement/scale_check_500_agents"; cp $M/figures/scale_check_N500_2027/*.png "$OUT/04_supplement/scale_check_500_agents/"; cp $M/tables/scale_check_N500_2027/*.md "$OUT/04_supplement/scale_check_500_agents/"
# 5. references: metadata and non-PubMed list
cp $M/references_extra_2026-10-21.csv "$OUT/05_references/pubmed_metadata_cited.csv"
cp $M/references_from_pmids.csv "$OUT/05_references/pubmed_metadata_all_project_sources.csv"
cp $M/non_pubmed_references_2026-10-21.yml "$OUT/05_references/non_pubmed_references.yml"
cp $M/non_pubmed_sources_2026-10-14.md "$OUT/05_references/non_pubmed_sources_inventory.md"
# 6. notes for the writers
cp $M/questions_for_mofan_2026-10-21.md "$OUT/06_notes/"
cp $M/MATERIALS.md "$OUT/06_notes/materials_map.md"
cp $M/results_package_2026-10-14.md "$OUT/06_notes/results_package_dev_set_2026-10-14_superseded.md"
cp $M/draft_2026-10-16_v2.md "$OUT/06_notes/draft_v2_before_citation_markup.md"
cat > "$OUT/00_README.md" <<'EOR'
# Manuscript package — HIV prevention strategies in primary care for people with opioid use disorder, Rhode Island (TITAN-RI)

Assembled 2026-10-21 from the archived primary analysis (2,000 simulated people, strategies from January 2027,
horizon 2027–2036; `calibration_archive/phase7_chain4_2026-10-13/strategy_only_2027/`). Every number in the text,
tables and figures comes from that archive or its `sensitivities/` folder; the 500-agent round is the scale check.

## What is where
- `01_manuscript/manuscript_numbered.md` — full draft (abstract, introduction, methods, results, discussion) with
  numbered citations; `references.md` is the matching numbered list (Vancouver style; up to six authors then et al.).
  `manuscript_with_citation_keys.md` is the same text with the citation keys visible (`[@PMID]`, `[@NP:key]`) for
  editing; re-run `analysis/build_references.py` after changes. `table1.md/csv` is Table 1; `figure_captions.md`
  holds the legends for every figure.
- `02_figures/` — PNG (300 dpi) and PDF. Main text: fig1 (model and strategies), fig2 (status quo 2012–2036),
  fig3 (effects), fig6 (attribution, intended as Figure 4). Supplement: fig4 (chain), fig5 (sensitivity),
  fig7 (cash flow), fig8 (projection against 2023–2025).
- `03_tables/` — every table as CSV and Markdown: effects, chain, cash flow, sensitivity, attribution, arms
  (strategy specifications with sources = Table S2), unit costs (S3), members (calibrated settings, S6 in the
  old numbering), parameters (S1, 178 inputs with source and status), status quo by year.
- `04_supplement/` — validation table (ISPOR–SMDM types), AdViSHE checklist, ODD outline and the reported-checks
  table (`supplement_validation_odd_*.md`); the model validation notes; unit-cost file; the 500-agent scale-check
  figures and tables.
- `05_references/` — PubMed metadata for the cited sources (full author lists), the project-wide PMID inventory,
  the non-PubMed references (RIDOH, CDC, APCD, TITAN repository, ODD) with items marked "to confirm" for URL and
  access date, and the non-PubMed source inventory.
- `06_notes/` — two data questions for the claims analyst; the materials map; the superseded development-set
  results package (2023–2031 at 500 agents; do not quote); the pre-markup draft.

## Status and open items
- Results, figures and tables are final for the primary analysis; the draft is readable but not yet trimmed to a
  journal word limit (main text about 6,500 words) and has no title page, author list or funding statement.
- Citations: 60-odd sources are keyed in the text; some Methods statements rely on the parameter table (S1) for
  their sources rather than an inline citation — writers may promote those as the journal requires.
- Non-PubMed references need URLs and access dates confirmed (`non_pubmed_references.yml`).
- Two definitional questions for the claims analyst are in `06_notes/`; neither changes a result.
- Target journal not yet chosen (drafted with JAIDS in mind; AJPH or Clinical Infectious Diseases are alternatives).
- The all-payer claims extracts are not in this package and must not be distributed.
EOR
( cd "$(dirname "$OUT")" && rm -f "$(basename "$OUT").zip" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")" )
echo "package: $OUT ($(find "$OUT" -type f | wc -l) files); zip: ${OUT}.zip ($(du -h "${OUT}.zip" | cut -f1))"
