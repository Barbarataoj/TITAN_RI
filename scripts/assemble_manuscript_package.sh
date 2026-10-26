#!/bin/bash
# Assemble the manuscript package for the writing team into one folder (and a zip).
# Usage: scripts/assemble_manuscript_package.sh [out_dir]   (from the repo, titan_env active)
# Nothing from docs/apcd_incoming/ is copied (unsuppressed small cells); the package carries only
# archived, aggregated results.
set -euo pipefail
cd /oscar/data/jtao9/TITAN_RI
OUT=${1:-docs/manuscript/package_$(date +%F)}; rm -rf "$OUT"; mkdir -p "$OUT"/{01_manuscript,02_figures,03_tables,04_supplement,05_references,06_notes}
M=docs/manuscript
# 1. manuscript: numbered text + reference list
python analysis/build_references.py $M/draft_2026-10-21_v4.md "$OUT/01_manuscript/manuscript_numbered.md" "$OUT/01_manuscript/references.md"
cp $M/draft_2026-10-21_v4.md "$OUT/01_manuscript/manuscript_with_citation_keys.md"
cp $M/tables/table1_main.md "$OUT/01_manuscript/table1.md"; cp $M/tables/table1_main.csv "$OUT/01_manuscript/table1.csv"
cp $M/figures/CAPTIONS.md "$OUT/01_manuscript/figure_captions.md"
strip_comments() { python - "$1" <<'EOP'
import re, sys
p = sys.argv[1]; t = open(p).read(); open(p, "w").write(re.sub(r"<!--.*?-->\s*", "", t, flags=re.S))
EOP
}
strip_comments "$OUT/01_manuscript/manuscript_numbered.md"; strip_comments "$OUT/01_manuscript/figure_captions.md"
# 2. figures (main: 1, 2, 3, 6->4; supplement: 4, 5, 7, 8)
for f in fig1_model_schematic fig2_status_quo fig3_effects fig4_chain fig5_sensitivity fig6_attribution fig7_cashflow fig8_forward_check figS9_model_modifications figS10_sdh_insurance_effects figS11_on_vs_off; do
  for ext in png pdf; do [ -f $M/figures/$f.$ext ] && cp $M/figures/$f.$ext "$OUT/02_figures/"; done; done
# 3. tables (csv + md)
cp $M/tables/table_*.csv $M/tables/table_*.md "$OUT/03_tables/"
# 4. supplement
cp $M/supplement_validation_odd_2026-10-21.md "$OUT/04_supplement/" 2>/dev/null || cp $M/supplement_validation_odd_2026-10-20.md "$OUT/04_supplement/"
cp $M/supplementary_methods_2026-10-21.md "$OUT/04_supplement/supplementary_methods.md"; strip_comments "$OUT/04_supplement/supplementary_methods.md"
cp $M/supplementary_table_s0_additions_2026-10-21.md "$OUT/04_supplement/supplementary_table_s0_additions.md"
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
cp $M/audit_figures_tables_methods_2026-10-21.md "$OUT/06_notes/"
cp $M/MATERIALS.md "$OUT/06_notes/materials_map.md"
cp $M/results_package_2026-10-14.md "$OUT/06_notes/results_package_dev_set_2026-10-14_superseded.md"
cp $M/draft_2026-10-16_v2.md "$OUT/06_notes/draft_v2_before_citation_markup.md"
cp $M/draft_2026-10-21_v3.md "$OUT/06_notes/draft_v3_before_readability_pass.md"
cat > "$OUT/00_README.md" <<'EOR'
# Manuscript package — HIV prevention strategies in primary care for people with opioid use disorder, Rhode Island (TITAN-RI)

Assembled ASSEMBLED_DATE. The Methods, Supplementary Methods and figure captions are current for the reported model:
the insurance and social-determinant (SDH) layers, their fit check with the layers on and off (Supplementary Figure
S11), and the one-to-one simulation design. The strategy results — the Results text, Figures 2–8 and the result
tables — still come from the 2,000-person primary analysis of 2026-10-20 (`calibration_archive/phase7_chain4_2026-10-13/
strategy_only_2027/`), run before the two layers were added. The same analysis on the reported model (primary round,
one-to-one round, fit checks and eight sensitivity rounds; `round2027_2026-10-25/`) is running and will replace every
result number, figure and table within about two days. Use this package to draft the framing, Introduction, Methods and
the structure of the Discussion; do not finalise any number yet.

## What is where
- `01_manuscript/manuscript_numbered.md` (and `.docx`) — the manuscript to read: abstract through conclusion, with
  numbered citations and the internal notes removed; `references.md` (`.docx`) is the matching numbered list (Vancouver
  style; up to six authors then et al.). `manuscript_with_citation_keys.md` is the editing master with the citation keys
  visible (`[@PMID]`, `[@NP:key]`) and the authors' internal notes as HTML comments; re-run `analysis/build_references.py`
  after changes. `table1.md/csv` is Table 1; `figure_captions.md` (`.docx`) holds the legends for every figure.
- `02_figures/` — PNG (300 dpi) and PDF. Main text: fig1 (model and strategies), fig2 (status quo 2012–2036 and
  cumulative infections averted and extra PrEP starts by strategy), fig3 (effects), fig6 (attribution, intended as
  Figure 4). Supplement: fig4 (chain), fig5 (sensitivity), fig7 (cash flow), fig8 (projection against 2023–2025),
  figS9 (model modules by origin), figS10 (insurance and SDH effects on transitions), figS11 (fit with and without the
  layers).
- `03_tables/` — every table as CSV and Markdown: effects, chain, cash flow, sensitivity, attribution, horizons,
  cumulative by arm, arms (strategy specifications with sources = Table S2), unit costs (S3), members (calibrated
  settings = S6), parameters (S1, inputs with source and status), status quo by year.
- `04_supplement/` — **Supplementary Methods** (`supplementary_methods.md`, `.docx`): engine, each added component with
  its values, data and scale, calibration objectives and gates, ensemble design and estimator, strategies, cost ledgers,
  sensitivities, validation (including the 500-person and one-to-one scale checks), software and archive paths;
  `supplementary_table_s0_additions.md`; the validation table, AdViSHE checklist, ODD outline and reported checks
  (`supplement_validation_odd_*.md`); the model validation notes; unit-cost file; the 500-person scale-check figures and
  tables. Supplementary table numbering: S0 what was added, S1 parameters, S2 strategy levers, S3 unit costs, S4 chain,
  S5 sensitivity, S6 calibrated settings, S7 cash flow, S8 validation by type, S9 AdViSHE, S10 reported checks.
- `05_references/` — PubMed metadata for the cited sources, the project-wide PMID inventory, the non-PubMed references
  (RIDOH, CDC, APCD, TITAN repository, ODD; URL and access date to confirm) and the non-PubMed source inventory.
- `06_notes/` — the figure/table/Methods audit of 2026-10-21; the data questions for the claims analyst
  (`questions_for_mofan_2026-10-21.md`, three definitional points, no new extracts needed); the materials map; the
  superseded development-set results package (do not quote); the two earlier drafts.

## Status and open items
- Methods: current. Results: provisional (see above); the reported model's results replace them within about two days.
- Pending in the text: the one-to-one result (placeholder in "How robust the findings are" and Supplementary Methods
  S9); the cohort size and scale sentences (14,700 and "about seven" become 14,440 and 7.2 with the reported insured
  share 0.949); the overdose fit check with the layers on reads "not shown" (1.6 % higher, within tolerance) pending the
  senior author's decision between reporting it so and recentring the rate.
- Not yet done: trimming to a journal word limit (main text about 8,000 words), title page, author list, funding and
  data-use statements, target journal (drafted with JAIDS in mind; AJPH or Clinical Infectious Diseases are
  alternatives). Non-PubMed references need URLs and access dates confirmed.
- The all-payer claims extracts are not in this package and must not be distributed; no cell below the claims'
  suppression threshold appears in any table.
EOR
sed -i "s/ASSEMBLED_DATE/$(date +%F)/" "$OUT/00_README.md"
if module load pandoc/2.19.2-b5wd 2>/dev/null; then
  for f in "$OUT/01_manuscript/manuscript_numbered.md" "$OUT/01_manuscript/references.md" "$OUT/01_manuscript/figure_captions.md" "$OUT/01_manuscript/table1.md" "$OUT/04_supplement/supplementary_methods.md"; do pandoc "$f" -o "${f%.md}.docx" 2>>"$OUT/06_notes/pandoc.log" || echo "pandoc failed on $f"; done
else echo "pandoc module not available; no .docx produced"; fi
( cd "$(dirname "$OUT")" && rm -f "$(basename "$OUT").zip" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")" )
echo "package: $OUT ($(find "$OUT" -type f | wc -l) files); zip: ${OUT}.zip ($(du -h "${OUT}.zip" | cut -f1))"
