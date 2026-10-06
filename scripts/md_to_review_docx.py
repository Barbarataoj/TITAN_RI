#!/usr/bin/env python
"""
Turn a review sheet written in Markdown (docs/manuscript/*_review_*.md) into a
Word file for reviewers without a Claude account.

pandoc does the conversion; the package XML is then adjusted so the tables
print well: borders, fixed column widths that fill the text width, a shaded
header row, US Letter with 0.75-inch margins, Calibri 11 pt. The repo-only note
between the title and the first `## ` section is dropped, and "PMID nnnnnnnn"
becomes a PubMed link.

Needs pandoc (on OSCAR: `module load pandoc`). The output is checked for
well-formed XML; open it once in Word before sending (no renderer on OSCAR).

Usage:
    python scripts/md_to_review_docx.py <sheet.md> <out.docx> \
        --title "..." --subtitle "..." --date "6 October 2026"
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile
import xml.dom.minidom
import zipfile
from pathlib import Path

PAGE_W, MARGIN = 12240, 1080          # twips: US Letter, 0.75 in
TEXT_W = PAGE_W - 2 * MARGIN
COL_SHARES = {3: (46, 32, 22), 4: (26, 26, 26, 22)}
HEADER_FILL = "E8EEF4"


def prepare_markdown(src, title, subtitle, date):
    body = "## " + src.split("\n## ", 1)[1]
    out = ["---", f'title: "{title}"', f'subtitle: "{subtitle}"', f'date: "{date}"', "---", ""]
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*-{3,}", lines[i + 1]):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("| " + " | ".join(f"**{c}**" for c in cells) + " |")
            out.append("|" + "|".join("-" * w for w in COL_SHARES[len(cells)]) + "|")
            i += 2
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    return re.sub(r"PMID\s+(\d{8})",
                  lambda m: f"[PMID {m.group(1)}](https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}/)", text)


def fix_tables(doc):
    borders = "<w:tblBorders>" + "".join(
        f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="8C8C8C"/>'
        for s in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>"
    margins = ('<w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
               '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar>')

    def one_table(m):
        t = m.group(0)
        grid = [int(x) for x in re.findall(r'<w:gridCol w:w="(\d+)" ?/>', t)]
        widths = [round(g * TEXT_W / sum(grid)) for g in grid]
        widths[-1] += TEXT_W - sum(widths)
        t = re.sub(r"<w:tblW [^>]*/>",
                   f'<w:tblW w:w="{TEXT_W}" w:type="dxa"/>{borders}<w:tblLayout w:type="fixed"/>{margins}',
                   t, count=1)
        t = re.sub(r"<w:tblGrid>.*?</w:tblGrid>",
                   "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>",
                   t, count=1, flags=re.S)

        def one_row(rm):
            row = rm.group(0)
            fill = f'<w:shd w:val="clear" w:color="auto" w:fill="{HEADER_FILL}"/>' if "<w:tblHeader" in row else ""
            col = iter(widths)
            return re.sub(r"<w:tcPr ?/>",
                          lambda _: f'<w:tcPr><w:tcW w:w="{next(col)}" w:type="dxa"/>{fill}</w:tcPr>', row)

        return re.sub(r"<w:tr>.*?</w:tr>", one_row, t, flags=re.S)

    doc, n = re.subn(r"<w:tbl>.*?</w:tbl>", one_table, doc, flags=re.S)
    sect = (f'<w:sectPr><w:pgSz w:w="{PAGE_W}" w:h="15840"/><w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" '
            f'w:bottom="{MARGIN}" w:left="{MARGIN}" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>')
    assert doc.count("<w:sectPr />") == 1
    return doc.replace("<w:sectPr />", sect), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.md").write_text(prepare_markdown(Path(args.src).read_text(), args.title, args.subtitle, args.date))
        subprocess.run(["pandoc", str(tmp / "in.md"), "-f", "markdown", "-t", "docx", "-o", str(tmp / "raw.docx")],
                       check=True)
        raw = tmp / "raw"
        with zipfile.ZipFile(tmp / "raw.docx") as z:
            z.extractall(raw)
        doc_p = raw / "word" / "document.xml"
        doc, n_tables = fix_tables(doc_p.read_text(encoding="utf8"))
        doc_p.write_text(doc, encoding="utf8")
        theme_p = raw / "word" / "theme" / "theme1.xml"
        theme = theme_p.read_text(encoding="utf8")
        theme_p.write_text(theme.replace('<a:minorFont><a:latin typeface="Cambria" />',
                                         '<a:minorFont><a:latin typeface="Calibri" />'), encoding="utf8")
        styles_p = raw / "word" / "styles.xml"
        styles = styles_p.read_text(encoding="utf8")
        styles_p.write_text(re.sub(r'(<w:rPrDefault>.*?<w:sz w:val=")24(" />\s*<w:szCs w:val=")24(" />)',
                                   r"\g<1>22\g<2>22\g<3>", styles, count=1, flags=re.S), encoding="utf8")
        names = sorted((os.path.relpath(os.path.join(r, f), raw) for r, _, fs in os.walk(raw) for f in fs),
                       key=lambda x: (x != "[Content_Types].xml", x))
        out = Path(args.out)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name in names:
                z.write(raw / name, name)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith((".xml", ".rels")):
                xml.dom.minidom.parseString(z.read(name))
    print(f"wrote {out} ({out.stat().st_size} bytes, {n_tables} tables, XML well-formed)")


if __name__ == "__main__":
    main()
