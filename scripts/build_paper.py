"""Build the paper and run the submission checks.

Compiles ``paper/main.tex`` (tectonic if present, otherwise MiKTeX
pdflatex+bibtex), then verifies what HRFEST actually enforces:

* page count within the 6-page limit,
* no undefined references or citations,
* **double-blind hygiene**: no author name, institution, local path or
  repository URL anywhere in the produced PDF, including its metadata.

The anonymity check is the one that cannot be skipped: a leaked name is an
administrative rejection regardless of the science.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path("paper")
MAIN = "main"

# Strings that must never appear in the review PDF.  Extend as needed.
FORBIDDEN = [
    "Segundo", "Manayay", "Jose Luis", "José Luis",
    "Universidad Nacional de Ingenier", "UNI ", "Lima",
    "github.com", "gitlab", "orcid",
    "C:\\Users", "/c/Users", "Cubli_control",
    "@gmail", "@uni.edu",
]


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="ignore")


def stage_figures() -> None:
    """Copy the figures next to the manuscript so the source tree is self-contained.

    IEEE PDF eXpress at camera-ready time takes a single directory; keeping the
    figures inside ``paper/`` from the start avoids a last-minute path fix.
    """
    src, dst = Path("figures"), PAPER / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.pdf"):
        shutil.copy2(f, dst / f.name)


def compile_paper() -> bool:
    tectonic = Path(os.environ.get("TEMP", "")) / "tectonic" / "tectonic.exe"
    if tectonic.exists():
        r = run([str(tectonic), "-X", "compile", f"{MAIN}.tex"], PAPER)
        ok = (PAPER / f"{MAIN}.pdf").exists() and r.returncode == 0
        if not ok:
            print(r.stdout[-4000:]); print(r.stderr[-4000:])
        return ok

    if shutil.which("pdflatex") is None:
        print("no LaTeX toolchain found")
        return False
    # pdflatex, bibtex, then enough passes for labels *and* the bibliography to
    # settle.  Three trailing passes is one more than usually needed; the cost
    # is a few seconds and the alternative is a PDF with dangling references.
    run(["pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex"], PAPER)
    run(["bibtex", MAIN], PAPER)
    for _ in range(3):
        run(["pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex"], PAPER)
    return (PAPER / f"{MAIN}.pdf").exists()


def pdf_text(path: Path) -> str:
    """Extract text without extra dependencies (best effort)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        meta = " ".join(str(v) for v in (reader.metadata or {}).values())
        return meta + "\n" + "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        raw = path.read_bytes().decode("latin-1", errors="ignore")
        return raw


def n_pages(path: Path) -> int:
    try:
        import pypdf
        return len(pypdf.PdfReader(str(path)).pages)
    except Exception:
        raw = path.read_bytes()
        return raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")


def main() -> int:
    stage_figures()
    if not compile_paper():
        print("BUILD FAILED")
        return 1
    pdf = PAPER / f"{MAIN}.pdf"
    pages = n_pages(pdf)
    print(f"compiled {pdf}  ({pages} pages)")

    log = (PAPER / f"{MAIN}.log")
    problems = []
    overfull = 0
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="ignore")
        overfull = len(re.findall(r"Overfull \\hbox \((\d+\.\d+)pt", text))

    # Tectonic does not put undefined-ref warnings in main.log, so check the
    # sources and the rendered PDF directly instead of trusting the log.
    src = ""
    for t in sorted(PAPER.glob("*.tex")):
        src += t.read_text(encoding="utf-8", errors="ignore")
    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs = set(re.findall(r"\\(?:eq|c)?ref\{([^}]+)\}", src))
    undef_ref = sorted(refs - labels)

    bib = (PAPER / "refs.bib").read_text(encoding="utf-8", errors="ignore")
    keys = set(re.findall(r"@\s*[A-Za-z]+\s*\{\s*([^,]+),", bib))
    cited = set()
    for grp in re.findall(r"\\cite\s*\{([^}]+)\}", src):
        cited |= {k.strip() for k in grp.split(",")}
    undef_cite = sorted(cited - keys)
    unused = sorted(keys - cited)

    if undef_ref:
        problems.append(f"undefined reference(s): {undef_ref}")
    if undef_cite:
        problems.append(f"undefined citation(s): {undef_cite}")
    print(f"  undefined refs {len(undef_ref)}, undefined cites {len(undef_cite)}, "
          f"overfull hboxes {overfull}")
    if unused:
        print(f"  note: {len(unused)} bib entries never cited: {unused}")

    if pages > 6:
        problems.append(f"{pages} pages: over the 6-page limit "
                        f"({pages - 6} extra page(s) cost $50 each, max 2)")

    body = pdf_text(pdf)
    leaks = [s for s in FORBIDDEN if s.lower() in body.lower()]
    if leaks:
        problems.append(f"DOUBLE-BLIND LEAK: {leaks}")
    else:
        print("  anonymity check: clean")

    if problems:
        print("\nISSUES:")
        for p in problems:
            print(f"  - {p}")
        return 2
    print("\nready for submission")
    return 0


if __name__ == "__main__":
    sys.exit(main())
