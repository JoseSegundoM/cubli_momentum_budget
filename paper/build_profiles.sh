#!/usr/bin/env bash
# Build both PDF profiles of the paper from the same source:
#   paper_223_ieee_candidate.pdf     -- no link annotations, for IEEE PDF eXpress
#   paper_223_review_interactive.pdf -- blue DOI/ORCID/e-mail links, for review
set -euo pipefail
cd "$(dirname "$0")"
# inside the repository the figures live in ../figures; a packaged source
# already ships them in ./figures
mkdir -p figures
if [ -d ../figures ]; then cp ../figures/fig_*.pdf figures/; fi
build() {  # $1 profile, $2 output name
  local job="$2" pre="\\def\\PDFPROFILE{$1}\\input{main.tex}"
  pdflatex -interaction=nonstopmode -halt-on-error -jobname="$job" "$pre" >/dev/null
  bibtex "$job" >/dev/null
  for _ in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error -jobname="$job" "$pre" >/dev/null; done
  echo "$job.pdf: $(pdfinfo "$job.pdf" | awk '/^Pages/{print $2}') pages"
}
build ieee paper_223_ieee_candidate
build interactive paper_223_review_interactive
