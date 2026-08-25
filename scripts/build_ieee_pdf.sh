#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_md="$repo_root/docs/PAPER_Faithfulness_Decoupling.md"
style_css="$repo_root/docs/ieee-conference.css"
work_dir="$repo_root/tmp/pdfs/ieee-paper"
output_dir="$repo_root/output/pdf"
output_pdf="$output_dir/PAPER_Faithfulness_Decoupling_IEEE.pdf"
chrome_bin="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

mkdir -p "$work_dir" "$output_dir"

pandoc "$source_md" \
  --from markdown+fenced_divs+tex_math_dollars \
  --to html5 \
  --standalone \
  --embed-resources \
  --mathml \
  --css "$style_css" \
  --resource-path "$repo_root/docs:$repo_root" \
  --metadata title="" \
  --output "$work_dir/paper.html"

"$chrome_bin" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$output_pdf" \
  "file://$work_dir/paper.html"

printf '%s\n' "$output_pdf"
