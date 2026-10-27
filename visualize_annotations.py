#!/usr/bin/env python3
"""Create a PNG visualization from annotated_dataset.jsonl.

Requires: pip install matplotlib
Run: python visualize_annotations.py [path/to/annotated_dataset.jsonl]
"""
import json
import sys
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "annotated_dataset.jsonl"
if not source.exists():
    raise SystemExit(f"Annotated dataset not found: {source}. Run annotate_openrouter.py first.")
rows = []
for line in source.read_text(encoding="utf-8").splitlines():
    if line.strip():
        try:
            row = json.loads(line)
            if row.get("status") == "success":
                rows.append(row)
        except json.JSONDecodeError:
            pass
if not rows:
    raise SystemExit("No successful annotations found in dataset.")

def counts(field, default="unclear"):
    return Counter((r.get("annotation") or {}).get(field) or default for r in rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, field, title, ylabel in [
    (axes[0], "court_disposition", "Court disposition", "Documents"),
    (axes[1], "document_type", "Document type", "Documents"),
]:
    c = counts(field)
    labels, values = zip(*c.most_common())
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
fig.suptitle(f"OpenRouter annotations of court documents (n={len(rows)})")
fig.tight_layout()
out = source.with_name("annotation_visualization.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved visualization to {out}")
