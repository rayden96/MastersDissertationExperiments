"""
Build the experiment ledger from experiments.json.

Renders two views that never drift (same source):
  experiment_ledger.csv   — spreadsheet (open in Excel / Google Sheets)
  EXPERIMENT_LEDGER.md    — GitHub-viewable table, grouped by chapter

`experiments.json` is the single source of truth — edit THAT (status, section,
figure, headline); this script only renders. When the master tables are
reachable (Drive on Colab, repo-local, or --results-root PATH), each axis row's
Status is upgraded pending->done if it appears, and a `live` column is filled
with the best speed-up / delta from the master table. So a refresh after a Colab
batch is one cell, not 35 hand edits.

Run:
    python build_ledger.py
    python build_ledger.py --results-root /content/drive/MyDrive/dissertation/results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent                 # PaperReadyExperiments
_REPO = _PRE.parent

COLUMNS = ["id", "chapter", "section", "title", "scope", "status", "priority",
           "live", "headline", "figure", "artifacts", "notes"]
STATUS_ICON = {"done": "[done]", "partial": "[partial]", "pending": "[ ]", "needs-rerun": "[rerun]"}


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _esc(s):
    return str(s).replace("|", "\\|")


def _find_master_tables(results_root):
    """{'bograd': rows, 'cosgd': rows} from whatever master tables are reachable
    (results_root arg, get_results_root() Drive, then repo-local)."""
    roots = []
    if results_root:
        roots.append(Path(results_root))
    try:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from common.storage import get_results_root  # noqa: E402
        roots.append(Path(get_results_root()))
    except Exception:
        pass
    out = {"bograd": [], "cosgd": []}
    rels = {"bograd": "10_bograd_ablation/09_cross_summary/master_table.json",
            "cosgd": "20_cosgd_ablation/08_cross_summary/master_table.json"}
    for key, rel in rels.items():
        for p in [r / rel for r in roots] + [_PRE / rel]:
            try:
                if p.exists():
                    out[key] = _load(p).get("rows", [])
                    break
            except Exception:
                pass
    return out


def _axis(folder):
    return Path(folder).name  # '20_cosgd_ablation/05_combine' -> '05_combine'


def _live_for(axis, tables):
    """(live_status, live_string) for an axis folder across the master tables."""
    for key in ("bograd", "cosgd"):
        rows = [r for r in tables[key] if r.get("folder") == axis]
        if not rows:
            continue
        sp = [r.get("epoch_speedup") for r in rows if isinstance(r.get("epoch_speedup"), (int, float))]
        dkey = "delta_vs_baseline" if key == "bograd" else "delta_acc"
        d = [r.get(dkey) for r in rows if isinstance(r.get(dkey), (int, float))]
        parts = []
        if sp:
            parts.append(f"spd<={max(sp):.2f}x")
        if d:
            parts.append(f"d-acc {min(d):+.3f}..{max(d):+.3f}")
        return ("done", "; ".join(parts) if parts else "rows present")
    return (None, "")


def render(data, tables):
    rows = []
    for e in data["experiments"]:
        live_status, live = _live_for(_axis(e["folder"]), tables)
        status = e.get("status", "pending")
        if live_status == "done" and status == "pending":
            status = "done"
        rows.append({
            "id": e["id"], "chapter": e.get("chapter", ""),
            "section": f'{e.get("section_num", "")} {e.get("section_label", "")}'.strip(),
            "title": e.get("title", ""), "scope": e.get("scope", ""),
            "status": status, "priority": e.get("priority", ""), "live": live,
            "headline": e.get("headline", ""), "figure": e.get("figure", ""),
            "artifacts": " | ".join(e.get("artifacts", [])), "notes": e.get("notes", ""),
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_md(rows, data, path):
    lines = ["# Experiment ledger", "",
             "Generated from `experiments.json` by `build_ledger.py` — **edit the JSON, not this file.**", "",
             "Status: " + "  ".join(f"`{v}` {k}" for k, v in STATUS_ICON.items()), ""]
    by_chapter = {}
    for r in rows:
        by_chapter.setdefault(r["chapter"], []).append(r)
    chap_titles = data.get("meta", {}).get("chapters", {})
    for chap in sorted(by_chapter):
        lines += [f"## {chap} — {chap_titles.get(chap, '')}".rstrip(" —"), "",
                  "| ID | Section | Title | Scope | Status | Live | Headline | Fig | Artifacts |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in by_chapter[chap]:
            lines.append("| {id} | {sec} | {title} | {scope} | {icon} | {live} | {head} | {fig} | {art} |".format(
                id=r["id"], sec=_esc(r["section"]), title=_esc(r["title"]), scope=_esc(r["scope"]),
                icon=STATUS_ICON.get(r["status"], r["status"]), live=_esc(r["live"]),
                head=_esc(r["headline"]), fig=_esc(r["figure"]) or "—", art=_esc(r["artifacts"])))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=None, help="dir holding the {10,20}_*/0*_cross_summary master tables")
    args = ap.parse_args()
    data = _load(_HERE / "experiments.json")
    tables = _find_master_tables(args.results_root)
    rows = render(data, tables)
    write_csv(rows, _HERE / "experiment_ledger.csv")
    write_md(rows, data, _HERE / "EXPERIMENT_LEDGER.md")
    c = Counter(r["status"] for r in rows)
    print(f"ledger: {len(rows)} experiments  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"enriched from master tables: {sum(1 for r in rows if r['live'])} rows "
          f"(bograd rows={len(tables['bograd'])}, cosgd rows={len(tables['cosgd'])})")
    print(f"wrote {_HERE / 'experiment_ledger.csv'}\nwrote {_HERE / 'EXPERIMENT_LEDGER.md'}")


if __name__ == "__main__":
    main()
