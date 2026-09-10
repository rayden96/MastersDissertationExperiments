"""
Turn the persisted ablation records into the LaTeX tabulars the dissertation's
method chapters input.

Training and plotting are separate (docs/experiment_design.md): this script
trains nothing. It reads what the aggregators already wrote --

  20_cosgd_ablation/08_cross_summary/{speedup_cells,master_table}.json
  10_bograd_ablation/09_cross_summary/{speedup_cells,master_table}.json
  20_cosgd_ablation/07_scalability/results/scalability_real.json

-- and emits one bare `tabular` per axis (no table environment, no caption:
the chapter's \\cosgdtable / \\bogradtable macro supplies those, and shows a
PENDING box while a file is absent).

Every axis table reads the same way: one row per ablation cell, and for each
dataset in the shared testbed the epoch speed-up, the wall-clock speed-up, and
the final test accuracy. The baseline row carries the target accuracy the
speed-ups are measured against.

Run:
    python make_thesis_tables.py                     # both chapters
    python make_thesis_tables.py --chapters cosgd    # one
    python make_thesis_tables.py --out /tmp/tables   # override destination
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent          # PaperReadyExperiments/
_REPO = _HERE.parent
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.storage import read_json, get_results_root  # noqa: E402

# The dissertation lives in a SIBLING folder outside this repo.
_DEFAULT_THESIS = _REPO.parent / "dissertation" / "chapters"

DATASETS = ["covertype", "cifar10"]
DS_LABEL = {"covertype": "Covertype", "cifar10": "CIFAR-10",
            "cifar100": "CIFAR-100", "emnist_balanced": "EMNIST-Bal."}
BASE_LABEL = {"sgd": "SGD", "signsgd": "SignSGD",
              "rmsprop": "RMSprop", "adam": "Adam"}


# --- cell label -> human-readable row name ---------------------------------
COSGD_LABELS = {
    "baseline": "baseline",
    # 20.05 combine
    "sum(paper)": "sum (uncapped)", "sum+cap2": r"sum, cap $\kappa{=}2$",
    "sum+cap3": r"sum, cap $\kappa{=}3$", "mean": "mean", "freq": "frequency",
    # 20.01 GS variant
    "gram_schmidt_normal": "classical, full",
    "gram_schmidt_negative": "classical, negative-only",
    "modified_gs_normal": "modified, full",
    "modified_gs_negative": "modified, negative-only",
    # 20.02 class order
    "order_desc": "descending", "order_asc": "ascending",
    "order_random": "random", "order_fixed": "fixed",
    # 20.03 pre-normalisation
    "prenorm0": "off", "prenorm1": "on",
    # 20.04 step method
    "step_single_forward": "single forward",
    "step_multi_forward": "forward per class",
    "step_multi_forward_with_BN": "forward per class, frozen statistics",
    # 20.06 base optimiser
    "cosgd": "COSGD",
}

BOGRAD_LABELS = {
    "baseline": "baseline", "baseline_K0": "baseline ($K{=}0$)",
    # 10.03 projection mode
    "mode_negative": "negative-only", "mode_full": "full", "mode_positive": "positive",
    # 10.04 orthogonalisation method
    "sequential_negative": "sequential, negative-only",
    "sequential_full": "sequential, full", "qr_full": "QR, full",
    "householder_full": "Householder, full",
    # 10.05 scope
    "scope_per_tensor": "per-tensor", "scope_global": "global",
    # 10.06 magnitude
    "bograd": "BOGrad", "bograd_preserve_mag": "BOGrad, magnitude preserved",
    "random_proj": "random projection",
    "random_proj_preserve_mag": "random projection, magnitude preserved",
    # 10.07 momentum
    "mu0_bogradOff": r"$\mu{=}0$, off", "mu0_bogradOn": r"$\mu{=}0$, on",
    "mu0.9_bogradOff": r"$\mu{=}0.9$, off", "mu0.9_bogradOn": r"$\mu{=}0.9$, on",
}


def _pretty(label: str, table: Dict[str, str]) -> str:
    if label in table:
        return table[label]
    m = re.fullmatch(r"K(\d+)", label)                  # 10.01 / 10.08 buffer
    if m:
        return f"$K{{=}}{m.group(1)}$"
    m = re.fullmatch(r"neg_alpha([\d.]+)", label)       # 10.03 strength
    if m:
        return rf"negative, $\alpha{{=}}{m.group(1)}$"
    m = re.fullmatch(r"(baseline|bograd)_?K?(\d*)_?lr([\d.e-]+)", label)  # 10.02
    if m:
        who = "baseline" if m.group(1) == "baseline" else "BOGrad"
        return rf"{who}, $\eta{{=}}{m.group(3)}$"
    return label.replace("_", r"\_")


def _sort_key(label: str):
    """Baseline rows first, then natural order (K2 before K128)."""
    is_base = 0 if ("baseline" in label.lower()
                    or "bogradoff" in label.lower().replace("_", "")) else 1
    nums = [int(n) for n in re.findall(r"\d+", label)]
    return (is_base, nums or [0], label)


# --- formatting ------------------------------------------------------------
def _spd(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "--"


def _acc(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "--"


def _axis_table(cells: List[dict], master: List[dict], folder: str,
                datasets: Sequence[str], labels: Dict[str, str]) -> Optional[str]:
    """One bare tabular for an axis: rows = cells, per dataset (epoch, wall, acc)."""
    rows = [c for c in cells if c.get("folder") == folder
            and c.get("dataset") in datasets]
    if not rows:
        return None
    ds_present = [d for d in datasets if any(r["dataset"] == d for r in rows)]
    bases = sorted({r["base"] for r in rows},
                   key=lambda b: list(BASE_LABEL).index(b) if b in BASE_LABEL else 99)

    # baseline accuracy per (base, dataset), from the master table
    base_acc = {(m["base"], m["dataset"]): m.get("baseline_acc")
                for m in master if m.get("folder") == folder}

    ncol = 1 + 3 * len(ds_present)
    out = [r"\begin{tabular}{l" + "rrr" * len(ds_present) + "}", r"\hline"]
    hdr = [""] + [rf"\multicolumn{{3}}{{c}}{{{DS_LABEL.get(d, d)}}}" for d in ds_present]
    out.append(" & ".join(hdr) + r" \\")
    out.append("".join(rf"\cline{{{2+3*i}-{4+3*i}}}" for i in range(len(ds_present))))
    out.append("cell & " + " & ".join(["epoch & wall & acc"] * len(ds_present)) + r" \\")
    out.append(r"\hline")

    for base in bases:
        if len(bases) > 1:
            out.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{BASE_LABEL.get(base, base)}}}}} \\")
        brows = [r for r in rows if r["base"] == base]
        cell_names = sorted({r["cell"] for r in brows}, key=_sort_key)

        # the baseline reference row (never in speedup_cells: it IS the reference)
        cols = []
        for d in ds_present:
            a = base_acc.get((base, d))
            cols += ["ref", "ref", _acc(a)]
        if any(c != "--" for c in cols[2::3]):
            out.append(" & ".join([_pretty("baseline", labels)] + cols) + r" \\")

        for cell in cell_names:
            cols = []
            for d in ds_present:
                r = next((x for x in brows if x["cell"] == cell and x["dataset"] == d), None)
                if r is None:
                    cols += ["--", "--", "--"]
                else:
                    cols += [_spd(r.get("epoch_speedup")), _spd(r.get("wall_speedup")),
                             _acc(r.get("final_test_acc"))]
            out.append(" & ".join([_pretty(cell, labels)] + cols) + r" \\")
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out)


def _master_table(master: List[dict], datasets: Sequence[str],
                  interference_key: str, interference_head: str) -> Optional[str]:
    """The cross-axis summary: best cell per (axis, base, dataset) with its
    speed-up and the interference index the method targets."""
    rows = [m for m in master if m.get("dataset") in datasets]
    if not rows:
        return None
    out = [r"\begin{tabular}{llrrrr}", r"\hline",
           r"axis & base & dataset & epoch & wall & " + interference_head + r" \\",
           r"\hline"]
    for m in sorted(rows, key=lambda r: (r.get("folder", ""), r.get("dataset", ""),
                                         r.get("base", ""))):
        out.append(" & ".join([
            str(m.get("axis", "")).replace("_", r"\_"),
            BASE_LABEL.get(m.get("base"), str(m.get("base"))),
            DS_LABEL.get(m.get("dataset"), str(m.get("dataset"))),
            _spd(m.get("epoch_speedup")), _spd(m.get("wall_speedup")),
            _acc(m.get(interference_key)),
        ]) + r" \\")
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out)


def _timing_table(rows: List[dict], datasets: Sequence[str]) -> Optional[str]:
    """30.11 per-step cost: every arm timed back-to-back in one process.

    Reported as the median over `n_timed` steps after a warmup, so these are the
    citable overhead figures; the per-cell `mean_step_wall_time_s` recorded during
    training comes from separate sessions on separate devices and is not
    comparable across arms.
    """
    present = [d for d in datasets if any(r.get("dataset") == d for r in rows)]
    if not present:
        return None
    methods = ["cosgd", "bograd", "dropout"]
    out = [r"\begin{tabular}{ll" + "r" * len(present) + "}", r"\hline",
           "base & method & " + " & ".join(DS_LABEL.get(d, d) for d in present) + r" \\",
           r"\hline"]
    for base in ("sgd", "signsgd", "rmsprop", "adam"):
        brows = [r for r in rows if r.get("base") == base]
        if not brows:
            continue
        cells = []
        for m in methods:
            vals = []
            for d in present:
                r = next((x for x in brows
                          if x.get("method") == m and x.get("dataset") == d), None)
                ov = r.get("overhead_x") if r else None
                vals.append(rf"${ov:.2f}\times$" if isinstance(ov, (int, float)) else "--")
            cells.append((m, vals))
        for i, (m, vals) in enumerate(cells):
            lead = BASE_LABEL.get(base, base) if i == 0 else ""
            out.append(f"{lead} & {m} & " + " & ".join(vals) + r" \\")
        out.append(r"\hline")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def _scalability_table(pts: List[dict]) -> Optional[str]:
    if not pts:
        return None
    out = [r"\begin{tabular}{lrrrrr}", r"\hline",
           r"dataset & classes & baseline (ms) & COSGD (ms) & overhead & "
           r"COSGD peak (MB) \\", r"\hline"]
    def ms(v):
        return f"{v*1000:.1f}" if isinstance(v, (int, float)) and v == v else "--"

    for p in sorted(pts, key=lambda q: q.get("n_classes", 0)):
        b, c, mem = (p.get("baseline_sec_per_step"), p.get("cosgd_sec_per_step"),
                     p.get("cosgd_peak_mb"))
        ov = rf"${c/b:.1f}\times$" if b and c else "--"
        mb = f"{mem:.0f}" if isinstance(mem, (int, float)) and mem == mem else "--"
        name = DS_LABEL.get(p.get("dataset"), str(p.get("dataset")))
        out.append(f"{name} & {p.get('n_classes', '--')} & "
                   f"{ms(b)} & {ms(c)} & {ov} & {mb} \\\\")
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out)


# --- sources ---------------------------------------------------------------
def _find(*rel: str) -> Optional[Path]:
    """First existing path among the persistent root and the repo-local copies."""
    for base in (get_results_root(), _HERE):
        p = base.joinpath(*rel)
        if p.exists():
            return p
    return None


def _load(rel: Sequence[str], key: str) -> List[dict]:
    p = _find(*rel)
    if p is None:
        print(f"  (missing {'/'.join(rel)})")
        return []
    try:
        return read_json(p).get(key, [])
    except Exception as e:
        print(f"  (unreadable {p}: {e})")
        return []


AXES_COSGD = [
    ("05_combine", "combine_speedup.tex"),
    ("01_gs_variant", "gs_variant_speedup.tex"),
    ("02_class_order", "class_order_speedup.tex"),
    ("03_prenormalize", "prenorm_speedup.tex"),
    ("04_step_method", "step_method_speedup.tex"),
    ("06_base_optimizer", "base_optimizer_speedup.tex"),
]

AXES_BOGRAD = [
    ("01_buffer_K", "buffer_K_speedup.tex"),
    ("02_lr_retune", "lr_retune_speedup.tex"),
    ("03_projection_mode", "mode_speedup.tex"),
    ("04_orth_method", "orth_speedup.tex"),
    ("05_projection_scope", "scope_speedup.tex"),
    ("06_magnitude", "magnitude_speedup.tex"),
    ("07_momentum_2x2", "momentum_speedup.tex"),
    ("08_batch_K", "batch_K_speedup.tex"),
]


def build_chapter(which: str, out_dir: Path, datasets: Sequence[str]) -> int:
    if which == "cosgd":
        rel = ("20_cosgd_ablation", "08_cross_summary")
        axes, labels = AXES_COSGD, COSGD_LABELS
        ikey, ihead = "I_inter", r"$I_{\text{inter}}$"
    else:
        rel = ("10_bograd_ablation", "09_cross_summary")
        axes, labels = AXES_BOGRAD, BOGRAD_LABELS
        ikey, ihead = "I_between_K32", r"$I_{\text{between}}$"

    cells = _load((*rel, "speedup_cells.json"), "cells")
    master = _load((*rel, "master_table.json"), "rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for folder, fname in axes:
        tex = _axis_table(cells, master, folder, datasets, labels)
        if tex is None:
            print(f"  - {fname:<28} no cells yet (PENDING box stays)")
            continue
        (out_dir / fname).write_text(tex + "\n", encoding="utf-8")
        print(f"  + {fname}")
        written += 1

    tex = _master_table(master, datasets, ikey, ihead)
    if tex:
        (out_dir / "master_speedup.tex").write_text(tex + "\n", encoding="utf-8")
        print("  + master_speedup.tex")
        written += 1

    if which == "cosgd":
        # Two layouts: persistent_dir() writes <root>/20_cosgd_ablation/07_scalability/,
        # while the repo-local copy sits under .../07_scalability/results/.
        p = (_find("20_cosgd_ablation", "07_scalability", "scalability_real.json")
             or _find("20_cosgd_ablation", "07_scalability", "results", "scalability_real.json")
             or _HERE / "20_cosgd_ablation" / "07_scalability" / "results" / "scalability_real.json")
        if p.exists():
            try:
                tex = _scalability_table(read_json(p).get("points", []))
            except Exception:
                tex = None
            if tex:
                (out_dir / "scalability_real.tex").write_text(tex + "\n", encoding="utf-8")
                print("  + scalability_real.tex")
                written += 1
        else:
            print("  - scalability_real.tex        no timing run yet (PENDING box stays)")
    else:
        # 30.11 timing: the citable per-step overhead, all arms measured
        # back-to-back on one device.
        p = _find("30_main_comparison", "timing", "timing.json")
        if p is not None:
            try:
                td = read_json(p)
                tex = _timing_table(td.get("rows", []),
                                    list(datasets) + ["cifar100"])
            except Exception as e:
                tex = None
                print(f"  ! timing.json unreadable: {e}")
            if tex:
                (out_dir / "overhead_measured.tex").write_text(tex, encoding="utf-8")
                print("  + overhead_measured.tex")
                written += 1
        else:
            print("  - overhead_measured.tex      no timing run yet (PENDING box stays)")

        # 10.10 already emits a finished tabular; carry it into the chapter so
        # one command fills every table the chapter inputs.
        p = _find("10_bograd_ablation", "10_scale_transfer", "scale_transfer.tex")
        if p is None:
            p = _HERE / "10_bograd_ablation" / "10_scale_transfer" / "scale_transfer.tex"
        if p.exists():
            # CIFAR-100 transfer is reported in the experiments chapter, not
            # the ablation chapter, so this one lands next door.
            exp_dir = out_dir.parent.parent / "experiments" / "tables"
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "scale_transfer.tex").write_text(
                p.read_text(encoding="utf-8"), encoding="utf-8")
            print("  + scale_transfer.tex")
            written += 1
        else:
            print("  - scale_transfer.tex          run 10_scale_transfer first "
                  "(PENDING box stays)")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", nargs="+", default=["cosgd", "bograd"],
                    choices=["cosgd", "bograd"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", default=None,
                    help="destination root (default: the sibling dissertation "
                         "chapters/ folder). Tables always land in "
                         "<root>/<chapter>/tables, so the two chapters never "
                         "collide on a shared filename such as master_speedup.tex")
    args = ap.parse_args()

    root = Path(args.out) if args.out else _DEFAULT_THESIS
    total = 0
    for which in args.chapters:
        out_dir = root / which / "tables"
        print(f"\n[{which}] -> {out_dir}")
        total += build_chapter(which, out_dir, args.datasets)
    print(f"\nwrote {total} table(s). Rebuild the thesis to pick them up.")


if __name__ == "__main__":
    main()
