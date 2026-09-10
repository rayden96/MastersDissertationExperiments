"""
BOGrad's per-step cost against buffer size K, on the three models of the
comparison chapter, as the tabular for Chapter 5's overhead table.

Reads the JSON written by

    python timing.py --datasets covertype cifar10 cifar100 --bases sgd \
        --methods baseline bograd --bograd_K 8 32 128 --out timing_K.json

and writes only the tabular, so the chapter keeps its own caption and label.
One column pair per model, headed by the model and its parameter count, since
the point of the table is how the projection's cost compares with the
network's own step as the network grows.

    python timing_table.py
    python timing_table.py --json path/to/timing_K.json --out path/to/overhead.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

MODELS = [("covertype", "Covertype MLP"), ("cifar10", "CIFAR-10 CNN"),
          ("cifar100", "CIFAR-100 ResNet-18")]
BASE_LABEL = {"sgd": "SGD", "signsgd": "SignSGD", "rmsprop": "RMSprop", "adam": "Adam"}


def _params(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def _ms(seconds: float) -> str:
    ms = seconds * 1000
    return f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}"


def build(data: dict) -> str:
    rows = data.get("rows", [])
    failed = data.get("failed", [])
    present = [(d, name) for d, name in MODELS
               if any(r.get("dataset") == d for r in rows)]
    Ks = sorted({r["hp"]["K"] for r in rows if r.get("method") == "bograd"}
                | {f["K"] for f in failed if f.get("K")})
    base = next((r["base"] for r in rows), "sgd")

    def find(ds: str, method: str, K: Optional[int] = None) -> Optional[dict]:
        return next((r for r in rows
                     if r.get("dataset") == ds and r.get("method") == method
                     and (K is None or r.get("hp", {}).get("K") == K)), None)

    n = len(present)
    out = [r"\small", r"\begin{tabular}{l" + "rr" * n + "}", r"\hline"]
    out.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}"
                                  for _, name in present) + r" \\")
    sizes = []
    for ds, _ in present:
        r = find(ds, "baseline")
        label = f"{_params(r['n_params'])} parameters" if r and r.get("n_params") else ""
        sizes.append(rf"\multicolumn{{2}}{{c}}{{{label}}}")
    out.append(" & " + " & ".join(sizes) + r" \\")
    out.append("".join(rf"\cline{{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n)))
    out.append("config" + " & ms/step & overhead" * n + r" \\")
    out.append(r"\hline")

    vals = []
    for ds, _ in present:
        r = find(ds, "baseline")
        vals += [_ms(r["median_step_s"]) if r else "n/a", "---"]
    out.append(f"baseline ({BASE_LABEL.get(base, base)}) & " + " & ".join(vals) + r" \\")
    for K in Ks:
        vals = []
        for ds, _ in present:
            r = find(ds, "bograd", K)
            if r is None or r.get("overhead_x") is None:
                # a buffer that did not fit on the device, recorded by timing.py
                vals += ["n/a", "n/a"]
            else:
                vals += [_ms(r["median_step_s"]),
                         rf"$+{(r['overhead_x'] - 1) * 100:.0f}\%$"]
        out.append(rf"$K{{=}}{K}$ & " + " & ".join(vals) + r" \\")
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(_HERE / "timing_K.json"))
    ap.add_argument("--out", default=str(_REPO.parent / "dissertation" / "chapters"
                                         / "bograd" / "tables" / "overhead.tex"))
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    tex = build(data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(tex, encoding="utf-8")
    print(f"timed on: {data.get('gpu') or data.get('device')}, "
          f"median of {data.get('n_steps')} steps after a full buffer\n")
    print(tex)
    for f in data.get("failed", []):
        print(f"  n/a  {f['dataset']} K={f['K']}: {f['error']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
