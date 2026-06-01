"""
COSGD improvement study — ROUND 2, motivated by round-1 findings:

Round-1 takeaways that shape this:
  - A (preserve_magnitude) was the only clear win + it decoupled LR (CIFAR best
    lr 0.2 -> 0.05). Adopt, and BUILD ON IT.
  - Full orthogonalisation barely helped accuracy (often ~tie/slightly worse) ->
    hypothesis: full orth OVER-CORRECTS. Test soft orth (strength a).
  - D (clustering) destroyed accuracy -> per-class granularity carries real
    signal; don't merge. (Not revisited here.)
  - COSGD reduces interference but rarely converts to accuracy (F5 echo) ->
    hypothesis: it orthogonalises even when classes don't conflict. Test a
    conflict GATE (only orthogonalise when mean pairwise cos < -threshold).

Round-2 changes (all opt-in, plain COSGD unchanged):
  F orth_strength a in [0,1]   : combined = (1-a) raw + a orth   (a=1 = COSGD)
  G conflict_gate              : skip orth when classes don't conflict
  H pcgrad method              : symmetric (order-free) projection vs sequential GS
  STACK best-of               : freq + preserve_magnitude + best(a, gate, method)

Always paired against: baseline SGD, and round-1 best COSGD
(freq + preserve_magnitude, full orth). All tuned over an LR grid, 3 seeds.

Run:
    python improvements_round2.py --which F G H STACK --problem 5cls cifar10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# reuse the round-1 problem defs + harness
import importlib.util
_spec = importlib.util.spec_from_file_location("ae1", _HERE / "improvements_AE.py")
ae1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ae1)

PROBLEMS = ae1.PROBLEMS
EPOCHS = ae1.EPOCHS
LRS = ae1.LRS
SEEDS = ae1.SEEDS
_sweep_lr = ae1._sweep_lr
cosgd_factory = ae1.cosgd_factory
sgd_factory = ae1.sgd_factory

# canonical round-1 winner config (the thing to beat)
R1 = dict(orthogonalization_method="modified_gs_negative", combine="freq", preserve_magnitude=True)


def _best(make, tr, te, ep, lrs, factory, is_cosgd=True):
    _, blr, bacc = _sweep_lr(make, tr, te, ep, lrs, factory, is_cosgd=is_cosgd)
    return blr, bacc


def test_F(problem):
    make, tr, te, C = PROBLEMS[problem](); ep, lrs = EPOCHS[problem], LRS[problem]
    print(f"\n### F soft orthogonalisation (strength a) on {problem} (C={C}) ###")
    bl_curve, bl_lr, bl_acc = _sweep_lr(make, tr, te, ep, lrs, lambda lr: sgd_factory(lr), is_cosgd=False)
    print(f"  baseline SGD            best lr={bl_lr:<5} acc={bl_acc:.4f}")
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, alr, aacc = _sweep_lr(make, tr, te, ep, lrs,
            lambda lr, a=a: cosgd_factory(lr=lr, orth_strength=a, **R1))
        tag = "(=raw)" if a == 0 else ("(=full COSGD)" if a == 1 else "")
        print(f"  COSGD orth_strength={a:<4} best lr={alr:<5} acc={aacc:.4f} {tag}")
    print("  -> is the accuracy peak at a<1 (full orth over-corrects) or a=1?")


def test_G(problem):
    make, tr, te, C = PROBLEMS[problem](); ep, lrs = EPOCHS[problem], LRS[problem]
    print(f"\n### G conflict gating on {problem} (C={C}) ###")
    bl_curve, bl_lr, bl_acc = _sweep_lr(make, tr, te, ep, lrs, lambda lr: sgd_factory(lr), is_cosgd=False)
    _, ulr, uacc = _sweep_lr(make, tr, te, ep, lrs, lambda lr: cosgd_factory(lr=lr, **R1))
    print(f"  baseline SGD            best lr={bl_lr:<5} acc={bl_acc:.4f}")
    print(f"  COSGD ungated           best lr={ulr:<5} acc={uacc:.4f}")
    for thr in [0.05, 0.1, 0.2, 0.4]:
        _, glr, gacc = _sweep_lr(make, tr, te, ep, lrs,
            lambda lr, thr=thr: cosgd_factory(lr=lr, conflict_gate=True, conflict_threshold=thr, **R1))
        print(f"  COSGD gate(thr={thr:<4})     best lr={glr:<5} acc={gacc:.4f}")
    print("  -> does gating retain accuracy while orthogonalising fewer steps (cheaper)?")


def test_H(problem):
    make, tr, te, C = PROBLEMS[problem](); ep, lrs = EPOCHS[problem], LRS[problem]
    print(f"\n### H PCGrad (symmetric) vs sequential GS on {problem} (C={C}) ###")
    bl_curve, bl_lr, bl_acc = _sweep_lr(make, tr, te, ep, lrs, lambda lr: sgd_factory(lr), is_cosgd=False)
    _, slr, sacc = _sweep_lr(make, tr, te, ep, lrs, lambda lr: cosgd_factory(lr=lr, **R1))
    _, plr, pacc = _sweep_lr(make, tr, te, ep, lrs,
        lambda lr: cosgd_factory(lr=lr, orthogonalization_method="pcgrad",
                                 combine="freq", preserve_magnitude=True))
    print(f"  baseline SGD            best lr={bl_lr:<5} acc={bl_acc:.4f}")
    print(f"  COSGD seq-GS negative   best lr={slr:<5} acc={sacc:.4f}")
    print(f"  COSGD PCGrad symmetric  best lr={plr:<5} acc={pacc:.4f}")
    print("  -> does order-free symmetric projection beat sequential GS?")


def test_STACK(problem):
    make, tr, te, C = PROBLEMS[problem](); ep, lrs = EPOCHS[problem], LRS[problem]
    print(f"\n### STACK best-of-round-2 on {problem} (C={C}) ###")
    bl_curve, bl_lr, bl_acc = _sweep_lr(make, tr, te, ep, lrs, lambda lr: sgd_factory(lr), is_cosgd=False)
    blm_curve, blm_lr, blm_acc = _sweep_lr(make, tr, te, ep, lrs,
        lambda lr: sgd_factory(lr, momentum=0.9), is_cosgd=False)
    _, r1lr, r1acc = _sweep_lr(make, tr, te, ep, lrs, lambda lr: cosgd_factory(lr=lr, **R1))
    # stacked: freq + preserve + soft(0.75) + gate + (keep seq-GS negative)
    _, stlr, stacc = _sweep_lr(make, tr, te, ep, lrs,
        lambda lr: cosgd_factory(lr=lr, orthogonalization_method="modified_gs_negative",
                                 combine="freq", preserve_magnitude=True,
                                 orth_strength=0.75, conflict_gate=True, conflict_threshold=0.0))
    print(f"  baseline SGD (no mom)   best lr={bl_lr:<5} acc={bl_acc:.4f}")
    print(f"  baseline SGD+mom0.9     best lr={blm_lr:<5} acc={blm_acc:.4f}")
    print(f"  COSGD round-1 best      best lr={r1lr:<5} acc={r1acc:.4f}")
    print(f"  COSGD STACK round-2     best lr={stlr:<5} acc={stacc:.4f}")
    print(f"  -> STACK vs no-mom baseline: {stacc-bl_acc:+.4f}; vs round-1 COSGD: {stacc-r1acc:+.4f}")


TESTS = {"F": test_F, "G": test_G, "H": test_H, "STACK": test_STACK}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=list(TESTS.keys()))
    ap.add_argument("--problem", nargs="+", default=["5cls", "cifar10"])
    args = ap.parse_args()
    print(f"device={ae1.DEV}  round2={args.which}  problems={args.problem}")
    for prob in args.problem:
        print(f"\n{'='*70}\nPROBLEM: {prob}\n{'='*70}")
        for w in args.which:
            try:
                TESTS[w](prob)
            except Exception as e:
                import traceback
                print(f"  !! {w} on {prob} failed: {type(e).__name__}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
