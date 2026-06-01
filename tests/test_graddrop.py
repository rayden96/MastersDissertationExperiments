"""
Verification for GradDrop (sign-purity per-class gradient dropout).

    python tests/test_graddrop.py

Checks:
  1. Sign-purity combine matches a hand-computed case (deterministic edges):
     - a sign-consistent coordinate survives as the full sum,
     - a perfectly-conflicted coordinate resolves to exactly one sign (never the
       damped mix), with the mask determined by the sampled uniform vs purity.
  2. leak=1 recovers the plain per-class sum; leak=0 is standard GradDrop.
  3. Base-optimizer composition (Adam / SignSGD / SGD+momentum) runs, finite, moves.
  4. Seeded determinism via an explicit generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.optimizers import GradDrop  # noqa: E402


def _problem(seed=0, n=36, n_classes=4):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, n_classes))
    data = torch.randn(n, 8)
    labels = torch.randint(0, n_classes, (n,))
    labels[:n_classes] = torch.arange(n_classes)
    return model, data, labels


def _flat(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def test_sign_purity_math():
    model, _, _ = _problem()
    gd = GradDrop(model.parameters(), model=model, criterion=nn.CrossEntropyLoss(), lr=0.1)

    # Construct a 2-class x 3-coordinate stack with known structure:
    #  coord 0: both +  (purity=1)   -> always kept, combined = 1+3 = 4
    #  coord 1: both -  (purity=0)   -> always kept, combined = -2 + -5 = -7
    #  coord 2: +2 and -2 (purity=0.5) -> conflicted; resolves to one sign
    G = torch.tensor([[1.0, -2.0,  2.0],
                      [3.0, -5.0, -2.0]])
    counts = torch.tensor([10.0, 10.0])

    # Force the uniform draw so the conflicted coord is deterministic.
    # purity = [1.0, 0.0, 0.5]; with u = 0.4 < purity at coords {0,2}, >= at {1}:
    #   keep_pos = [True, False, True] -> coord2 keeps the +2 (class 0), drops -2.
    gd.generator = None
    torch.manual_seed(123)
    # monkey-patch rand to a fixed vector for this assertion
    real_rand = torch.rand
    try:
        torch.rand = lambda *a, **k: torch.tensor([0.4, 0.4, 0.4])
        combined = gd._combine(G, counts)
    finally:
        torch.rand = real_rand

    expected = torch.tensor([4.0, -7.0, 2.0])  # coord2 -> +2 kept, -2 dropped
    assert torch.allclose(combined, expected, atol=1e-6), \
        f"sign-purity combine wrong: {combined} != {expected}"
    print("  [1] sign-purity combine matches hand-computed case  OK")


def test_leak_extremes():
    model, _, _ = _problem()
    G = torch.randn(5, 12)
    counts = torch.ones(5)

    gd1 = GradDrop(model.parameters(), model=model, criterion=nn.CrossEntropyLoss(),
                   lr=0.1, leak=1.0)
    torch.manual_seed(0)
    combined_leak1 = gd1._combine(G, counts)
    assert torch.allclose(combined_leak1, G.sum(dim=0), atol=1e-6), \
        "leak=1 must recover the plain per-class sum"

    gd0 = GradDrop(model.parameters(), model=model, criterion=nn.CrossEntropyLoss(),
                   lr=0.1, leak=0.0)
    torch.manual_seed(0)
    combined_leak0 = gd0._combine(G, counts)
    assert not torch.allclose(combined_leak0, G.sum(dim=0)), \
        "leak=0 (standard) should differ from the plain sum on conflicted coords"
    print("  [2] leak=1 recovers plain sum; leak=0 is standard GradDrop  OK")


def test_base_composition():
    from common.optimizers import SignSGD
    for name, kwargs in [("adam", dict(base_optimizer_cls=torch.optim.Adam, lr=1e-2)),
                         ("signsgd", dict(base_optimizer_cls=SignSGD, lr=1e-2)),
                         ("sgd+mom", dict(base_optimizer_cls=torch.optim.SGD, lr=1e-2, momentum=0.9))]:
        model, data, labels = _problem(seed=1)
        gd = GradDrop(model.parameters(), model=model, criterion=nn.CrossEntropyLoss(),
                      step_method="single_forward", **kwargs)
        before = _flat(model)
        for _ in range(3):
            gd.step(data, labels, torch.unique(labels))
        after = _flat(model)
        assert torch.isfinite(after).all(), f"{name}: non-finite"
        assert not torch.allclose(after, before), f"{name}: did not move"
    print("  [3] base-optimizer composition (Adam / SignSGD / SGD+momentum)  OK")


def test_seeded_determinism():
    def run():
        model, data, labels = _problem(seed=2)
        gen = torch.Generator().manual_seed(999)
        gd = GradDrop(model.parameters(), model=model, criterion=nn.CrossEntropyLoss(),
                      lr=0.1, generator=gen)
        for _ in range(5):
            gd.step(data, labels, torch.unique(labels))
        return _flat(model)
    a, b = run(), run()
    assert torch.allclose(a, b, atol=0), "GradDrop not deterministic with seeded generator"
    print("  [4] seeded-generator determinism  OK")


if __name__ == "__main__":
    print("GradDrop verification:")
    test_sign_purity_math()
    test_leak_extremes()
    test_base_composition()
    test_seeded_determinism()
    print("\nAll GradDrop assertions passed.")
