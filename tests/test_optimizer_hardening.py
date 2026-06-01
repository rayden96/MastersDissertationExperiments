"""
Verification for the M0 optimizer-hardening changes.

Runnable as a plain script (no pytest dependency):

    python tests/test_optimizer_hardening.py

Checks:
  1. BoGrad._project_sequential is numerically identical to an independent
     reference implementation, across modes and store_normalised (regression guard).
  2. projection_strength alpha interpolates baseline (0) <-> full projection (1).
  3. BoGrad per-tensor buffers survive a state_dict -> load_state_dict round-trip.
  4. End-to-end determinism: two identically-seeded BoGrad runs match.
  5. SignSGD vanilla and Signum update rules are correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.optimizers import BoGrad, SignSGD  # noqa: E402


def _ref_sequential(x_flat, buffer, mode, store_normalised, eps=1e-12):
    """Faithful reimplementation of the ORIGINAL `.item()`-branch sequential GS."""
    projected = x_flat.clone()
    for b in buffer:
        b_vec = b
        if store_normalised:
            denom = 1.0
        else:
            bb = float(torch.dot(b_vec, b_vec).item())
            if bb <= eps:
                continue
            denom = bb
        dot_val = torch.dot(projected, b_vec)
        if mode == "negative" and dot_val.item() >= 0:
            continue
        if mode == "positive" and dot_val.item() <= 0:
            continue
        projected = projected - (dot_val / denom) * b_vec
    return projected


def _make_bograd(mode, store_normalised, strength=1.0, orth="sequential"):
    p = torch.nn.Parameter(torch.zeros(8))
    return BoGrad(
        [p], base_optimizer_cls=torch.optim.SGD, lr=0.1,
        buffer_size=6, projection_mode=mode, store_normalised=store_normalised,
        orth_method=orth, projection_strength=strength, collect_stats=False,
    )


def test_masked_sequential_equivalence():
    torch.manual_seed(0)
    n_fail = 0
    for mode in ("full", "negative", "positive"):
        for store_normalised in (True, False):
            opt = _make_bograd(mode, store_normalised)
            for _ in range(20):
                x = torch.randn(8)
                raw_buf = [torch.randn(8) for _ in range(opt.buffer_size)]
                if store_normalised:
                    buf = [b / (b.norm() + opt.eps) for b in raw_buf]
                else:
                    buf = raw_buf
                got = opt._project_sequential(x.clone(), buf)
                ref = _ref_sequential(x.clone(), buf, mode, store_normalised, opt.eps)
                if not torch.allclose(got, ref, atol=1e-6, rtol=0):
                    n_fail += 1
    assert n_fail == 0, f"sequential projection diverged from reference in {n_fail} cases"
    print("  [1] sequential projection == reference  (full/negative/positive, norm on/off)  OK")


def test_projection_strength():
    torch.manual_seed(1)
    full_opt = _make_bograd("full", True, strength=1.0)
    x = torch.randn(8)
    buf = [b / b.norm() for b in (torch.randn(8) for _ in range(4))]
    proj_full, _ = full_opt._project(x.clone(), buf)

    zero_opt = _make_bograd("full", True, strength=0.0)
    proj_zero, _ = zero_opt._project(x.clone(), buf)
    assert torch.allclose(proj_zero, x, atol=1e-6), "alpha=0 must return the input unchanged"

    half_opt = _make_bograd("full", True, strength=0.5)
    proj_half, _ = half_opt._project(x.clone(), buf)
    expected_half = x + 0.5 * (proj_full - x)
    assert torch.allclose(proj_half, expected_half, atol=1e-6), "alpha=0.5 must be the midpoint"
    print("  [2] projection_strength alpha in {0, 0.5, 1} interpolates correctly  OK")


def _tiny_train(opt, model, steps=15, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(32, 4)
    y = torch.randint(0, 3, (32,))
    lossfn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossfn(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()


def test_state_dict_roundtrip():
    torch.manual_seed(2)
    model = torch.nn.Linear(4, 3)
    opt = BoGrad(model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.1,
                 buffer_size=4, project_stage="update", projection_mode="negative",
                 collect_stats=False, momentum=0.9)
    _tiny_train(opt, model, steps=12)

    # snapshot per-tensor buffers
    before = {id(p): [b.clone() for b in opt.state[p]["buffer"]]
              for g in opt.param_groups for p in g["params"]}
    sd = opt.state_dict()

    model2 = torch.nn.Linear(4, 3)
    model2.load_state_dict(model.state_dict())
    opt2 = BoGrad(model2.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.1,
                  buffer_size=4, project_stage="update", projection_mode="negative",
                  collect_stats=False, momentum=0.9)
    opt2.load_state_dict(sd)

    restored = {id(p): opt2.state[p]["buffer"]
                for g in opt2.param_groups for p in g["params"]}

    # match by position (ids differ across models)
    before_lists = list(before.values())
    restored_lists = list(restored.values())
    ok = len(before_lists) == len(restored_lists) and all(
        len(a) == len(b) and all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))
        for a, b in zip(before_lists, restored_lists)
    )
    if ok:
        print("  [3] per-tensor buffers survive state_dict round-trip  OK (no serialization fix needed)")
    else:
        print("  [3] !! per-tensor buffers NOT preserved on load_state_dict -- serialization fix REQUIRED")
    return ok


def test_end_to_end_determinism():
    def run():
        torch.manual_seed(7)
        model = torch.nn.Linear(4, 3)
        opt = BoGrad(model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.1,
                     buffer_size=8, projection_mode="negative", collect_stats=False)
        _tiny_train(opt, model, steps=20, seed=7)
        return torch.cat([p.detach().view(-1) for p in model.parameters()])
    a, b = run(), run()
    assert torch.allclose(a, b, atol=0), "BoGrad runs not deterministic"
    print("  [4] end-to-end BoGrad determinism  OK")


def test_signsgd():
    # vanilla: theta <- theta - lr * sign(g)
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.0]))
    opt = SignSGD([p], lr=0.1)
    p.grad = torch.tensor([3.0, -5.0, 0.0])
    opt.step()
    expected = torch.tensor([1.0, -2.0, 0.0]) - 0.1 * torch.sign(torch.tensor([3.0, -5.0, 0.0]))
    assert torch.allclose(p.data, expected, atol=1e-7), f"SignSGD vanilla wrong: {p.data}"

    # signum: buf = mu*buf + g ; update = sign(buf)
    p2 = torch.nn.Parameter(torch.zeros(3))
    opt2 = SignSGD([p2], lr=1.0, momentum=0.5)
    g1 = torch.tensor([1.0, -1.0, 2.0]); p2.grad = g1.clone(); opt2.step()  # buf=g1
    after1 = -1.0 * torch.sign(g1)
    assert torch.allclose(p2.data, after1, atol=1e-7), "Signum step 1 wrong"
    g2 = torch.tensor([-3.0, 1.0, 1.0]); p2.grad = g2.clone(); opt2.step()  # buf=0.5*g1+g2
    buf2 = 0.5 * g1 + g2
    after2 = after1 - 1.0 * torch.sign(buf2)
    assert torch.allclose(p2.data, after2, atol=1e-7), "Signum step 2 wrong"
    print("  [5] SignSGD vanilla + Signum momentum update rules  OK")


if __name__ == "__main__":
    print("Optimizer hardening verification:")
    test_masked_sequential_equivalence()
    test_projection_strength()
    serialization_ok = test_state_dict_roundtrip()
    test_end_to_end_determinism()
    test_signsgd()
    print("\nAll assertions passed." if serialization_ok else
          "\nAssertions passed EXCEPT serialization (see [3] above).")
