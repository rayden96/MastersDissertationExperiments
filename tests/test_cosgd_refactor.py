"""
Verification for the COSGD wrapper refactor.

    python tests/test_cosgd_refactor.py

Checks:
  1. Back-compat: COSGD(base=SGD, combine=sum, class_order=fixed, prenormalize=False)
     applies exactly  theta <- theta - lr * sum(orthogonalised per-class grads)
     (i.e. identical math to the legacy SGD-only COSGD).
  2. Base-optimizer composition: COSGD(base=Adam) and COSGD(base=SignSGD) run and
     produce finite, non-trivial updates.
  3. New knobs change behaviour: combine in {sum,mean,freq}, prenormalize, class_order.
  4. collect_timing gates the timer (off => no data, on => data).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.optimizers import COSGD, SignSGD  # noqa: E402
from common.optimizers.COSGD import gram_schmidt_normal  # noqa: E402


def _problem(seed=0, n=36, n_classes=4):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, n_classes))
    data = torch.randn(n, 8)
    labels = torch.randint(0, n_classes, (n,))
    # guarantee >=2 classes present
    labels[:n_classes] = torch.arange(n_classes)
    return model, data, labels


def _flat(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def test_backcompat_sgd_math():
    model, data, labels = _problem()
    lr = 0.1
    opt = COSGD(model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=lr,
                model=model, criterion=nn.CrossEntropyLoss(),
                orthogonalization_method='gram_schmidt_normal',
                step_method='single_forward', combine='sum', class_order='fixed')
    valid = torch.unique(labels)

    before = _flat(model)
    # expected combined from the SAME internal per-class computation (params unchanged)
    cg, _, _ = opt._grads_single_forward(data, labels, valid)
    expected_combined = gram_schmidt_normal(cg).sum(dim=0)
    expected_after = before - lr * expected_combined

    opt.step(data, labels, valid)
    after = _flat(model)
    assert torch.allclose(after, expected_after, atol=1e-5), \
        f"back-compat SGD math mismatch (max diff {(after-expected_after).abs().max().item():.2e})"
    print("  [1] base=SGD, combine=sum == legacy 'theta -= lr*sum(ortho per-class)'  OK")


def test_base_composition():
    for name, kwargs in [("adam", dict(base_optimizer_cls=torch.optim.Adam, lr=1e-2)),
                         ("signsgd", dict(base_optimizer_cls=SignSGD, lr=1e-2)),
                         ("sgd+mom", dict(base_optimizer_cls=torch.optim.SGD, lr=1e-2, momentum=0.9))]:
        model, data, labels = _problem(seed=1)
        opt = COSGD(model.parameters(), model=model, criterion=nn.CrossEntropyLoss(),
                    step_method='single_forward', **kwargs)
        valid = torch.unique(labels)
        before = _flat(model)
        for _ in range(3):
            opt.step(data, labels, valid)
        after = _flat(model)
        assert torch.isfinite(after).all(), f"{name}: non-finite params"
        assert not torch.allclose(after, before), f"{name}: params did not move"
    print("  [2] base-optimizer composition (Adam / SignSGD / SGD+momentum)  OK")


def test_knobs_change_behaviour():
    def step_once(**extra):
        model, data, labels = _problem(seed=2)
        opt = COSGD(model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.1,
                    model=model, criterion=nn.CrossEntropyLoss(),
                    step_method='single_forward', **extra)
        opt.step(data, labels, torch.unique(labels))
        return _flat(model)

    base = step_once(combine='sum', class_order='fixed', prenormalize=False)
    mean = step_once(combine='mean', class_order='fixed', prenormalize=False)
    freq = step_once(combine='freq', class_order='fixed', prenormalize=False)
    prenorm = step_once(combine='sum', class_order='fixed', prenormalize=True)
    desc = step_once(combine='sum', class_order='desc', prenormalize=False)

    assert not torch.allclose(base, mean), "combine=mean should differ from sum"
    assert not torch.allclose(base, freq), "combine=freq should differ from sum"
    assert not torch.allclose(base, prenorm), "prenormalize should change the update"
    assert not torch.allclose(base, desc), "class_order=desc should change the update"
    print("  [3] knobs change behaviour (combine sum/mean/freq, prenormalize, class_order)  OK")


def test_timing_gate():
    model, data, labels = _problem(seed=3)
    off = COSGD(model.parameters(), model=model, criterion=nn.CrossEntropyLoss(),
                lr=0.1, collect_timing=False)
    off.step(data, labels, torch.unique(labels))
    _, total_off = off.get_time_stats()
    assert total_off == 0, "collect_timing=False must record no timing"

    model2, data2, labels2 = _problem(seed=3)
    on = COSGD(model2.parameters(), model=model2, criterion=nn.CrossEntropyLoss(),
               lr=0.1, collect_timing=True)
    on.step(data2, labels2, torch.unique(labels2))
    _, total_on = on.get_time_stats()
    assert total_on > 0, "collect_timing=True must record timing"
    print("  [4] collect_timing gate (off=0 overhead, on=records)  OK")


if __name__ == "__main__":
    print("COSGD refactor verification:")
    test_backcompat_sgd_math()
    test_base_composition()
    test_knobs_change_behaviour()
    test_timing_gate()
    print("\nAll COSGD assertions passed.")
