"""What the per-class subgradients of a real network actually look like.

Section 4.4 argues that per-class subgradients are neither mutually orthogonal
(the high-dimensional null) nor freely rotatable, and that their magnitude
spread is what makes the ordering and pre-normalisation axes matter. That
argument was previously made in prose against a figure containing only random
vectors. This produces the picture from real subgradients.

Panels:
  (a) the C per-class subgradients of one batch, projected onto the plane of
      their own first two principal components, drawn as arrows from the origin
  (b) the same vectors after Gram-Schmidt, in the same plane
  (c) the per-class norms, which set who dominates the subtraction

The 2D projection is a projection: it exaggerates spread, because it picks the
plane of greatest variance. The measured pairwise cosines in full dimension are
therefore printed on the panel so the reader compares against the number rather
than the picture.

    python subgradient_geometry.py --steps 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.models import get_model
from common.datasets import get_dataset          # noqa: F401  (may be unused)


def per_class_subgradients(model, crit, x, y, num_classes, device):
    """One gradient per class present in the batch, flattened."""
    params = [p for p in model.parameters() if p.requires_grad]
    out, present = [], []
    for c in range(num_classes):
        m = y == c
        if int(m.sum()) == 0:
            continue
        loss = crit(model(x[m]), y[m])
        g = torch.autograd.grad(loss, params, retain_graph=False)
        out.append(torch.cat([t.reshape(-1) for t in g]).detach())
        present.append(c)
    return torch.stack(out), present


def gram_schmidt(G, order):
    """Classical Gram-Schmidt in the given row order, full projection."""
    out = G.clone()
    done = []
    for i in order:
        v = out[i].clone()
        for j in done:
            b = out[j]
            nb = b.dot(b)
            if nb > 1e-12:
                v = v - (v.dot(b) / nb) * b
        out[i] = v
        done.append(i)
    return out


def pairwise_cos(G):
    Gn = G / (G.norm(dim=1, keepdim=True) + 1e-12)
    C = (Gn @ Gn.T).cpu().numpy()
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200,
                    help="training steps before the snapshot is taken")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train = torchvision.datasets.CIFAR10(str(_REPO / "data"), train=True,
                                         download=True, transform=tf)
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True,
                                         num_workers=0, drop_last=True)

    model = get_model("small_cifar_cnn", num_classes=10).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    it = iter(loader)
    for step in range(args.steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader); xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        crit(model(xb), yb).backward()
        opt.step()
    print(f"snapshot after {args.steps} steps")

    xb, yb = next(it)
    xb, yb = xb.to(device), yb.to(device)
    G, present = per_class_subgradients(model, crit, xb, yb, 10, device)
    print(f"subgradients: {tuple(G.shape)}  classes present: {len(present)}")

    norms = G.norm(dim=1)
    order = torch.argsort(norms, descending=True).tolist()      # canonical order
    O = gram_schmidt(G, order)

    cos_before, cos_after = pairwise_cos(G), pairwise_cos(O)
    print(f"pairwise cosine before GS: mean {cos_before.mean():+.4f}, "
          f"conflicting {100*(cos_before < 0).mean():.1f}%, "
          f"|max| {np.abs(cos_before).max():.4f}")
    print(f"pairwise cosine after  GS: mean {cos_after.mean():+.4f}, "
          f"|max| {np.abs(cos_after).max():.2e}")
    print(f"norms: min {norms.min():.4f}  max {norms.max():.4f}  "
          f"max/min {norms.max()/norms.min():.2f}")

    # 2D plane: principal components of the subgradient set itself
    A = G.cpu().numpy().astype(np.float64)
    A = A - A.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    P = Vt[:2]                                                  # 2 x p
    B = (G.cpu().numpy() @ P.T)
    Bo = (O.cpu().numpy() @ P.T)
    print(f"plane captures {100*var[:2].sum():.1f}% of the set's variance")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    cmap = plt.get_cmap("tab10")

    for ax, V, title in ((axes[0], B, "(a) per-class subgradients"),
                         (axes[1], Bo, "(b) after Gram-Schmidt")):
        lim = np.abs(np.vstack([B, Bo])).max() * 1.15
        for i, c in enumerate(present):
            ax.annotate("", xy=(V[i, 0], V[i, 1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=cmap(c), lw=1.8))
            ax.text(V[i, 0] * 1.07, V[i, 1] * 1.07, str(c),
                    color=cmap(c), fontsize=8, ha="center", va="center")
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC 1", fontsize=9); ax.set_ylabel("PC 2", fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].text(0.03, 0.03,
                 f"mean cos {cos_before.mean():+.3f}\n"
                 f"{100*(cos_before < 0).mean():.0f}% conflicting",
                 transform=axes[0].transAxes, fontsize=8, va="bottom")
    axes[1].text(0.03, 0.03, f"max |cos| {np.abs(cos_after).max():.1e}",
                 transform=axes[1].transAxes, fontsize=8, va="bottom")

    ax = axes[2]
    idx = np.argsort(-norms.cpu().numpy())
    ax.bar(range(len(present)), norms.cpu().numpy()[idx],
           color=[cmap(present[i]) for i in idx])
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([str(present[i]) for i in idx], fontsize=8)
    ax.set_xlabel("class (descending norm)", fontsize=9)
    ax.set_ylabel(r"$\|g_c\|$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"(c) per-class norms, max/min = {norms.max()/norms.min():.1f}",
                 fontsize=10)

    fig.tight_layout()
    out = Path(args.out) if args.out else (
        _REPO.parent / "dissertation" / "chapters" / "cosgd" / "figures"
        / "subgradient_geometry.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    stats = dict(steps=args.steps, batch=args.batch, seed=args.seed,
                 n_classes=len(present), p=int(G.shape[1]),
                 cos_before_mean=float(cos_before.mean()),
                 cos_before_frac_neg=float((cos_before < 0).mean()),
                 cos_before_absmax=float(np.abs(cos_before).max()),
                 cos_after_absmax=float(np.abs(cos_after).max()),
                 norm_max_min=float(norms.max() / norms.min()),
                 plane_var=float(var[:2].sum()))
    import json
    (out.parent / "subgradient_geometry.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def apply_style():
    try:
        from common.plotting import apply_thesis_rcparams
        apply_thesis_rcparams()
    except Exception:
        pass


if __name__ == "__main__":
    main()
