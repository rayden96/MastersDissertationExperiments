# COSGD.py
"""
COSGD — Class-Orthogonalised Gradient Descent.

Computes per-class subgradients on the current batch, **orthogonalises** them
against each other (Gram-Schmidt), combines them into a single vector, writes
that into each parameter's `.grad`, then lets a base optimiser apply the step.

The shared per-class machinery (parameter layout, gradient extraction, the
three forward/backward strategies, timing, write-grad + base step) lives in
`_per_class.PerClassGradientOptimizer`. COSGD only implements the fold step
`_combine`: Gram-Schmidt, then sum / mean / freq-weight. (GradDrop is the
sibling that swaps GS for sign-purity masking — same extraction, different fold.)

Composes with SGD, SignSGD, RMSprop, Adam via `base_optimizer_cls`.

Design axes (driven by PaperReadyExperiments/20_cosgd_ablation):
  - orthogonalization_method : {gram_schmidt_normal, gram_schmidt_negative,
                                modified_gs_normal, modified_gs_negative}
  - step_method              : {single_forward, multi_forward, multi_forward_with_BN}
  - class_order              : {fixed, desc, asc, random}  (GS is order-dependent)
  - prenormalize             : unit-normalise per-class grads before GS (thesis 3.1/3.2)
  - combine                  : {sum, mean, freq}  (freq = §02 class-frequency weighting)

Back-compat: `COSGD(params, lr=..., model=..., criterion=...,
orthogonalization_method=..., step_method=...)` with defaults
(base=SGD, combine="sum", class_order="fixed", prenormalize=False) reproduces
the previous SGD-only behaviour exactly. Step signature unchanged:
`step(data, labels, unique_labels)`.
"""

from typing import Type

import torch
from torch.optim.optimizer import Optimizer

from common.optimizers._per_class import PerClassGradientOptimizer


###############################
# Orthogonalization Utilities #
###############################

def gram_schmidt_normal(vectors):
    """Standard Gram-Schmidt. vectors: [num_vectors, dim]."""
    num_vectors, _ = vectors.shape
    ortho = torch.zeros_like(vectors)
    for i in range(num_vectors):
        v = vectors[i].clone()
        if i > 0:
            prev = ortho[:i]
            dots = torch.mv(prev, v)
            norms_sq = torch.clamp(torch.sum(prev ** 2, dim=1), min=1e-12)
            v = v - torch.sum((dots / norms_sq).unsqueeze(1) * prev, dim=0)
        ortho[i] = v
    return ortho


def gram_schmidt_negative(vectors):
    """Gram-Schmidt removing only negative (destructive) projections."""
    n = vectors.shape[0]
    ortho = torch.zeros_like(vectors)
    for i in range(n):
        v = vectors[i].clone()
        for j in range(i):
            u = ortho[j]
            dot = torch.dot(v, u)
            if dot < 0:
                v = v - (dot / torch.clamp(torch.dot(u, u), min=1e-12)) * u
        ortho[i] = v
    return ortho


def modified_gram_schmidt_normal(vectors):
    """Modified Gram-Schmidt (more numerically stable)."""
    n = vectors.shape[0]
    ortho = torch.zeros_like(vectors)
    for i in range(n):
        v = vectors[i].clone()
        for j in range(i):
            u = ortho[j]
            u_norm = torch.norm(u)
            if u_norm > 1e-12:
                un = u / u_norm
                v = v - torch.dot(v, un) * un
        ortho[i] = v
    return ortho


def modified_gram_schmidt_negative(vectors):
    """Modified Gram-Schmidt removing only negative projections."""
    n = vectors.shape[0]
    ortho = torch.zeros_like(vectors)
    for i in range(n):
        v = vectors[i].clone()
        for j in range(i):
            u = ortho[j]
            u_norm = torch.norm(u)
            if u_norm > 1e-12:
                un = u / u_norm
                dot = torch.dot(v, un)
                if dot < 0:
                    v = v - dot * un
        ortho[i] = v
    return ortho


def pcgrad_project(vectors, generator=None):
    """PCGrad-style SYMMETRIC conflict projection (Yu et al. 2020), adapted to
    per-class gradients. For each vector g_i, iterate over the others j in a
    random order and, whenever <g_i, g_j> < 0, remove g_i's projection onto g_j:
        g_i <- g_i - (<g_i,g_j>/||g_j||^2) g_j.
    Unlike sequential Gram-Schmidt this projects EACH vector against ALL others
    (not just earlier ones), so it has no privileged first row — addressing the
    order-dependence of GS. Returns the modified [C,P] stack (caller sums/combines).
    """
    n = vectors.shape[0]
    out = vectors.clone()
    idx = list(range(n))
    for i in range(n):
        gi = out[i].clone()
        order = torch.randperm(n, generator=generator).tolist() if generator is not None else idx
        for j in order:
            if j == i:
                continue
            gj = vectors[j]            # project against the ORIGINAL others (PCGrad convention)
            dot = torch.dot(gi, gj)
            if dot < 0:
                gi = gi - (dot / torch.clamp(torch.dot(gj, gj), min=1e-12)) * gj
        out[i] = gi
    return out


def modified_gram_schmidt_negative_inplace(vectors):
    """In-place modified GS, negative-only. Mutates `vectors` (no second [C,P]
    buffer). Used by COSGD's low_memory path — behaviourally identical to
    modified_gram_schmidt_negative since modified GS only ever reads rows < i,
    which are already orthogonalised when processed in order."""
    n = vectors.shape[0]
    for i in range(n):
        for j in range(i):
            u = vectors[j]
            u_norm = u.norm()
            if u_norm > 1e-12:
                un = u / u_norm
                dot = torch.dot(vectors[i], un)
                if dot < 0:
                    vectors[i].sub_(dot * un)
    return vectors


def modified_gram_schmidt_normal_inplace(vectors):
    """In-place modified GS, full projection. Mutates `vectors`."""
    n = vectors.shape[0]
    for i in range(n):
        for j in range(i):
            u = vectors[j]
            u_norm = u.norm()
            if u_norm > 1e-12:
                un = u / u_norm
                vectors[i].sub_(torch.dot(vectors[i], un) * un)
    return vectors


ORTHOGONALIZATION_METHODS = {
    "gram_schmidt_normal": gram_schmidt_normal,
    "gram_schmidt_negative": gram_schmidt_negative,
    "modified_gs_normal": modified_gram_schmidt_normal,
    "modified_gs_negative": modified_gram_schmidt_negative,
    "pcgrad": pcgrad_project,
}

_INPLACE_METHODS = {
    "modified_gs_negative": modified_gram_schmidt_negative_inplace,
    "modified_gs_normal": modified_gram_schmidt_normal_inplace,
}


def orthogonalize_gradients(gradient_list, method):
    """Orthogonalise a [n, dim] stack (or list) and return the summed vector.

    Retained for back-compat / external callers. COSGD itself calls the
    per-method function directly so it can apply its own combine rule.
    """
    if isinstance(gradient_list, list):
        if not gradient_list:
            raise ValueError("Empty gradient list provided")
        gradients = torch.stack(gradient_list, dim=0)
    else:
        gradients = gradient_list
    fn = ORTHOGONALIZATION_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown orthogonalization method: {method}")
    return torch.sum(fn(gradients), dim=0)


###############################
# COSGD                       #
###############################

class COSGD(PerClassGradientOptimizer):
    _VALID_ORDER = ("fixed", "desc", "asc", "random")
    _VALID_COMBINE = ("sum", "mean", "freq")

    def __init__(
        self,
        params,
        *,
        base_optimizer_cls: Type[Optimizer] = torch.optim.SGD,
        lr: float = 1e-3,
        model=None,
        criterion=None,
        orthogonalization_method: str = "gram_schmidt_normal",
        step_method: str = "single_forward",
        class_order: str = "fixed",
        prenormalize: bool = False,
        combine: str = "sum",
        preserve_magnitude: bool = False,
        max_rescale: float = 10.0,
        low_memory: bool = False,
        cluster_k: int = 0,
        cluster_iters: int = 5,
        orth_strength: float = 1.0,
        conflict_gate: bool = False,
        conflict_threshold: float = 0.0,
        collect_timing: bool = False,
        eps: float = 1e-12,
        **base_optimizer_kwargs,
    ):
        if orthogonalization_method not in ORTHOGONALIZATION_METHODS:
            raise ValueError(f"Unknown orthogonalization method: {orthogonalization_method}")
        if class_order not in self._VALID_ORDER:
            raise ValueError(f"class_order must be one of {self._VALID_ORDER}")
        if combine not in self._VALID_COMBINE:
            raise ValueError(f"combine must be one of {self._VALID_COMBINE}")

        super().__init__(
            params, base_optimizer_cls=base_optimizer_cls, lr=lr,
            model=model, criterion=criterion, step_method=step_method,
            collect_timing=collect_timing, eps=eps, **base_optimizer_kwargs,
        )
        self.orthogonalization_method = orthogonalization_method
        self.class_order = class_order
        self.prenormalize = bool(prenormalize)
        self.combine = combine
        # A: rescale the combined direction back to the freq-weighted raw sum's
        #    norm, so the effective step magnitude (and thus optimal LR) tracks
        #    the baseline regardless of how much orthogonalisation/combine shrank it.
        self.preserve_magnitude = bool(preserve_magnitude)
        self.max_rescale = float(max_rescale)
        # B: in-place GS + no second [C,P] alloc (modified_gs_* methods only).
        self.low_memory = bool(low_memory)
        # D: cosine k-means the C per-class grads into K<=C clusters, orthogonalise
        #    the K cluster-sums (O(K^2) GS instead of O(C^2)). 0 = off (per-class).
        self.cluster_k = int(cluster_k)
        self.cluster_iters = int(cluster_iters)
        # Round 2:
        # F (soft orth): interpolate combined = (1-a)*raw + a*orthogonalised.
        #    a=1 = full COSGD, a=0 = plain batch grad. Tests whether full
        #    orthogonalisation over-corrects (round 1: full was the only knob).
        self.orth_strength = float(orth_strength)
        # G (conflict gating): only orthogonalise when the per-class grads
        #    actually conflict (mean pairwise cosine < -threshold); otherwise pass
        #    the raw batch grad through. Saves cost + avoids stripping averageable
        #    signal when there's no harmful interference (F5 motivation).
        self.conflict_gate = bool(conflict_gate)
        self.conflict_threshold = float(conflict_threshold)

    def _row_order(self, class_grads):
        if self.class_order == "fixed":
            return None
        if self.class_order == "random":
            return torch.randperm(class_grads.shape[0], device=class_grads.device)
        norms = class_grads.norm(dim=1)
        return torch.argsort(norms, descending=(self.class_order == "desc"))

    def _cluster(self, class_grads, counts):
        """D: cosine k-means the rows into <=cluster_k groups; return
        (cluster_sums [K,P], cluster_counts [K]) where each cluster vector is the
        count-weighted sum of its members and cluster_counts is the summed
        membership (so downstream freq-weighting still sees true frequencies)."""
        C = class_grads.shape[0]
        K = min(self.cluster_k, C)
        if K <= 0 or K >= C:
            return class_grads, counts
        with self.timer.time_context("cluster"):
            # normalise rows for cosine assignment
            norms = class_grads.norm(dim=1, keepdim=True).clamp_min(self.eps)
            units = class_grads / norms
            # init centroids = K rows with largest gradient norm (most informative)
            init = torch.argsort(norms.squeeze(1), descending=True)[:K]
            centroids = units[init].clone()
            assign = torch.zeros(C, dtype=torch.long, device=class_grads.device)
            for _ in range(self.cluster_iters):
                sim = units @ centroids.t()           # [C,K] cosine similarity
                assign = sim.argmax(dim=1)
                for k in range(K):
                    members = units[assign == k]
                    if members.shape[0] > 0:
                        c = members.mean(dim=0)
                        cn = c.norm()
                        if cn > self.eps:
                            centroids[k] = c / cn
            # build cluster sums (count-weighted) + summed counts
            csum = torch.zeros(K, class_grads.shape[1], device=class_grads.device)
            ccnt = torch.zeros(K, device=class_grads.device)
            for k in range(K):
                m = assign == k
                if m.any():
                    w = counts[m].unsqueeze(1)
                    csum[k] = (w * class_grads[m]).sum(dim=0)
                    ccnt[k] = counts[m].sum()
            # drop empty clusters
            keep = ccnt > 0
            return csum[keep], ccnt[keep]

    def _fold(self, rows, counts):
        """Combine [K,P] rows -> [P] under the active combine rule."""
        if self.combine == "sum":
            return rows.sum(dim=0)
        if self.combine == "mean":
            return rows.mean(dim=0)
        w = (counts / counts.sum().clamp_min(1.0)).unsqueeze(1)  # freq
        return (w * rows).sum(dim=0)

    def _mean_pairwise_cos(self, class_grads):
        """Mean pairwise cosine across the per-class rows (for conflict gating)."""
        n = class_grads.shape[0]
        if n < 2:
            return 1.0
        u = class_grads / class_grads.norm(dim=1, keepdim=True).clamp_min(self.eps)
        sim = u @ u.t()
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        return float(off.mean().item())

    def _combine(self, class_grads, counts):
        # D: optional clustering BEFORE ordering/GS (reduces C -> K)
        if self.cluster_k > 0:
            class_grads, counts = self._cluster(class_grads, counts)

        with self.timer.time_context("ordering"):
            order = self._row_order(class_grads)
            if order is not None:
                class_grads = class_grads[order]
                counts = counts[order]
        if self.prenormalize:
            class_grads = class_grads / class_grads.norm(dim=1, keepdim=True).clamp_min(self.eps)

        # raw (un-orthogonalised) combined direction — needed for G gate + F blend + A target
        raw_combined = self._fold(class_grads, counts)

        # G (conflict gating): if the classes don't meaningfully conflict, skip
        # orthogonalisation entirely and descend the raw batch gradient.
        if self.conflict_gate:
            if self._mean_pairwise_cos(class_grads) >= -self.conflict_threshold:
                return raw_combined

        with self.timer.time_context("orthogonalization"):
            if self.low_memory and self.orthogonalization_method in _INPLACE_METHODS:
                ortho = _INPLACE_METHODS[self.orthogonalization_method](class_grads)
            else:
                ortho = ORTHOGONALIZATION_METHODS[self.orthogonalization_method](class_grads)

        combined = self._fold(ortho, counts)

        # F (soft orth): interpolate raw <-> fully-orthogonalised. a=1 -> COSGD,
        # a=0 -> plain batch gradient. Tests whether full orthogonalisation
        # over-corrects.
        if self.orth_strength != 1.0:
            combined = (1.0 - self.orth_strength) * raw_combined + self.orth_strength * combined

        # A: rescale combined direction back to the raw target norm (clipped)
        if self.preserve_magnitude:
            target_norm = raw_combined.norm()
            cn = combined.norm()
            if cn > self.eps:
                scale = (target_norm / cn).clamp(max=self.max_rescale)
                combined = combined * scale
        return combined


__all__ = ["COSGD", "orthogonalize_gradients", "ORTHOGONALIZATION_METHODS"]
