# 10.10 — Scale transfer (CIFAR-100 / ResNet-18)

**Claim.** The per-knob findings of the BOGrad ablation study (established on
the covertype + CIFAR-10 testbed) survive contact with a realistic modern
problem: CIFAR-100 with a ResNet-18. BOGrad in its canonical configuration,
tuned only in learning rate and buffer size K, still reaches the baseline's
final accuracy in fewer epochs.

**Design.** This is an *aggregator, not a sweep* — it trains nothing. The Ch6
bakeoff (`30_main_comparison`) already runs baseline vs BOGrad on CIFAR-100
across all four base optimisers at tuned hyperparameters, three-plus seeds,
paired data order. This axis reads those canonical records
(`30_main_comparison/_core/results/<campaign>/cifar100/cell_<base>__{baseline,bograd}.json`)
and produces the Chapter 5 transfer table: per base, final accuracy
(mean ± std), epochs-to-target, epoch and wall-clock speed-up, and the tuned
(lr, K). Reading the tuned bakeoff cells is deliberately the *fairer* form of
the claim than re-running the fixed-LR ablation protocol at scale.

**Expected outcome.** Epoch speed-up ≥ 1 on the bases where the ablations
predicted a benefit, with the per-base K optimum consistent with axis 10.01's
finding (small K for RMSprop, large for Adam).

Run (after the bakeoff's cifar100 cells exist):

    python run.py                     # defaults: campaign=main, dataset=cifar100
    python run.py --campaign main --dataset cifar100

Artefacts: `scale_transfer.json` + `scale_transfer.tex` (a LaTeX tabular for
`chapters/bograd`), written both here and to the persistent results root.
