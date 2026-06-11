# Experiment ledger

Generated from `experiments.json` by `build_ledger.py` — **edit the JSON, not this file.**

Status: `[done]` done  `[partial]` partial  `[ ]` pending  `[rerun]` needs-rerun

## Ch3 — Interference (written)

| ID | Section | Title | Scope | Status | Live | Headline | Fig | Artifacts |
|---|---|---|---|---|---|---|---|---|
| VAL-2D | 3.x sec:interference:validation | 2D mixture metric validation | synthetic 2D Gaussian mixture | [ ] |  |  | — | run: 01_small_2d/run.py \| plot: 01_small_2d/plot.py |
| VAL-C10 | 3.x sec:interference:validation | CIFAR-10 metric validation | CIFAR-10 / small CNN | [ ] |  |  | — | run: 02_medium_cifar10/run.py |
| VAL-C100 | 3.x sec:interference:validation | CIFAR-100 metric validation | CIFAR-100 / ResNet-18 | [ ] |  |  | — | run: 03_large_cifar100/run.py |

## Ch4 — COSGD

| ID | Section | Title | Scope | Status | Live | Headline | Fig | Artifacts |
|---|---|---|---|---|---|---|---|---|
| 20.05 | 4.6 sec:cosgd:results | Combine rule + norm cap (central axis) | iris/wine/bc/digits (+mnist/cifar10); SGD; fixed lr | [done] | d-acc -0.029..+0.046 | sum is the speed mechanism; freq/mean sit near baseline; cap needed at high dim | — | master: 20_cosgd_ablation/08_cross_summary/master_table.json (folder 05_combine) |
| 20.01 | 4.6 sec:cosgd:results | Gram-Schmidt variant | iris/wine/bc/digits; SGD | [done] | d-acc -0.243..+0.002 | negative variants more stable; mechanism (ΔI_inter>0) confirmed on low-dim | — | master: 20_cosgd_ablation/08_cross_summary/master_table.json (folder 01_gs_variant) |
| 20.02 | 4.3 subsec:cosgd:ordering | Class (magnitude) ordering | iris/wine/bc/digits; SGD | [partial] | d-acc -0.197..+0.002 | desc/asc similar on low-dim; high-dim SGD collapses at un-retuned LR | — | master: 20_cosgd_ablation/08_cross_summary/master_table.json (folder 02_class_order) |
| 20.03 | 4.4 sec:cosgd:highdim | Pre-normalisation before GS | iris/wine/bc/digits; SGD | [partial] | d-acc -0.022..+0.000 | prenorm rescues high-dim digits (ΔI_inter +0.27, best high-dim lever) | — | master: 20_cosgd_ablation/08_cross_summary/master_table.json (folder 03_prenormalize) |
| DIM-SWEEP | 4.4 subsec:cosgd:cosine | Cosine-vs-dim + L2-after-GS (synthetic) | synthetic dim ladder (no training) | [done] |  | \|cos\| 0.70→0.04 as dim grows; L2 change small-normalised vs large-raw (thesis 3.1/3.2) | — | script: 20_cosgd_ablation/03_prenormalize/synthetic_dim_sweep.py |
| 20.04 | 4.6 sec:cosgd:results | Step method / BatchNorm handling | cifar10 (no-BN), cifar100 (BN); SGD | [partial] | d-acc -0.313..-0.313 | per-class forward corrupts BN; multi_forward_with_BN catastrophic on cifar10 at fixed LR | — | master: 20_cosgd_ablation/08_cross_summary/master_table.json (folder 04_step_method) |
| 20.06 | 4.6 sec:cosgd:results | Base optimizer × COSGD | wine/digits; sgd/signsgd/rmsprop/adam | [done] | d-acc -0.258..+0.054 | adaptive bases robust; plain SGD collapses on digits (LR mismatch), adam fine | — | master: 20_cosgd_ablation/08_cross_summary/master_table.json (folder 06_base_optimizer) |
| 20.07 | 4.7 sec:cosgd:limits | Class-count scalability (O(n^2) wall) | synthetic CIFAR-10 {2..100} + real cifar10/emnist47/cifar100 | [ ] |  | smoke: 0.9x→2.9x→18.1x overhead at 2→10→50 classes — full run pending | — | run: 20_cosgd_ablation/07_scalability/run.py [+--real] |
| 20.08 | 4.6 sec:cosgd:results | COSGD summary + COSGD↔BOGrad contrast | aggregator (reads 20.01-06 + BoGrad table) | [done] |  | COSGD raises I_inter (+0.06 avg); mechanism dissociation from BOGrad (now Δ-vs-Δ) | — | master: 20_cosgd_ablation/08_cross_summary/master_table.json \| contrast: 20_cosgd_ablation/08_cross_summary/mechanism_contrast.json |

## Ch5 — BOGrad

| ID | Section | Title | Scope | Status | Live | Headline | Fig | Artifacts |
|---|---|---|---|---|---|---|---|---|
| 10.01 | 5.2 subsec:bograd:buffer | Buffer size K | cifar10; sgd/signsgd/rmsprop/adam | [done] | d-acc +0.016..+0.169 | finite per-optimiser best K (sgd/rmsprop 32, adam 128, signsgd 64); signsgd benefits most | — | master: 10_bograd_ablation/09_cross_summary/master_table.json (folder 01_buffer_K) |
| 10.02 | 5.2 sec:bograd:algorithm | Learning-rate retune | cifar10; 4 opt | [done] | d-acc +0.108..+0.292 | BoGrad's best LR is lower; deltas vs single-LR baseline are inflated (read with care) | — | master: 10_bograd_ablation/09_cross_summary/master_table.json (folder 02_lr_retune) |
| 10.03 | 5.2 sec:bograd:algorithm | Projection mode + strength | cifar10; 4 opt | [done] | d-acc +0.015..+0.169 | negative / neg-alpha wins for all optimisers; full/positive collapse when aligned with descent | — | master: 10_bograd_ablation/09_cross_summary/master_table.json (folder 03_projection_mode) |
| 10.04 | 5.2 subsec:bograd:ortho_methods | Orthogonalisation method | cifar10; 4 opt | [done] | d-acc +0.000..+0.169 | soft sequential_negative > true-orthogonal (qr ≡ householder) for all opt | — | master: 10_bograd_ablation/09_cross_summary/master_table.json (folder 04_orth_method) |
| 10.05 | 5.2 sec:bograd:algorithm | Projection scope (per-tensor vs global) | cifar10; 4 opt | [done] | d-acc +0.017..+0.169 | minor knob; sgd/rmsprop prefer global, signsgd/adam per_tensor | — | master: 10_bograd_ablation/09_cross_summary/master_table.json (folder 05_projection_scope) |
| 10.06 | 5.3 sec:bograd:theory | Direction vs magnitude | cifar10; 4 opt | [done] | d-acc +0.021..+0.199 | preserve_mag wins (3/4); direction matters, magnitude inert; random-proj control fails | — | master: 10_bograd_ablation/09_cross_summary/master_table.json (folder 06_magnitude) |
| 10.07 | 5.3 sec:bograd:theory | Momentum × BoGrad (2×2) | cifar10; sgd/signsgd/rmsprop (no adam) | [done] | rows present | BoGrad helps on top of momentum (bogradOn wins all present bases) | — | master: 10_bograd_ablation/09_cross_summary/master_table.json (folder 07_momentum_2x2) |
| 10.08 | 5.2 subsec:bograd:batch_size | Batch size × K | cifar10; sgd (default) | [done] | d-acc +0.002..+0.043 | K32 wins; relative gain grows as batch shrinks (+0.002 big → +0.044 small) | — | master: 10_bograd_ablation/09_cross_summary/master_table.json (folder 08_batch_K) |
| 10.09 | 5.6 sec:bograd:summary | BoGrad cross-optimizer/arch summary | aggregator (reads 10.01-08) | [done] |  | BoGrad helps all 4 optimisers; benefit tracks between-batch interference level | — | master: 10_bograd_ablation/09_cross_summary/master_table.json |

## Ch6 — Experiments

| ID | Section | Title | Scope | Status | Live | Headline | Fig | Artifacts |
|---|---|---|---|---|---|---|---|---|
| 30.00 | 6.2 sec:experiments:main_results | Convergence speed-up (headline) | 6 ds × 4 opt × 5 method × 5 seed | [ ] |  |  | — | view: 30_main_comparison/views.py view_speedup |
| 30.01 | 6.2 sec:experiments:main_results | Test-accuracy trajectories | 6 ds, 5 seeds (mean±std band) | [ ] |  |  | — | view: 30_main_comparison/views.py |
| 30.02 | 6.2 sec:experiments:main_results | Fixed-budget tables (10/50/100%) | 6 ds | [ ] |  |  | — | view: 30_main_comparison/views.py |
| 30.03 | 6.2 sec:experiments:main_results | Final-accuracy summary bars | 6 ds × 4 opt × 5 arms | [ ] |  |  | — | view: 30_main_comparison/views.py |
| IMPL | 6.2 sec:experiments:main_results | Implicit-method comparison | dropout/clipping/Adam/momentum | [ ] |  |  | — | research/04_implicit_comparisons |
| 30.04 | 6.3 sec:experiments:sensitivity | LR sensitivity | cifar10 + cifar100, per method | [ ] |  |  | — |  |
| 30.05 | 6.3 sec:experiments:sensitivity | K sensitivity (BoGrad) | per dataset | [ ] |  |  | — |  |
| 30.06 | 6.3 sec:experiments:sensitivity | Batch-size sensitivity | cifar10 + emnist | [ ] |  |  | — |  |
| 30.07 | 6.4 sec:experiments:scale | Model-scale study | width-scaled CNN + ResNet-18/34 | [ ] |  |  | — |  |
| 30.08 | 6.5 sec:experiments:gradient_stats | Gradient statistics during training | all 5 arms | [ ] |  |  | — |  |
| 30.09 | 6.4 sec:experiments:scale | Accuracy-vs-wall-clock Pareto | per dataset | [ ] |  |  | — |  |
| 40.01 | 6.x sec:experiments:synthesis | Does interference reduction predict accuracy? | regress Δacc on Δinterference across suite | [ ] |  |  | — | 40_synthesis/ |

## Ch7 — Conclusions/Synthesis

| ID | Section | Title | Scope | Status | Live | Headline | Fig | Artifacts |
|---|---|---|---|---|---|---|---|---|
| 40.02 | 7.x sec:conclusions:wprime | W' separability at scale | cos(g,g̃) + alignment fingerprint | [ ] |  |  | — | 40_synthesis/ |
