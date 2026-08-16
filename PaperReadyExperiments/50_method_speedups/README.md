# 50 — Method speed-up tests

Not a paper experiment. A cheap harness for deciding whether an implementation
change to BOGrad or COSGD is worth adopting, before spending compute on the
real ablations (10_/20_).

Three checks, cheapest first:

1. **Correctness** — sequential and batched projection must agree exactly on an
   orthogonal buffer, and both must respect the non-increase property on a
   correlated one. One row is informational: it demonstrates that the unguarded
   batched operator *does* lengthen the step, which is why `batched_norm_guard`
   exists.
2. **Speed** — median per-step cost, warmup discarded, all arms in one process
   on one device (same protocol as 30.11).
3. **Does it still train** — a few hundred steps, one seed. Enough to catch a
   variant that diverges or stalls; **not** enough to rank variants on accuracy.

Anything that passes goes to the real ablations for validation. Nothing here
belongs in the dissertation as evidence.

    python test_speedups.py --quick          # 1 dataset, 1 base, fast
    python test_speedups.py                  # covertype + cifar10 + cifar100
