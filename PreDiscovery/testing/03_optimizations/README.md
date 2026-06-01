# Test 03 — Optimization ablation

## Aim

Identify which speed/memory optimisations to BoGrad's projection actually help, on what architectures, without compromising accuracy. Each candidate optimisation is tested individually and in combination.

We want to answer:
1. Which optimisations should become BoGrad's defaults?
2. How does the BoGrad overhead scale with model size?
3. Are any optimisations *only* helpful on certain architectures?

## Candidate optimisations

| Tag | What it changes | Expected impact |
|---|---|---|
| `current` | Baseline — no changes (current BoGrad) | Reference |
| `sync_removed` | Replace `.item()` in projection inner loop with `torch.where` mask | Eliminates per-iteration GPU→CPU sync |
| `fp16_buffer` | Set `buffer_dtype=torch.float16` | Halve buffer memory; sometimes speed-up via tensor cores |
| `global_scope` | Set `projection_scope="global"` | Single buffer for all params; fewer Python-level iterations |
| `combined` | sync_removed + fp16_buffer + global_scope all on | Stacked best |

(`stacked_buffer` and `foreach_outer_loop` are deferred — both are bigger refactors that we'll add only if `combined` doesn't deliver enough.)

## Method

Two parts:

### Part A — Microbenchmark on SmallCNN

For each optimisation, run a short training (1–2 epochs) and measure:

- **Wall-clock per step** — median of last 1000 steps (warmup excluded).
- **Peak GPU memory** — via `torch.cuda.max_memory_allocated()`.
- **Final test accuracy at 5 epochs** — sanity that the optimisation didn't change the math.

Cross-checked against a no-BoGrad baseline (just SGD+momentum) so we can quote BoGrad's relative overhead.

### Part B — Architecture scaling

Run `combined` (the recommended config from Part A) on three architectures:

- `SmallCNN` (~93k params) — the existing reference.
- `ResNet8` (~250k params) — medium.
- `ResNet18CIFAR` (~11M params) — bigger; CIFAR-adapted.

Same metrics. Tells us how BoGrad's overhead scales with parameter count.

## Decision criteria

For each optimisation: is it strictly Pareto-better (faster or lighter, no accuracy regression)? If yes, set as default. If accuracy regresses by >0.5pt, document as "off by default" with a flag.

For Part B: produce an "overhead vs params" curve. We expect overhead as a fraction to *decrease* with model size (training step gets longer faster than projection cost). If it stays high, BoGrad is fundamentally costly and we need to redesign.

## Important notes

- **Part A and Part B require code edits** to implement `sync_removed` etc. Currently only the flags `buffer_dtype` and `projection_scope` are user-facing; `sync_removed` is a math-equivalent change that needs to be applied to `_project_sequential`. The run script applies these via patching the BoGrad instance for cleanliness, OR via a separate `BoGradOptimised` subclass.
- Each optimisation is tested *one at a time first* (sweep over individuals), then `combined`. Tells us if any pair is non-additive.
- For Part B, ResNet18 + K=128 needs ~5GB GPU memory just for the buffer. Tune K downward if you hit OOM, and document.

## Setup

- Part A: 5 variants × 3 trials × {1 short benchmark + 5-epoch sanity training} ≈ **15 short + 15 full runs ≈ 90 min.**
- Part B: 1 variant × 3 architectures × 3 trials × {benchmark + 5-epoch training} ≈ **9 short + 9 full runs ≈ 60–90 min** (longer for ResNet-18).
- Total: ~3 hours.

## Run

```bash
# Part A — microbenchmark optimisations on SmallCNN
python testing/03_optimizations/run.py --part A

# Part B — best-combined config on multiple architectures
python testing/03_optimizations/run.py --part B

# Both
python testing/03_optimizations/run.py --part both

# Quicker sanity (1/4 train, 2 epochs)
python testing/03_optimizations/run.py --part A --quick --epochs 2
```

## Output

- `results/run_<timestamp>/results.json` — full benchmark + accuracy data.
- Two summary tables printed to console:
  - Part A: variant × {wall-clock/step, peak-mem, accuracy} → recommend defaults.
  - Part B: arch × overhead → quote in thesis Chapter 4 scalability discussion.
