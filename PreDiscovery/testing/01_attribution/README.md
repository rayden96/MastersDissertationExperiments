# Test 01 — Attribution: direction or magnitude?

## Aim

Decompose BoGrad's accuracy gain into three possible sources:

1. **Direction effect** — BoGrad's projection produces step directions that better avoid recent interference, so the optimizer takes more useful steps.
2. **Magnitude effect** — BoGrad's projection systematically reduces ‖update‖, which is equivalent to a hidden learning-rate cut. Any method that reduces step magnitude similarly would help.
3. **Both** — direction matters AND the implicit step-size regularisation matters.

We need to know which it is, because:
- If purely magnitude: drop BoGrad, just lower the LR.
- If purely direction: keep current default (no rescaling), the magnitude reduction is a side-effect.
- If both: use rescaling to capture direction effect, and tune LR independently.

## Why "just run with smaller LR" isn't enough

The average step magnitude reduction varies per step (data-dependent), so there's no single matched LR. Also, scaling LR uniformly doesn't capture the direction-dependent component. We need orthogonal controls.

## Method — three controls

For each variant, run a 3-point LR sweep so we compare best-tuned to best-tuned:

| Variant | Description | Tests |
|---|---|---|
| `baseline` | Pure SGD+momentum | Reference |
| `bograd` | Update-stage K=32 negative, no rescaling | Current default — direction + (implicit) magnitude reduction |
| `bograd_rescale` | Same but `preserve_magnitude=True` | Direction only (magnitude restored) |
| `bograd_random` | Same with `random_projection=True` | Magnitude reduction with arbitrary random direction |

LR sweep: `{0.025, 0.05, 0.1}` for SGD+momentum (centred on the standard 0.05).

## Decision criteria

| Outcome | Interpretation |
|---|---|
| `bograd_rescale` ≈ `bograd` ≫ `bograd_random` ≈ `baseline` | **Direction matters; magnitude doesn't.** Default to rescaling. |
| `bograd` ≫ `bograd_rescale` ≈ `bograd_random` ≈ `baseline-low-lr` | **Magnitude matters; direction doesn't.** Drop BoGrad, just lower LR. |
| `bograd` ≫ `bograd_rescale` ≫ `bograd_random` ≈ `baseline-low-lr` | **Both matter.** Keep current default (no rescaling); document the magnitude effect. |
| All variants ≈ baseline at matched LR | **No real BoGrad effect.** Honest negative result; thesis reframe. |

## Setup

- Architecture: `SmallCNN` (CIFAR-10, ~93k params) — matches prior phase results for direct comparability.
- Optimizer family: SGD+momentum=0.9 (where prior sweep showed +4.2pts at K=32 update-stage).
- BoGrad config: `project_stage="update"`, `buffer_size=32`, `projection_mode="negative"`.
- Training: 5 epochs, batch_size=128, 3 trials per (variant, LR) cell.
- Total runs: 4 variants × 3 LRs × 3 trials = **36 runs ≈ 90 min on T4.**

## Output

- `results/run_<timestamp>/results.json` — full per-trial data.
- `results/run_<timestamp>/per_trial/<variant>__lr<LR>__t<trial>.json` — individual run logs.
- Console summary table grouped by variant, with delta-vs-baseline at each LR.

## Run

```bash
python testing/01_attribution/run.py
python testing/01_attribution/run.py --epochs 5 --trials 3   # explicit defaults
python testing/01_attribution/run.py --quick                 # 1/4 train set, faster
python testing/01_attribution/run.py --lrs 0.01,0.025,0.05,0.1,0.2   # wider sweep
```
