# Testing — five focused tests informing the thesis story

This folder holds the systematic experiments that decide what BoGrad actually
does, what its right hyperparameters are, and how it behaves at scale. Each
subfolder is one test with its own `README.md`, `run.py`, and `results/`.

The order below is intentional: each test answers a question whose answer
informs how the next test should be set up.

## Tests

| # | Folder | Question | Decides |
|---|---|---|---|
| 1 | `01_attribution/` | Is BoGrad's accuracy gain from direction or implicit step-size reduction? | Whether to keep BoGrad at all; whether to rescale by default |
| 2 | `02_rescaling/` | Which form of magnitude rescaling works best? | The rescale knob's default behaviour |
| 3 | `03_optimizations/` | Which speed/memory optimisations actually help, on what model size? | New defaults for `BoGrad`; scalability claims |
| 4 | `04_inpipeline_diagnosis/` | Why does in-pipeline projection work for Adam but not RMSprop / SignSGD? | Recommended projection point per optimiser family |
| 5 | `05_2x2_grid/` | For each optimiser, decompose contribution of momentum vs BoGrad and their interaction | The headline Chapter 4 result |

## Recommended execution order

1. `01_attribution` — make-or-break for the thesis story.
2. `02_rescaling` *if* attribution shows direction matters.
3. `04_inpipeline_diagnosis` — small and informative; informs Test 5 choices.
4. `03_optimizations` — characterise scaling on bigger CNNs.
5. `05_2x2_grid` — final headline, run last with the best version of BoGrad.

## Shared utilities

[`_common.py`](_common.py) contains:
- `SmallCNN` (the standard CIFAR-10 small CNN we've been using)
- `ResNet8` (medium model, ~250k params)
- `ResNet18CIFAR` (CIFAR-adapted ResNet-18, ~11M params)
- `build_cifar10`, `make_train_loader`, `evaluate`, `train_run`, `aggregate`

Each test's `run.py` imports from `_common` to keep the per-test code focused
on the test's specific variants and configuration.

## Conventions

- All run scripts seed deterministically (`torch.manual_seed(seed)`,
  `torch.cuda.manual_seed_all(seed)`, `Generator(seed)` for the dataloader).
- Each test runs `--trials` independent trials at seeds `base_seed + trial * 1000`.
- Within a trial, all configs share the same data-loader generator state for
  paired comparison.
- Output goes to `<test_dir>/results/run_<timestamp>/` with `results.json` plus
  `per_trial/<config>__t<trial>.json`.
- Each test prints a summary table at the end, with markers for ++/+/~/-/--
  improvement vs the baseline cell.

## Cross-test findings index

A running list of confirmed conclusions, updated as each test completes:

(none yet — populate as tests finish)
