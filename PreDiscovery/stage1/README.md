# Stage 1 — Real-NN interference characterisation (meeting prep)

A focused experiment for the meeting tomorrow. Demonstrates the two types
of interference and the methods proposed to address each.

## The framing

| Type | What | Method | Metric |
|---|---|---|---|
| **Inter-batch (intra-batch)** | Class gradients within a batch conflict; the averaged batch gradient is a degraded direction | COSGD | `cos(g_c, g_{c'})` for class pairs c, c' inside a batch |
| **Between-batch (inter-batch)** | Training on batch t damages classes/examples not in batch t | BoGrad | OOB forgetting magnitude |

## What this folder produces

1. **`inter_batch_metric.py`** — `measure_inter_batch_interference()`. For
   each class c: forward + backward on probe-set examples of class c → get
   per-class gradient g_c. Then compute pairwise cos(g_c, g_{c'}) across
   all class pairs. Aggregates: frac_negative, mean_cos, max_conflict.

2. **`run_comparison.py`** — runs three configurations on CIFAR-10:
   - SGD+momentum baseline
   - SGD+momentum + BoGrad (between-batch interference reducer)
   - COSGD (inter-batch interference reducer)
   Each captures: standard training metrics, OOB forgetting (from
   `InterferenceTracker`), inter-batch interference (from this folder's
   new metric).

3. **`results/run_<timestamp>/`** — JSON outputs per run with the full
   per-step diagnostics and final summary table.

## How to run

```bash
# Default: 3 methods × 1 epoch CIFAR-10 with both metrics
python stage1/run_comparison.py

# Quicker (1/4 train set)
python stage1/run_comparison.py --quick

# Custom inter-batch measurement cadence
python stage1/run_comparison.py --inter-batch-every 50
```

## What to present

The headline output is a comparison table:

```
Method          final_acc   inter-batch %neg   OOB_forget    WW_K
baseline        0.4xxx      0.5x               xxx           0.6x
+ BoGrad        0.5xxx      ~same              ↓ x.xx        0.7x   ← addresses between-batch
+ COSGD         0.5xxx      ↓ ~                ~same         ?      ← addresses inter-batch
```

Plus a per-step plot showing inter-batch frac_negative and OOB forgetting
over training, with method curves overlaid.
