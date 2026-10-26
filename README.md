# FedMIPT release3

Modular public implementation of the recovered original **Task1 Phone–Sports**
model. This release follows the `main.py` / `fedmipt/` / `configs/` style of the
earlier public release. Its default is fixed-cost Null-aware OT.

| Setting | Task1 default |
|---|---|
| Fusion | masked mean of available projected text/image features |
| Null cost | fixed 0.7; no learned Null parameter |
| Matched-mass exponent gamma | 0.5 |
| Interests K | 3 |
| ItemCL / UserCL | 0.04 / 0.01 |
| OT epsilon / iterations | 0.07 / 50 |
| Pretraining / transfer rounds | 2 / at most 50 |
| Early-stopping patience | 5 |
| Optimizer | Adam, learning rate 0.001, weight decay 1e-5 |
| Batch size / training negatives | 512 / 5 |
| Evaluation | one positive + 99 fixed negatives |

The recovered five-seed historical Task1 Test result is
**0.329338 ± 0.002428 NDCG@10** (sample standard deviation). It identifies the
source configuration; it is not a newly trained result from this release.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the verification boundary.
Task2/Task3, Learned Null, and outcome-filtered data are not bundled as
reproduced results of release3.

## Setup and data

Use Python 3.10 and install `requirements.txt` into your environment. GPU users
should install a PyTorch build compatible with their CUDA setup.

```bash
conda activate py310
pip install -r requirements.txt
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

Provide the original, unfiltered Task1 split at
`data/task1_i10_u5/splits/seed2026/`, or set `data_dir` explicitly.
The feature arrays, domain-local item mapping, shared user mapping, fixed
candidates, and split membership must match the intended experiment.
See [data/README.md](data/README.md). Raw data and trained weights are not
distributed here.

## Train and evaluate

```bash
# Validation selects and restores the best checkpoint. It does not score Test.
python main.py train --config configs/task1_phone_sports.json --seed 42 --device cuda

# Evaluate the selected checkpoint after freezing the experiment setting.
python main.py test --run_dir runs/task1_phone_sports_seed42 --device cuda

# Historical five-seed set: 1, 2, 42, 99, 2026.
bash scripts/train_main.sh task1_phone_sports
python scripts/summarize.py runs --split validation
```

The training seed changes initialization and sampling, not the saved split or
candidate files. For host-specific thread limits, pass `--cpu_threads N` and
set all four thread environment variables to the same N. The included shell
scripts also accept `CPU_THREADS=N`. No machine-specific paths, credentials,
SSH commands, or background schedulers are required.

Each run saves the resolved config, source/data hashes, Validation history,
best checkpoint, `summary.json`, and `status/complete.json`. Test first
reproduces the checkpoint's Validation score and writes a separate
`test_summary.json` and `status/test_complete.json`. Existing run directories
and Test results are never overwritten.

## Controls

All controls are explicit config overrides and require independent training:

```bash
# Balanced OT: remove the Null row/column, retain interest matching/readout.
python main.py train --config configs/task1_phone_sports.json --seed 42 --tag balanced --set ot_use_null=false

# Strict no cross-domain transfer: no packages and no projector FedAvg.
python main.py train --config configs/task1_phone_sports.json --seed 42 --tag no_cross --set cross_transfer=false

# Read source interests directly without OT.
python main.py train --config configs/task1_phone_sports.json --seed 42 --tag no_ot --set ot_enable=false
```

`scripts/train_ablations.sh` contains the original content/contrastive controls.
The gamma value remains 0.5 in the Balanced control; when all valid real mass
is matched, its matched-mass scaling becomes approximately one.

## Implementation

`fedmipt/models.py` contains the private model; `interests.py` builds detached
interest packages; `transport.py` solves fixed-cost Null-aware/Balanced OT;
`readout.py` scores candidate-conditioned transfer; `federated.py` averages
only the semantic encoder. `trainer.py` implements recommendation losses,
Validation selection, and checkpoint restoration.

Raw interactions, item features and local embeddings stay within each domain.
Only the shared semantic projector is averaged and per-user interest packages
are exchanged. This is a simulation of that boundary, not a networked secure
federated deployment or a formal privacy guarantee. Matched mass is a transport
statistic and historical scaling factor, not calibrated confidence.
