# Provenance and verification

release3 targets the recovered 2026-09-06 Task1 model, with the original
Task1 K=3 content-retention-100% runs (seeds 1, 2, 42, 99, 2026).
All eight domain-level HR/NDCG@5/@10 historical averages match the supplied
table after rounding to four decimals. The verified historical overall
Test NDCG@10 is 0.32933833826633363, sample std 0.002428447506635901.

Recovered source SHA-256:

| Original file | SHA-256 |
|---|---|
| main.py | 9aeed1f3fee31f90ede599ace7b2d44459e4108e57340e2ae5a4bf97207de811 |
| model/client.py | 6070749e184f1b121712101df4d269b85f8d3ac5c12c3c5d98a95b93762bd7b4 |
| model/engine.py | 5aaed1787adfde628bd54ba53d1b6d123a79d181f57f1d951f2f8599d670fc25 |

These are hashes of surviving source snapshots, not immutable hashes recorded
at historical training time. Full historical Task1 checkpoints were not
recovered. Historical results must therefore remain labeled historical until
this release is independently trained and evaluated on the intended data.

The refactoring preserves initialization order, original model state keys
(`encoder`, `semantic_scale`, `interest_scale`, `mlp`), masked-mean arithmetic,
recomputed semantic branches in candidate scoring, interest clustering,
fixed-cost Null transport, gamma scaling, BCE/ItemCL/UserCL, negative sampling
and positive removal from training histories. Detached package construction
remains detached. No learnable rejection-cost parameter is added.

Intentional engineering differences:

- Explicit config files, separate train/test commands, source/data hashes and
  completion markers replace machine-specific launch procedures.
- Training reads Train/Validation labels and derives the catalog size from
  feature metadata. Test is available only to the separate Test command.
- Zero-history packages use a finite zero direction rather than divide by
  zero. This guard is outside the recovered Task1 nonempty-history contract.
- Balanced OT is an additional control; it is not presented as a recovered
  historical result. Gamma is not silently changed for that control.
- Nonzero package noise is outside the recovered original configuration and
  is rejected by config validation. Task2/Task3 are not claimed reproduced.

Verification in this delivery is synthetic CPU source equivalence:
initialization, semantics, user vectors, interest packages, OT, candidate
forward outputs, loss components and parameter gradients. It is not real-data
training, a CUDA/MPS numerical equivalence test, checkpoint migration, or a
fresh reproduction of the paper's metric. Runtime checks and source hashes
are recorded separately under `VERIFICATION.json` and `SOURCE_MANIFEST.json`.

The balanced-mode verification passed 18 cases: three seeds (1,42,2026),
three transfer paths (Null-aware, source without OT, strict no-transfer),
and both the historical zero-output scorer and a nonzero scorer to exercise
all parameter gradients. Initialization/state keys matched exactly; maximum
absolute numerical difference was 5.960464477539063e-8. The environment was
Python 3.10 on CPU with four threads; versions are in `VERIFICATION.json`.
The check used in-memory synthetic tensors without optimizer steps, datasets,
checkpoints or remote file creation. It did not alter any experiment process.
