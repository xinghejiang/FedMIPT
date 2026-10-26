# Data

Datasets and CLIP features are not included. release3 packages the recovered
original Task1 Phone–Sports configuration. `data_dir` points to its split:

```text
data/
  task1_i10_u5/splits/seed2026/{phone,sport}/
```

## Construction

The original Task1 uses the Phone and Sports categories with shared users.
Ratings become implicit feedback; items with fewer
than 10 interactions and users with fewer than 5 are removed, and only users
present in both domains are kept. For each user one interaction is held out
for Validation and one for Test; the rest form the training set. Item text
(title, brand, category, description) and the primary product image are
encoded offline with a frozen CLIP ViT-B/32 (512-dimensional features).
`seed2026` identifies the split; it is independent of the training seed.

## Files in each domain directory

| File | Content |
|---|---|
| `train.csv`, `vali.csv`, `test.csv` | columns `uid,iid,rating`; IDs contiguous from 0; user IDs shared by both domains; exactly one row per user in `vali.csv` and `test.csv` |
| `text_clip.npy`, `image_clip.npy` | float32 `[num_items, 512]`, row = item ID |
| `modality_mask.npy` | bool `[num_items, 2]`, (text, image) availability; missing content is expressed only here |
| `vali_candidates.npz`, `test_candidates.npz` | array `items` int64 `[num_users, 100]`; column 0 is the held-out positive, columns 1–99 are fixed negatives not observed by the user. Required. |
| `item_seen_in_train_mask.npy` | optional bool `[num_items]`; checked against `train.csv` |
| `vali_positive_is_cold.npy`, `test_positive_is_cold.npy` | optional bool `[num_users]`; default: positive not in `train.csv` |

Validation is scored with train-only histories, Test with train + Validation
histories. The SHA-256 of every file is recorded in each run and checked
again before Test evaluation.
