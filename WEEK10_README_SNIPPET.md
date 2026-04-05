## Week-10 Query Generator (Hyperparameter- & Robustness-aware)

This script reads the running log in `input and outout values weekly iteration.xlsx` and proposes the next round of BBO queries while enforcing the hard format constraint (`0.xxxxxx` for every coordinate).

- Filters out invalid historical inputs (anything outside `[0,1)` when formatted).
- Anchors on the best observed valid point and applies a small local step.
- Uses the last-two-points trend when the latest observation improved; otherwise pulls toward the best point and adds tiny jitter.

Run:
```bash
python week10_bbo_submission.py --xlsx "input and outout values weekly iteration.xlsx"
```

Outputs:
- `week<N>_queries.txt` (paste-ready)
- `week<N>_queries.json` (machine-readable)
