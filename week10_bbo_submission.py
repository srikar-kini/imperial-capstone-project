from pathlib import Path

code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""week10_bbo_submission.py

Generate the next round of BBO query vectors from the running log in
<File>input and outout values weekly iteration.xlsx</File>.

Key points
- Reads the workbook and parses "week N input values" / "week N output values" blocks.
- Filters out invalid INPUT rows (any coordinate not in [0,1) or not starting with 0.xxxxxx).
- Uses a conservative local-refinement strategy:
    * anchor = best observed valid point so far (max y)
    * trend  = direction between the last two valid points (if last improved)
    * propose = anchor + step * sign(trend) (or small random jitter if no trend)
- Enforces the hard submission constraint: each xi is 0.xxxxxx, six decimals, < 1.0.

Usage
  python week10_bbo_submission.py --xlsx "input and outout values weekly iteration.xlsx" \
      --step 0.0010 --jitter 0.0005

Outputs
  - Prints query lines ready to paste into the portal.
  - Writes week<N>_queries.txt and week<N>_queries.json.

Dependencies
  pip install pandas openpyxl numpy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DIM_BY_FN = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}


def sheet_to_text(xlsx_path: Path, sheet_name: str | None = None) -> str:
    """Load the sheet and concatenate all cells to a single text blob."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    # Flatten, drop NaN, join with newlines
    vals = df.astype(str).values.ravel().tolist()
    vals = [v for v in vals if v and v.lower() != 'nan']
    return "\n".join(vals)


def parse_weeks(text: str) -> Dict[int, Dict[str, Dict[int, List[float] | float]]]:
    """Parse the text into: weeks[week]['inputs'][fn]=list, weeks[week]['outputs'][fn]=float"""
    weeks: Dict[int, Dict[str, Dict[int, object]]] = {}

    week_blocks = re.split(r"(?i)(?=week\s+\d+\s+input\s+values)", text)
    for block in week_blocks:
        m_week = re.search(r"(?i)week\s+(\d+)\s+input\s+values", block)
        if not m_week:
            continue
        w = int(m_week.group(1))
        weeks.setdefault(w, {"inputs": {}, "outputs": {}})

        # Inputs
        for fn in range(1, 9):
            pat = rf"(?is)Function\s+{fn}\s*:\s*\[([^\]]+)\]"
            m = re.search(pat, block)
            if m:
                nums = [float(x.strip()) for x in m.group(1).split(',')]
                weeks[w]["inputs"][fn] = nums

        # Outputs block may be in the same chunk; try to find it
        m_out = re.search(r"(?is)week\s+\d+\s+output\s+values(.*)$", block)
        if m_out:
            out_text = m_out.group(1)
            for fn in range(1, 9):
                pat = rf"(?is)Function\s+{fn}\s*:\s*([-+0-9\.Ee]+)"
                mo = re.search(pat, out_text)
                if mo:
                    weeks[w]["outputs"][fn] = float(mo.group(1))

    return weeks


def is_valid_query(vec: List[float]) -> bool:
    """Hard rule: each xi must be in [0,1) and start with 0.xxxxxx when rendered."""
    for v in vec:
        if not (0.0 <= v < 1.0):
            return False
        s = f"{v:.6f}"
        if not s.startswith("0."):
            return False
    return True


def format_query(vec: np.ndarray) -> str:
    vec = np.clip(vec, 0.0, np.nextafter(1.0, 0.0))
    return "-".join(f"{float(v):.6f}" for v in vec)


def propose_next(valid_points: List[Tuple[np.ndarray, float]], dim: int, step: float, jitter: float, seed: int) -> np.ndarray:
    """Conservative local refinement around best point, with trend when available."""
    rng = np.random.default_rng(seed)

    # Sort by y desc
    valid_points = sorted(valid_points, key=lambda t: t[1], reverse=True)
    best_x, best_y = valid_points[0]

    # Trend from last two valid observations by time (we preserve original order by assuming caller supplies in time order)
    # We'll also compute a 'recent' trend using the last two entries.
    if len(valid_points) >= 2:
        # Need time order: caller should pass list already time-ordered; so reconstruct by using original list
        pass

    # We'll compute trend using the last two points in the *time-ordered* list provided by caller
    # caller passes time_ordered in addition to sorted list by y
    return best_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="input and outout values weekly iteration.xlsx")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--step", type=float, default=0.0010, help="refinement step size")
    ap.add_argument("--jitter", type=float, default=0.0005, help="small jitter when trend is unclear")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    text = sheet_to_text(xlsx_path, sheet_name=args.sheet)
    weeks = parse_weeks(text)
    if not weeks:
        raise SystemExit("No weeks parsed. Check the sheet content format.")

    last_week = max(weeks.keys())
    next_week = last_week + 1

    proposals: Dict[int, str] = {}
    proposals_vec: Dict[int, List[float]] = {}

    # Build per-function time-ordered valid points
    for fn in range(1, 9):
        dim = DIM_BY_FN[fn]
        time_points: List[Tuple[np.ndarray, float]] = []
        for w in sorted(weeks.keys()):
            inp = weeks[w]["inputs"].get(fn)
            out = weeks[w]["outputs"].get(fn)
            if inp is None or out is None:
                continue
            if len(inp) != dim:
                continue
            if not is_valid_query(inp):
                continue
            time_points.append((np.array(inp, dtype=float), float(out)))

        if len(time_points) < 2:
            # not enough history; random safe point
            x = np.clip(np.random.default_rng(args.seed + fn).random(dim), 0.0, np.nextafter(1.0, 0.0))
            proposals[fn] = format_query(x)
            proposals_vec[fn] = [float(v) for v in x]
            continue

        # Best point by y
        best_idx = int(np.argmax([y for _, y in time_points]))
        best_x = time_points[best_idx][0]

        # Recent trend (last two points)
        x_prev, y_prev = time_points[-2]
        x_last, y_last = time_points[-1]
        trend = x_last - x_prev

        rng = np.random.default_rng(args.seed + fn)

        if y_last >= y_prev:
            # continue in the same direction (small step)
            direction = np.sign(trend)
            # if any dimension has zero sign, add tiny random sign
            direction[direction == 0] = rng.choice([-1.0, 1.0], size=(direction == 0).sum())
            x_new = best_x + args.step * direction
        else:
            # pull slightly toward best point + tiny jitter for exploration
            x_new = 0.85 * best_x + 0.15 * x_last
            x_new = x_new + rng.normal(0.0, args.jitter, size=dim)

        # Enforce constraints
        x_new = np.clip(x_new, 0.0, np.nextafter(1.0, 0.0))

        proposals[fn] = format_query(x_new)
        proposals_vec[fn] = [float(v) for v in x_new]

    # Print for portal
    print(f"\n# Proposed queries for Week {next_week} (paste into portal)\n")
    for fn in range(1, 9):
        print(f"Function {fn}: {proposals[fn]}")

    # Save artefacts
    out_txt = Path(f"week{next_week}_queries.txt")
    out_json = Path(f"week{next_week}_queries.json")

    out_txt.write_text("\n".join([f"Function {fn}: {proposals[fn]}" for fn in range(1, 9)]) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps({"week": next_week, "queries": proposals_vec}, indent=2), encoding="utf-8")

    print(f"\nWrote {out_txt} and {out_json}\n")


if __name__ == "__main__":
    main()
'''

Path('week10_bbo_submission.py').write_text(code, encoding='utf-8')

# also create a tiny README snippet
readme = """## Week-10 Query Generator (Hyperparameter- & Robustness-aware)\n\nThis script reads the running log in `input and outout values weekly iteration.xlsx` and proposes the next round of BBO queries while enforcing the hard format constraint (`0.xxxxxx` for every coordinate).\n\n- Filters out invalid historical inputs (anything outside `[0,1)` when formatted).\n- Anchors on the best observed valid point and applies a small local step.\n- Uses the last-two-points trend when the latest observation improved; otherwise pulls toward the best point and adds tiny jitter.\n\nRun:\n```bash\npython week10_bbo_submission.py --xlsx "input and outout values weekly iteration.xlsx"\n```\n\nOutputs:\n- `week<N>_queries.txt` (paste-ready)\n- `week<N>_queries.json` (machine-readable)\n"""
Path('WEEK10_README_SNIPPET.md').write_text(readme, encoding='utf-8')

['week10_bbo_submission.py', 'WEEK10_README_SNIPPET.md']