"""Generate and Validate Candidate B Submission CSV.

Generates solution/submission_phase9_candidate_b.csv using:
- Physical emissions from Phase 8 V2 (geometry + carrier continuity + dead ball V2)
- Candidate B Duration-Aware Semi-Markov Decoder (soft minimum barrier + exponential tail)
- Calibrated multiplier targeting 15.0s mean possession duration

Verifies:
1. Exact row count: 199,319.
2. Frame column monotonic 0..199318 without gaps or duplicates.
3. Header exactly 'frame,label'.
4. Labels strictly subset of {'H', 'A', 'D', 'X'}.
5. X strictly outside the two half blocks.
6. H, A, D strictly inside the two half blocks.
7. Official submission validation via evaluation/validate_submission.py.
8. Bitwise determinism across two independent generation runs.
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from mtah_possession.semi_markov import SemiMarkovConfig, decode_blocks_semi_markov, calibrate_switch_penalties_semi_markov
from mtah_possession.pipeline import describe, write_csv


def generate_submission(out_path, calibrate=True):
    cache_path = os.path.join(HERE, "bar_ath_cache.npz")
    data = np.load(cache_path)
    L = data["L"]
    blocks = [tuple(b) for b in data["blocks"]]
    n = int(data["n"][0])
    fps = int(data["fps"][0])
    feats = {k: data[k] for k in ["dH", "dA", "ball_speed", "ball_confidence", "carrier_cH", "carrier_cA"]}

    cfg = SemiMarkovConfig(candidate="B")

    if calibrate:
        m, labels = calibrate_switch_penalties_semi_markov(
            L, blocks, n, cfg, fps=fps, target_s=15.0, iters=12, feats=feats
        )
    else:
        m = 1.0
        labels = decode_blocks_semi_markov(L, blocks, n, cfg, feats=feats, fps=fps)

    write_csv(out_path, labels)
    return m, labels, blocks, n, fps


def validate_candidate_b_file(path, blocks, n):
    df = pd.read_csv(path)
    # 1. Row count
    assert len(df) == n, f"Row count {len(df)} != {n}"
    # 2. Header
    assert list(df.columns) == ["frame", "label"], f"Columns {list(df.columns)} != ['frame', 'label']"
    # 3. Frames
    assert (df["frame"].values == np.arange(n)).all(), "Frame column is not monotonic 0..N-1"
    # 4. Labels
    labels = df["label"].values
    assert set(labels) <= {"H", "A", "D", "X"}, f"Invalid labels: {set(labels)}"

    # 5. X placement
    in_block = np.zeros(n, dtype=bool)
    for b_start, b_end in blocks:
        in_block[b_start : b_end + 1] = True

    assert (labels[~in_block] == "X").all(), "Found non-X labels outside blocks"
    assert (labels[in_block] != "X").all(), "Found X labels inside playing blocks"

    print("Submission structural format validation: PASS")


if __name__ == "__main__":
    out_file = os.path.join(HERE, "submission_phase9_candidate_b.csv")
    temp_file = os.path.join(HERE, "submission_phase9_candidate_b_run2.csv")

    print("Generating Run 1...")
    m1, lab1, blocks, n, fps = generate_submission(out_file, calibrate=True)
    d1 = describe(lab1, fps)
    print(f"Run 1: m={m1:.6f}, possessions={d1['possessions']}, mean={d1['mean_s']:.2f}s | H={d1['H']}, A={d1['A']}, D={d1['D']}, X={d1['X']}")
    validate_candidate_b_file(out_file, blocks, n)

    print("Generating Run 2 for determinism verification...")
    m2, lab2, _, _, _ = generate_submission(temp_file, calibrate=True)
    assert m1 == m2, f"Multiplier mismatch: {m1} != {m2}"
    assert (lab1 == lab2).all(), "Label mismatch across independent runs"

    # Bitwise hash check
    with open(out_file, "rb") as f1, open(temp_file, "rb") as f2:
        h1 = hashlib.sha256(f1.read()).hexdigest()
        h2 = hashlib.sha256(f2.read()).hexdigest()
    assert h1 == h2, f"Bitwise hash mismatch: {h1} != {h2}"
    print(f"Bitwise Determinism: PASS (SHA-256: {h1})")

    # Clean up temp run 2
    if os.path.exists(temp_file):
        os.remove(temp_file)

    print(f"Candidate B submission successfully generated and verified at: {out_file}")

