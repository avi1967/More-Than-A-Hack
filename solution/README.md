# Duration-Aware Semi-Markov Possession Inference (Candidate B)

This directory contains the complete source code, tests, and documentation for the football possession inference system for **More than a Hack 2027 (Phase 1: Ball Possession)**.

The system decodes continuous optical tracking into discrete possession spells ($H$ for Home, $A$ for Away, $D$ for Dead Ball, and $X$ for Halftime/Non-playing blocks) on every frame ($199,319$ frames) of the match.

---

## Quickstart

Run from the root of the repository (`more-than-a-hack-2027/`):

```bash
# 1. Install dependencies
pip install -r solution/requirements.txt

# 2. Run automated test suite (27 unit and synthetic tests, no match data required)
python -m pytest solution/tests/

# 3. Generate the verified competition submission (Candidate B)
python solution/generate_candidate_b_submission.py

# 4. Validate output file structure and contracts
python evaluation/validate_submission.py solution/submission_phase9_candidate_b.csv
```

---

## System Architecture

The pipeline consists of seven decoupled stages designed from first physical principles:

1. **Tracking Ingestion (`atd.py`)**: Parses raw camera-derived ASCII tracking data and stadium geometry coordinates into pitch meters.
2. **Confidence-Aware Feature Extraction (`features.py`)**: Computes kinematic metrics, short-gap ball interpolation ($\le 1.5\text{ s}$), velocity clamping, and continuous sensor confidence.
3. **Carrier Continuity Mechanism (`carrier_continuity.py`)**: Prevents false possession turnover when the ball carrier is momentarily lost from optical tracking, maintaining virtual carrier radius ($0.7\text{ m}$) with exponential decay ($\tau = 2.0\text{ s}$).
4. **Refined Probabilistic Dead-Ball Detection V2 (`dead_ball.py`)**: 6-stream evidence fusion including out-of-bounds sigmoids, aerial flight suppression, carrier-holding exceptions, and collective team dynamics.
5. **Physical Emission Log-Likelihoods (`model.py`)**: Generates normalized frame-level emission log-potentials for states $H, A, D$.
6. **Duration-Aware Semi-Markov Decoder (`semi_markov.py`)**: Segment-level dynamic programming with soft minimum duration barriers ($d_{\min} = 1.0\text{ s}$), eliminating sub-second 1-frame chatter artifacts.
7. **Bisection Calibration (`pipeline.py`)**: Aligns overall spell duration to a nominal $15.0\text{ s}$ prior ($m = 0.472147$).

---

## File Structure

```
solution/
├── mtah_possession/                 # Core Python package
│   ├── __init__.py
│   ├── atd.py                       # ATD tracking parser & coordinate normalization
│   ├── carrier_continuity.py        # Ball-carrier persistence & touch barriers
│   ├── dead_ball.py                 # Multi-stream dead-ball probabilistic detector
│   ├── features.py                  # Kinematic features & confidence estimation
│   ├── model.py                     # Emission models & Viterbi reference
│   ├── pipeline.py                  # Full match pipeline & bisection search
│   ├── semi_markov.py               # Exact semi-Markov segment dynamic program
│   └── smart_tagging.py             # Tactical tag alignment (stoppage contexts)
├── tests/                           # Automated test suite
│   ├── test_phase9_semi_markov.py   # 10 adversarial unit tests for semi-Markov DP
│   └── test_synthetic.py            # 17 synthetic benchmark tests
├── generate_candidate_b_submission.py # Verified one-command submission generator
├── run.py                           # CLI entry point for full pipeline
├── requirements.txt                 # Pinned Python package dependencies
├── TECHNICAL.md                     # Complete scientific technical documentation
└── *.md                             # Engineering audit and design phase reports
```

---

## Verified Submission Integrity

Running `python solution/generate_candidate_b_submission.py` generates the frozen competition candidate:
- **Output File**: `solution/submission_phase9_candidate_b.csv`
- **Rows**: 199,319
- **Labels**: Home ($H$): 59,269 (29.7%), Away ($A$): 38,964 (19.5%), Dead ($D$): 60,624 (30.4%), Unknown/Halftime ($X$): 40,462 (20.3%)
- **Possession Spells**: 262 possessions, mean duration 15.00s
- **Deterministic SHA-256 Digest**: `8d476d796f006dfb52be87e4a0d89c01d74375cd0665451f28877d8b54ad9d03`
- **Validation**: Passed official `evaluation/validate_submission.py`
