# Technical Document: Possession Inference System (Phase 9 Candidate B)

**Challenge:** More than a Hack 2027 (Football Possession Prediction)  
**Accepted Model Candidate:** Candidate B (Duration-Aware Semi-Markov Sequence Decoder)  
**Output Submission File:** `solution/submission_phase9_candidate_b.csv`  
**Fallback Submission File:** `solution/submission_phase8_v2.csv`  
**Status:** Frozen Competition Candidate — Ready for Submission  
**Verification:** Bitwise Deterministic (SHA-256: `8d476d796f006dfb52be87e4a0d89c01d74375cd0665451f28877d8b54ad9d03`)

---

## 1. System Architecture Overview

The competition ranks submissions primarily on **spellF1** (mean $F_1$ score across temporal IoU thresholds $\tau \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$) where possessions are maximal contiguous spells of Home ($H$) or Away ($A$), separated by Dead-ball stoppages ($D$) or unobserved intervals ($X$). The secondary tie-break metric is frame-level **macroF1**.

The final production pipeline consists of seven cleanly decoupled components:
1. **Raw Tracking & Metadata Ingestion**: Parsing raw ATD tracking and XML metadata.
2. **Confidence-Aware Tracking Features**: Physical kinematic filtering, imputation gating, and continuous confidence.
3. **Carrier Continuity Mechanism**: Probabilistic memory protecting against ball-carrier tracking dropout.
4. **Refined Probabilistic Dead-Ball Detection (V2)**: Independent estimation of $P(D \mid \text{evidence})$ with flight counter-evidence and stationary-ball carrier holding logic.
5. **Physical Emission Log-Likelihoods**: Physically grounded $H, A, D$ evidence fusion without label fitting.
6. **Duration-Aware Semi-Markov Sequence Decoding (Candidate B)**: Exact segment-level dynamic programming with soft minimum duration barriers.
7. **Bisection Calibration**: Domain-prior alignment to nominal $15.0\text{ s}$ mean possession duration.

```
  [Raw ATD Tracking] + [XML Metadata] + [Smart Tagging (Contextual Only)]
                            │
                            ▼
        [1. Feature Extraction & Quality Indicators]
         - Short-gap kinematic imputation (<= 1.5s, <= 35 m/s)
         - Continuous ball confidence in [0.0, 1.0]
         - Reliable velocity estimation & finite smoothing
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
 [2. Carrier Continuity]         [3. Dead-Ball Detector V2]
  - Carrier touch (<= 1.2m)       - Boundary proximity & outside sigmoid
  - Tackle barrier (<= 0.8m)      - Inactivity vs carrier holding
  - Exponential decay (tau=2.0s)   - Flight counter-evidence
  - Virtual carrier radius (0.7m) - Calm, repositioning, box density
            └───────────────┬───────────────┘
                            ▼
           [4. Physical Log-Emissions (L)]
            - H / A control contrast & dead suppression
            - D log-odds relative to base stoppage rate
                            │
                            ▼
     [5. Duration-Aware Semi-Markov Decoder (Candidate B)]
      - Exact segment DP: max_{d in [1, D_max]}
      - Centered soft minimum barrier (d_min = 1.0s)
      - Dynamic touch discounts & restart costs
      - Bisection calibration (target = 15.0s, m = 0.472147)
                            │
                            ▼
           [Output Submission CSV (199,319 rows)]
            - 262 possessions, mean duration 15.00s
            - 0 one-frame blips; Validated by official checker
```

---

## 2. Component Details

### 2.1 Tracking Data Ingestion & Geometry
- **Inputs**: Raw ATD text tracking files ($199,319$ frames at $25\text{ fps}$), XML match metadata, and tactical Smart Tagging XML.
- **Pitch Dimensions**: Calibrated to match stadium pitch ($105.0\text{ m} \times 68.0\text{ m}$).
- **Halves & Blocks**: Play is segmented into two active blocks (Half 1: frames $11,185 \dots 90,651$; Half 2: frames $110,808 \dots 190,197$). Halftime ($20,156$ frames) and pre/post-match frames are strictly labeled `"X"`.

### 2.2 Confidence-Aware Features (Implemented in `features.py`)
- **Imputation Policy**: Imputes missing ball positions strictly across short gaps ($\le 1.5\text{ s}$, $37$ frames). Gaps with displacement speed $> 35\text{ m/s}$ or crossing out-of-bounds boundaries are rejected as unphysical.
- **Continuous Ball Confidence**: Assigns $1.0$ to directly observed frames, decays smoothly towards $0.1$ based on gap length and edge distance, and sets $0.0$ for missing/dead-ball frames.
- **Kinematic Filtering**: Missing ball speeds default to neutral rolling velocity ($3.0\text{ m/s}$) rather than $0.0\text{ m/s}$, preventing artificial control spikes. Player velocities $> 12\text{ m/s}$ (sensor slot swaps) are filtered to NaN.
- **Player Quality Metrics**: Computes active player counts (`valid_home_players`, `valid_away_players`), missing player fraction, and median moving-window activity.

### 2.3 Robust Carrier Continuity (Implemented in `carrier_continuity.py`)
- **Carrier Acquisition**: Player within intimate contact radius $r_{\text{touch}} = 1.2\text{ m}$ with ball speed $< 6.0\text{ m/s}$ and high observation confidence establishes carrier status ($c_H = 1.0$ or $c_A = 1.0$).
- **Dropout Mitigation**: If tracking temporarily loses the ball carrier, carrier confidence decays exponentially with timescale $\tau_{\text{carrier}} = 2.0\text{ s}$. During dropout, a virtual carrier distance $r_{\text{virt}} = 0.7\text{ m}$ is maintained, preventing the nearest opponent from inheriting possession merely due to sensor occlusion.
- **Intimate Tackle Barrier**: When opponent carrier continuity is active ($c > 0.2$), taking possession requires intimate tackle distance $r_{\text{tackle}} = 0.8\text{ m}$ rather than casual proximity ($1.2\text{ m}$).
- **Kick-Decay**: When ball velocity exceeds physical dribbling speed ($> 6.0\text{ m/s}$), carrier continuity decays rapidly ($\tau_{\text{kick}} = 0.4\text{ s}$).

### 2.4 Refined Probabilistic Dead-Ball Detection V2 (Implemented in `dead_ball.py`)
- **Signal Streams**:
  1. *Boundary Proximity & Out-of-Bounds Probability*: Sigmoid mapping signed boundary distance ($\text{margin} = -0.3\text{ m}, \tau = 0.25\text{ m}$).
  2. *Ball Inactivity Duration*: Elapsed stillness ($v < 0.5\text{ m/s}$).
  3. *Carrier Holding Exception*: When an active player holds the ball ($d_{\min} \le 1.0\text{ m}$, e.g. goalkeeper holding ball or defender shielding), dead-ball inactivity accumulation is inhibited.
  4. *Live Inbounds Flight Counter-Evidence*: When ball travels at high speed ($> 6.0\text{ m/s}$) inside pitch boundaries, $P(D)$ is actively suppressed by up to $-4.0\text{ log-odds}$, protecting aerial passes from false stoppage segmentation.
  5. *Collective Player Dynamics*: Team centroid repositioning velocities, median walking calmness, and penalty box packing ($D_{\text{box}} \ge 0.35$).
  6. *Smart Tagging Context*: Restart tags (throw-in, corner, goal kick, free kick, kick-off) provide contextual stoppage windows, but **never assign team possession** (addressing the fact that 32 of 63 tags have `Team=N/A`).

### 2.5 Physical Emission Log-Likelihoods (Implemented in `model.py`)
- **Spatial Control Contrast**:
  $$\text{ctrl} = \frac{\exp(-d_{\min} / 4.0\text{ m})}{1 + (v_{\text{ball}} / 10.0\text{ m/s})^2} \cdot \text{ball\_confidence}$$
  $$\text{side} = \tanh\left(\frac{d_A - d_H}{2.0\text{ m}}\right)$$
- **Dead Suppression on Live Play**: High dead-ball evidence softly suppresses live emissions:
  $$L_t(H) = 0.5 \cdot \text{gain}_{\text{side}} \cdot \text{ctrl} \cdot \text{side} - 0.5 \cdot \max(0, L_t(D))$$
  $$L_t(A) = -0.5 \cdot \text{gain}_{\text{side}} \cdot \text{ctrl} \cdot \text{side} - 0.5 \cdot \max(0, L_t(D))$$
  $$L_t(D) = \text{gain}_{\text{dead}} \cdot (\text{logit}(P(D_t)) - \text{logit}(P_0(D)))$$

### 2.6 Duration-Aware Semi-Markov Sequence Decoder (Implemented in `semi_markov.py`)
- **Model Formulation**: Exact segment-level Hidden Semi-Markov Model (HSMM) dynamic program over durations $d \in [1, D_{\max}]$ ($D_{\max} = 6,000$ frames = $240\text{ s}$).
- **Duration Potential**: Centered soft minimum barrier:
  $$\ln \tilde{P}(d) = -5.0 \cdot \left[\max\left(0, \frac{25 - d}{25}\right)\right]^2 - 0.003 \cdot d - \max_{d'} \text{pen}(d')$$
  Mode at $d = 25$ frames ($1.0\text{ s}$) pays $0.0\text{ nats}$ penalty. Sub-second chatter ($d < 25$) pays quadratic penalty ($-4.54\text{ nats}$ at 1 frame).
- **Transition Costs**:
  - $H \leftrightarrow A$: $\text{switch\_team} \times (0.4 \text{ if controlled touch else } 1.0)$.
  - $H/A \to D$: $\text{switch\_dead}$.
  - $D \to H/A$: $\text{switch\_dead} \times 0.5$.
- **Boundary Precision**: Completely eliminates 1-frame ($0.04\text{ s}$) artifacts natively without heuristic boundary refinement.

### 2.7 Calibration Procedure (Implemented in `semi_markov.py` and `pipeline.py`)
- Bisection search in log-space over $m \in [0.02, 2.0]$ in 12 iterations.
- Finds scalar multiplier $m$ such that decoded full-match mean possession duration matches nominal $15.0\text{ s}$.
- For Candidate B on `BAR-ATH`: $m = 0.472147$, yielding $262$ possessions of mean duration $15.00\text{ s}$.

---

## 3. Scientific Limitations & Critical Caveats

### 3.1 Synthetic Benchmark vs. Unseen Match Reality
Synthetic matches provide controlled verification of edge cases (aerial duels, carrier loss, rapid turnovers), but do not perfectly replicate the physical complexity of real matches. Superiority on synthetic data is strong evidence of algorithmic soundness, but is **not** an empirical guarantee of a specific score on the unseen competition match.

### 3.2 The Calibration Fragility Phenomenon
The Phase 9.1 audit revealed that calibrating switch penalties to a $15.0\text{ s}$ target is an empirical domain prior, not a physical law. If an unseen evaluation match exhibits an atypical tactical tempo:
- In matches dominated by long possessions ($> 40\text{ s}$), forcing a 15-second mean duration drives $m \to 0.02$, over-fragmenting genuine possessions.
- In matches dominated by rapid turnovers ($1-3\text{ s}$), forcing a 15-second mean duration raises $m \to 2.0$, artificially merging separate possessions.
The accepted submission retains calibration because `BAR-ATH` is an authentic La Liga match whose rhythm matches standard domain statistics.

### 3.3 Sensor Occlusion & Tracking Dropout
Carrier continuity bridges tracking dropouts up to $1.5-2.0\text{ seconds}$. Occlusions exceeding $3-4\text{ seconds}$ in contested scrums inevitably decay continuity towards neutral, relying on subsequent spatial recapture.

### 3.4 Bounded Temporal Horizons
The semi-Markov decoder uses non-causal dynamic programming within playing halves (offline decoding). No future information leaks across halves or beyond the match boundaries.

### 3.5 Absence of Private Ground-Truth Tuning
No model parameter, heuristic threshold, or evidence weight was fitted or tuned against public (the 17-spell validation slice) or private ground truth. All thresholds were set a priori from physical principles.

---

## 4. Exact Reproduction & Verification

### 4.1 System Requirements
- Python: $3.9+$ (Verified on Python 3.13.3)
- Core dependencies: `numpy` ($\ge 1.24$), `pandas` ($\ge 1.5$)
- OS: OS-independent (Windows, Linux, macOS)
- Memory: $< 200\text{ MB}$ dynamic RAM
- Single Decode Runtime: $\approx 7.6\text{ seconds}$ on `BAR-ATH`

### 4.2 Data Access & Directory Setup
Per competition terms, match tracking and video assets are licensed exclusively for enrolled participants and must not be redistributed in public repositories:
1. Enrol at the official portal: [More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027).
2. Download the match archive for `BAR-ATH`.
3. Unpack into the project root `./data/` folder, or export the environment variable:
   ```bash
   export MTAH_DATA_DIR=/path/to/extracted/data
   # On Windows PowerShell:
   # $env:MTAH_DATA_DIR="D:\path\to\extracted\data"
   ```
4. Verify files match the structure described in `DATA.md` (`tracking-atd.txt`, `tracking-atd-metadata.xml`, `smart-tagging-tags.xml`).

### 4.3 Reproduction Commands
To regenerate the final Candidate B submission from scratch:
```bash
python solution/generate_candidate_b_submission.py
```
Or via the primary CLI entry point:
```bash
python solution/run.py --out solution/submission_phase9_candidate_b.csv --decoder semi_markov
```

To validate the generated submission using official competition tools:
```bash
python evaluation/validate_submission.py solution/submission_phase9_candidate_b.csv
```

To run the complete automated test suite (27 passing unit and integration tests):
```bash
python -m pytest solution/tests/
```

### 4.4 Verification & Deterministic Integrity
Candidate B produces bitwise deterministic output:
- **Rows**: Exactly 199,319 frames ($0 \dots 199318$)
- **Label Distribution**: $H = 59,269$ ($29.7\%$), $A = 38,964$ ($19.5\%$), $D = 60,624$ ($30.4\%$), $X = 40,462$ ($20.3\%$)
- **Possession Segments**: 262 possessions with mean duration $15.00\text{ s}$
- **Bisection Multiplier**: $m = 0.472147$
- **SHA-256 Digest**: `8d476d796f006dfb52be87e4a0d89c01d74375cd0665451f28877d8b54ad9d03`
- **Validation**: Passed official `evaluation/validate_submission.py` format checks.
