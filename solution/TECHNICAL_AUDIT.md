# Technical Audit: More than a Hack 2027 Ball Possession Solution

**Author:** Lead ML Researcher  
**Date:** November 2026  
**Target:** Competition-Grade Ball Possession Inference (`spellF1` Optimization)  
**Current Baseline Status:**
- `spellF1`: **0.3771** (ranking metric)
- `macroF1`: **0.5469**
- Best Provided Baseline: **0.4242** (`baseline_smooth125` on public slice) / **0.2433** (`baseline_smooth50` full-match floor)
- Predicted Possessions (Public 5-min): **18** (Ground Truth: **17**, mean duration: **14.2 s**)
- Mean Possession Duration (Public 5-min): **16.5 s**
- Full-Match Output: **336 possessions**, mean **15.0 s**
- Submission Validation: **PASSED** (199,319 frames, 0-indexed, no NaNs)

---

## Executive Summary

Our current system employs a three-state sticky Viterbi decoder with geometric distance-based likelihoods and a log-space bisection calibration targeting 15.0 seconds mean duration. While this lifted `spellF1` from 0.2435 to 0.3771 and aligned macro possession counts (336 match possessions, ~15 s mean), our comprehensive audit reveals critical structural bottlenecks:

1. **Dead Ball Collapse (`recall_D = 6.5%`)**: The dead ball class is severely evidence-starved. Ground truth has 1,469 dead-ball frames on the public test slice; our model captures only 96. Because `oob` and `still` require `ball_ok == True` (and tracking drops the ball the moment it goes out of play or rests), 93.5% of dead-ball stretches are misclassified as live play, merging distinct possessions and degrading temporal IoU.
2. **Half 2 Pre-Kickoff False Play**: The second-half tracking block begins at frame 110808, but kickoff is at frame 113741. In those 2,933 frames (~117 seconds) before kickoff, the unconstrained Viterbi decoder manufactures **2,726 frames of live possession** (`H: 1882`, `A: 844`), introducing spurious possessions before the ball is even in play.
3. **Boundary Jitter Destroying High-IoU Matches**: At $\text{IoU} = 0.5$, our model already matches 8 possessions ($F_1 = 0.4571$), outperforming `baseline_smooth125` (7 matched, $F_1 = 0.4242$). However, at $\text{IoU} = 0.7$, our $F_1$ collapses to $0.1143$ because transition boundaries suffer from 2–4 seconds of latency due to sluggish filters and missing ball gaps.
4. **Coupled Multiplier Bias**: Scaling `switch_team` and `switch_dead` with a single multiplier $m$ creates a false dependency between team turnovers and dead-ball stoppages.

Addressing these issues directly will comfortably surpass the 0.4242 threshold while maintaining strict compliance with competition rules (no ground-truth fitting, label-free calibration, fully deterministic).

---

## Detailed Findings

```
Severity Levels:
- CRITICAL: Severe logical or alignment defect corrupting scores or manufacturing spurious possessions.
- HIGH: Major structural defect causing significant loss of metric performance.
- MEDIUM: Suboptimal signal representation, filter distortion, or heuristic bias.
- LOW / INFORMATIONAL: Minor inconsistency or verified positive sanity check.
```

---

### Area A: Frame Alignment & Block Handling

#### Finding A.1 — Half 2 Pre-Kickoff False Possession Generation
- **Severity**: CRITICAL
- **Location**: `solution/mtah_possession/model.py` (lines 75, 93–96); `solution/run.py` (line 28)
- **Description**: The video blocks passed to the decoder (`half_video_blocks`) define Half 2 as frame `110808..190197`. However, the official kickoff of the second half (`GAMES[GAME]["kickoff_2nd"]`) occurs at frame `113741` (75 min 49 s 16 frames). During the 2,933 frames (117.3 seconds) between block start and kickoff, players are walking onto the pitch and the ball rests on the centre spot. Because Viterbi initializes unconstrained (`score = L[0].copy()`), proximity to the stationary ball causes the model to predict **1,882 frames of H and 844 frames of A** (2,726 live frames) before the half has even kicked off. In `possession_metrics.py`, `valid_runs` grades the entire half from 110808; these false possessions directly hurt precision and `spellF1`.
- **Recommended Correction**: Enforce a strict pre-kickoff dead-ball prior: for any frame before `kickoff` in Half 1 or `kickoff_2nd` in Half 2, clamp emission likelihoods to force state `D` (or initialize Viterbi state score with $[-\infty, -\infty, 0]$).
- **Dependencies / Data**: Standard library / existing `games.py` constants only (`kickoff`, `kickoff_2nd`). Zero new dependencies.

#### Finding A.2 — End-of-Half Terminal Boundary Bleed
- **Severity**: MEDIUM
- **Location**: `solution/mtah_possession/model.py` (lines 89–96)
- **Description**: At the end of each half (`90651` and `190197`), play terminates with the referee's whistle, but tracking continues up to the boundary. In `myteam.csv`, the final 1,000 frames of Half 2 contain 620 frames of live possession (`A`) that run right into the file boundary, producing truncated possessions with poor temporal IoU.
- **Recommended Correction**: Decay live ball emissions or bias towards `D` during the terminal 150 frames (6 seconds) if ball movement ceases or tracking degrades.
- **Dependencies / Data**: Pure algorithmic adjustment. No new data.

#### Finding A.3 — ATD Frame Indexing Verification
- **Severity**: INFORMATIONAL (VERIFIED CORRECT)
- **Location**: `solution/mtah_possession/atd.py` (lines 53–58)
- **Description**: Tactical ATD raw file was audited: contains exactly 199,319 records numbered `1..199319`. Video frames are 0-indexed (`0..199318`). The line `vid = ids[ok].astype(np.int64) - 1` performs an exact, contiguous 1-to-1 mapping with 0 duplicate frames, 0 dropped frames, and 0 non-unit steps. Frame alignment is mathematically sound.

---

### Area B: Feature Extraction & Physical Geometry

#### Finding B.1 — Unsmoothed Ball Velocity and Acceleration Absence
- **Severity**: HIGH
- **Location**: `solution/mtah_possession/features.py` (lines 52–57)
- **Description**: The docstring claims `# ball speed (m/s) over a short lag, then smoothed`. However, no smoothing function is actually called: `ball_speed = np.where(np.isfinite(v), v, np.nan)`. The raw displacement `disp / (lag / fps)` contains high-frequency single-frame tracker jitter. This noise propagates directly into `ctrl = ... / (1 + (v / 10)^2)`, causing control likelihoods to flicker violently. Furthermore, ball acceleration is never computed.
- **Recommended Correction**: Apply `_smooth(v, width=int(0.4 * fps))` on finite velocity segments, and compute acceleration magnitude $\Delta v / \Delta t$ to detect abrupt velocity impulses characteristic of kicks and tackles.
- **Dependencies / Data**: Numpy only.

#### Finding B.2 — Discarded High-Precision Tracker Player Speeds
- **Severity**: HIGH
- **Location**: `solution/mtah_possession/atd.py` (lines 46, 59–63); `solution/mtah_possession/features.py` (lines 59–65)
- **Description**: Metrica's raw ATD export provides instantaneous 2D player velocity in the 3rd field of each player slot (`<x, y, speed>`), populated on >95% of active tracks. `atd.py` completely discards this value (`players` array only stores $x, y$). `features.py` then attempts to re-estimate speed using 5-frame finite differences across track slots (`players[lag:] - players[:-lag]`). Because track slots frequently swap identities or drop out, cross-slot subtraction introduces massive artificial velocity spikes (>50 m/s).
- **Recommended Correction**: Retain slot speeds from the raw ATD file, or clamp finite-difference displacements to physical human limits (< 11.5 m/s) before computing team activity.
- **Dependencies / Data**: Existing ATD raw file format; zero external dependencies.

#### Finding B.3 — Hyper-Conservative Ball Gap Interpolation
- **Severity**: MEDIUM
- **Location**: `solution/mtah_possession/features.py` (lines 37–38, 7–23)
- **Description**: `_fill_short_gaps` uses `max_gap = cfg.ball_gap_s * fps = 37` frames (1.5 s). Out of 85,831 missing ball frames in the match, this bridges only 211 frames (0.24%). Over 40% of graded frames have no ball coordinates. During these missing intervals, all team evidence is shut to 0.0. When passes or clearances occur, the tracker frequently drops the ball for 2–4 seconds; because evidence is zero, the model cannot detect the turnover until seconds after the receiver controls it.
- **Recommended Correction**: Allow gap filling up to 2.5–3.0 seconds only when start and end points demonstrate continuous trajectory kinematics, while leaving long stoppages unfilled.
- **Dependencies / Data**: Numpy only.

#### Finding B.4 — 2D Planar Projection Artifacts During Aerial Duels
- **Severity**: MEDIUM
- **Location**: `solution/mtah_possession/features.py` (lines 44–50); `solution/mtah_possession/model.py` (lines 51–53)
- **Description**: In 2D tactical tracking, high balls (crosses, clearances, goal kicks) project onto players standing underneath them. At the trajectory apex, 2D planar velocity $v \to 0$. The model calculates $v \approx 0$ and small 2D distance $d_{\min}$, causing `ctrl` to surge near 1.0, falsely assigning possession to players who are 15 metres below the flight path.
- **Recommended Correction**: Suppress `ctrl` when a ball's 2D speed drops to near zero in the middle of a high-speed arc without an accompanying deceleration signature of a foot/chest trap.
- **Dependencies / Data**: Algorithmic heuristic on ball speed time-series.

---

### Area C: Probabilistic Model (Log-Emissions)

#### Finding C.1 — Catastrophic Dead-Ball Evidence Deficit (`recall_D = 6.5%`)
- **Severity**: CRITICAL
- **Location**: `solution/mtah_possession/model.py` (lines 59–65); `solution/mtah_possession/features.py` (lines 70–76)
- **Description**: 
  - On the public test slice, true dead ball constitutes 1,469 frames. Our model predicts `D` on only 96 frames ($F_{1, D} = 0.1226$, Recall = 0.0654).
  - Root Cause: Both `oob` and `still` are gated behind `ball_ok == True` (`oob = ball_ok & ...`; `slow = ... & ball_ok`). But whenever the ball leaves the pitch or goes dead, the camera and tracker lose the ball (`ball_ok = False`). The exact frames where dead-ball evidence should be strongest receive $0.0$ emission bonus!
  - The only active cue during an occlusion is `activity` ($\text{gain} = 0.02$). Even if all players stand completely still, it accumulates only $0.02 \times 25 = 0.5$ nats/sec, unable to overcome the transition penalty ($0.866$ nats) until several seconds have elapsed.
  - As a result, 93.5% of dead-ball time is mislabelled as `H` or `A`, bridging distinct possessions and heavily degrading IoU.
- **Recommended Correction**:
  1. Detect boundary exits: If the last valid ball coordinate was within 2 m of a touchline/byline and the ball disappears, assign persistent `oob` evidence.
  2. Sustained absence prior: An unseen ball lasting > 4.0 seconds with low player activity is almost certainly dead ball.
  3. Integrate Smart Tagging: Set pieces in `BAR-ATH_tactical_smart_tagging.xml` (`Throw-in`, `Corner Kick`, `Goal Kick`, `Free Kick`) define exact dead-ball intervals leading up to restarts.
- **Dependencies / Data**: ATD kinematics + optional Smart Tagging XML parsing (using Python standard library `xml.etree`).

#### Finding C.2 — Arbitrary 60-Metre Penalty on Asymmetric Tracking Dropouts
- **Severity**: HIGH
- **Location**: `solution/mtah_possession/model.py` (lines 44–46)
- **Description**:
  ```python
  dH[np.isnan(dH) & ~np.isnan(dA)] = 60.0
  dA[np.isnan(dA) & ~np.isnan(dH)] = 60.0
  ```
  On 899 frames, Home players are missing while Away players are tracked; on 170 frames, vice versa. Artificially setting the unseen side's distance to 60.0 m yields $d_A - d_H \approx -58$ m, instantly saturating $\tanh((d_A - d_H)/2) = -1.0$. If a Home player is shielding the ball and temporarily unassigned by the tracker, this penalty immediately awards full control to Away, triggering false turnover spikes.
- **Recommended Correction**: When one team has zero tracked players near the ball, clamp distance difference to a neutral cap ($\le 3.0$ m) or fall back to previous state persistence rather than manufacturing extreme artificial contrast.
- **Dependencies / Data**: Numpy only.

#### Finding C.3 — Proximity vs Controlled Touch Confusion
- **Severity**: MEDIUM
- **Location**: `solution/mtah_possession/model.py` (lines 53–54)
- **Description**: Team evidence is governed purely by distance difference: $\text{side} = \tanh((d_A - d_H) / 2.0)$. In a tight duel or contested press where $d_H = 0.8$ m and $d_A = 0.7$ m, possession flips between frames based on centimetre-level tracking noise.
- **Recommended Correction**: Incorporate a directional/displacement consensus: a player possesses the ball if their velocity vector aligns with ball displacement, preventing non-possessing nearby defenders from snatching ownership.
- **Dependencies / Data**: Numpy vector mathematics.

---

### Area D: Temporal Decoder & Calibration

#### Finding D.1 — Coupled Multiplier Imposing Rigid Topological Bias
- **Severity**: HIGH
- **Location**: `solution/mtah_possession/pipeline.py` (lines 22–23)
- **Description**: Calibration optimizes a single multiplier $m \approx 0.1732$ that scales both `switch_team` ($6.0 \to 1.039$) and `switch_dead` ($5.0 \to 0.866$). Team-to-team handovers ($H \leftrightarrow A$) and live-to-dead transitions ($H/A \leftrightarrow D$) represent completely different physical processes. Coupling them means that making the model responsive to turnovers simultaneously makes it volatile to dead-ball chattering.
- **Recommended Correction**: Parameterize calibration with two distinct scales: $m_{\text{team}}$ for team switches and $m_{\text{dead}}$ for dead ball, or optimize $m_{\text{team}}$ while setting $m_{\text{dead}}$ via stoppage duration statistics.
- **Dependencies / Data**: Standard library bisection / grid search.

#### Finding D.2 — Duration Prior vs Boundary IoU Tradeoff
- **Severity**: HIGH
- **Location**: `solution/mtah_possession/pipeline.py` (lines 10–38)
- **Description**: The 12-iteration bisection minimizes $| \text{mean\_s} - 15.0 |$. While this correctly tunes the first moment of the distribution, it is indifferent to boundary placement. On the public slice:
  - $\text{IoU} @ 0.5$: $F_1 = 0.4571$ (8 matched spells) — beating the baseline.
  - $\text{IoU} @ 0.7$: $F_1 = 0.1143$ (only 2 matched spells).
  The steep drop-off at IoU 0.6 and 0.7 occurs because transition boundaries are smeared by 2–4 seconds of filter latency. `spellF1` ranks on the *unweighted average* across $[0.3, 0.4, 0.5, 0.6, 0.7]$. Boundary precision is just as vital as possession count.
- **Recommended Correction**: Reduce filter lag (shorten window widths for transition detection, use causal/half-normal kernels) so that possession onsets and offsets snap sharply to real physical events.
- **Dependencies / Data**: Algorithmic only.

#### Finding D.3 — Suppression of Short, Genuine Possessions
- **Severity**: MEDIUM
- **Location**: `solution/mtah_possession/model.py` (lines 72–85)
- **Description**: A 2-to-3-second possession (50–75 frames, e.g. clearance, quick pass sequence, intercepted counter) must accumulate $> 2 \times 1.039 = 2.08$ nats to switch in and out. If the ball is contested or moving fast, evidence per frame is low ($\le 0.02$), accumulating at most $1.5$ nats. The Viterbi decoder completely smooths over these legitimate short spells.
- **Recommended Correction**: Introduce transition bonus on high-confidence ball kicks/touches to lower the barrier for entering a new possession state.
- **Dependencies / Data**: Numpy only.

---

### Area E: Reproducibility & Pipeline Integrity

#### Finding E.1 — Determinism, Packaging and Interface
- **Severity**: INFORMATIONAL (VERIFIED COMPLIANT)
- **Location**: `solution/run.py`; `solution/tests/test_synthetic.py`
- **Description**:
  - The pipeline is fully deterministic: no random seeds, stochastic sampling, or multithreaded race conditions.
  - Reproducibility is clean via `python solution/run.py --out myteam.csv`.
  - Dependencies are strictly standard library, `numpy`, and `pandas`.
  - Synthetic test suite (`solution/tests/test_synthetic.py`) runs in < 15 seconds and verifies calibration constraints within 20% of target.
  - File format validation passes 100% of checks on row count (199,319), column schema (`frame,label`), integer frame continuity, and label set $\{H, A, D, X\}$.

---

## Benchmark Metrics Comparison

| Submission / Metric | `macroF1` | `spellF1` (Ranking) | Possessions (Public) | Mean Dur (Public) | `transF1` ($\tau=25$) | Latency | `frag` |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ground Truth (Public)** | — | **1.0000** | **17** | **14.2 s** | — | 0.0 | 1.00 |
| `baseline_smooth125` (Best Baseline) | 0.5812 | **0.4242** | 16 | 18.5 s | 0.1667 | -7.7 f | 0.81 |
| **`myteam` (Current)** | 0.5469 | **0.3771** | **18** | **16.5 s** | **0.2105** | **-4.0 f** | **0.90** |
| `baseline_smooth50` (Floor) | 0.6207 | 0.3256 | 26 | 10.8 s | 0.0816 | +4.0 f | 1.43 |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 | 7.6 s | 0.2034 | +8.2 f | 1.90 |
| `baseline_raw` | 0.5972 | 0.0378 | 131 | 2.0 s | 0.1410 | -3.1 f | 6.52 |

---

## Proposed Implementation Plan

To systematically surpass `spellF1 = 0.4242` without violating competition constraints or overfitting the public 5-minute slice, we recommend the following phased implementation:

### Phase 1: Pre-Kickoff Clamping & Boundary Fixes (Immediate Win)
- Enforce strict state `D` prior to `GAMES[GAME]["kickoff"]` (frame 11186) and `GAMES[GAME]["kickoff_2nd"]` (frame 113741).
- Prevent the synthesis of 2,726 false live frames in Half 2 pre-kickoff.
- Expected impact: Significant reduction in false positive spells and immediate precision boost.

### Phase 2: Dead-Ball Evidence Reconstruction
- Remove the strict `ball_ok` gating on `oob`: If ball was within 2 m of boundary when last seen, sustain `oob` evidence across missing frames.
- Re-balance `dead` emission formula: Increase `oob_gain` and `activity_gain` sensitivity so that genuine stoppages accumulate decisive log-likelihood within 1.5 seconds.
- Expected impact: Resolves the 93.5% dead-ball recall deficit, splitting mistakenly merged possessions and boosting IoU @ 0.6 and 0.7.

### Phase 3: Kinematic Feature Refinement
- Smooth ball speed with a short causal/symmetric Gaussian kernel (width 7–9 frames) to eliminate tracker displacement jitter.
- Replace synthetic finite-difference player speed with ATD slot speeds or apply velocity capping (< 11.5 m/s) to eliminate identity-swap spikes.
- Restrict extreme 60 m asymmetric distance penalty to a neutral saturation cap ($\le 3.0$ m).

### Phase 4: Verification & Score Validation
- Run synthetic unit test suite `test_synthetic.py` to ensure zero regressions.
- Execute full match generation `run.py --out candidate.csv`.
- Run `validate_submission.py`.
- Run single evaluation scoring against `run_evaluation.py` and document progress.

