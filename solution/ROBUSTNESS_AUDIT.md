# Robustness Report Audit and Independent Validation

**Author:** Independent Validation & Technical Audit Team  
**Subject:** Technical audit of `solution/ROBUSTNESS_REPORT.md` and associated robustness experiments  
**Scope:** Strict verification of experimental validity, metric calculations, synthetic vs real data domains, parameter invariance, and dropout failure modes  
**Compliance Notice:** Conducted without modifying model logic, tuning hyperparameters, or referencing ground-truth labels.

---

## 1. Executive Summary

This audit independently assesses the experimental protocols, code implementations, metric formulations, and claims documented in `solution/ROBUSTNESS_REPORT.md`.

While the underlying pipeline operates deterministically and without code errors, the audit revealed several **critical methodological limitations and overstated claims** in the original report:
1. **Domain Conflation**: Experiments 1, 2, 3, and 4 were conducted exclusively on an idealized synthetic match generator (`make()`), where artificial geometry ($|d_A - d_H| \approx 10$ m) severely saturated evidence ($75.3\%$ of frames having $\tanh > 0.99$). The report presented these results as general properties of the match inference system.
2. **False Parameter Invariance**: The reported "100.00% label invariance across 36 parameter perturbations" is an artifact of synthetic evidence saturation combined with coarse 12-step bisection calibration. On real match tracking data, uncalibrated parameter perturbations alter **$5.4\% - 8.6\%$** of frame emissions and **$1.1\% - 1.5\%$** of decoded labels after calibration.
3. **Inertia Forward-Fill Misidentified as Recovery**: The reported $99.93\%$ agreement under $40\%$ ball dropout reflects state persistence (forward-filling), not possession recovery. When ball dropout is concentrated at possession transitions or occurs in bursts $> 5$ seconds, handover detection degrades significantly: transition boundaries displace by $1.89$ s on average, and $20$-second bursts cause complete sequence collapse (16 possessions collapse into 1).
4. **Critical Fragility to Ball-Carrier Loss**: While the system tolerates random slot dropouts, losing the track of the **nearest player to the ball** causes accuracy to crash from **$88.42\%$ to $25.25\%$**, because the opponent's nearest defender immediately inherits possession.
5. **Unsupported Tactical Narratives**: The original report inferred tactical concepts (e.g. "Athletic pressing", "Barcelona patient build-up", "defending leads") without independent validation. These narratives have been removed and replaced with empirical sequence metrics.

### Original Claim Classification Matrix

| Original Claim in Report | Audit Classification | Justification & Summary of Findings |
| :--- | :---: | :--- |
| **"100% label identity across 36 parameter permutations"** | **Invalid** | True only on synthetic data due to saturated evidence. On real tracking, emissions shift by $5-8\%$ and labels shift by $1.1-1.5\%$. |
| **"Exceptional resilience to ball dropouts (99.93% agreement)"** | **Valid with qualifications** | Holds for i.i.d. single-frame drops inside steady possessions via inertia. Fails under consecutive bursts $> 5$ s or dropouts at transitions. |
| **"High resilience to player tracking loss"** | **Valid with qualifications** | Valid for non-carrier player slots ($88\%$ acc). Completely invalid if the ball carrier drops ($25.25\%$ acc, catastrophic failure). |
| **"Structural topology invariance under spatial noise"** | **Validated** | Invariant transition count ($18$ vs $18$) maintained up to $\sigma = 0.50$ m noise via boundary refinement and transition penalties. |
| **"Independent half-boundary resets"** | **Validated** | Verified: zero temporal leakage across halves; each half starts from independent initial conditions. |
| **"Tactical asymmetry between halves"** | **Invalid** | Tactical interpretations (pressing, defensive tactics) are unverified speculation; measurable physical statistics are valid. |
| **"Peak memory under 1.1 GB"** | **Validated** | Verified: $1,094.3$ MB peak RAM on 199,319 frames. |
| **"Calibration runtime ~17 minutes"** | **Validated** | Verified: $1,004$ s spent in calibration loop due to unvectorized boundary refinement. |

---

## 2. Experiment-by-Experiment Validity Assessment

Every experiment in `run_robustness_experiments.py` was inspected for data domain, perturbation timing, pipeline completeness, caching issues, seed determinism, and metric validity:

| Exp # | Experiment Title | Tested Data Domain | Perturbation Timing | Reruns Complete Pipeline? | Caching / Leakage Issues | Metric Validity | Audit Finding |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | Ball Dropout | Synthetic (`make()`, $N=6000$) | Raw ball array before `compute()` | Yes | None detected | Valid | Measures i.i.d. dropout only; misses burst/transition vulnerability. |
| **2** | Tracking Dropout | Synthetic (`make()`, $N=6000$) | Raw player array before `compute()` | Yes | None detected | Valid | Measures random slot loss only; ignores ball-carrier loss. |
| **3** | Parameter Perturbation | Synthetic (`make()`, $N=6000$) | `Config` modified before `compute()` | Yes | None detected | Valid | Saturated synthetic data masked real parameter sensitivity. |
| **4** | Temporal Stability | Synthetic (`make()`, $N=6000$) | Coordinate array before `compute()` | Yes | None detected | Valid | Correctly measured boundary jitter and transition counts. |
| **5** | Half Symmetry | Real Match (`BAR-ATH`, $N=199319$) | None (observational) | Yes | None detected | Valid | Quantitative metrics valid; tactical interpretations unsupported. |
| **6** | Duration Distribution | Real Match (`BAR-ATH`, $N=199319$) | None (observational) | Yes | None detected | Valid | Quantiles correctly computed over maximal contiguous spells. |
| **7** | Fragmentation | Real Match (`BAR-ATH`, $N=199319$) | None (observational) | Yes | None detected | Valid | Correctly audited micro-spells ($<1$s) and long spells ($>45$s). |
| **8** | Runtime Profiling | Real Match (`BAR-ATH`, $N=199319$) | None (observational) | Yes | None detected | Valid | Memory tracked via `tracemalloc`; timing profiled per module. |

---

## 3. Reproducibility and Random Seed Checks

1. **Seed Management**:
   - Synthetic fixtures used explicitly fixed NumPy RNG seeds (`seed=101`, `102`, `103`, `104`, `105`, `106`).
   - Bit-for-bit determinism was confirmed: running repeated identical seeds produces identical floating-point feature arrays and discrete integer Viterbi paths.
2. **Deterministic Decoder Execution**:
   - Both `viterbi` and `viterbi_duration_aware` use deterministic dynamic programming without stochastic components or tie-break randomness.
3. **Environment Independence**:
   - All routines rely strictly on Python 3.9+ standard library, NumPy, and Pandas, with no system-dependent random seeds.

---

## 4. In-Depth Investigation: Parameter Invariance

### The Problem
`ROBUSTNESS_REPORT.md` reported $100.00\%$ sequence agreement across all 36 parameter permutations ($\pm 5\%$, $\pm 10\%$, $\pm 20\%$) for `ctrl_scale_m`, `side_scale_m`, `ball_speed_ref`, `walk_speed`, `dead_gain`, and `switch_team`.

### Empirical Audit Analysis
We extracted the exact emission differences and compared synthetic benchmark tracking against real match tracking (`BAR-ATH`, frames 20,000–30,000):

#### A. Synthetic Data Analysis
- Synthetic player positions generated by `make()` place the ball carrier at $d \approx 0.4$ m from the ball, while all opponents are $8 - 25$ m away.
- As a result, $|d_A - d_H|$ averages **$9.82$ m** (median $8.88$ m).
- **$75.3\%$ of synthetic frames have $\tanh(|d_A - d_H| / 2.0) > 0.99$**!
- Because the evidence is completely saturated at $+1.0$ or $-1.0$, log-emissions $|dL|$ changed by less than $0.001$ nats under scale perturbations.
- Even before calibration, uncalibrated agreement on synthetic data was $93.07\% - 94.30\%$. The coarse 12-step bisection calibration simply absorbed the remaining difference, yielding an artificial $100.00\%$ agreement.

#### B. Real Match Data Analysis
On real match tracking, player-ball geometry is nuanced:
- Real $|d_A - d_H|$ averages **$4.88$ m** (median **$3.57$ m**).
- Only **$35.9\%$** of frames have $\tanh > 0.99$ (less than half the synthetic rate).

#### Quantitative Parameter Sensitivity (Real Match Data, Frames 20,000–30,000):
| Parameter Perturbation | Mean Uncalibrated Agreement | Calibrated Agreement | Calibrated Multiplier $m$ Shift | Effective Team Switch Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Unperturbed)** | $100.00\%$ | $100.00\%$ | $m = 0.3557$ | $2.134$ nats |
| `ctrl_scale_m` ($-20\%$) | **$92.95\%$** | **$99.28\%$** | $m = 0.2860$ | $1.716$ nats |
| `ctrl_scale_m` ($+20\%$) | **$94.56\%$** | **$99.98\%$** | $m = 0.3557$ | $2.134$ nats |
| `side_scale_m` ($-20\%$) | **$94.56\%$** | **$99.67\%$** | $m = 0.3557$ | $2.134$ nats |
| `side_scale_m` ($+20\%$) | **$92.97\%$** | **$100.00\%$** | $m = 0.3557$ | $2.134$ nats |
| `dead_gain` ($-20\%$) | **$92.95\%$** | **$98.88\%$** | $m = 0.2971$ | $1.783$ nats |
| `dead_gain` ($+20\%$) | **$92.95\%$** | **$98.86\%$** | $m = 0.3136$ | $1.881$ nats |
| `switch_team` ($-20\%$) | **$94.59\%$** | **$98.53\%$** | $m = 0.3687$ | $1.770$ nats |
| `switch_team` ($+20\%$) | **$91.37\%$** | **$98.88\%$** | $m = 0.2667$ | $1.920$ nats |

### Audit Conclusion on Invariance
1. **The claim of 100% parameter invariance is refuted**: On real tracking data, uncalibrated emissions cause $5.4\% - 8.6\%$ of frame labels to shift.
2. **Calibration partially compensates**: Bisection calibration dynamically adjusts multiplier $m$ to absorb roughly $80\%$ of the parameter drift, but **$1.1\% - 1.5\%$** of frame labels still genuinely change on real match data.
3. Calibration stabilizes the target mean possession duration, but does not provide complete parameter immunity.

---

## 5. In-Depth Investigation: Ball Dropout Robustness

### The Problem
The original report stated that the model achieved $99.93\%$ agreement under $40\%$ ball dropout. We investigated whether the model genuinely detects possession during dropouts or simply relies on forward-fill inertia.

### Empirical Modality Breakdown
We evaluated four distinct ball dropout regimes:
1. **Random i.i.d. Dropout** (10%, 20%, 40%)
2. **Consecutive Burst Dropouts** (1.0 s, 2.0 s, 5.0 s, 10.0 s, 20.0 s)
3. **Transition-Concentrated Dropout** ($\pm 1.5$ s around true possession handovers)
4. **Clustered Dropout** (loss during scrums)

| Dropout Modality | Label Agreement | Detected Possessions (Base=16) | Mean Duration | Mean Transition Displacement | Max Transition Displacement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Clean)** | $100.00\%$ | 16 | $13.07$ s | $0.00$ s | $0.00$ s |
| **Random Dropout 10%** | $99.97\%$ | 16 | $13.07$ s | $0.00$ s | $0.00$ s |
| **Random Dropout 20%** | $99.98\%$ | 16 | $13.07$ s | $0.00$ s | $0.00$ s |
| **Random Dropout 40%** | $99.80\%$ | 16 | $13.05$ s | $0.00$ s | $0.00$ s |
| **Consecutive Burst 1.0 s** | $94.03\%$ | 14 | $14.94$ s | $1.69$ s | $17.72$ s |
| **Consecutive Burst 2.0 s** | $94.03\%$ | 14 | $14.94$ s | $1.69$ s | $17.72$ s |
| **Consecutive Burst 5.0 s** | $93.85\%$ | 14 | $14.94$ s | $1.73$ s | $17.72$ s |
| **Consecutive Burst 10.0 s** | $81.17\%$ | 13 | $14.95$ s | $2.84$ s | $17.72$ s |
| **Consecutive Burst 20.0 s** | **$18.88\%$** | **1 (collapsed)**| **$28.80$ s** | **$92.89$ s** | **$207.24$ s** |
| **Transition-Concentrated** | **$92.30\%$** | **13 (3 missed)**| **$15.94$ s** | **$1.89$ s** | **$17.72$ s** |

### Audit Finding
- **Mechanism Disclosed**: When the ball is missing, `ball_confidence = 0.0`, setting control emissions to zero ($L[H] = 0, L[A] = 0$). The Viterbi decode pays a transition penalty to switch states, so it **always forward-fills the previous state**.
- In i.i.d. dropout, the previous state happens to be the current possessor, creating an illusion of $99.9\%$ robustness.
- However, if the ball is lost **during a turnover**, the model cannot detect the handover. The transition is either delayed until tracking resumes or completely erased (3 out of 16 possessions lost).
- Under prolonged 20-second occlusions, sequence segmentation completely collapses into a single merged possession.

---

## 6. In-Depth Investigation: Player Tracking Robustness

### The Problem
The report claimed "high resilience to player tracking loss" based on randomly removing individual player slots. We audited realistic non-random tracking failure modes.

### Empirical Failure Mode Analysis
| Tracking Failure Mode | Sequence Accuracy | Agreement vs Clean Base | Detected Possessions (Base=13) | Failure Severity |
| :--- | :---: | :---: | :---: | :---: |
| **Random Slot Loss 20%** | $89.30\%$ | $97.95\%$ | 13 | Negligible |
| **Random Slot Loss 40%** | $87.78\%$ | $96.25\%$ | 13 | Low |
| **All Away Players Lost (5 s)** | $88.42\%$ | $100.00\%$ | 13 | Low (absorbed by transition cost) |
| **Loss of Nearest Player to Ball** | **$25.25\%$** | **$30.32\%$** | **3 (10 missed)** | **Catastrophic Failure** |
| **Cluster Drop (< 5 m from Ball)**| **$25.25\%$** | **$31.07\%$** | **3 (10 missed)** | **Catastrophic Failure** |
| **Player ID Swaps (H vs A)** | $87.38\%$ | $98.97\%$ | 13 | Low (if isolated to outfield players) |

### Audit Finding
1. **The model is fundamentally fragile to ball-carrier tracking dropout**: If the single player closest to the ball drops out of tracking, the opposing defender automatically becomes the nearest tracked player. The model flips possession to the opponent, crashing accuracy from $88.42\%$ to $25.25\%$.
2. **Cluster Vulnerability**: Losing players within 5 metres of the ball causes the same catastrophic failure.
3. The claim of general player-tracking resilience in `ROBUSTNESS_REPORT.md` is **invalid without explicit qualification**: the system is resilient only to non-carrier slot dropouts.

---

## 7. Metric Implementation Audit

We inspected the codebase to verify whether evaluation metrics align with competition rules:
1. **Contiguous Possession Spells**:
   - `describe()` in `pipeline.py` correctly computes runs of contiguous `H` and `A` labels.
   - `D` and `X` strictly terminate possession runs.
2. **Abstention (`X`) Handling**:
   - `blocks` restricts prediction to valid tactical video halves (`half_video_blocks(GAME)`).
   - Pre-match, halftime, and post-match frames are assigned `X`.
   - Within active playing blocks, $0$ frames are labeled `X`, ensuring full compliance with competition scoring.
3. **Transition Displacement**:
   - Frame displacement $\Delta t$ is computed as $|\tau_{\text{pred}} - \tau_{\text{true}}|$, divided by $25$ fps to yield physical seconds.
   - Verified mathematically correct.
4. **Duration Calculations**:
   - Mean duration is computed as $\sum \text{len}(\text{runs}) / (\text{fps} \times \text{count})$, strictly excluding $D$ and $X$.
   - Verified mathematically correct.

---

## 8. Unsupported and Overstated Claims in Original Report

The following statements in `ROBUSTNESS_REPORT.md` were found to be overstated or unsupported by direct evidence:

1. **"Across all 36 parameter permutations, the final decoded sequence achieved 100.00% bit-for-bit label identity."**
   - *Correction*: This occurred only on an idealized synthetic match where evidence was $75.3\%$ saturated. On real match tracking data, uncalibrated parameter shifts alter $5.4\% - 8.6\%$ of frame emissions, and $1.1\% - 1.5\%$ after calibration.
2. **"The model displays exceptional resilience to ball dropouts."**
   - *Correction*: The model does not recover ball possession; it forward-fills prior state inertia. It is fragile to dropouts at possession turnovers (displacing transitions by $1.89$ s) and collapses under bursts $> 10$ s.
3. **"High resilience to player tracking loss."**
   - *Correction*: The model is resilient only to non-carrier player slot loss. Losing the ball carrier's track causes catastrophic failure (accuracy drops to $25.25\%$).
4. **"Reflecting Barcelona early dominance, later defending leads... and Athletic Club pressing."**
   - *Correction*: These are speculative tactical narratives inferred without external match event ground truth. The report must state only empirical measurements.

---

## 9. Corrected Empirical Findings

Below are the corrected, rigorously verified statistics for match `BAR-ATH` ($199,319$ frames):

### Match Sequence Diagnostics (Measured Strictly Without Tactical Inference)
| Physical & Sequence Metric | First Half | Second Half | Empirical Difference |
| :--- | :---: | :---: | :---: |
| **Graded Playing Frames** | $79,467$ ($52.98$ min) | $79,390$ ($52.93$ min) | $-77$ frames ($-0.05$ min) |
| **Possession Spells Count** | 99 | 108 | $+9$ spells ($+9.1\%$) |
| **Mean Possession Duration** | $16.63$ s | $13.39$ s | $-3.24$ s ($-19.5\%$) |
| **Median Possession Duration** | $9.16$ s | $8.96$ s | $-0.20$ s ($-2.2\%$) |
| **Home Possession ($H$)** | $32.24\%$ ($25,588$ frames) | $27.23\%$ ($21,653$ frames) | $-5.01\%$ |
| **Away Possession ($A$)** | $19.56\%$ ($15,576$ frames) | $18.31\%$ ($14,505$ frames) | $-1.24\%$ |
| **Dead-Ball Stoppage ($D$)** | $48.20\%$ ($38,303$ frames) | $54.46\%$ ($43,232$ frames) | $+6.26\%$ |
| **Turnover Rate** | $2.93$ transitions / min | $3.44$ transitions / min | $+0.51$ transitions / min ($+17.4\%$) |
| **Tracking Missing Ball Rate** | $40.63\%$ | $40.71\%$ | $+0.08\%$ (identical sensor quality) |
| **Mean Tracked Players** | $20.12$ / 22 | $20.85$ / 22 | $+0.73$ players |
| **Mean Composite Quality $Q$** | $0.814$ | $0.835$ | $+0.021$ |

### Corrected Real-Data Parameter Sensitivity
- Parameter variations of $\pm 20\%$ cause an uncalibrated emission shift of **$5.4\% - 8.6\%$**.
- After label-free bisection calibration to the $15.0$ s target, **$98.5\% - 99.9\%$** of frame labels remain identical (a real-world label shift of **$0.1\% - 1.5\%$**).

---

## 10. Prioritised Recommendations

Based on the empirical failure modes identified in this audit, the following engineering recommendations are prioritised:

1. **High Priority: Carrier Loss Mitigation (Prevent Catastrophic $25\%$ Failure)**:
   - When the ball is observed moving at speed or under control, but the nearest tracked player is $> 2.5$ m away, do not immediately assign possession to whichever opponent is nearest.
   - Check whether a player track recently disappeared at the ball's coordinates; maintain a decay inertia for the disappearing carrier.
2. **High Priority: Transition Ball Dropout Awareness**:
   - When ball tracking drops out while two opposing players are within $2$ metres of each other (contested zone), do not forward-fill with full confidence. Flag the interval as contested and allow player velocity divergence to resolve the handover.
3. **Medium Priority: Calibration Speed Optimization**:
   - In `model.py`, `calibrate_switch_penalties` reruns `refine_possession_boundaries` across all $\sim 400$ transitions on every one of the 12 bisection iterations, causing calibration to take $1,004$ s ($16.7$ minutes).
   - *Fix*: Run the 12 bisection iterations using standard Viterbi ($< 1$ s total), and apply `refine_possession_boundaries` only once on the final selected multiplier. This will reduce pipeline execution time from $17$ minutes to under $35$ seconds.
4. **Low Priority: Remove Default 60m Distance Fallback**:
   - Replace the static `dH = 60.0` rule when one team is completely missing with a neutral emission, avoiding artificial home-advantage bias during one-sided tracking dropouts.

