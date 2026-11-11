# Phase 7 — Robust Ball-Carrier Loss and Possession Continuity: System Design

**Author:** Antigravity Machine Learning Research Team  
**Module:** `solution/mtah_possession/carrier_continuity.py`  
**Integration:** `features.py`, `model.py`, `pipeline.py`  
**Status:** Implemented & Independently Validated  

---

## 1. Executive Summary & Problem Formulation

The Phase 6 Technical Audit (`solution/ROBUSTNESS_AUDIT.md`) uncovered a severe structural fragility in the baseline possession inference pipeline:
- While the system exhibited satisfactory tolerance to random non-carrier slot dropouts ($88\%$ sequence accuracy), the loss of the **actual ball carrier's track** resulted in catastrophic failure: sequence accuracy collapsed to **$25.25\%$**, with agreement falling to **$30.32\%$**.
- In the baseline architecture, possession evidence is derived purely from instantaneous Euclidean proximity:
  $$dH = \min_{h \in \text{Home}} \|p_h - b\|, \quad dA = \min_{a \in \text{Away}} \|p_a - b\|$$
  $$side = \tanh\left(\frac{dA - dH}{\text{scale}}\right)$$
- When the ball carrier's tracking slot drops out (due to visual occlusion, tackle clumping, or camera perspective shift), $dH$ instantaneously jumps from intimate contact ($< 1.0$ m) to the next nearest teammate ($> 4.5$ m).
- Any opposing defender in marking or pressing range ($1.5 - 2.5$ m) becomes the closest tracked player to the ball ($dA < dH$). Consequently, $side$ flips negative, and the model assigns strong, spurious positive possession log-likelihood to the opponent.
- In calibrated sequence decoding, this creates false turnovers, fractures contiguous possession runs, and artificially inflates transition counts.

Phase 7 designs, validates, and deploys a **continuous carrier-loss confidence mechanism** and a **decaying carrier-continuity hypothesis**. The mechanism prevents spurious possession inheritance by opponents when a carrier disappears, while maintaining prompt, zero-lag detection of genuine tackles and turnovers.

---

## 2. Root Cause Analysis: The Instantaneous Proximity Flaw

The failure mechanism stems from three false assumptions in naive tracking-based possession models:

1. **The Spatial Monopoly Fallacy**: Assuming that the closest tracked player is necessarily the player with the ball. When tracking data drops an individual player slot, the player does not vanish physically from the pitch. Treating the absence of a track as proof of absence on the pitch creates artificial vacancies that opponents incorrectly inherit.
2. **Proximity Equated to Control at Non-Physical Distances**: In the baseline, `ctrl_scale_m = 4.0 m`. At a distance of $2.2$ m, control evidence remains substantial:
   $$\text{ctrl} = \exp(-2.2 / 4.0) \approx 0.577$$
   A defender $2.2$ m away from the ball does not have physical control. Awarding positive possession log-odds to a player who is over two metres away violates physical mechanics.
3. **Symmetric Turnover Thresholds**: In physical football, possession is fundamentally asymmetric: the incumbent possessor retains the ball until an opponent actively challenges and establishes physical contact. In the baseline, an opponent could win possession merely by having $dA < dH$, even if both players were metres away from the ball.

---

## 3. The Five Operational Regimes

To resolve these failure modes, the system partitions tracking observations into five mutually exclusive physical regimes:

```
+---------------------------------------------------------------------------------------------------+
|                                      TRACKING OBSERVATION SPACE                                   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                 [Ball Observed]                                   [Ball Unobserved]
                         |                                                 |
             +-----------+-----------+                         +-----------+-----------+
             |                       |                         |                       |
       [Player Near]          [No Player Near]          [Steady Inactive]     [Contested Dropout]
      (dH/dA <= 1.2m)         (dH/dA > 1.2m)             (Low Motion /         (Carrier Loss /
             |                       |                     No Duel)              Tackle / OOB)
     +-------+-------+               |                         |                       |
     |               |               |                         v                       v
[Unambiguous]   [Contested]          v                    Forward-fill             Attenuate
(Clear Holder)  (Tackle Duel)   Carrier Continuity         Incumbent               Continuity;
     |               |          Hypothesis (Decay)        State (D/H/A)           Early Handover
     v               v
  Direct        Contested
  Evidence      Dampening
 (Regime 1)    (Regime 3)           (Regime 2)             (Regime 4)               (Regime 5)
```

### Regime 1: Ball Observed and Near a Tracked Player (Direct Control)
- **Criteria**: $d_{\min} \le r_{\text{touch}} = 1.2$ m, $|dA - dH| \ge \delta_{\text{sep}} = 0.5$ m, ball speed $v_b \le 7.0$ m/s, and ball confidence $c_b > 0.4$.
- **Behavior**: Direct, unambiguous physical control. The controlling team receives maximum carrier confidence ($c_T = 1.0$), the opponent is set to $0.0$, and raw physical distances are passed directly without modification ($dH_{\text{eff}} = dH$, $dA_{\text{eff}} = dA$).
- **Integrity**: Standard clean play is 100% unaltered.

### Regime 2: Ball Observed but No Tracked Player is Close (Carrier Loss / Loose Ball)
- **Criteria**: Both $dH > r_{\text{touch}}$ and $dA > r_{\text{touch}}$ ($> 1.2$ m).
- **Subcase 2A (Carrier Disappearance)**: If a team held control immediately prior ($c_T > 0$) and the ball did not accelerate into flight ($v_b \le 7.0$ m/s), the carrier has dropped from tracking. The carrier-continuity confidence decays with characteristic half-life $\tau_{\text{carrier}} = 2.0$ s. The effective distance for team $T$ is clamped to virtual carrier contact ($r_{\text{virt}} = 0.7$ m), preventing nearby opponents ($dA \in [1.3, 2.5]$ m) from inheriting possession.
- **Subcase 2B (Loose Ball / Pass / Clearance)**: Ball speed is elevated ($v_b > 9.0$ m/s) or out of bounds. The continuity decays rapidly ($\tau_{\text{kick}} = 0.4$ s). Both teams exceed control radius, control evidence drops to near zero, and the decode preserves sequence continuity via transition penalties without spurious opposing nats.

### Regime 3: Nearby Players from Both Teams Create Ambiguity (Contested Duel)
- **Criteria**: Both $dH \le 2.0$ m and $dA \le 2.0$ m, with separation $|dA - dH| < \delta_{\text{sep}} = 0.5$ m.
- **Behavior**: Physical 50-50 challenge or tackle scrum. Neither team satisfies unambiguous control. The incumbent possessor's continuity decays moderately ($\tau \approx 1.5$ s), preventing high-frequency chattering or deflection-induced possession flipping until one player separates cleanly.

### Regime 4: Ball Unobserved While Previous State is Known (Steady Inactive)
- **Criteria**: $ball\_ok$ is False ($c_b = 0$), but no contested event preceded the gap.
- **Behavior**: Direct control evidence is set to 0. Continuity decays gracefully. The Viterbi decoder forward-fills the previous state via transition penalties. If players stop moving, the independent $P(D)$ detector cleanly transitions the sequence into dead ball ($D$).

### Regime 5: Ball Unobserved During a Potential Team Transition (Contested Dropout)
- **Criteria**: Ball tracking is lost immediately following a contested duel ($|dH - dA| < 1.0$ m) or sudden deceleration.
- **Behavior**: Carrier continuity is already attenuated ($c \le 0.3$). Effective distances are not artificially depressed. When the ball reappears with the new possessor, the decode transitions immediately without boundary lag.

---

## 4. Mathematical Formulation of Carrier Continuity

Let $c_{H, t} \in [0, 1]$ and $c_{A, t} \in [0, 1]$ represent the continuous carrier-continuity confidence for Home and Away at discrete frame $t$.

### 4.1 Update Equations

At each frame $t$, given ball state $(b_t, v_{b, t}, c_{b, t})$ and nearest player distances $(dH_t, dA_t)$:

1. **Direct Control Triggers**:
   $$\text{live}_t = (c_{b, t} > 0.4) \land (\text{ball\_ok}_t)$$
   
   To prevent an opponent from falsely claiming control when the incumbent carrier's slot drops out, the distance required to claim possession is asymmetric:
   $$r_{\text{req}, H} = \begin{cases} r_{\text{tackle}} = 0.8\text{ m} & \text{if } c_{A, t-1} > 0.2 \\ r_{\text{touch}} = 1.2\text{ m} & \text{otherwise} \end{cases}$$
   $$r_{\text{req}, A} = \begin{cases} r_{\text{tackle}} = 0.8\text{ m} & \text{if } c_{H, t-1} > 0.2 \\ r_{\text{touch}} = 1.2\text{ m} & \text{otherwise} \end{cases}$$

   The effective separation check accounts for incumbent continuity:
   $$dH^{\text{prev\_eff}} = \begin{cases} \min(dH_t, r_{\text{virt}}) & \text{if } c_{H, t-1} > 0.2 \\ dH_t & \text{otherwise} \end{cases}$$
   $$dA^{\text{prev\_eff}} = \begin{cases} \min(dA_t, r_{\text{virt}}) & \text{if } c_{A, t-1} > 0.2 \\ dA_t & \text{otherwise} \end{cases}$$

   Direct control is assigned if:
   $$\text{ctrl}_{H, t} = \text{live}_t \land (dH_t \le r_{\text{req}, H}) \land (dA^{\text{prev\_eff}} - dH_t \ge \delta_{\text{sep}}) \land (v_{b, t} \le 7.0\text{ m/s})$$
   $$\text{ctrl}_{A, t} = \text{live}_t \land (dA_t \le r_{\text{req}, A}) \land (dH^{\text{prev\_eff}} - dA_t \ge \delta_{\text{sep}}) \land (v_{b, t} \le 7.0\text{ m/s})$$

2. **Continuity Propagation and Decay**:
   $$\alpha_t = \begin{cases} \exp\left(-\frac{1}{\text{FPS} \cdot \tau_{\text{kick}}}\right) & \text{if } v_{b, t} > 9.0\text{ m/s} \\ \exp\left(-\frac{1}{\text{FPS} \cdot \tau_{\text{carrier}}}\right) & \text{otherwise} \end{cases}$$

   State transitions:
   $$(c_{H, t}, c_{A, t}) = \begin{cases} (1.0, 0.0) & \text{if } \text{ctrl}_{H, t} \land \neg \text{ctrl}_{A, t} \\ (0.0, 1.0) & \text{if } \text{ctrl}_{A, t} \land \neg \text{ctrl}_{H, t} \\ (\alpha_t \cdot c_{H, t-1}, \alpha_t \cdot c_{A, t-1}) & \text{otherwise} \end{cases}$$

### 4.2 Effective Distance Mapping

Rather than introducing hard heuristic possession flags into the decode, the continuity state is projected into the geometric distance space:
$$dH^{\text{eff}}_t = (1 - c_{H, t}) \cdot dH_t + c_{H, t} \cdot \min(dH_t, r_{\text{virt}})$$
$$dA^{\text{eff}}_t = (1 - c_{A, t}) \cdot dA_t + c_{A, t} \cdot \min(dA_t, r_{\text{virt}})$$
where $r_{\text{virt}} = 0.7$ metres.

**Properties**:
- When $c_{H, t} = 1.0$ and $dH_t = 0.5$ m: $dH^{\text{eff}}_t = 0.5$ m (exact identity).
- When $c_{H, t} = 1.0$ and $dH_t = 5.0$ m (carrier missing): $dH^{\text{eff}}_t = 0.7$ m.
- As $c_{H, t} \to 0$ over time: $dH^{\text{eff}}_t \to dH_t$ continuously without step discontinuities.

---

## 5. Architectural Separation of Uncertainty

In strict compliance with competition design principles:
1. **Uncertainty is NOT Encoded as a Discrete State**: The decoder operates strictly over the valid state space $\{H, A, D, X\}$. Carrier loss confidence modulates continuous emissions and effective distances; it never outputs an ad-hoc "uncertain" token.
2. **Deterministic Dynamic Programming**: The forward recursion for $c_{H, t}$ and $c_{A, t}$ is strictly causal and bit-for-bit deterministic.
3. **Decoupled Boundary Refinement**: Boundary refinement operates on the calibrated sequence using local likelihood contrasts, maintaining sharp transitions at genuine turnovers.

