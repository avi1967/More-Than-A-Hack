# Phase 8 — Probabilistic Dead-Ball Detection and Restart-Aware Possession Inference: System Design

**Author:** Antigravity ML Research Team  
**Module:** `solution/mtah_possession/dead_ball.py`, `smart_tagging.py`, `features.py`, `model.py`  
**Status:** Implemented & Verified  

---

## 1. Executive Summary & Audit of Existing Dead-Ball Logic

In professional football tracking analysis, correctly identifying stoppages (labeled `D`) is critical because dead balls act as the fundamental separators between Home (`H`) and Away (`A`) possession spells. Under the competition's ranking metric (`spellF1`), a single false dead ball splits an unbroken possession into two, while a missed dead ball incorrectly merges distinct possessions or obscures genuine restart handovers.

### 1.1 Technical Audit of Existing Phase 7 Dead-Ball Logic

Our technical audit identified **five structural failure modes** in the prior dead-ball implementation:

1. **The `ball_speed_reliable` Boolean Type Bug**:
   In `dead_ball.py`:
   ```python
   ball_speed = feats.get("ball_speed_reliable", feats["ball_speed"])
   ```
   `feats["ball_speed_reliable"]` is a boolean array indicating observation reliability, not velocity magnitude. When the ball was reliably tracked, `ball_speed` evaluated to `True` ($1.0$ m/s in float arithmetic). Consequently, `is_still = ball_ok & (speed_safe < 0.5)` could **never evaluate to True** for an observed ball. Ball inactivity accumulation (`p_inactivity`) was completely non-functional for observed stationary balls, causing **$0.0\%$ detection accuracy** on stationary dead balls waiting for free kicks.

2. **Classification of Live Ball Flight as Dead Ball**:
   When a ball is kicked into high-speed aerial flight ($v_b \in [10, 25]$ m/s) across the pitch, no player is within $2.5$ m of the ball (`live_ctrl = 0.0`), and outfield players are watching the flight (`p_calm = 1.0`). In the baseline, `dead_logit` summed `base_logit + 1.5 * p_calm = -1.65 + 1.5 = -0.15`, generating elevated $P(D) \approx 0.46 - 0.74$. As a result, aerial crosses and clearances were falsely classified as dead-ball stoppages.

3. **Midfield Ball Occlusion Falsely Triggering Stoppages**:
   The baseline flagged missing ball frames as stoppages if `p_calm > 0.4`. Whenever a momentary tracking occlusion occurred in mid-pitch while players were jogging, the model falsely declared a stoppage.

4. **Inability to Distinguish Live Carrier Holding from Abandoned Dead Ball**:
   When a goalkeeper holds the ball inside the penalty box for $4-5$ seconds or a defender stands over the ball during a slow build-up, the ball is motionless ($v_b < 0.5$ m/s), but **strictly live**. The previous detector accumulated inactivity indiscriminately, threatening false dead-ball transitions during live goalkeeper possession.

5. **Smart Tagging Phase Bleed**:
   Smart Tagging instance windows often span $35$ seconds and begin $2-5$ seconds before the ball crosses the boundary. The baseline injected static $+2.5$ log-odds across the entire window without requiring physical tracking corroboration, displacing stoppage boundaries.

---

## 2. Physical Principles of Stoppages and Restarts

To eliminate these vulnerabilities, Phase 8 establishes four physical principles:

1. **Kinematic Immunity of Inbounds Ball Flight**: A ball traveling at high speed ($v_b > 3.0$ m/s) entirely within pitch boundaries ($p_{\text{ball\_out}} < 0.2$) is physically live (shot, pass, clearance). Live flight evidence must actively and decisively suppress $P(D)$.
2. **Proximity-Gated Inactivity**: A stationary ball is only a candidate stoppage if **no player is in intimate possession** ($d_{\min} > 1.2$ m) or if the ball is out of bounds. When a goalkeeper or player is within $1.0$ m inside the pitch, ball stillness represents live control, not a stoppage.
3. **Corroborated Boundary Exits**: A missing ball represents a stoppage if and only if it was heading towards or crossed the boundary line, or if player activity drops to walking pace across the team. Interior occlusions during active play are tracking dropouts, not dead balls.
4. **Physical Restart Resumption**: Play resumes from a stoppage when the ball accelerates into the pitch ($v_b > 2.5$ m/s) or when a player takes a controlled touch following restart positioning. Smart Tagging priors must immediately extinguish upon physical resumption.

---

## 3. Mathematical Formulation of the Refined Detector (v2)

Let $t$ index the discrete video frame ($25$ fps). The detector computes continuous geometric and physical signals:

### 3.1 Out-of-Bounds and Boundary Proximity
- Signed boundary distance: $\delta_t = \min(x_t, L - x_t, y_t, W - y_t)$ (negative when outside pitch).
- Boundary proximity: $\pi_{\text{bound}, t} = \exp(-\max(0, \delta_t) / 3.0)$.
- Outside-pitch probability via smooth sigmoid:
  $$p_{\text{out}, t} = \frac{1}{1 + \exp((\delta_t + 0.3) / 0.20)}$$

### 3.2 Inactivity with Carrier-Holding Modulation
- True physical ball speed: $v_{b, t} = \text{feats}[\text{"ball\_speed"}]_t$.
- Candidate stillness: $\text{still}_t = \text{ball\_ok}_t \land (v_{b, t} < 0.6\text{ m/s})$.
- Carrier holding check:
  $$\text{holding}_t = \text{still}_t \land (d_{\min, t} \le 1.0\text{ m}) \land (\delta_t > 2.0\text{ m})$$
  $$\text{stoppage\_candidate}_t = \text{still}_t \land \neg\text{holding}_t \land (p_{\text{out}, t} < 0.8)$$
- Continuous inactivity accumulation $\Delta \tau_t$:
  $$\Delta \tau_t = \begin{cases} \Delta \tau_{t-1} + \Delta t & \text{if } \text{stoppage\_candidate}_t \\ 0.0 & \text{if } \text{ball\_ok}_t \land v_{b, t} \ge 0.8\text{ m/s} \\ \max(0, \Delta \tau_{t-1} - 2\Delta t) & \text{if } \text{holding}_t \\ \Delta \tau_{t-1} + \Delta t & \text{if } \neg\text{ball\_ok}_t \land \Delta \tau_{t-1} \ge 1.0\text{ s} \\ 0.0 & \text{otherwise} \end{cases}$$
- Inactivity probability:
  $$p_{\text{inact}, t} = \text{clip}\left(\frac{\Delta \tau_t - 1.2}{2.5}, 0.0, 1.0\right)$$

### 3.3 Live Ball Flight Counter-Evidence
When the ball travels inbounds at speed, it directly refutes dead ball:
$$p_{\text{flight}, t} = \text{ball\_ok}_t \cdot (1 - p_{\text{out}, t}) \cdot \text{clip}\left(\frac{v_{b, t} - 2.5}{5.0}, 0.0, 1.0\right)$$

### 3.4 Live Ball Control Counter-Evidence
$$p_{\text{ctrl}, t} = \text{ball\_ok}_t \cdot \text{clip}\left(1.0 - \frac{d_{\min, t}}{2.0}, 0.0, 1.0\right) \cdot c_{b, t} \cdot (1 - p_{\text{out}, t})$$

### 3.5 Corroborated Missing Ball Cue
A missing ball is only counted as stoppage evidence if physically corroborated:
$$\text{miss\_stop}_t = \neg\text{ball\_ok}_t \land (p_{\text{calm}, t} > 0.6 \lor \pi_{\text{bound}, t} > 0.5 \lor \text{tag\_prior}_t > 0.5)$$
$$p_{\text{miss\_dead}, t} = \text{miss\_stop}_t \cdot \max(p_{\text{calm}, t}, \text{tag\_prior}_t)$$

### 3.6 Log-Odds Evidence Fusion
The composite log-odds of a stoppage are fused as:
$$\text{logit}(P(D_t)) = \text{logit}(p_0) + 6.5 \cdot p_{\text{out}, t} + 4.0 \cdot p_{\text{inact}, t} + 1.2 \cdot p_{\text{calm}, t} + 1.0 \cdot p_{\text{rep}, t} + 1.0 \cdot p_{\text{box}, t} + 2.0 \cdot \text{tag\_prior}_t + 1.5 \cdot p_{\text{miss\_dead}, t} - 3.5 \cdot p_{\text{ctrl}, t} - 4.5 \cdot p_{\text{flight}, t}$$
where $p_0 = 0.16$ is the uninformative prior. Continuous probability:
$$P(D_t) = \frac{1}{1 + \exp(-\text{logit}(P(D_t)))}$$

---

## 4. Modeling Stoppage and Restart Situations (Step 3)

The refined architecture systematically handles the 8 canonical situations:

- **Situation A: Ball Out of Play $\to$ Restart**: As ball crosses boundary, $p_{\text{out}} \to 1.0$, rapidly engaging $D$. As the restart is struck into the pitch ($v_b > 2.5$ m/s), $p_{\text{flight}}$ activates, dropping $P(D) < 0.10$ and enabling the receiving team to establish possession without lag.
- **Situation B: Temporary Ball Inactivity $\to$ Continued Play**: Ball stops briefly ($< 1.2$ s) while player scans. $\Delta \tau_t$ remains below threshold ($p_{\text{inact}} = 0$), preventing false dead-ball insertion.
- **Situation C: Contested Loose Ball $\to$ Team Takeover**: Midfield duel with bouncing ball. $p_{\text{flight}}$ and carrier continuity maintain live play; neither team triggers false stoppage.
- **Situation D: Ball Missing $\to$ Players Repositioning (Stoppage)**: Ball kicked into crowd; players walk and drift ($p_{\text{calm}} > 0.6$). Corroborated missing cue engages $D$.
- **Situation E: Ball Missing $\to$ Players Moving (Live Play)**: Camera occlusion during sprint. Because $p_{\text{calm}} < 0.3$ and ball is in interior, missing stoppage cue remains $0.0$. Live possession forward-fills cleanly.
- **Situation F: Goal $\to$ Kickoff**: Prolonged celebration stoppage. Centroids drift to center, box density is low, and $P(D)$ persists until kickoff pass is executed.
- **Situation G: Throw-in/Corner with Smart Tagging `Team=N/A`**: Smart Tagging provides only weak contextual evidence for the existence of the stoppage. Restart possession ownership is determined strictly by which player strikes the ball in tracking data.
- **Situation H: Goalkeeper Possession / Slow Build-Up**: Goalkeeper holds ball for 5 seconds inside penalty box ($d_{\min} \le 0.8$ m, $\delta_t > 2.0$ m). $\text{holding}_t = \text{True}$ prevents inactivity accumulation, preserving continuous goalkeeper possession.

