"""Duration-Aware Semi-Markov Decoder for Possession Spell Segmentation.

Implements explicit state-duration modeling (HSMM / EDHMM) using dynamic programming
over segment durations d in [1, D_max], supporting:
- Candidate A: Phase 8 V2 baseline (first-order Viterbi with boundary refinement).
- Candidate B: Semi-Markov decoder with simple duration prior (soft minimum + exponential decay).
- Candidate C: Semi-Markov decoder with flexible Gamma duration distributions.
- Candidate D: Semi-Markov decoder with state-specific duration distributions and transition costs.
- Candidate E: Segment DP ablation without duration priors (transition costs only).
"""
import math
from dataclasses import dataclass, replace
import numpy as np

from .model import H, A, D, refine_possession_boundaries


@dataclass
class SemiMarkovConfig:
    # Model type
    candidate: str = "D"  # 'A', 'B', 'C', 'D', 'E'
    # Base transition costs (in nats)
    switch_team: float = 6.0        # H <-> A base cost
    switch_dead: float = 5.0        # H/A -> D stoppage cost
    restart_cost_ratio: float = 0.5  # D -> H/A restart cost = switch_dead * restart_cost_ratio
    touch_discount: float = 0.4     # discount on turnover cost when controlled touch is observed
    # Duration bounds
    max_duration_s: float = 240.0   # 240 seconds = 6000 frames at 25 fps
    # Candidate B (Simple duration prior)
    simple_min_dur_s: float = 1.0
    simple_penalty: float = 5.0
    simple_decay: float = 0.003
    # Candidate C (Flexible Gamma shared)
    gamma_shape: float = 2.2
    gamma_scale_s: float = 6.0
    # Candidate D (State-specific)
    poss_shape: float = 2.2
    poss_scale_s: float = 6.0       # mode ~7.2s, mean ~13.2s
    poss_min_dur_s: float = 1.0     # soft penalty for possession < 1.0s
    poss_min_penalty: float = 3.0
    dead_shape: float = 1.3         # heavier tail for stoppages
    dead_scale_s: float = 11.5      # mode ~3.5s, mean ~15.0s
    dead_min_dur_s: float = 2.0     # soft penalty for stoppage < 2.0s
    dead_min_penalty: float = 4.0
    # Carrier continuity parameters (inherited from Phase 7/8)
    carrier_touch_m: float = 1.2
    carrier_tackle_m: float = 0.8
    # Refinement option
    refine_boundaries: bool = False  # Semi-Markov jointly optimizes boundaries; refinement is optional


def compute_duration_log_probs(cfg: SemiMarkovConfig, state: int, max_d: int, fps: int = 25) -> np.ndarray:
    """Compute normalized log P(d | state) for d in [1, max_d].

    Returns array of shape (max_d + 1,), where index d corresponds to duration d frames,
    and index 0 is -inf.
    """
    log_p = np.full(max_d + 1, -np.inf, dtype=np.float64)
    d_frames = np.arange(1, max_d + 1, dtype=np.float64)
    d_s = d_frames / fps

    cand = cfg.candidate.upper()

    if cand == "E":
        # Candidate E: Ablation with uniform duration prior
        log_p[1:] = 0.0
        return log_p

    if cand == "B":
        # Candidate B: Simple duration prior with soft minimum duration penalty
        min_frames = cfg.simple_min_dur_s * fps
        under = np.maximum(0.0, (min_frames - d_frames) / max(1.0, min_frames))
        pen = -cfg.simple_penalty * (under ** 2) - cfg.simple_decay * d_frames
        # Center so that modal/peak duration pays 0 penalty
        log_p[1:] = pen - np.max(pen)
        return log_p

    if cand == "C":
        # Candidate C: Shared Gamma duration distribution
        k = cfg.gamma_shape
        theta = cfg.gamma_scale_s * fps
        # Gamma log-pdf: (k - 1)*log(d) - d/theta
        raw = (k - 1.0) * np.log(d_frames) - (d_frames / theta)
        # Soft minimum penalty for sub-second chatter
        min_frames = 1.0 * fps
        under = np.maximum(0.0, (min_frames - d_frames) / min_frames)
        raw -= 3.0 * (under ** 2)
        # Center so that modal duration pays 0 penalty
        log_p[1:] = raw - np.max(raw)
        return log_p

    if cand == "D":
        # Candidate D: State-specific duration distributions
        if state in (H, A):
            k = cfg.poss_shape
            theta = cfg.poss_scale_s * fps
            raw = (k - 1.0) * np.log(d_frames) - (d_frames / theta)
            min_frames = cfg.poss_min_dur_s * fps
            under = np.maximum(0.0, (min_frames - d_frames) / max(1.0, min_frames))
            raw -= cfg.poss_min_penalty * (under ** 2)
        else:  # state == D
            k = cfg.dead_shape
            theta = cfg.dead_scale_s * fps
            raw = (k - 1.0) * np.log(d_frames) - (d_frames / theta)
            min_frames = cfg.dead_min_dur_s * fps
            under = np.maximum(0.0, (min_frames - d_frames) / max(1.0, min_frames))
            raw -= cfg.dead_min_penalty * (under ** 2)

        # Center so that modal duration pays 0 penalty
        log_p[1:] = raw - np.max(raw)
        return log_p

    # Default fallback
    log_p[1:] = 0.0
    return log_p


def viterbi_semi_markov(L: np.ndarray, cfg: SemiMarkovConfig, feats: dict = None, fps: int = 25) -> np.ndarray:
    """Exact Duration-Aware Semi-Markov (Segment-level HSMM) Viterbi Decoder.

    Args:
        L: (N, 3) log-emissions for H (0), A (1), D (2).
        cfg: SemiMarkovConfig with transition costs and duration prior specs.
        feats: dictionary of tracking features (for controlled touch and restart context).
        fps: frame rate (default 25).

    Returns:
        (N,) int8 array of decoded state labels (0=H, 1=A, 2=D).
    """
    n = len(L)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    max_d = min(n, int(cfg.max_duration_s * fps))

    # Precompute duration log-priors for all 3 states: shape (3, max_d + 1)
    log_p_dur = np.zeros((3, max_d + 1), dtype=np.float64)
    for s in (H, A, D):
        log_p_dur[s] = compute_duration_log_probs(cfg, s, max_d, fps=fps)

    # Transition costs into states at frame t:
    # When transitioning into H at frame t: cost depends on predecessor (A vs D)
    # When transitioning into A at frame t: cost depends on predecessor (H vs D)
    # When transitioning into D at frame t: cost depends on predecessor (H vs A)
    base_team_cost = cfg.switch_team
    cost_dead = cfg.switch_dead
    cost_restart = cfg.switch_dead * cfg.restart_cost_ratio

    if feats is not None:
        dH = np.nan_to_num(feats.get("dH", np.full(n, np.nan)), nan=50.0)
        dA = np.nan_to_num(feats.get("dA", np.full(n, np.nan)), nan=50.0)
        b_spd = np.nan_to_num(feats.get("ball_speed", np.full(n, np.nan)), nan=3.0)
        b_conf = np.nan_to_num(feats.get("ball_confidence", np.ones(n)), nan=1.0)
        cH = feats.get("carrier_cH", np.zeros(n))
        cA = feats.get("carrier_cA", np.zeros(n))

        r_touch = getattr(cfg, "carrier_touch_m", 1.2)
        r_tackle = getattr(cfg, "carrier_tackle_m", 0.8)
        thresh_H = np.where(cA > 0.2, r_tackle, r_touch)
        thresh_A = np.where(cH > 0.2, r_tackle, r_touch)

        touch_H = (dH <= thresh_H) & (dH < dA) & (b_spd < 6.0) & (b_conf > 0.4)
        touch_A = (dA <= thresh_A) & (dA < dH) & (b_spd < 6.0) & (b_conf > 0.4)

        cost_A_to_H = np.where(touch_H, base_team_cost * cfg.touch_discount, base_team_cost)
        cost_H_to_A = np.where(touch_A, base_team_cost * cfg.touch_discount, base_team_cost)
    else:
        cost_A_to_H = np.full(n, base_team_cost)
        cost_H_to_A = np.full(n, base_team_cost)

    # Prefix cumulative sum of log emissions: shape (n + 1, 3)
    # S_L[0] = 0; S_L[t] = sum_{i=0}^{t-1} L[i]
    S_L = np.zeros((n + 1, 3), dtype=np.float64)
    S_L[1:] = np.cumsum(L, axis=0)

    # Dynamic programming matrices:
    # V[t, s]: best score ending at frame t in state s
    V = np.full((n + 1, 3), -1e12, dtype=np.float64)
    V[0, :] = 0.0

    # Backpointers:
    # back_d[t, s]: optimal duration of state s ending at frame t
    # trans_s[t, s]: predecessor state that transitioned into state s
    back_d = np.zeros((n + 1, 3), dtype=np.int32)
    trans_s = np.full((n + 1, 3), -1, dtype=np.int8)

    # Precomputed predecessor scores buffer B:
    # B[t', s] = max_{s' != s} [V[t', s'] - C(s' -> s, t' + 1)] - S_L[t', s]
    # For t'=0, start of match is unconstrained:
    B = np.full((n + 1, 3), -1e12, dtype=np.float64)
    for s in (H, A, D):
        B[0, s] = 0.0 - S_L[0, s]

    # Main DP loop over end frames t = 1..n
    for t in range(1, n + 1):
        cur_max_d = min(t, max_d)
        # Vectorized slice of B from t-1 down to t-cur_max_d: shape (cur_max_d, 3)
        # Index k (0 to cur_max_d - 1) corresponds to duration d = k + 1, predecessor frame t' = t - (k + 1)
        b_slice = B[t - cur_max_d : t][::-1]

        for s in (H, A, D):
            dur_prior = log_p_dur[s, 1 : cur_max_d + 1]
            scores = b_slice[:, s] + dur_prior
            best_idx = int(np.argmax(scores))
            V[t, s] = scores[best_idx] + S_L[t, s]
            back_d[t, s] = best_idx + 1

        # Compute B[t] for transitions occurring after frame t (starting at frame t + 1)
        if t < n:
            # Transitions into H: from A (1) with cost_A_to_H[t], from D (2) with cost_restart
            val_from_A = V[t, A] - cost_A_to_H[t]
            val_from_D = V[t, D] - cost_restart
            if val_from_A >= val_from_D:
                B[t, H] = val_from_A - S_L[t, H]
                trans_s[t, H] = A
            else:
                B[t, H] = val_from_D - S_L[t, H]
                trans_s[t, H] = D

            # Transitions into A: from H (0) with cost_H_to_A[t], from D (2) with cost_restart
            val_from_H = V[t, H] - cost_H_to_A[t]
            val_from_D = V[t, D] - cost_restart
            if val_from_H >= val_from_D:
                B[t, A] = val_from_H - S_L[t, A]
                trans_s[t, A] = H
            else:
                B[t, A] = val_from_D - S_L[t, A]
                trans_s[t, A] = D

            # Transitions into D: from H (0) with cost_dead, from A (1) with cost_dead
            val_from_H = V[t, H] - cost_dead
            val_from_A = V[t, A] - cost_dead
            if val_from_H >= val_from_A:
                B[t, D] = val_from_H - S_L[t, D]
                trans_s[t, D] = H
            else:
                B[t, D] = val_from_A - S_L[t, D]
                trans_s[t, D] = A

    # Backtracking to reconstruct optimal state sequence
    path = np.empty(n, dtype=np.int8)
    curr_s = int(np.argmax(V[n]))
    curr_t = n

    while curr_t > 0:
        d = int(back_d[curr_t, curr_s])
        prev_t = curr_t - d
        path[prev_t:curr_t] = curr_s
        if prev_t > 0:
            curr_s = int(trans_s[prev_t, curr_s])
        curr_t = prev_t

    if cfg.refine_boundaries:
        path = refine_possession_boundaries(path, L, window_frames=int(1.5 * fps))

    return path


def decode_blocks_semi_markov(L: np.ndarray, blocks: list, n_frames: int, cfg: SemiMarkovConfig,
                              feats: dict = None, fps: int = 25) -> np.ndarray:
    """Label array of 'H'/'A'/'D' inside playing blocks, 'X' everywhere else.

    Resets temporal state appropriately between halves.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    letters = np.array(["H", "A", "D"])

    for first, last in blocks:
        sub_L = L[first : last + 1]
        sub_feats = None
        if feats is not None:
            sub_feats = {
                k: v[first : last + 1] if isinstance(v, np.ndarray) and len(v) == n_frames else v
                for k, v in feats.items()
            }
        path = viterbi_semi_markov(sub_L, cfg, feats=sub_feats, fps=fps)
        labels[first : last + 1] = letters[path]

    return labels


def calibrate_switch_penalties_semi_markov(L: np.ndarray, blocks: list, n_frames: int,
                                           cfg: SemiMarkovConfig, fps: int = 25,
                                           target_s: float = 15.0, iters: int = 12,
                                           m_range: tuple = (0.02, 2.0), feats: dict = None):
    """Calibrate switch penalties by bisection in log-space over m_range to match target_s mean possession duration."""
    from .pipeline import describe

    low = math.log(m_range[0])
    high = math.log(m_range[1])
    best_m = None
    best_labels = None
    best_err = float("inf")

    for _ in range(iters):
        mid = (low + high) / 2.0
        m = math.exp(mid)
        cfg_m = replace(cfg, switch_team=cfg.switch_team * m, switch_dead=cfg.switch_dead * m)
        labels = decode_blocks_semi_markov(L, blocks, n_frames, cfg_m, feats=feats, fps=fps)
        d = describe(labels, fps=fps)
        dur = d["mean_s"]
        err = abs(dur - target_s)
        if err < best_err:
            best_err = err
            best_m = m
            best_labels = labels

        if dur < target_s:
            low = mid
        else:
            high = mid

    return best_m, best_labels
