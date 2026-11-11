"""Turn per-frame evidence into H / A / D labels with a sticky three-state Viterbi decode.

Why a decode and not a per-frame vote: the ranking metric (spellF1) counts whole possessions, so
the number of hand-overs matters far more than frame accuracy. Evidence is a log-likelihood per
state per frame; changing state costs a fixed penalty. A hand-over therefore needs a few seconds of
consistent evidence, and a frame with no evidence (ball unseen) simply keeps the previous state --
which is also what the scorer does with an abstention.
"""
from dataclasses import dataclass

import numpy as np

H, A, D = 0, 1, 2


@dataclass
class Config:
    # -- geometry / signal (physical scales, chosen a priori, not fitted to any labels) --
    ball_gap_s: float = 1.5        # interpolate ball gaps up to this long
    speed_lag_s: float = 0.2       # lag for velocity estimates
    activity_window_s: float = 3.0
    oob_margin_m: float = 0.5
    still_speed: float = 0.5       # m/s: ball "not moving"
    still_window_s: float = 3.0
    ctrl_scale_m: float = 4.0      # how quickly 'control' evidence fades with distance
    side_scale_m: float = 2.0      # distance gap (A minus H) that counts as decisive
    ball_speed_ref: float = 10.0   # m/s: a ball flying this fast is nobody's
    walk_speed: float = 1.0        # m/s: median player speed below which players are standing
    # -- evidence strength, nats per frame at 25 fps --
    side_gain: float = 0.20
    alive_gain: float = 0.10
    oob_gain: float = 0.30
    still_gain: float = 0.05
    activity_gain: float = 0.02
    dead_gain: float = 0.06        # nats per frame per unit log-odds P(D)
    # -- decode: nats paid per state change. Duration prior, not fitted: a possession lasts
    # ~10-20 s (~375 frames), so a change costs about ln(375) = 6 nats. --
    switch_team: float = 6.0       # H <-> A
    switch_dead: float = 5.0       # H/A <-> D
    # -- Phase 7: Carrier continuity & loss robustness --
    carrier_continuity: bool = True
    carrier_tau_s: float = 2.0         # memory timescale for carrier continuity
    carrier_kick_tau_s: float = 0.4    # rapid decay timescale when ball is kicked
    carrier_touch_m: float = 1.2       # distance threshold for direct carrier touch
    carrier_tackle_m: float = 0.8      # intimate distance required to tackle from incumbent
    carrier_sep_m: float = 0.5         # separation margin required for clear ownership
    carrier_virt_m: float = 0.7        # nominal carrier proximity during tracking loss
    # -- Phase 8: Refined probabilistic dead-ball detection --
    dead_ball_v2: bool = True


def log_emissions(f, cfg):
    """(N, 3) log-likelihoods for H, A, D."""
    n = len(f["dH"])
    dH, dA = f["dH"].copy(), f["dA"].copy()
    dH[np.isnan(dH) & ~np.isnan(dA)] = 60.0   # one side unseen: the other is nearer by default
    dA[np.isnan(dA) & ~np.isnan(dH)] = 60.0
    have = f["ball_ok"] & np.isfinite(dH) & np.isfinite(dA)

    dmin = np.where(have, np.minimum(dH, dA), np.inf)
    # Use ball speed if finite; if missing on an active ball, use neutral reference speed (3.0 m/s)
    # rather than assuming zero velocity
    v = np.where(np.isfinite(f["ball_speed"]), f["ball_speed"], 3.0)
    ctrl = np.exp(-dmin / cfg.ctrl_scale_m) / (1.0 + (v / cfg.ball_speed_ref) ** 2)
    ctrl = np.where(have, ctrl, 0.0)

    # Continuous ball confidence modulates control evidence:
    # Directly observed ball has full control strength; imputed ball is softly discounted;
    # Missing ball has zero control evidence.
    ball_conf = f.get("ball_confidence")
    if ball_conf is not None:
        ctrl = ctrl * ball_conf

    side = np.tanh(np.where(have, dA - dH, 0.0) / cfg.side_scale_m)   # >0: home nearer

    L = np.zeros((n, 3))
    if "p_dead" in f:
        # Phase 3: Independent probabilistic dead-ball detector integration
        p_dead = np.clip(f["p_dead"], 0.001, 0.999)
        p0_dead = 0.16
        logit_p = np.log(p_dead / (1.0 - p_dead))
        logit_p0 = np.log(p0_dead / (1.0 - p0_dead))
        d_evidence = logit_p - logit_p0
        dead = cfg.dead_gain * d_evidence

        # When dead ball evidence is active, softly suppress live team emissions
        dead_suppr = 0.5 * np.maximum(0.0, dead)
        L[:, H] = 0.5 * cfg.side_gain * ctrl * side - dead_suppr
        L[:, A] = -0.5 * cfg.side_gain * ctrl * side - dead_suppr
        L[:, D] = dead
    else:
        # Backward-compatible fallback
        L[:, H] = 0.5 * cfg.side_gain * ctrl * side
        L[:, A] = -0.5 * cfg.side_gain * ctrl * side

        dead = np.zeros(n)
        dead -= cfg.alive_gain * ctrl                          # someone is playing the ball
        dead += cfg.oob_gain * f["oob"]                        # ball left the pitch
        dead += cfg.still_gain * f["still"]                    # ball lying still
        dead += cfg.activity_gain * np.clip(                   # everybody walking / standing
            (cfg.walk_speed - f["activity"]) / cfg.walk_speed, -1.0, 1.0)
        L[:, D] = dead
    return L


def viterbi(L, cfg):
    """Most likely state path under switch penalties. Start is unconstrained."""
    n = len(L)
    cost = np.array([[0.0, cfg.switch_team, cfg.switch_dead],
                     [cfg.switch_team, 0.0, cfg.switch_dead],
                     [cfg.switch_dead, cfg.switch_dead, 0.0]])
    score = L[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - cost          # cand[i, j]: be in i, move to j
        best = cand.argmax(axis=0)
        back[t] = best
        score = cand[best, np.arange(3)] + L[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def refine_possession_boundaries(path, L, window_frames=38):
    """Refine possession boundaries using local changes in control evidence."""
    n = len(path)
    refined = path.copy()
    transitions = np.flatnonzero(path[1:] != path[:-1]) + 1

    for t_star in transitions:
        s_prev = path[t_star - 1]
        s_next = path[t_star]
        w0 = max(0, t_star - window_frames)
        w1 = min(n, t_star + window_frames)
        if w1 - w0 < 4:
            continue

        seg_prev = L[w0:w1, s_prev]
        seg_next = L[w0:w1, s_next]
        cum_prev = np.cumsum(seg_prev)
        tot_next = np.sum(seg_next)
        cum_next = np.cumsum(seg_next)

        scores = cum_prev[:-1] + (tot_next - cum_next[:-1])
        best_k = int(np.argmax(scores))
        best_tau = w0 + best_k

        refined[w0 : best_tau + 1] = s_prev
        refined[best_tau + 1 : w1] = s_next

    return refined


def viterbi_duration_aware(L, cfg, feats=None, fps=25):
    """Duration-aware inertial Viterbi decoder with controlled-touch detection and boundary refinement."""
    n = len(L)
    if feats is not None:
        dH = np.nan_to_num(feats.get("dH", np.full(n, np.nan)), nan=50.0)
        dA = np.nan_to_num(feats.get("dA", np.full(n, np.nan)), nan=50.0)
        b_spd = np.nan_to_num(feats.get("ball_speed", np.full(n, np.nan)), nan=3.0)
        b_conf = np.nan_to_num(feats.get("ball_confidence", np.ones(n)), nan=1.0)
        cH = feats.get("carrier_cH", feats.get("cH", np.zeros(n)))
        cA = feats.get("carrier_cA", feats.get("cA", np.zeros(n)))

        # Controlled touch: close proximity, physical velocity (<6.0 m/s), reliable observation.
        # If opponent has active carrier continuity, taking possession requires intimate tackle distance.
        r_touch = getattr(cfg, "carrier_touch_m", 1.2)
        r_tackle = getattr(cfg, "carrier_tackle_m", 0.8)
        thresh_H = np.where(cA > 0.2, r_tackle, r_touch)
        thresh_A = np.where(cH > 0.2, r_tackle, r_touch)

        touch_H = (dH <= thresh_H) & (dH < dA) & (b_spd < 6.0) & (b_conf > 0.4)
        touch_A = (dA <= thresh_A) & (dA < dH) & (b_spd < 6.0) & (b_conf > 0.4)

        base_team_cost = cfg.switch_team
        cost_H_to_A = np.where(touch_A, base_team_cost * 0.4, base_team_cost)
        cost_A_to_H = np.where(touch_H, base_team_cost * 0.4, base_team_cost)
    else:
        cost_H_to_A = np.full(n, cfg.switch_team)
        cost_A_to_H = np.full(n, cfg.switch_team)

    cost_dead = cfg.switch_dead
    cost_restart = cfg.switch_dead * 0.5  # Lower penalty out of stoppage

    score = L[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)

    for t in range(1, n):
        c00 = score[0]
        c10 = score[1] - cost_A_to_H[t]
        c20 = score[2] - cost_restart

        c01 = score[0] - cost_H_to_A[t]
        c11 = score[1]
        c21 = score[2] - cost_restart

        c02 = score[0] - cost_dead
        c12 = score[1] - cost_dead
        c22 = score[2]

        b0 = 0 if c00 >= c10 and c00 >= c20 else (1 if c10 >= c20 else 2)
        b1 = 0 if c01 >= c11 and c01 >= c21 else (1 if c11 >= c21 else 2)
        b2 = 0 if c02 >= c12 and c02 >= c22 else (1 if c12 >= c22 else 2)

        back[t, 0] = b0
        back[t, 1] = b1
        back[t, 2] = b2

        score[0] = (c00 if b0 == 0 else (c10 if b0 == 1 else c20)) + L[t, 0]
        score[1] = (c01 if b1 == 0 else (c11 if b1 == 1 else c21)) + L[t, 1]
        score[2] = (c02 if b2 == 0 else (c12 if b2 == 1 else c22)) + L[t, 2]

    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    # Refine boundaries using local control evidence
    refined_path = refine_possession_boundaries(path, L, window_frames=int(1.5 * fps))
    return refined_path


def decode_blocks(L, blocks, n_frames, cfg, feats=None, decoder="duration_aware", fps=25):
    """Label array of 'H'/'A'/'D' inside the playing blocks, 'X' everywhere else.

    Resets temporal state appropriately between halves.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    letters = np.array(["H", "A", "D"])
    for first, last in blocks:
        sub_L = L[first : last + 1]
        if decoder == "duration_aware" and feats is not None:
            sub_feats = {
                k: v[first : last + 1] if isinstance(v, np.ndarray) and len(v) == n_frames else v
                for k, v in feats.items()
            }
            path = viterbi_duration_aware(sub_L, cfg, feats=sub_feats, fps=fps)
        elif decoder in ("semi_markov", "candidate_b"):
            from .semi_markov import SemiMarkovConfig, viterbi_semi_markov
            sm_cfg = SemiMarkovConfig(candidate="B", switch_team=cfg.switch_team, switch_dead=cfg.switch_dead)
            sub_feats = {
                k: v[first : last + 1] if isinstance(v, np.ndarray) and len(v) == n_frames else v
                for k, v in feats.items()
            } if feats is not None else None
            path = viterbi_semi_markov(sub_L, sm_cfg, feats=sub_feats, fps=fps)
        else:
            path = viterbi(sub_L, cfg)
        labels[first : last + 1] = letters[path]
    return labels


def calibrate_switch_penalties(*args, **kwargs):
    from .pipeline import calibrate_switch_penalties as _cal
    return _cal(*args, **kwargs)
