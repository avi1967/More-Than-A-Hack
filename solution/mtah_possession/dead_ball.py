"""Independent probabilistic dead-ball detector.

Estimates P(D | tracking evidence) independently from team possession classification.
Produces a continuous per-frame P(D) in [0.0, 1.0] using:
1. Ball proximity to pitch boundaries.
2. Ball outside-pitch probability.
3. Ball observation confidence.
4. Ball inactivity duration (carrier-holding aware).
5. Home and away player activity and calmness.
6. Collective player repositioning.
7. Player density and formation changes.
8. Smart Tagging restart intervals (treated as noisy contextual evidence).
9. Live inbounds ball flight counter-evidence.
"""
import warnings
import numpy as np

from .atd import HOME, AWAY
from .smart_tagging import compute_tag_dead_prior


def _smooth(x, width):
    if width <= 1:
        return x
    k = np.ones(width) / width
    pad = width // 2
    xp = np.pad(x, (pad, width - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def _compute_dead_ball_detector_v1(track, feats, pitch, fps, smart_tag_instances=None):
    """Phase 3/7 baseline dead-ball detector preserved for exact reproducible ablation."""
    length, width = pitch
    raw_ball = track["ball"].astype(np.float64)
    players = track["players"].astype(np.float64)
    team = track["team"]
    n = len(raw_ball)

    ball_ok = feats["ball_ok"]
    ball_conf = feats.get("ball_confidence", np.where(ball_ok, 1.0, 0.0)).astype(np.float32)
    ball_speed = feats.get("ball_speed_reliable", feats["ball_speed"])

    bx = raw_ball[:, 0]
    by = raw_ball[:, 1]
    has_pos = np.isfinite(bx) & np.isfinite(by)

    dx_left = bx
    dx_right = length - bx
    dy_bot = by
    dy_top = width - by

    dist_to_boundary = np.full(n, np.nan, dtype=np.float32)
    min_dist = np.minimum(np.minimum(dx_left, dx_right), np.minimum(dy_bot, dy_top))
    dist_to_boundary[has_pos] = min_dist[has_pos].astype(np.float32)

    pos_d = np.maximum(0.0, np.nan_to_num(dist_to_boundary, nan=50.0))
    boundary_proximity = np.exp(-pos_d / 3.0).astype(np.float32)
    boundary_proximity[~has_pos] = 0.0

    margin = -0.3
    tau = 0.25
    p_ball_out = np.zeros(n, dtype=np.float32)
    p_ball_out[has_pos] = (1.0 / (1.0 + np.exp((min_dist[has_pos] - margin) / tau))).astype(np.float32)

    speed_safe = np.nan_to_num(ball_speed, nan=99.0)
    is_still = ball_ok & (speed_safe < 0.5) & (~(p_ball_out > 0.8))

    ball_inactivity_s = np.zeros(n, dtype=np.float32)
    dt = 1.0 / fps
    accum = 0.0
    for t in range(n):
        if is_still[t]:
            accum += dt
        elif ball_ok[t] and speed_safe[t] >= 0.5:
            accum = 0.0
        elif not ball_ok[t]:
            if accum >= 1.0:
                accum += dt
            else:
                accum = 0.0
        ball_inactivity_s[t] = accum

    p_inactivity = np.clip((ball_inactivity_s - 1.5) / 3.0, 0.0, 1.0).astype(np.float32)

    home_act = feats.get("home_activity", feats["activity"])
    away_act = feats.get("away_activity", feats["activity"])
    all_act = 0.5 * (home_act + away_act)
    all_act_safe = np.nan_to_num(all_act, nan=1.5)
    p_calm = np.clip((1.3 - all_act_safe) / 0.6, 0.0, 1.0).astype(np.float32)

    lag_rep = max(1, int(0.5 * fps))
    pos_h = players[:, team == HOME, :]
    pos_a = players[:, team == AWAY, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            cent_h = np.nanmedian(pos_h, axis=1)
            cent_a = np.nanmedian(pos_a, axis=1)

    default_h = np.array([length / 3.0, width / 2.0], dtype=np.float64)
    default_a = np.array([2.0 * length / 3.0, width / 2.0], dtype=np.float64)
    cent_h = np.where(np.isfinite(cent_h), cent_h, default_h)
    cent_a = np.where(np.isfinite(cent_a), cent_a, default_a)

    dt_rep = (2 * lag_rep) / fps
    v_cent_h = np.zeros(n, dtype=np.float32)
    v_cent_a = np.zeros(n, dtype=np.float32)

    dh = np.sqrt(((cent_h[2 * lag_rep:] - cent_h[:-2 * lag_rep]) ** 2).sum(axis=1))
    da = np.sqrt(((cent_a[2 * lag_rep:] - cent_a[:-2 * lag_rep]) ** 2).sum(axis=1))
    v_cent_h[lag_rep : lag_rep + len(dh)] = np.nan_to_num(dh / dt_rep, nan=0.0)
    v_cent_a[lag_rep : lag_rep + len(da)] = np.nan_to_num(da / dt_rep, nan=0.0)

    repositioning_speed = 0.5 * (v_cent_h + v_cent_a)
    repositioning_speed = _smooth(repositioning_speed, int(0.5 * fps))

    p_repositioning = (np.clip((repositioning_speed - 0.25) / 0.6, 0.0, 1.0) * p_calm).astype(np.float32)

    in_box_h = ((pos_h[:, :, 0] < 16.5) | (pos_h[:, :, 0] > length - 16.5)) & (
        np.abs(pos_h[:, :, 1] - width / 2.0) < 20.15
    )
    in_box_a = ((pos_a[:, :, 0] < 16.5) | (pos_a[:, :, 0] > length - 16.5)) & (
        np.abs(pos_a[:, :, 1] - width / 2.0) < 20.15
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            n_in_box = np.nansum(in_box_h, axis=1) + np.nansum(in_box_a, axis=1)
    valid_players = np.maximum(1.0, feats.get("valid_players", 20.0))
    box_density = np.nan_to_num(n_in_box / valid_players, nan=0.0).astype(np.float32)

    p_box_density = (np.clip((box_density - 0.25) / 0.20, 0.0, 1.0) * p_calm).astype(np.float32)

    tag_dead_prior = compute_tag_dead_prior(
        smart_tag_instances, n, ball_speed, ball_ok, feats["oob"], all_act_safe, fps=fps
    )

    dH = feats["dH"]
    dA = feats["dA"]
    d_min = np.minimum(dH, dA)
    d_min_safe = np.nan_to_num(d_min, nan=99.0)
    live_ctrl = np.where(
        ball_ok,
        np.clip(1.0 - d_min_safe / 2.5, 0.0, 1.0) * ball_conf * (1.0 - p_ball_out),
        0.0,
    ).astype(np.float32)

    ball_missing = feats.get("ball_missing", ~ball_ok)
    missing_stoppage = ball_missing & ((p_calm > 0.4) | (boundary_proximity > 0.5) | (tag_dead_prior > 0.4))
    p_missing_dead = (missing_stoppage.astype(np.float32) * np.maximum(p_calm, tag_dead_prior)).astype(np.float32)

    p0_dead = 0.16
    base_logit = float(np.log(p0_dead / (1.0 - p0_dead)))

    dead_logit = (
        base_logit
        + 6.0 * p_ball_out
        + 3.5 * p_inactivity
        + 1.5 * p_calm
        + 1.2 * p_repositioning
        + 1.0 * p_box_density
        + 2.5 * tag_dead_prior
        + 1.5 * p_missing_dead
        - 3.5 * live_ctrl
    )

    p_dead = (1.0 / (1.0 + np.exp(-dead_logit))).astype(np.float32)

    return {
        "p_dead": p_dead,
        "dist_to_boundary": dist_to_boundary,
        "boundary_proximity": boundary_proximity,
        "p_ball_out": p_ball_out,
        "ball_inactivity_s": ball_inactivity_s,
        "p_inactivity": p_inactivity,
        "p_calm": p_calm,
        "repositioning_speed": repositioning_speed,
        "p_repositioning": p_repositioning,
        "box_density": box_density,
        "p_box_density": p_box_density,
        "tag_dead_prior": tag_dead_prior,
    }


def _compute_dead_ball_detector_v2(track, feats, pitch, fps, smart_tag_instances=None):
    """Phase 8 refined probabilistic dead-ball detector.

    Resolves:
    1. Boolean type bug on ball_speed (uses true float velocity).
    2. Ball in flight live counter-evidence (prevents aerial passes from triggering D).
    3. Missing ball interior tracking dropout vs stoppage distinction.
    4. Stationary live ball (carrier holding/dribbling) vs unpossessed dead ball.
    5. Corroborated Smart Tagging prior integration.
    """
    length, width = pitch
    raw_ball = track["ball"].astype(np.float64)
    players = track["players"].astype(np.float64)
    team = track["team"]
    n = len(raw_ball)

    ball_ok = feats["ball_ok"]
    ball_conf = feats.get("ball_confidence", np.where(ball_ok, 1.0, 0.0)).astype(np.float32)
    # FIX 1: Use actual physical ball speed array, not the boolean mask!
    ball_speed = feats["ball_speed"].astype(np.float64)

    # 1. Ball proximity to pitch boundaries
    bx = raw_ball[:, 0]
    by = raw_ball[:, 1]
    has_pos = np.isfinite(bx) & np.isfinite(by)

    dx_left = bx
    dx_right = length - bx
    dy_bot = by
    dy_top = width - by

    dist_to_boundary = np.full(n, np.nan, dtype=np.float32)
    min_dist = np.minimum(np.minimum(dx_left, dx_right), np.minimum(dy_bot, dy_top))
    dist_to_boundary[has_pos] = min_dist[has_pos].astype(np.float32)

    # Boundary proximity: decays smoothly away from touchlines/goallines
    pos_d = np.maximum(0.0, np.nan_to_num(dist_to_boundary, nan=50.0))
    boundary_proximity = np.exp(-pos_d / 3.0).astype(np.float32)
    boundary_proximity[~has_pos] = 0.0

    # 2. Ball outside pitch probability: sharp sigmoid at -0.3m
    margin = -0.3
    tau = 0.20
    p_ball_out = np.zeros(n, dtype=np.float32)
    p_ball_out[has_pos] = (1.0 / (1.0 + np.exp((min_dist[has_pos] - margin) / tau))).astype(np.float32)

    # 3. Ball inactivity duration:
    # FIX 4: Distinguish stationary live ball (carrier holding/standing on ball) from stationary dead ball
    speed_safe = np.nan_to_num(ball_speed, nan=99.0)
    dH = feats.get("dH_raw", feats["dH"])
    dA = feats.get("dA_raw", feats["dA"])
    d_min = np.minimum(np.nan_to_num(dH, nan=50.0), np.nan_to_num(dA, nan=50.0))

    # A still ball is only a candidate stoppage if:
    # Either nobody is in intimate contact (d_min > 1.0m), OR ball is near/outside boundary
    is_still_candidate = ball_ok & (speed_safe < 0.6)
    carrier_holding = is_still_candidate & (d_min <= 1.0) & (pos_d > 2.0)
    is_still_stoppage = is_still_candidate & (~carrier_holding) & (~(p_ball_out > 0.8))

    ball_inactivity_s = np.zeros(n, dtype=np.float32)
    dt = 1.0 / fps
    accum = 0.0
    for t in range(n):
        if is_still_stoppage[t]:
            accum += dt
        elif ball_ok[t] and speed_safe[t] >= 0.8:
            accum = 0.0
        elif carrier_holding[t]:
            # Slowly decay inactivity if carrier is holding live ball
            accum = max(0.0, accum - 2.0 * dt)
        elif not ball_ok[t]:
            if accum >= 1.0:
                accum += dt
            else:
                accum = 0.0
        ball_inactivity_s[t] = accum

    p_inactivity = np.clip((ball_inactivity_s - 1.2) / 2.5, 0.0, 1.0).astype(np.float32)

    # 4. Player activity and calmness
    home_act = feats.get("home_activity", feats["activity"])
    away_act = feats.get("away_activity", feats["activity"])
    all_act = 0.5 * (home_act + away_act)
    all_act_safe = np.nan_to_num(all_act, nan=1.5)
    p_calm = np.clip((1.2 - all_act_safe) / 0.5, 0.0, 1.0).astype(np.float32)

    # 5. Collective player repositioning
    lag_rep = max(1, int(0.5 * fps))
    pos_h = players[:, team == HOME, :]
    pos_a = players[:, team == AWAY, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            cent_h = np.nanmedian(pos_h, axis=1)
            cent_a = np.nanmedian(pos_a, axis=1)
    default_h = np.array([length / 3.0, width / 2.0], dtype=np.float64)
    default_a = np.array([2.0 * length / 3.0, width / 2.0], dtype=np.float64)
    cent_h = np.where(np.isfinite(cent_h), cent_h, default_h)
    cent_a = np.where(np.isfinite(cent_a), cent_a, default_a)

    dt_rep = (2 * lag_rep) / fps
    v_cent_h = np.zeros(n, dtype=np.float32)
    v_cent_a = np.zeros(n, dtype=np.float32)
    dh = np.sqrt(((cent_h[2 * lag_rep:] - cent_h[:-2 * lag_rep]) ** 2).sum(axis=1))
    da = np.sqrt(((cent_a[2 * lag_rep:] - cent_a[:-2 * lag_rep]) ** 2).sum(axis=1))
    v_cent_h[lag_rep : lag_rep + len(dh)] = np.nan_to_num(dh / dt_rep, nan=0.0)
    v_cent_a[lag_rep : lag_rep + len(da)] = np.nan_to_num(da / dt_rep, nan=0.0)

    repositioning_speed = 0.5 * (v_cent_h + v_cent_a)
    repositioning_speed = _smooth(repositioning_speed, int(0.5 * fps))
    p_repositioning = (np.clip((repositioning_speed - 0.25) / 0.6, 0.0, 1.0) * p_calm).astype(np.float32)

    # 6. Box density
    in_box_h = ((pos_h[:, :, 0] < 16.5) | (pos_h[:, :, 0] > length - 16.5)) & (
        np.abs(pos_h[:, :, 1] - width / 2.0) < 20.15
    )
    in_box_a = ((pos_a[:, :, 0] < 16.5) | (pos_a[:, :, 0] > length - 16.5)) & (
        np.abs(pos_a[:, :, 1] - width / 2.0) < 20.15
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            n_in_box = np.nansum(in_box_h, axis=1) + np.nansum(in_box_a, axis=1)
    valid_players = np.maximum(1.0, feats.get("valid_players", 20.0))
    box_density = np.nan_to_num(n_in_box / valid_players, nan=0.0).astype(np.float32)
    p_box_density = (np.clip((box_density - 0.28) / 0.20, 0.0, 1.0) * p_calm).astype(np.float32)

    # 7. Smart Tagging prior
    tag_dead_prior = compute_tag_dead_prior(
        smart_tag_instances, n, ball_speed, ball_ok, feats["oob"], all_act_safe, fps=fps
    )

    # FIX 2: Live ball flight counter-evidence (suppresses dead ball during aerial passes/shots)
    flight_live = np.where(
        ball_ok & (p_ball_out < 0.2),
        np.clip((speed_safe - 2.5) / 5.0, 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    # Live ball control counter-evidence
    live_ctrl = np.where(
        ball_ok,
        np.clip(1.0 - d_min / 2.0, 0.0, 1.0) * ball_conf * (1.0 - p_ball_out),
        0.0,
    ).astype(np.float32)

    # FIX 3: Missing ball during live play vs stoppage:
    # A missing ball is only a stoppage cue if near boundary or corroborated by calm players
    ball_missing = feats.get("ball_missing", ~ball_ok)
    missing_stoppage = ball_missing & (
        (p_calm > 0.6) | (boundary_proximity > 0.5) | (tag_dead_prior > 0.5)
    )
    p_missing_dead = (missing_stoppage.astype(np.float32) * np.maximum(p_calm, tag_dead_prior)).astype(np.float32)

    # Log-odds fusion
    p0_dead = 0.16
    base_logit = float(np.log(p0_dead / (1.0 - p0_dead)))

    dead_logit = (
        base_logit
        + 6.5 * p_ball_out
        + 4.0 * p_inactivity
        + 1.2 * p_calm
        + 1.0 * p_repositioning
        + 1.0 * p_box_density
        + 2.0 * tag_dead_prior
        + 1.5 * p_missing_dead
        - 3.5 * live_ctrl
        - 4.5 * flight_live   # Strong negative log-odds during inbounds ball flight
    )

    p_dead = (1.0 / (1.0 + np.exp(-dead_logit))).astype(np.float32)

    return {
        "p_dead": p_dead,
        "dist_to_boundary": dist_to_boundary,
        "boundary_proximity": boundary_proximity,
        "p_ball_out": p_ball_out,
        "ball_inactivity_s": ball_inactivity_s,
        "p_inactivity": p_inactivity,
        "p_calm": p_calm,
        "repositioning_speed": repositioning_speed,
        "p_repositioning": p_repositioning,
        "box_density": box_density,
        "p_box_density": p_box_density,
        "tag_dead_prior": tag_dead_prior,
        "flight_live": flight_live,
    }


def compute_dead_ball_detector(track, feats, pitch, fps, smart_tag_instances=None, cfg=None):
    """Estimate P(D | evidence) continuously per frame without any NaN propagation.

    Parameters:
        track: ATD tracking dict containing 'ball', 'players', 'team'.
        feats: Dict of computed intermediate features.
        pitch: (length, width) tuple in metres.
        fps: Sampling rate (typically 25).
        smart_tag_instances: Optional parsed Smart Tagging set-piece events list.
        cfg: Optional Config instance. If cfg.dead_ball_v2 is False, executes baseline v1 detector.

    Returns:
        dict containing continuous probabilistic dead-ball features.
    """
    if cfg is not None and not getattr(cfg, "dead_ball_v2", True):
        return _compute_dead_ball_detector_v1(track, feats, pitch, fps, smart_tag_instances=smart_tag_instances)
    return _compute_dead_ball_detector_v2(track, feats, pitch, fps, smart_tag_instances=smart_tag_instances)
