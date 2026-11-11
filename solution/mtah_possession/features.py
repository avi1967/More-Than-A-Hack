"""Per-frame evidence from tracking. Everything here is geometry; nothing is fitted to labels."""
import warnings
import numpy as np

from .atd import HOME, AWAY
from .carrier_continuity import compute_carrier_continuity
from .dead_ball import compute_dead_ball_detector


def _smooth(x, width):
    if width <= 1:
        return x
    k = np.ones(width) / width
    pad = width // 2
    xp = np.pad(x, (pad, width - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def _smooth_finite(x, width):
    """Smooth 1D array across finite observations while strictly preserving missing NaNs."""
    if width <= 1:
        return x
    out = np.full_like(x, np.nan)
    finite = np.isfinite(x)
    if not finite.any():
        return out
    filled = np.interp(np.arange(len(x)), np.flatnonzero(finite), x[finite])
    smoothed = _smooth(filled, width)
    out[finite] = smoothed[finite]
    return out


def _fill_short_gaps(raw_ball, max_gap_frames, fps, pitch=None):
    """Linear interpolation of short NaN gaps, avoiding long gaps, unphysical speeds, and dead-ball boundaries.

    Returns:
        filled_ball: (N, 2) float64 array of coordinates, NaN where unavailable.
        imputed_mask: (N,) boolean array, True where position was interpolated.
        confidence: (N,) float32 array in [0.0, 1.0], 1.0 for directly observed,
                    degrading smoothly with gap length and distance to observation,
                    and 0.0 for missing/unfilled frames.
    """
    n = len(raw_ball)
    filled_ball = raw_ball.copy()
    imputed_mask = np.zeros(n, dtype=bool)
    observed_mask = np.isfinite(raw_ball[:, 0]) & np.isfinite(raw_ball[:, 1])
    confidence = np.zeros(n, dtype=np.float32)
    confidence[observed_mask] = 1.0

    if observed_mask.sum() < 2:
        return filled_ball, imputed_mask, confidence

    diff = np.diff(np.concatenate(([0], observed_mask.astype(np.int8), [0])))
    obs_starts = np.flatnonzero(diff == 1)
    obs_ends = np.flatnonzero(diff == -1)

    length = pitch[0] if pitch is not None else 105.0
    width = pitch[1] if pitch is not None else 68.0

    for k in range(len(obs_ends) - 1):
        g_start = int(obs_ends[k])
        g_end = int(obs_starts[k + 1])
        g_len = g_end - g_start
        if g_len <= 0 or g_len > max_gap_frames:
            continue

        t_prev = g_start - 1
        t_next = g_end
        p_prev = raw_ball[t_prev]
        p_next = raw_ball[t_next]

        # Kinematic check: reject unphysical speeds (> 35 m/s)
        dist = np.linalg.norm(p_next - p_prev)
        dt = (g_len + 1) / fps
        if dist / max(dt, 1e-4) > 35.0:
            continue

        # Boundary check: avoid interpolating across likely dead-ball boundary exits
        margin = -0.5
        oob_prev = (p_prev[0] < margin or p_prev[0] > length - margin or
                    p_prev[1] < margin or p_prev[1] > width - margin)
        oob_next = (p_next[0] < margin or p_next[0] > length - margin or
                    p_next[1] < margin or p_next[1] > width - margin)
        if oob_prev or oob_next:
            continue

        for i, t in enumerate(range(g_start, g_end)):
            alpha = (i + 1.0) / (g_len + 1.0)
            filled_ball[t] = (1.0 - alpha) * p_prev + alpha * p_next
            imputed_mask[t] = True

            # Continuous confidence feature decaying with gap length and edge distance
            d_edge = min(t - t_prev, t_next - t)
            gap_penalty = 1.0 - 0.5 * (g_len / max_gap_frames)
            u = d_edge / ((g_len + 1.0) / 2.0)
            center_penalty = 0.7 + 0.3 * u
            confidence[t] = float(np.clip(gap_penalty * center_penalty, 0.1, 0.95))

    return filled_ball, imputed_mask, confidence


def compute(track, pitch, fps, cfg, smart_tag_instances=None):
    """dict of per-frame arrays with confidence-aware tracking signals."""
    length, width = pitch
    raw_ball = track["ball"].astype(np.float64)
    n = len(raw_ball)

    # 1. Preserve original observation mask
    ball_observed = np.isfinite(raw_ball[:, 0]) & np.isfinite(raw_ball[:, 1])

    # 2 & 3 & 5. Distinguish observed, imputed and missing; compute continuous confidence;
    # avoid interpolating long gaps or likely dead-ball exits
    max_gap_frames = int(cfg.ball_gap_s * fps)
    filled_ball, ball_imputed, ball_confidence = _fill_short_gaps(
        raw_ball, max_gap_frames, fps, pitch=(length, width)
    )
    ball_missing = ~np.isfinite(filled_ball[:, 0])
    ball_ok = ~ball_missing

    players = track["players"].astype(np.float64)
    team = track["team"]

    # Nearest home / away player to the ball
    d = np.sqrt(((players - filled_ball[:, None, :]) ** 2).sum(axis=2))
    d = np.where(np.isfinite(d), d, np.inf)
    dH = d[:, team == HOME].min(axis=1)
    dA = d[:, team == AWAY].min(axis=1)
    dH[~np.isfinite(dH)] = np.nan
    dA[~np.isfinite(dA)] = np.nan

    # 4. Ball velocity across reliable observation intervals
    lag = max(1, int(round(cfg.speed_lag_s * fps)))
    v = np.full(n, np.nan, dtype=np.float64)
    ball_speed_reliable = np.zeros(n, dtype=bool)

    half_lag = lag // 2
    dt = lag / fps

    disp = np.sqrt(((filled_ball[lag:] - filled_ball[:-lag]) ** 2).sum(axis=1))
    v[half_lag : half_lag + len(disp)] = disp / dt

    obs_both = ball_observed[lag:] & ball_observed[:-lag]
    ball_speed_reliable[half_lag : half_lag + len(obs_both)] = obs_both

    # Filter unphysical displacement spikes (> 35 m/s)
    v = np.where(v <= 35.0, v, np.nan)
    ball_speed = _smooth_finite(v, max(1, int(0.2 * fps)))

    # 6 & 7. Team-specific player activity and tracking-quality indicators
    pos_valid = np.isfinite(players[:, :, 0]) & np.isfinite(players[:, :, 1])
    valid_home_players = pos_valid[:, team == HOME].sum(axis=1)
    valid_away_players = pos_valid[:, team == AWAY].sum(axis=1)
    valid_players = valid_home_players + valid_away_players

    pv = np.full((n, players.shape[1]), np.nan, dtype=np.float64)
    p_disp = np.sqrt(((players[lag:] - players[:-lag]) ** 2).sum(axis=2))
    p_speed = p_disp / dt
    p_speed = np.where(p_speed <= 12.0, p_speed, np.nan)  # filter slot ID swaps (> 12 m/s)
    pv[half_lag : half_lag + len(p_speed)] = p_speed

    vel_valid = np.isfinite(pv)
    valid_player_velocities = vel_valid.sum(axis=1)
    missing_player_fraction = np.clip(1.0 - (valid_players / 22.0), 0.0, 1.0)
    tracking_quality = np.clip(0.6 * (valid_players / 20.0) + 0.4 * ball_confidence, 0.0, 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            home_speed = np.nanmedian(pv[:, team == HOME], axis=1)
            away_speed = np.nanmedian(pv[:, team == AWAY], axis=1)
            all_speed = np.nanmedian(pv, axis=1)

    home_speed_filled = np.where(np.isfinite(home_speed), home_speed, np.nanmedian(home_speed))
    away_speed_filled = np.where(np.isfinite(away_speed), away_speed, np.nanmedian(away_speed))
    all_speed_filled = np.where(np.isfinite(all_speed), all_speed, np.nanmedian(all_speed))

    act_w = int(cfg.activity_window_s * fps)
    home_activity = _smooth(home_speed_filled, act_w)
    away_activity = _smooth(away_speed_filled, act_w)
    activity_diff = home_activity - away_activity
    activity = _smooth(all_speed_filled, act_w)

    m = cfg.oob_margin_m
    oob = ball_ok & ((filled_ball[:, 0] < -m) | (filled_ball[:, 0] > length + m)
                     | (filled_ball[:, 1] < -m) | (filled_ball[:, 1] > width + m))

    slow = (np.nan_to_num(ball_speed, nan=99.0) < cfg.still_speed) & ball_ok
    still = _smooth(slow.astype(float), int(cfg.still_window_s * fps)) > 0.95

    feats = {
        # Core existing keys (backward compatible)
        "ball_ok": ball_ok,
        "dH": dH,
        "dA": dA,
        "ball_speed": ball_speed,
        "activity": activity,
        "oob": oob,
        "still": still,
        # Phase 2 confidence-aware ball features
        "ball_observed": ball_observed,
        "ball_imputed": ball_imputed,
        "ball_missing": ball_missing,
        "ball_confidence": ball_confidence,
        "ball_speed_reliable": ball_speed_reliable,
        # Phase 2 team-specific player activity
        "home_speed": home_speed,
        "away_speed": away_speed,
        "home_activity": home_activity,
        "away_activity": away_activity,
        "activity_diff": activity_diff,
        "valid_home_players": valid_home_players,
        "valid_away_players": valid_away_players,
        # Phase 2 tracking-quality indicators
        "valid_players": valid_players,
        "valid_player_velocities": valid_player_velocities,
        "missing_player_fraction": missing_player_fraction,
        "tracking_quality": tracking_quality,
    }

    # Phase 3/8 independent probabilistic dead-ball detector
    db_feats = compute_dead_ball_detector(track, feats, pitch, fps, smart_tag_instances, cfg=cfg)
    feats.update(db_feats)

    # Phase 7 independent carrier-loss confidence & possession continuity mechanism
    if getattr(cfg, "carrier_continuity", True):
        cH, cA, dH_eff, dA_eff, carrier_loss_H, carrier_loss_A = compute_carrier_continuity(
            dH=feats["dH"],
            dA=feats["dA"],
            ball_speed=feats["ball_speed"],
            ball_ok=feats["ball_ok"],
            ball_confidence=feats.get("ball_confidence"),
            fps=fps,
            tau_carrier=getattr(cfg, "carrier_tau_s", 2.0),
            tau_kick=getattr(cfg, "carrier_kick_tau_s", 0.4),
            r_touch=getattr(cfg, "carrier_touch_m", 1.2),
            r_tackle=getattr(cfg, "carrier_tackle_m", 0.8),
            r_sep=getattr(cfg, "carrier_sep_m", 0.5),
            r_virt=getattr(cfg, "carrier_virt_m", 0.7),
        )
        feats["dH_raw"] = feats["dH"]
        feats["dA_raw"] = feats["dA"]
        feats["dH"] = dH_eff
        feats["dA"] = dA_eff
        feats["carrier_cH"] = cH
        feats["carrier_cA"] = cA
        feats["carrier_loss_H"] = carrier_loss_H
        feats["carrier_loss_A"] = carrier_loss_A

    return feats
