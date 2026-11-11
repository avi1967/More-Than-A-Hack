"""Carrier-loss confidence and continuous possession continuity mechanism.

Mitigates spurious possession handover to opponents when a ball carrier's track
is temporarily lost due to occlusion, cluster tackles, or camera perspective changes.
"""
import numpy as np


def compute_carrier_continuity(dH, dA, ball_speed, ball_ok, ball_confidence, fps=25,
                               tau_carrier=2.0, tau_kick=0.4, r_touch=1.2, r_tackle=0.8,
                               r_sep=0.5, r_virt=0.7):
    """Computes continuous carrier continuity confidence and effective player distances.

    Parameters:
        dH: (N,) float array, distance from ball to nearest Home player (metres)
        dA: (N,) float array, distance from ball to nearest Away player (metres)
        ball_speed: (N,) float array, estimated ball speed (m/s)
        ball_ok: (N,) bool array, True where ball position is available
        ball_confidence: (N,) float array in [0, 1]
        fps: frames per second (typically 25)
        tau_carrier: memory timescale (seconds) for carrier continuity during steady play
        tau_kick: rapid decay timescale (seconds) when ball is kicked into flight (> 9 m/s)
        r_touch: radius (metres) for direct unconstrained carrier control
        r_tackle: intimate radius (metres) required to tackle from an incumbent carrier
        r_sep: distance separation (metres) required to confirm decisive ownership
        r_virt: nominal proximity (metres) assigned to virtual carrier during tracking loss

    Returns:
        cH: (N,) float32 array, Home carrier-continuity confidence in [0, 1]
        cA: (N,) float32 array, Away carrier-continuity confidence in [0, 1]
        dH_eff: (N,) float64 array, effective Home distance to ball
        dA_eff: (N,) float64 array, effective Away distance to ball
        carrier_loss_H: (N,) bool array, True where Home carrier is lost from tracking
        carrier_loss_A: (N,) bool array, True where Away carrier is lost from tracking
    """
    n = len(dH)
    cH = np.zeros(n, dtype=np.float32)
    cA = np.zeros(n, dtype=np.float32)

    alpha_carrier = float(np.exp(-1.0 / (fps * max(tau_carrier, 1e-3))))
    alpha_kick = float(np.exp(-1.0 / (fps * max(tau_kick, 1e-3))))

    curr_cH = 0.0
    curr_cA = 0.0

    for t in range(n):
        b_ok = bool(ball_ok[t]) if ball_ok is not None else True
        b_conf = float(ball_confidence[t]) if ball_confidence is not None else 1.0
        v_b = float(ball_speed[t]) if np.isfinite(ball_speed[t]) else 3.0

        dh = float(dH[t]) if np.isfinite(dH[t]) else 99.0
        da = float(dA[t]) if np.isfinite(dA[t]) else 99.0

        is_live = b_ok and (b_conf > 0.4)

        # Asymmetric control thresholds: taking control from an incumbent requires
        # intimate proximity (r_tackle), while open ball requires r_touch.
        thresh_H = r_tackle if curr_cA > 0.2 else r_touch
        thresh_A = r_tackle if curr_cH > 0.2 else r_touch

        # Effective separation: compare against virtual carrier distance if incumbent has continuity
        dh_prev_eff = min(dh, r_virt) if curr_cH > 0.2 else dh
        da_prev_eff = min(da, r_virt) if curr_cA > 0.2 else da

        ctrl_H = is_live and (dh <= thresh_H) and (da_prev_eff - dh >= r_sep) and (v_b <= 7.0)
        ctrl_A = is_live and (da <= thresh_A) and (dh_prev_eff - da >= r_sep) and (v_b <= 7.0)

        # Decay timescale: kicked/flying ball decays quickly, steady dribble decays smoothly
        decay = alpha_kick if v_b > 9.0 else alpha_carrier

        if ctrl_H and not ctrl_A:
            curr_cH = 1.0
            curr_cA = 0.0
        elif ctrl_A and not ctrl_H:
            curr_cA = 1.0
            curr_cH = 0.0
        else:
            curr_cH *= decay
            curr_cA *= decay

        cH[t] = curr_cH
        cA[t] = curr_cA

    # Effective distance mapping: blend raw distance toward virtual carrier distance
    # proportional to continuity confidence
    valid_H = np.isfinite(dH)
    dH_eff = np.where(
        valid_H,
        (1.0 - cH) * dH + cH * np.minimum(dH, r_virt),
        np.where(cH > 0.1, r_virt, dH)
    )

    valid_A = np.isfinite(dA)
    dA_eff = np.where(
        valid_A,
        (1.0 - cA) * dA + cA * np.minimum(dA, r_virt),
        np.where(cA > 0.1, r_virt, dA)
    )

    # Carrier loss indicators: continuity remains active while tracked player is missing or far (>2.2m)
    carrier_loss_H = (cH > 0.2) & (~np.isfinite(dH) | (dH > 2.2))
    carrier_loss_A = (cA > 0.2) & (~np.isfinite(dA) | (dA > 2.2))

    return cH, cA, dH_eff, dA_eff, carrier_loss_H, carrier_loss_A

