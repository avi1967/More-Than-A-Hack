"""Smart Tagging XML loader and contextual restart prior for dead-ball detection.

Treats Smart Tagging events as noisy contextual evidence, never ground truth.
Accounts for Team=N/A, does not assign possession ownership from tags, and
distinguishes the stoppage preceding a restart from live possession following it.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

RESTART_CODES = {"SET PIECES", "GOALS"}
RESTART_TYPES = {
    "Throw-in",
    "Corner Kick",
    "Goal Kick",
    "Free Kick",
    "Kick Off (Start)",
    "Kick Off (After Goal)",
    "GOALS",
}


def load_smart_tagging(xml_path, fps=25):
    """Parse Smart Tagging XML and extract restart and goal intervals.

    Returns:
        list of dict: Each entry contains 'type', 'code', 'team',
                      'start_s', 'end_s', 'f_start', 'f_end'.
    """
    if not xml_path or not os.path.exists(xml_path):
        return []

    try:
        tree = ET.parse(xml_path)
    except Exception:
        return []

    root = tree.getroot()
    instances = []

    for elem in root.iter("instance"):
        code_elem = elem.findtext("code")
        if not code_elem or code_elem.strip() not in RESTART_CODES:
            continue

        code = code_elem.strip()
        labels = {}
        for lbl in elem.findall("label"):
            g = lbl.findtext("group")
            t = lbl.findtext("text")
            if g and t:
                labels[g.strip()] = t.strip()

        ev_type = labels.get("Type", code)
        team = labels.get("Team", "N/A")

        try:
            start_s = float(elem.findtext("start"))
            end_s = float(elem.findtext("end"))
        except (TypeError, ValueError):
            continue

        f_start = int(round(start_s * fps))
        f_end = int(round(end_s * fps))

        instances.append({
            "code": code,
            "type": ev_type,
            "team": team,
            "start_s": start_s,
            "end_s": end_s,
            "f_start": f_start,
            "f_end": f_end,
        })

    return instances


def compute_tag_dead_prior(instances, n_frames, ball_speed, ball_ok, oob,
                           player_activity, fps=25):
    """Compute continuous per-frame prior P(tag_dead) in [0.0, 1.0].

    Distinguishes the stoppage preceding a restart from live possession following it:
    - Pre-restart stoppage: ball is still, out of bounds, or missing, and players
      are walking or repositioning -> prior is elevated (0.5 to 0.9).
    - Post-restart live play: once the ball moves with speed (>2.5 m/s) on the pitch
      or players accelerate into active play, the prior decays rapidly to 0.0.
    """
    prior = np.zeros(n_frames, dtype=np.float32)
    if not instances:
        return prior

    # Precompute physical activity cues
    calm_players = np.clip((1.4 - player_activity) / 0.6, 0.0, 1.0)
    safe_speed = np.nan_to_num(ball_speed, nan=0.0)
    moving_ball = ball_ok & (~oob) & (safe_speed > 2.5)

    for inst in instances:
        f0 = max(0, inst["f_start"])
        f1 = min(n_frames, inst["f_end"])
        if f0 >= f1:
            continue

        ev_type = inst["type"]
        is_goal = (ev_type == "GOALS") or (inst["code"] == "GOALS")
        is_kickoff = "Kick Off" in ev_type

        # Base contextual strength by event type
        if is_goal:
            base_p = 0.90
        elif is_kickoff:
            base_p = 0.85
        elif ev_type in ("Corner Kick", "Goal Kick"):
            base_p = 0.75
        elif ev_type in ("Throw-in", "Free Kick"):
            base_p = 0.65
        else:
            base_p = 0.50

        # Scan for restart execution within the window:
        # A restart occurs when the ball is kicked/thrown into active motion
        # or players transition from calm to high intensity.
        active_restart = False
        decay = 1.0

        for t in range(f0, f1):
            if is_goal:
                # Goals are prolonged stoppages; check if play restarts
                # before kickoff
                p_t = base_p * (0.5 + 0.5 * calm_players[t])
            elif is_kickoff:
                # 5-second kickoff anchor: dead until the ball is kicked
                if moving_ball[t]:
                    active_restart = True
                if active_restart:
                    decay *= 0.85
                p_t = base_p * decay
            else:
                # 35-second set-piece window (stoppage followed by restart)
                # If ball is moving fast inbounds and players are active, play resumed
                if moving_ball[t] and calm_players[t] < 0.3:
                    active_restart = True
                if active_restart:
                    decay *= 0.88  # decay over ~15-20 frames (0.6-0.8s)
                # Stoppage evidence: ball still, out of bounds, missing, or players calm
                stoppage_cue = (oob[t] or (~ball_ok[t]) or (safe_speed[t] < 1.0)
                                or (calm_players[t] > 0.5))
                modulator = 0.5 + 0.5 * float(stoppage_cue)
                p_t = base_p * modulator * decay

            if p_t > prior[t]:
                prior[t] = p_t

    return prior

