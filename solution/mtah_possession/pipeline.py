import math
from dataclasses import replace
import numpy as np

from .atd import load_tracking
from .features import compute
from .model import Config, log_emissions, decode_blocks
from .smart_tagging import load_smart_tagging


def calibrate_switch_penalties(L, blocks, n_frames, cfg, fps=25, target_s=15.0,
                               iters=12, m_range=(0.02, 2.0), feats=None,
                               decoder="duration_aware"):
    """Calibrate switch penalties by bisection in log-space over m_range to match target_s mean possession duration."""
    low = math.log(m_range[0])
    high = math.log(m_range[1])
    best_m = None
    best_labels = None
    best_err = float("inf")

    for _ in range(iters):
        mid = (low + high) / 2.0
        m = math.exp(mid)
        cfg_m = replace(cfg, switch_team=cfg.switch_team * m, switch_dead=cfg.switch_dead * m)
        labels = decode_blocks(L, blocks, n_frames, cfg_m, feats=feats, decoder=decoder, fps=fps)
        d = describe(labels, fps=fps)
        dur = d["mean_s"]
        err = abs(dur - target_s)
        if err < best_err:
            best_err = err
            best_m = m
            best_labels = labels

        # Mean duration falls as m falls; update bounds accordingly
        if dur < target_s:
            low = mid
        else:
            high = mid

    return best_m, best_labels


def build_labels(raw_path, meta_path, n_frames, pitch, blocks, fps=25, cfg=None,
                 target_possession_s=15.0, calibrate=True, smart_tag_path=None,
                 decoder="duration_aware"):
    """frame-indexed array of H/A/D/X. `blocks` = [(first_frame, last_frame)] of graded play."""
    cfg = cfg or Config()
    track = load_tracking(raw_path, meta_path, n_frames, pitch)
    instances = load_smart_tagging(smart_tag_path, fps=fps) if smart_tag_path else None
    feats = compute(track, pitch, fps, cfg, smart_tag_instances=instances)
    L = log_emissions(feats, cfg)
    if calibrate:
        m, labels = calibrate_switch_penalties(
            L, blocks, n_frames, cfg, fps=fps, target_s=target_possession_s,
            feats=feats, decoder=decoder
        )
        feats["multiplier"] = m
        return labels, feats
    else:
        feats["multiplier"] = 1.0
        return decode_blocks(L, blocks, n_frames, cfg, feats=feats, decoder=decoder, fps=fps), feats


def write_csv(path, labels):
    with open(path, "w") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def describe(labels, fps=25):
    runs, prev, start = [], None, 0
    for i, l in enumerate(list(labels) + ["#"]):
        if l != prev:
            if prev in ("H", "A"):
                runs.append(i - start)
            prev, start = l, i
    alive = sum(runs)
    return {"possessions": len(runs),
            "mean_s": alive / fps / max(1, len(runs)),
            "H": int((labels == "H").sum()), "A": int((labels == "A").sum()),
            "D": int((labels == "D").sum()), "X": int((labels == "X").sum())}
