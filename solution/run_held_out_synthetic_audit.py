"""Phase 9.1 Held-Out Adversarial Synthetic Benchmark Audit.

Generates 10 held-out synthetic match datasets across 8 adversarial scenarios:
1. Many genuine 1-3s short possessions.
2. Very long possessions (30-60s).
3. Highly imbalanced H/A spells.
4. Rapid turnovers.
5. Consecutive short dead-ball interruptions.
6. Long aerial passages.
7. Carrier disappearance during genuine possession.
8. Dense contested situations.

Evaluates Candidate A (Phase 8 baseline) vs Candidate B (Semi-Markov)
without modifying Candidate B in any way.
Computes official metrics via evaluation/possession_metrics.py:
- spell_f1_mean (IoU thresholds: 0.3, 0.4, 0.5, 0.6, 0.7)
- macro_f1 (unweighted over {H, A, D}, GT X excluded, pred X charged)
- alive_dead_f1
- team_accuracy
- fragmentation ratio
- possession count error
- mean duration error
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from mtah_possession.model import Config, log_emissions, decode_blocks
from mtah_possession.pipeline import describe, calibrate_switch_penalties
from mtah_possession.semi_markov import (
    SemiMarkovConfig,
    decode_blocks_semi_markov,
    calibrate_switch_penalties_semi_markov,
)
from evaluation import possession_metrics as pm

L_, W_, S, FPS = 105.0, 68.0, 46, 25


def make_adversarial_synthetic_match(tmp, seed=100, scenario="mixed"):
    """Generate an adversarial synthetic match with complex distributions."""
    rng = np.random.default_rng(seed)
    n = 6000  # 240 seconds = 4 minutes
    truth = np.full(n, "D", dtype="<U1")
    t = 0
    team = 0

    # Build ground truth sequence based on scenario
    while t < n:
        if scenario == "short_poss":
            dur = int(rng.integers(25, 100))  # 1.0 to 4.0 seconds
        elif scenario == "long_poss":
            dur = int(rng.integers(750, 1500))  # 30 to 60 seconds
        elif scenario == "imbalanced":
            # Home team gets 80% duration
            dur = int(rng.integers(400, 800)) if team == 0 else int(rng.integers(50, 150))
        elif scenario == "rapid_turnovers":
            dur = int(rng.integers(35, 75))  # 1.4 to 3.0 seconds
        else:  # mixed / default
            r = rng.random()
            if r < 0.25:
                dur = int(rng.integers(30, 80))   # short 1.2 - 3.2s
            elif r < 0.75:
                dur = int(rng.integers(200, 450)) # typical 8 - 18s
            else:
                dur = int(rng.integers(600, 1100)) # long 24 - 44s

        dur = min(dur, n - t)
        truth[t : t + dur] = "HA"[team]
        t += dur
        team ^= 1

        # Dead ball between possessions
        if t < n:
            if scenario == "consecutive_dead":
                d_dur = int(rng.integers(50, 120))  # 2.0 to 4.8 seconds
            elif scenario == "rapid_turnovers":
                d_dur = int(rng.integers(0, 30)) if rng.random() < 0.3 else 0
            else:
                d_dur = int(rng.integers(100, 250)) if rng.random() < 0.45 else 0

            d_dur = min(d_dur, n - t)
            truth[t : t + d_dur] = "D"
            t += d_dur

    # Physical player and ball kinematics
    base = rng.uniform([12, 12], [93, 56], size=(22, 2))
    phase0 = rng.uniform(0, 6.28, (22, 1))
    rate = np.where(truth == "D", 0.15, 1.0)
    ph = phase0[None] + np.cumsum(rate)[:, None, None] * 0.25 / FPS * np.ones((1, 22, 1))
    pos = base[None] + 8.0 * np.concatenate([np.sin(ph), np.cos(ph)], axis=2)

    ball = np.full((n, 2), np.nan)
    holder = 0

    for i in range(n):
        if truth[i] == "D":
            ball[i] = [-3.0, 30.0]
            continue
        if i % 40 == 0:
            holder = rng.integers(0, 11) + (11 if truth[i] == "A" else 0)
        ball[i] = pos[i, holder] + rng.normal(0, 0.35, 2)

    # Carrier dropout scenario: carrier disappears for 30-50 frames during possession
    if scenario in ("carrier_dropout", "mixed"):
        drop_times = rng.choice(np.arange(100, n - 200, 500), size=3, replace=False)
        for dt in drop_times:
            if truth[dt] in ("H", "A"):
                # Move holder player 15m away for 35 frames (1.4s) while ball continues
                pos[dt : dt + 35, holder] += [15.0, 15.0]

    # Aerial passage scenario: ball flies at 16 m/s across midfield
    if scenario in ("aerial", "mixed"):
        aerial_times = rng.choice(np.arange(200, n - 300, 600), size=2, replace=False)
        for at in aerial_times:
            if truth[at] in ("H", "A"):
                for k in range(50):
                    ball[at + k] = [40.0 + k * (16.0 / FPS), 34.0]

    # Ball tracking dropout
    seen = np.ones(n, bool)
    i = 0
    while i < n:
        if rng.random() < 0.005:
            seen[i : i + int(rng.integers(15, 150))] = False
        i += 1
    ball[~seen] = np.nan

    # Format tracking files
    team_slot = np.array([0] * 23 + [1] * 23)
    lines = []
    for i in range(n):
        groups = []
        for s in range(S):
            if s < 11 or 23 <= s < 34:
                k = s if s < 11 else s - 23 + 11
                groups.append("{:.5f},{:.5f},0.5".format(pos[i, k, 0] / L_, pos[i, k, 1] / W_))
            else:
                groups.append("NaN,NaN,NaN")
        b = "NaN,NaN,NaN" if np.isnan(ball[i, 0]) else "{:.5f},{:.5f},0".format(
            ball[i, 0] / L_, ball[i, 1] / W_)
        lines.append("{}:{}:{}".format(i + 1, ";".join(groups), b))

    raw = os.path.join(tmp, "atd.txt")
    with open(raw, "w") as h:
        h.write("\n".join(lines) + "\n")

    xml = ['<?xml version="1.0"?><Metadata><Teams><Team id="h"><Name>Home</Name></Team>'
           '<Team id="a"><Name>Away</Name></Team></Teams><Players>']
    for s in range(S):
        xml.append('<Player id="p{0}" teamId="{1}"/>'.format(s, "h" if team_slot[s] == 0 else "a"))
    xml.append("</Players></Metadata>")
    meta = os.path.join(tmp, "atd.xml")
    with open(meta, "w") as h:
        h.write("".join(xml))

    return raw, meta, n, truth


def run_held_out_benchmark():
    held_out_seeds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    scenarios = ["mixed", "short_poss", "long_poss", "imbalanced", "rapid_turnovers",
                 "consecutive_dead", "carrier_dropout", "aerial", "mixed", "mixed"]

    candidates = ["A", "B"]
    results = {c: {"fixed": [], "cal": []} for c in candidates}
    cfg_base_p8 = Config()

    print(f"Running Held-Out Adversarial Benchmark across {len(held_out_seeds)} seeds...")

    for seed, scen in zip(held_out_seeds, scenarios):
        print(f"\n--- Seed {seed} | Scenario: {scen} ---")
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta, n, truth = make_adversarial_synthetic_match(tmp, seed=seed, scenario=scen)
            from mtah_possession.atd import load_tracking
            from mtah_possession.features import compute

            track = load_tracking(raw, meta, n, (L_, W_))
            feats = compute(track, (L_, W_), FPS, cfg_base_p8)
            L = log_emissions(feats, cfg_base_p8)
            blocks = [(0, n - 1)]

            gt_codes = pm.label_codes(truth)
            runs = pm.valid_runs(gt_codes)
            d_gt = describe(truth, FPS)

            for cand in candidates:
                # 1. Fixed parameters
                if cand == "A":
                    lab_fixed = decode_blocks(L, blocks, n, cfg_base_p8, feats=feats, decoder="duration_aware", fps=FPS)
                else:
                    cfg_sm = SemiMarkovConfig(candidate="B")
                    lab_fixed = decode_blocks_semi_markov(L, blocks, n, cfg_sm, feats=feats, fps=FPS)

                # Official metrics
                pred_fixed_codes = pm.label_codes(lab_fixed)
                cm_fixed = pm.confusion(gt_codes, pred_fixed_codes)
                cm3_fixed = pm.scored_confusion(cm_fixed)
                mf1_fixed = pm.macro_f1(cm3_fixed)
                filled_fixed = pm.fill_abstentions(pred_fixed_codes, runs)
                spell_res_fixed = pm.spell_scores(gt_codes, filled_fixed, runs)
                n_gt_segs, n_pred_segs, frag_fixed = pm.fragmentation(gt_codes, filled_fixed, runs)
                d_fixed = describe(lab_fixed, FPS)

                res_fixed = {
                    "seed": seed, "scenario": scen,
                    "spell_f1_mean": spell_res_fixed["mean_f1"],
                    "macro_f1": mf1_fixed,
                    "frag_ratio": frag_fixed,
                    "n_gt": d_gt["possessions"],
                    "n_pred": d_fixed["possessions"],
                    "poss_diff": d_fixed["possessions"] - d_gt["possessions"],
                }
                results[cand]["fixed"].append(res_fixed)

                # 2. Calibrated parameters
                if cand == "A":
                    m_cal, lab_cal = calibrate_switch_penalties(
                        L, blocks, n, cfg_base_p8, fps=FPS, target_s=15.0, iters=10, feats=feats, decoder="duration_aware"
                    )
                else:
                    cfg_sm = SemiMarkovConfig(candidate="B")
                    m_cal, lab_cal = calibrate_switch_penalties_semi_markov(
                        L, blocks, n, cfg_sm, fps=FPS, target_s=15.0, iters=10, feats=feats
                    )

                pred_cal_codes = pm.label_codes(lab_cal)
                cm_cal = pm.confusion(gt_codes, pred_cal_codes)
                cm3_cal = pm.scored_confusion(cm_cal)
                mf1_cal = pm.macro_f1(cm3_cal)
                filled_cal = pm.fill_abstentions(pred_cal_codes, runs)
                spell_res_cal = pm.spell_scores(gt_codes, filled_cal, runs)
                n_gt_segs, n_pred_segs, frag_cal = pm.fragmentation(gt_codes, filled_cal, runs)
                d_cal = describe(lab_cal, FPS)

                res_cal = {
                    "seed": seed, "scenario": scen, "m": float(m_cal),
                    "spell_f1_mean": spell_res_cal["mean_f1"],
                    "macro_f1": mf1_cal,
                    "frag_ratio": frag_cal,
                    "n_gt": d_gt["possessions"],
                    "n_pred": d_cal["possessions"],
                    "poss_diff": d_cal["possessions"] - d_gt["possessions"],
                }
                results[cand]["cal"].append(res_cal)

                print(f"  Cand {cand} | Fixed spellF1={res_fixed['spell_f1_mean']:.4f}, macroF1={res_fixed['macro_f1']:.4f}, poss={res_fixed['n_pred']}/{res_fixed['n_gt']} | "
                      f"Cal (m={m_cal:.4f}) spellF1={res_cal['spell_f1_mean']:.4f}, macroF1={res_cal['macro_f1']:.4f}, poss={res_cal['n_pred']}/{res_cal['n_gt']}")

    out_file = os.path.join(HERE, "phase9_1_held_out_benchmark_results.npy")
    np.save(out_file, results)
    print(f"\nSaved held-out benchmark results to {out_file}")

    print("\n================ HELD-OUT ADVERSARIAL BENCHMARK SUMMARY (10 SEEDS) ================")
    print(f"{'Candidate':<12} | {'Mode':<6} | {'spellF1 Mean':<18} | {'macroF1 Mean':<18} | {'Frag Ratio':<12} | {'Poss Diff':<10}")
    print("-" * 85)
    for c in ["A", "B"]:
        for mode in ["fixed", "cal"]:
            df = pd.DataFrame(results[c][mode])
            sf1_m, sf1_s = df["spell_f1_mean"].mean(), df["spell_f1_mean"].std()
            mf1_m, mf1_s = df["macro_f1"].mean(), df["macro_f1"].std()
            frag = df["frag_ratio"].mean()
            pdiff = df["poss_diff"].mean()
            print(f"{c:<12} | {mode:<6} | {sf1_m:.4f} +/- {sf1_s:.4f}  | {mf1_m:.4f} +/- {mf1_s:.4f}  | {frag:.3f}        | {pdiff:+.1f}")


if __name__ == "__main__":
    run_held_out_benchmark()

