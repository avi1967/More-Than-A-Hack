"""Phase 9 Synthetic Ground-Truth Benchmark.

Evaluates Candidates A through E on synthetic matches with known ground truth across multiple seeds,
computing the official competition metrics from evaluation/possession_metrics.py:
- spell_f1_mean (the primary ranking metric across IoU thresholds [0.3, 0.4, 0.5, 0.6, 0.7])
- macro_f1 (secondary per-frame metric over {H, A, D})
- alive_dead_f1
- team_accuracy
- fragmentation ratio
- possession count error
- mean duration error
"""
import os
import sys
import tempfile
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from tests.test_synthetic import make, L_, W_, FPS
from mtah_possession.model import Config, log_emissions, decode_blocks
from mtah_possession.pipeline import describe, calibrate_switch_penalties
from mtah_possession.semi_markov import (
    SemiMarkovConfig,
    decode_blocks_semi_markov,
    calibrate_switch_penalties_semi_markov,
)
from evaluation import possession_metrics as pm


def evaluate_synthetic_run(gt_labels, pred_labels, fps=25):
    """Compute official metrics against ground truth."""
    gt_codes = pm.label_codes(gt_labels)
    pred_codes = pm.label_codes(pred_labels)
    runs = pm.valid_runs(gt_codes)

    # Frame-level metrics
    cm = pm.confusion(gt_codes, pred_codes)
    cm3 = pm.scored_confusion(cm)
    mf1 = pm.macro_f1(cm3)
    acc = pm.accuracy(cm3)
    adf1 = pm.alive_dead_f1(cm3)
    strict_team_acc, live_team_acc = pm.team_accuracy(cm3)

    # Spell-level ranking metrics
    filled_pred = pm.fill_abstentions(pred_codes, runs)
    spell_res = pm.spell_scores(gt_codes, filled_pred, runs)
    spell_f1_mean = spell_res["mean_f1"]
    per_iou = spell_res["per_iou"]

    # Fragmentation and stats
    n_gt_segs, n_pred_segs, frag_ratio = pm.fragmentation(gt_codes, filled_pred, runs)
    d_pred = describe(pred_labels, fps)
    d_gt = describe(gt_labels, fps)

    return {
        "spell_f1_mean": spell_f1_mean,
        "spell_f1_03": per_iou[0.3]["f1"],
        "spell_f1_05": per_iou[0.5]["f1"],
        "spell_f1_07": per_iou[0.7]["f1"],
        "macro_f1": mf1,
        "frame_acc": acc,
        "alive_dead_f1": adf1,
        "team_acc": live_team_acc,
        "frag_ratio": frag_ratio,
        "n_gt_poss": d_gt["possessions"],
        "n_pred_poss": d_pred["possessions"],
        "poss_diff": d_pred["possessions"] - d_gt["possessions"],
        "mean_dur_gt": d_gt["mean_s"],
        "mean_dur_pred": d_pred["mean_s"],
        "dur_err": abs(d_pred["mean_s"] - d_gt["mean_s"]),
    }


def run_synthetic_benchmark(seeds=(0, 1, 2, 3, 4)):
    candidates = ["A", "B", "C", "D", "E"]
    results = {cand: {"fixed": [], "cal": []} for cand in candidates}

    cfg_base_p8 = Config()

    print(f"Starting Synthetic Benchmark across seeds: {seeds}...")

    for seed in seeds:
        print(f"\n--- Synthetic Seed {seed} ---")
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta, n, truth = make(tmp, seed=seed)
            # Load tracking and features
            from mtah_possession.atd import load_tracking
            from mtah_possession.features import compute
            track = load_tracking(raw, meta, n, (L_, W_))
            feats = compute(track, (L_, W_), FPS, cfg_base_p8)
            L = log_emissions(feats, cfg_base_p8)
            blocks = [(0, n - 1)]

            for cand in candidates:
                # 1. Fixed parameters
                if cand == "A":
                    lab_fixed = decode_blocks(L, blocks, n, cfg_base_p8, feats=feats, decoder="duration_aware", fps=FPS)
                else:
                    cfg_sm = SemiMarkovConfig(candidate=cand)
                    lab_fixed = decode_blocks_semi_markov(L, blocks, n, cfg_sm, feats=feats, fps=FPS)

                res_fixed = evaluate_synthetic_run(truth, lab_fixed, fps=FPS)
                results[cand]["fixed"].append(res_fixed)

                # 2. Calibrated parameters (target 15.0s)
                if cand == "A":
                    m_cal, lab_cal = calibrate_switch_penalties(
                        L, blocks, n, cfg_base_p8, fps=FPS, target_s=15.0, iters=10, feats=feats, decoder="duration_aware"
                    )
                else:
                    cfg_sm = SemiMarkovConfig(candidate=cand)
                    m_cal, lab_cal = calibrate_switch_penalties_semi_markov(
                        L, blocks, n, cfg_sm, fps=FPS, target_s=15.0, iters=10, feats=feats
                    )

                res_cal = evaluate_synthetic_run(truth, lab_cal, fps=FPS)
                res_cal["multiplier"] = float(m_cal)
                results[cand]["cal"].append(res_cal)

                print(f"  Seed {seed} | Cand {cand}: "
                      f"Fixed spellF1={res_fixed['spell_f1_mean']:.4f}, macroF1={res_fixed['macro_f1']:.4f}, poss={res_fixed['n_pred_poss']}/{res_fixed['n_gt_poss']} | "
                      f"Cal spellF1={res_cal['spell_f1_mean']:.4f}, macroF1={res_cal['macro_f1']:.4f}, poss={res_cal['n_pred_poss']}/{res_cal['n_gt_poss']}")

    # Aggregate statistics
    summary = {}
    for cand in candidates:
        summary[cand] = {}
        for mode in ["fixed", "cal"]:
            mode_data = results[cand][mode]
            df = pd.DataFrame(mode_data)
            agg = {}
            for col in df.columns:
                agg[f"{col}_mean"] = float(df[col].mean())
                agg[f"{col}_std"] = float(df[col].std())
            summary[cand][mode] = agg

    out_file = os.path.join(HERE, "phase9_synthetic_benchmark_results.npy")
    np.save(out_file, {"raw": results, "summary": summary})
    print(f"\nSaved synthetic benchmark results to {out_file}")

    print("\n================ SYNTHETIC BENCHMARK SUMMARY ================")
    print(f"{'Cand':<6} | {'Mode':<6} | {'spellF1 Mean':<14} | {'macroF1 Mean':<14} | {'Frag Ratio':<12} | {'Poss Err':<10}")
    print("-" * 75)
    for cand in candidates:
        for mode in ["fixed", "cal"]:
            s = summary[cand][mode]
            print(f"{cand:<6} | {mode:<6} | {s['spell_f1_mean_mean']:.4f} +/- {s['spell_f1_mean_std']:.4f} | "
                  f"{s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f} | "
                  f"{s['frag_ratio_mean']:.3f}        | {s['poss_diff_mean']:+.1f}")


if __name__ == "__main__":
    run_synthetic_benchmark()

