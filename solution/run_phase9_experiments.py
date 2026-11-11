"""Phase 9 Comprehensive Experimental Runner.

Runs and evaluates Candidates A through E on real match (BAR-ATH) and synthetic benchmarks:
- Candidate A: Phase 8 V2 baseline (first-order Viterbi + boundary refinement)
- Candidate B: Semi-Markov with simple duration prior
- Candidate C: Semi-Markov with flexible Gamma distribution
- Candidate D: Semi-Markov with state-specific duration distributions and transition costs
- Candidate E: Segment DP ablation without duration priors (transition costs only)

Evaluates:
- Real-match spell distributions, quantiles, and extreme lengths (<3s, >30s, 1-frame blips)
- Boundary displacements and label agreements relative to frozen Phase 8 V2
- Parameter sensitivity (+/-10%, +/-20%)
- Half 1 vs Half 2 symmetry and independence
- Computational runtime and memory
- Synthetic benchmark accuracy and spell metrics
"""
import os
import sys
import time
import math
from dataclasses import replace
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from mtah_possession.model import Config, log_emissions, decode_blocks
from mtah_possession.pipeline import describe, calibrate_switch_penalties
from mtah_possession.semi_markov import (
    SemiMarkovConfig,
    decode_blocks_semi_markov,
    calibrate_switch_penalties_semi_markov,
)


def load_cached_data():
    cache_path = os.path.join(HERE, "bar_ath_cache.npz")
    data = np.load(cache_path)
    L = data["L"]
    dH = data["dH"]
    dA = data["dA"]
    ball_speed = data["ball_speed"]
    ball_confidence = data["ball_confidence"]
    carrier_cH = data["carrier_cH"]
    carrier_cA = data["carrier_cA"]
    n = int(data["n"][0])
    fps = int(data["fps"][0])
    blocks = [tuple(b) for b in data["blocks"]]
    feats = {
        "dH": dH,
        "dA": dA,
        "ball_speed": ball_speed,
        "ball_confidence": ball_confidence,
        "carrier_cH": carrier_cH,
        "carrier_cA": carrier_cA,
    }
    return L, feats, blocks, n, fps


def compute_spell_stats(labels, fps=25):
    """Detailed spell duration statistics for H, A, and D."""
    runs = {"H": [], "A": [], "D": []}
    prev = None
    start = 0
    transitions = []
    trans_pairs = []

    for i, l in enumerate(list(labels) + ["#"]):
        if l != prev:
            if prev in runs:
                dur_frames = i - start
                runs[prev].append(dur_frames / fps)
            if prev is not None and l in ("H", "A", "D") and prev in ("H", "A", "D"):
                transitions.append(i)
                trans_pairs.append((prev, l))
            prev, start = l, i

    all_poss = runs["H"] + runs["A"]
    res = {}

    def _dist_summary(arr, prefix):
        a = np.array(arr)
        if len(a) == 0:
            return {
                f"{prefix}_count": 0, f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0, f"{prefix}_p10": 0.0, f"{prefix}_p25": 0.0,
                f"{prefix}_p50": 0.0, f"{prefix}_p75": 0.0, f"{prefix}_p90": 0.0,
                f"{prefix}_max": 0.0, f"{prefix}_sub1s": 0, f"{prefix}_sub3s": 0,
                f"{prefix}_sub5s": 0, f"{prefix}_over30s": 0, f"{prefix}_over45s": 0,
                f"{prefix}_one_frame": 0,
            }
        return {
            f"{prefix}_count": len(a),
            f"{prefix}_mean": float(np.mean(a)),
            f"{prefix}_std": float(np.std(a)),
            f"{prefix}_min": float(np.min(a)),
            f"{prefix}_p10": float(np.percentile(a, 10)),
            f"{prefix}_p25": float(np.percentile(a, 25)),
            f"{prefix}_p50": float(np.percentile(a, 50)),
            f"{prefix}_p75": float(np.percentile(a, 75)),
            f"{prefix}_p90": float(np.percentile(a, 90)),
            f"{prefix}_max": float(np.max(a)),
            f"{prefix}_sub1s": int((a < 1.0).sum()),
            f"{prefix}_sub3s": int((a < 3.0).sum()),
            f"{prefix}_sub5s": int((a < 5.0).sum()),
            f"{prefix}_over30s": int((a > 30.0).sum()),
            f"{prefix}_over45s": int((a > 45.0).sum()),
            f"{prefix}_one_frame": int((a <= 1.0 / fps + 1e-5).sum()),
        }

    res.update(_dist_summary(all_poss, "poss"))
    res.update(_dist_summary(runs["H"], "H"))
    res.update(_dist_summary(runs["A"], "A"))
    res.update(_dist_summary(runs["D"], "D"))

    # Transitions
    trans_df = pd.Series(trans_pairs).value_counts().to_dict() if len(trans_pairs) > 0 else {}
    res["trans_total"] = len(trans_pairs)
    res["trans_H_to_A"] = trans_df.get(("H", "A"), 0)
    res["trans_A_to_H"] = trans_df.get(("A", "H"), 0)
    res["trans_H_to_D"] = trans_df.get(("H", "D"), 0)
    res["trans_A_to_D"] = trans_df.get(("A", "D"), 0)
    res["trans_D_to_H"] = trans_df.get(("D", "H"), 0)
    res["trans_D_to_A"] = trans_df.get(("D", "A"), 0)

    # Frame totals
    res["frames_H"] = int((labels == "H").sum())
    res["frames_A"] = int((labels == "A").sum())
    res["frames_D"] = int((labels == "D").sum())
    res["frames_X"] = int((labels == "X").sum())

    return res


def compute_displacement_and_agreement(ref_labels, test_labels, fps=25):
    """Compute frame agreement, interval differences, and boundary displacements relative to ref_labels."""
    n = len(ref_labels)
    total_agree = float((ref_labels == test_labels).mean()) * 100.0

    active_mask = (ref_labels != "X")
    active_agree = float((ref_labels[active_mask] == test_labels[active_mask]).mean()) * 100.0

    # Differing intervals
    diff = (ref_labels != test_labels)
    diff_d = np.diff(np.concatenate(([0], diff.astype(np.int8), [0])))
    starts = np.flatnonzero(diff_d == 1)
    ends = np.flatnonzero(diff_d == -1)
    intervals_count = len(starts)

    # Transition displacements
    ref_trans = np.flatnonzero((ref_labels[1:] != ref_labels[:-1]) & (ref_labels[:-1] != "X") & (ref_labels[1:] != "X")) + 1
    test_trans = np.flatnonzero((test_labels[1:] != test_labels[:-1]) & (test_labels[:-1] != "X") & (test_labels[1:] != "X")) + 1

    displacements = []
    if len(ref_trans) > 0 and len(test_trans) > 0:
        for rt in ref_trans:
            dists = np.abs(test_trans - rt)
            min_d = np.min(dists)
            if min_d <= 5.0 * fps:  # within 5 seconds match
                displacements.append(min_d / fps)

    disp_arr = np.array(displacements) if len(displacements) > 0 else np.array([0.0])

    return {
        "agree_total_pct": total_agree,
        "agree_active_pct": active_agree,
        "diff_frames": int(diff.sum()),
        "diff_intervals": intervals_count,
        "disp_mean_s": float(np.mean(disp_arr)),
        "disp_median_s": float(np.median(disp_arr)),
        "disp_p90_s": float(np.percentile(disp_arr, 90)),
        "disp_max_s": float(np.max(disp_arr)),
    }


def run_all():
    print("Loading data...")
    L, feats, blocks, n, fps = load_cached_data()

    # Verify Frozen Reference
    ref_path = os.path.join(HERE, "submission_phase8_v2.csv")
    ref_df = pd.read_csv(ref_path)
    ref_labels = ref_df["label"].values
    ref_stats = compute_spell_stats(ref_labels, fps)
    print(f"Verified Reference Phase 8 V2: {ref_stats['poss_count']} possessions, mean {ref_stats['poss_mean']:.2f}s")
    assert ref_stats["poss_count"] == 259, f"Reference possession count mismatch: {ref_stats['poss_count']}"

    candidates = ["A", "B", "C", "D", "E"]
    results = {}

    cfg_base_p8 = Config()

    for cand in candidates:
        print(f"\n==========================================")
        print(f"Running Candidate {cand}")
        print(f"==========================================")

        # 1. Fixed parameters (uncalibrated m = 1.0)
        t0 = time.perf_counter()
        if cand == "A":
            lab_fixed = decode_blocks(L, blocks, n, cfg_base_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand)
            lab_fixed = decode_blocks_semi_markov(L, blocks, n, cfg_sm, feats=feats, fps=fps)
        t_fixed = time.perf_counter() - t0

        stats_fixed = compute_spell_stats(lab_fixed, fps)
        disp_fixed = compute_displacement_and_agreement(ref_labels, lab_fixed, fps)
        stats_fixed.update(disp_fixed)
        stats_fixed["runtime_s"] = t_fixed
        stats_fixed["multiplier"] = 1.0
        results[f"{cand}_fixed"] = stats_fixed

        print(f"  Fixed: possessions={stats_fixed['poss_count']}, mean={stats_fixed['poss_mean']:.2f}s, "
              f"median={stats_fixed['poss_p50']:.2f}s, 1-frame={stats_fixed['poss_one_frame']}, "
              f"sub3s={stats_fixed['poss_sub3s']}, agree={stats_fixed['agree_total_pct']:.2f}%, "
              f"runtime={t_fixed:.2f}s")

        # 2. Calibrated parameters (m tuned to 15.0s target)
        t0 = time.perf_counter()
        if cand == "A":
            m_cal, lab_cal = calibrate_switch_penalties(
                L, blocks, n, cfg_base_p8, fps=fps, target_s=15.0, iters=12, feats=feats, decoder="duration_aware"
            )
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand)
            m_cal, lab_cal = calibrate_switch_penalties_semi_markov(
                L, blocks, n, cfg_sm, fps=fps, target_s=15.0, iters=12, feats=feats
            )
        t_cal = time.perf_counter() - t0

        stats_cal = compute_spell_stats(lab_cal, fps)
        disp_cal = compute_displacement_and_agreement(ref_labels, lab_cal, fps)
        stats_cal.update(disp_cal)
        stats_cal["runtime_s"] = t_cal
        stats_cal["multiplier"] = float(m_cal)
        results[f"{cand}_cal"] = stats_cal

        print(f"  Calibrated (m={m_cal:.6f}): possessions={stats_cal['poss_count']}, mean={stats_cal['poss_mean']:.2f}s, "
              f"median={stats_cal['poss_p50']:.2f}s, 1-frame={stats_cal['poss_one_frame']}, "
              f"sub3s={stats_cal['poss_sub3s']}, agree={stats_cal['agree_total_pct']:.2f}%, "
              f"runtime={t_cal:.2f}s")

    # 3. Parameter Sensitivity Experiments (+/-10%, +/-20%)
    print("\nRunning Parameter Sensitivity Experiments on Candidate D and Candidate A...")
    sens_results = {}
    perturbations = [-0.20, -0.10, 0.0, 0.10, 0.20]

    for p in perturbations:
        mult = 1.0 + p
        # Candidate A
        cfg_pert_a = replace(cfg_base_p8, switch_team=cfg_base_p8.switch_team * mult, switch_dead=cfg_base_p8.switch_dead * mult)
        lab_pert_a = decode_blocks(L, blocks, n, cfg_pert_a, feats=feats, decoder="duration_aware", fps=fps)
        st_a = compute_spell_stats(lab_pert_a, fps)
        disp_a = compute_displacement_and_agreement(ref_labels, lab_pert_a, fps)
        st_a.update(disp_a)
        sens_results[f"A_pert_{int(p*100):+d}"] = st_a

        # Candidate D
        cfg_pert_d = SemiMarkovConfig(
            candidate="D",
            switch_team=6.0 * mult,
            switch_dead=5.0 * mult,
        )
        lab_pert_d = decode_blocks_semi_markov(L, blocks, n, cfg_pert_d, feats=feats, fps=fps)
        st_d = compute_spell_stats(lab_pert_d, fps)
        disp_d = compute_displacement_and_agreement(ref_labels, lab_pert_d, fps)
        st_d.update(disp_d)
        sens_results[f"D_pert_{int(p*100):+d}"] = st_d

        print(f"  Perturbation {int(p*100):+d}%: A poss={st_a['poss_count']}, mean={st_a['poss_mean']:.2f}s, agree={st_a['agree_total_pct']:.2f}% | "
              f"D poss={st_d['poss_count']}, mean={st_d['poss_mean']:.2f}s, agree={st_d['agree_total_pct']:.2f}%")

    # 4. Half Symmetry and Independence
    print("\nHalf Symmetry Analysis...")
    half_results = {}
    for cand in ["A", "D"]:
        for h_idx, (first, last) in enumerate(blocks):
            sub_L = L[first : last + 1]
            sub_feats = {k: v[first : last + 1] for k, v in feats.items()}
            sub_ref = ref_labels[first : last + 1]
            sub_n = len(sub_L)
            if cand == "A":
                cfg = replace(cfg_base_p8, switch_team=cfg_base_p8.switch_team * 0.509659, switch_dead=cfg_base_p8.switch_dead * 0.509659)
                sub_lab = decode_blocks(sub_L, [(0, sub_n - 1)], sub_n, cfg, feats=sub_feats, decoder="duration_aware", fps=fps)
            else:
                cfg = SemiMarkovConfig(candidate="D", switch_team=6.0 * results["D_cal"]["multiplier"], switch_dead=5.0 * results["D_cal"]["multiplier"])
                sub_lab = decode_blocks_semi_markov(sub_L, [(0, sub_n - 1)], sub_n, cfg, feats=sub_feats, fps=fps)

            st = compute_spell_stats(sub_lab, fps)
            disp = compute_displacement_and_agreement(sub_ref, sub_lab, fps)
            st.update(disp)
            half_results[f"{cand}_half_{h_idx + 1}"] = st
            print(f"  {cand} Half {h_idx + 1}: poss={st['poss_count']}, mean={st['poss_mean']:.2f}s, "
                  f"H_count={st['H_count']}, A_count={st['A_count']}, D_count={st['D_count']}, agree={st['agree_total_pct']:.2f}%")

    # Save all results to disk
    out_file = os.path.join(HERE, "phase9_experiment_results.npy")
    all_data = {
        "candidates": results,
        "sensitivity": sens_results,
        "half_symmetry": half_results,
    }
    np.save(out_file, all_data)
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    run_all()

