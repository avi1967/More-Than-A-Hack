"""Adversarial Synthetic Test Suite for Phase 9 Semi-Markov Possession Decoding.

Evaluates 10 adversarial challenge cases:
1. Genuine 1-3 second possession.
2. Genuine rapid turnover.
3. Short opponent marking spike.
4. Carrier dropout during possession.
5. Long aerial pass.
6. Genuine dead-ball interruption.
7. Two short possessions separated by D.
8. Long uninterrupted possession.
9. Restart followed by immediate turnover.
10. Dense contested-ball situation.
"""
import os
import sys
import pytest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mtah_possession.model import Config, log_emissions, decode_blocks
from mtah_possession.semi_markov import (
    SemiMarkovConfig,
    viterbi_semi_markov,
    decode_blocks_semi_markov,
)
from mtah_possession.pipeline import describe


def _make_dummy_feats(n, dH_val=5.0, dA_val=5.0, ball_spd=2.0, p_dead_val=0.05):
    """Helper to construct dummy tracking features."""
    return {
        "dH": np.full(n, dH_val, dtype=np.float64),
        "dA": np.full(n, dA_val, dtype=np.float64),
        "ball_speed": np.full(n, ball_spd, dtype=np.float64),
        "ball_ok": np.ones(n, dtype=bool),
        "oob": np.zeros(n, dtype=bool),
        "still": np.zeros(n, dtype=bool),
        "activity": np.full(n, 1.8, dtype=np.float64),
        "p_dead": np.full(n, p_dead_val, dtype=np.float64),
        "ball_confidence": np.ones(n, dtype=np.float32),
        "valid_players": np.full(n, 22, dtype=np.int32),
        "carrier_cH": np.zeros(n, dtype=np.float64),
        "carrier_cA": np.zeros(n, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Case 1: Genuine 1-3 second possession
# ---------------------------------------------------------------------------
def test_case1_genuine_short_possession():
    """Verify that a genuine 2.0-second possession is recognized and not erased."""
    fps = 25
    n = 200  # 8.0 seconds total
    # Team H possesses for 50 frames (2.0s), preceded and followed by dead ball
    feats = _make_dummy_feats(n, dH_val=10.0, dA_val=10.0, p_dead_val=0.85)

    # Frame 50..100: Team H cleanly controls ball (0.5m), ball is live
    feats["dH"][50:100] = 0.5
    feats["p_dead"][50:100] = 0.05
    feats["carrier_cH"][50:100] = 0.9

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        h_count = (lab == "H").sum()
        assert h_count >= 35, f"Candidate {cand} erased genuine 2.0s possession (only {h_count} frames H)"
        d = describe(lab, fps)
        assert d["possessions"] == 1, f"Candidate {cand} expected 1 possession, got {d['possessions']}"


# ---------------------------------------------------------------------------
# Case 2: Genuine rapid turnover
# ---------------------------------------------------------------------------
def test_case2_genuine_rapid_turnover():
    """Verify that a clean turnover from H to A is detected promptly without excessive delay."""
    fps = 25
    n = 200  # 8.0 seconds
    feats = _make_dummy_feats(n, dH_val=0.5, dA_val=4.0, p_dead_val=0.05)
    feats["carrier_cH"][:100] = 0.9

    # At frame 100, Away player tackles and takes possession
    feats["dH"][100:] = 4.0
    feats["dA"][100:] = 0.4
    feats["carrier_cH"][100:] = 0.0
    feats["carrier_cA"][100:] = 0.9

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D", "E"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        # Turnover should happen close to frame 100 (+/- 10 frames = 0.4s)
        transitions = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        assert len(transitions) == 1, f"Candidate {cand} had {len(transitions)} transitions, expected 1"
        tau = transitions[0]
        assert abs(tau - 100) <= 12, f"Candidate {cand} turnover lag too large: tau={tau} vs expected 100"
        assert lab[50] == "H" and lab[150] == "A"


# ---------------------------------------------------------------------------
# Case 3: Short opponent marking spike
# ---------------------------------------------------------------------------
def test_case3_opponent_marking_spike():
    """Verify that a brief 0.3-second (7 frames) proximity spike by an opponent does NOT trigger a false turnover."""
    fps = 25
    n = 200
    feats = _make_dummy_feats(n, dH_val=0.5, dA_val=4.0, p_dead_val=0.05)
    feats["carrier_cH"] = np.full(n, 0.9)

    # Away player runs past ball carrier for 7 frames (0.28s) getting to 0.7m
    feats["dA"][90:97] = 0.7

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D", "E"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        assert (lab == "H").all(), f"Candidate {cand} falsely switched possession during momentary opponent run-past"


# ---------------------------------------------------------------------------
# Case 4: Carrier dropout during possession
# ---------------------------------------------------------------------------
def test_case4_carrier_dropout():
    """Verify that carrier continuity prevents possession loss when carrier temporarily vanishes from tracking."""
    fps = 25
    n = 200
    feats = _make_dummy_feats(n, dH_val=0.5, dA_val=3.0, p_dead_val=0.05)
    # Frames 80..120 (1.6s): Home carrier is lost (dH becomes large), but continuity cH remains active
    feats["dH"][80:120] = 5.0
    feats["carrier_cH"][:80] = 1.0
    for t in range(80, 120):
        feats["carrier_cH"][t] = np.exp(-(t - 80) / (2.0 * fps))  # decaying continuity

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        assert (lab == "H").all(), f"Candidate {cand} lost possession during carrier dropout"


# ---------------------------------------------------------------------------
# Case 5: Long aerial pass
# ---------------------------------------------------------------------------
def test_case5_long_aerial_pass():
    """Verify that a 2.5-second inbounds aerial pass keeps the kicking team's possession."""
    fps = 25
    n = 200
    # Frames 0..60: Team H carries ball
    # Frames 60..120 (2.4s): ball in air at 15 m/s, dH=dA=8.0m, p_dead is low (live flight)
    # Frames 120..200: Team H teammate receives ball (dH=0.5m)
    feats = _make_dummy_feats(n, dH_val=8.0, dA_val=8.0, ball_spd=15.0, p_dead_val=0.02)
    feats["dH"][:60] = 0.5
    feats["dH"][120:] = 0.5
    feats["carrier_cH"][:60] = 0.9
    feats["carrier_cH"][120:] = 0.9

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        # Possession should stay H throughout; no spurious D or A
        assert (lab == "H").all(), f"Candidate {cand} fragmented long aerial pass into {(lab != 'H').sum()} non-H frames"


# ---------------------------------------------------------------------------
# Case 6: Genuine dead-ball interruption
# ---------------------------------------------------------------------------
def test_case6_genuine_dead_ball():
    """Verify that a genuine 6-second stoppage is correctly segmented with D."""
    fps = 25
    n = 300  # 12.0 seconds
    feats = _make_dummy_feats(n, dH_val=0.5, dA_val=4.0, p_dead_val=0.03)

    # Frames 100..250 (6.0s): stoppage, high p_dead, ball still, players unpossessed
    feats["dH"][100:250] = 12.0
    feats["dA"][100:250] = 12.0
    feats["p_dead"][100:250] = 0.92
    feats["ball_speed"][100:250] = 0.05
    feats["still"][100:250] = True

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        d_count = (lab[100:250] == "D").sum()
        assert d_count >= 130, f"Candidate {cand} failed to recognize dead ball: only {d_count}/150 frames D"
        assert lab[50] == "H" and lab[280] == "H"


# ---------------------------------------------------------------------------
# Case 7: Two short possessions separated by D
# ---------------------------------------------------------------------------
def test_case7_two_short_possessions_separated_by_dead():
    """Verify that a stoppage between two short possessions is preserved and not bridged."""
    fps = 25
    n = 250  # 10.0 seconds
    # Frame 0..50 (2.0s): Team H
    # Frame 50..150 (4.0s): Dead ball
    # Frame 150..200 (2.0s): Team A
    # Frame 200..250 (2.0s): Dead ball
    feats = _make_dummy_feats(n, dH_val=10.0, dA_val=10.0, p_dead_val=0.88)
    feats["dH"][:50] = 0.5
    feats["p_dead"][:50] = 0.05

    feats["dA"][150:200] = 0.5
    feats["p_dead"][150:200] = 0.05

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        # Verify that D is present in the middle interval [60..140]
        assert (lab[60:140] == "D").all(), f"Candidate {cand} failed to preserve dead ball separating short spells"
        assert (lab[:40] == "H").all()
        assert (lab[160:190] == "A").all()


# ---------------------------------------------------------------------------
# Case 8: Long uninterrupted possession
# ---------------------------------------------------------------------------
def test_case8_long_uninterrupted_possession():
    """Verify that a 40-second continuous possession is maintained as 1 single spell without fragmentation."""
    fps = 25
    n = 1000  # 40.0 seconds
    feats = _make_dummy_feats(n, dH_val=0.6, dA_val=4.5, p_dead_val=0.03)
    feats["carrier_cH"] = np.full(n, 0.95)

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D", "E"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        assert (lab == "H").all(), f"Candidate {cand} fragmented 40s possession into multiple spells"
        d = describe(lab, fps)
        assert d["possessions"] == 1


# ---------------------------------------------------------------------------
# Case 9: Restart followed by immediate turnover
# ---------------------------------------------------------------------------
def test_case9_restart_immediate_turnover():
    """Verify restart by H immediately intercepted by A after 0.6 seconds (15 frames)."""
    fps = 25
    n = 200
    feats = _make_dummy_feats(n, dH_val=10.0, dA_val=10.0, p_dead_val=0.88)
    # Restart at frame 50 by Team H
    feats["dH"][50:65] = 0.5
    feats["p_dead"][50:] = 0.04
    feats["carrier_cH"][50:65] = 0.8
    # Immediate interception at frame 65 by Team A
    feats["dH"][65:150] = 4.0
    feats["dA"][65:150] = 0.4
    feats["carrier_cA"][65:150] = 0.9

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        # Team A must take possession around frame 65..75 and hold it
        assert (lab[75:140] == "A").all(), f"Candidate {cand} failed to award possession to A after interception"


# ---------------------------------------------------------------------------
# Case 10: Dense contested-ball situation
# ---------------------------------------------------------------------------
def test_case10_dense_contested_ball():
    """Verify stability in a 3-second scramble (dH ~ dA ~ 0.8m) without rapid oscillation."""
    fps = 25
    n = 200
    feats = _make_dummy_feats(n, dH_val=0.5, dA_val=3.0, p_dead_val=0.04)
    feats["carrier_cH"][:50] = 0.9

    # Scramble from frame 50 to 125 (3.0s): dH and dA oscillate slightly around 0.8m
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, 75)
    feats["dH"][50:125] = 0.8 + noise
    feats["dA"][50:125] = 0.8 - noise
    feats["carrier_cH"][50:125] = 0.3
    feats["carrier_cA"][50:125] = 0.3

    # Team H emerges with the ball at frame 125
    feats["dH"][125:] = 0.5
    feats["dA"][125:] = 3.5
    feats["carrier_cH"][125:] = 0.9

    cfg_p8 = Config()
    L = log_emissions(feats, cfg_p8)

    for cand in ["A", "B", "C", "D"]:
        if cand == "A":
            lab = decode_blocks(L, [(0, n - 1)], n, cfg_p8, feats=feats, decoder="duration_aware", fps=fps)
        else:
            cfg_sm = SemiMarkovConfig(candidate=cand, switch_team=3.0, switch_dead=2.5)
            lab = decode_blocks_semi_markov(L, [(0, n - 1)], n, cfg_sm, feats=feats, fps=fps)

        transitions = np.flatnonzero(lab[1:] != lab[:-1])
        # There should be no rapid chatter / ping-pong (at most 2 transitions across whole sequence)
        assert len(transitions) <= 2, f"Candidate {cand} chattered in contested scramble: {len(transitions)} switches"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
