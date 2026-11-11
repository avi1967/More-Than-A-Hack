"""End-to-end check on a synthetic match in the real file format (no match data needed)."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mtah_possession.pipeline import build_labels, describe  # noqa: E402
from mtah_possession.features import compute  # noqa: E402
from mtah_possession.atd import load_tracking  # noqa: E402
from mtah_possession.model import Config, log_emissions, viterbi, calibrate_switch_penalties, decode_blocks  # noqa: E402

L_, W_, S, FPS = 105.0, 68.0, 46, 25


def make(tmp, seed=0):  # noqa
    rng = np.random.default_rng(seed)
    n = 6000
    # truth: possessions of 10-20 s alternating, one 6 s dead ball between some
    truth, t, team = np.full(n, "D", dtype="<U1"), 0, 0
    while t < n:
        dur = int(rng.integers(250, 500))
        truth[t:t + dur] = "HA"[team]
        t += dur
        team ^= 1
        if rng.random() < 0.4:
            t += 150
    # players: 11 per team, drifting around; ball glued to a player of the owning team
    base = rng.uniform([12, 12], [93, 56], size=(22, 2))
    phase0 = rng.uniform(0, 6.28, (22, 1))
    rate = np.where(truth == "D", 0.15, 1.0)             # players slow right down when dead
    ph = phase0[None] + np.cumsum(rate)[:, None, None] * 0.25 / FPS * np.ones((1, 22, 1))
    pos = base[None] + 8.0 * np.concatenate([np.sin(ph), np.cos(ph)], axis=2)  # ~2 m/s live
    ball = np.full((n, 2), np.nan)
    holder = 0
    for i in range(n):
        if truth[i] == "D":
            ball[i] = [-3.0, 30.0]
            continue
        if i % 50 == 0:
            holder = rng.integers(0, 11) + (11 if truth[i] == "A" else 0)
        ball[i] = pos[i, holder] + rng.normal(0, 0.4, 2)
    seen = np.ones(n, bool)
    i = 0
    while i < n:                       # drop ~40% of the ball in bursts
        if rng.random() < 0.004:
            seen[i:i + int(rng.integers(20, 200))] = False
        i += 1
    ball[~seen] = np.nan
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
    open(raw, "w").write("\n".join(lines) + "\n")
    xml = ['<?xml version="1.0"?><Metadata><Teams><Team id="h"><Name>Home</Name></Team>'
           '<Team id="a"><Name>Away</Name></Team></Teams><Players>']
    for s in range(S):
        xml.append('<Player id="p{0}" teamId="{1}"/>'.format(s, "h" if team_slot[s] == 0 else "a"))
    xml.append("</Players></Metadata>")
    meta = os.path.join(tmp, "atd.xml")
    open(meta, "w").write("".join(xml))
    return raw, meta, n, truth


def test_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        raw, meta, n, truth = make(tmp)
        target_s = 15.0
        labels, feats = build_labels(raw, meta, n, (L_, W_), [(0, n - 1)], FPS,
                                     target_possession_s=target_s, calibrate=True)
        d, t = describe(labels), describe(truth)
        acc = float((labels == truth).mean())
        print("truth", t["possessions"], "pred", d["possessions"], "frame acc %.3f" % acc,
              "mean_s %.2f" % d["mean_s"], "multiplier %.4f" % feats.get("multiplier", 1.0))
        assert set(labels) <= set("HADX")
        assert abs(d["possessions"] - t["possessions"]) <= max(3, 0.35 * t["possessions"])
        assert acc > 0.75
        assert abs(d["mean_s"] - target_s) <= 0.20 * target_s

        # Also verify uncalibrated mode works as before
        lab_uncal, _ = build_labels(raw, meta, n, (L_, W_), [(0, n - 1)], FPS, calibrate=False)
        d_uncal = describe(lab_uncal)
        acc_uncal = float((lab_uncal == truth).mean())
        assert set(lab_uncal) <= set("HADX")
        assert abs(d_uncal["possessions"] - t["possessions"]) <= max(3, 0.35 * t["possessions"])
        assert acc_uncal > 0.75


def test_short_and_long_gaps():
    n = 200
    fps = 25
    pitch = (105.0, 68.0)
    cfg = Config()

    ball = np.full((n, 2), np.nan)
    ball[10:30] = [50.0, 30.0]
    # Short gap 30..35 (5 frames < 1.5s)
    ball[35:50] = [55.0, 32.0]
    # Long gap 50..120 (70 frames > 1.5s)
    ball[120:150] = [60.0, 35.0]

    players = np.full((n, 46, 2), 50.0)
    team = np.array([0] * 23 + [1] * 23)
    track = {"ball": ball, "players": players, "team": team}

    feats = compute(track, pitch, fps, cfg)

    # 1. Short gap: properly imputed with continuous confidence
    assert feats["ball_imputed"][30:35].all(), "Short gap was not imputed"
    assert (feats["ball_confidence"][30:35] > 0.3).all(), "Short gap confidence too low"
    assert (feats["ball_confidence"][30:35] < 1.0).all(), "Imputed ball should not have 1.0 confidence"
    assert not feats["ball_missing"][30:35].any(), "Short gap marked as missing"

    # 2. Long gap: not imputed, strictly missing with 0 confidence
    assert feats["ball_missing"][50:120].all(), "Long gap was imputed"
    assert not feats["ball_imputed"][50:120].any(), "Long gap marked as imputed"
    assert (feats["ball_confidence"][50:120] == 0.0).all(), "Long gap confidence is non-zero"

    # 3. Exactly one observation state per frame, no phantom positions
    state_sum = (feats["ball_observed"].astype(int) +
                 feats["ball_imputed"].astype(int) +
                 feats["ball_missing"].astype(int))
    assert (state_sum == 1).all(), "Observation masks do not cleanly partition frames"


def test_missing_ball_does_not_generate_artificial_dead_ball():
    n = 150
    fps = 25
    pitch = (105.0, 68.0)
    cfg = Config()

    # Live play: home player has ball at frames 0..30
    ball = np.full((n, 2), np.nan)
    ball[0:30] = [50.0, 34.0]
    # Ball missing completely from 30..150!

    players = np.full((n, 46, 2), np.nan)
    # Home players actively running at 2.5 m/s (above walk_speed 1.0 m/s)
    for s in range(11):
        for t in range(n):
            players[t, s] = [20.0 + t * (2.5 / fps) + s * 2.0, 30.0]
    # Away players actively running
    for s in range(23, 34):
        for t in range(n):
            players[t, s] = [60.0 + t * (2.0 / fps) + (s - 23) * 2.0, 30.0]

    team = np.array([0] * 23 + [1] * 23)
    track = {"ball": ball, "players": players, "team": team}

    feats = compute(track, pitch, fps, cfg)
    L = log_emissions(feats, cfg)

    # During missing ball with running players, dead-ball log-likelihood should not be positive
    assert (L[30:150, 2] <= 0.0).all(), f"Dead emission is positive during active missing ball: {L[30:150, 2].max()}"

    # Viterbi should decode and persist live state H (0), NOT flip to D (2)
    path = viterbi(L, cfg)
    assert (path[30:150] == 0).all(), f"State flipped to D during active missing ball! Path: {path[30:150]}"


def test_simulated_ball_dropout():
    rng = np.random.default_rng(999)
    with tempfile.TemporaryDirectory() as tmp:
        raw, meta, n, truth = make(tmp, seed=123)
        track = load_tracking(raw, meta, n, (L_, W_))

        # Inject extra simulated ball dropout bursts (up to 40% additional loss)
        ball = track["ball"].copy()
        i = 0
        while i < n:
            if rng.random() < 0.02:
                burst_len = rng.integers(10, 80)
                ball[i:i + burst_len] = np.nan
                i += burst_len
            i += 1

        track_dropout = {"ball": ball, "players": track["players"], "team": track["team"]}
        cfg = Config()
        feats = compute(track_dropout, (L_, W_), FPS, cfg)

        # Confidence bounded and calibrated
        assert (feats["ball_confidence"] >= 0.0).all() and (feats["ball_confidence"] <= 1.0).all()
        assert (feats["ball_confidence"][feats["ball_missing"]] == 0.0).all()
        assert (feats["ball_confidence"][feats["ball_observed"]] == 1.0).all()

        # Decoding remains stable and bounded under heavy dropout
        L = log_emissions(feats, cfg)
        target_s = 15.0
        m, lab = calibrate_switch_penalties(L, [(0, n - 1)], n, cfg, fps=FPS, target_s=target_s)
        d = describe(lab, FPS)
        assert set(lab) <= set("HADX")
        assert abs(d["mean_s"] - target_s) <= 0.20 * target_s


def test_determinism():
    n = 100
    fps = 25
    pitch = (105.0, 68.0)
    cfg = Config()
    rng = np.random.default_rng(1234)

    ball = rng.uniform(0, 50, (n, 2))
    ball[rng.random(n) < 0.4] = np.nan
    players = rng.uniform(0, 50, (n, 46, 2))
    players[rng.random((n, 46)) < 0.2] = np.nan
    team = np.array([0] * 23 + [1] * 23)
    track = {"ball": ball, "players": players, "team": team}

    f1 = compute(track, pitch, fps, cfg)
    f2 = compute(track, pitch, fps, cfg)

    for k in f1:
        if np.issubdtype(f1[k].dtype, np.floating):
            assert np.array_equal(np.isnan(f1[k]), np.isnan(f2[k]))
            valid = ~np.isnan(f1[k])
            assert np.array_equal(f1[k][valid], f2[k][valid]), f"Key {k} not deterministic!"
        else:
            assert np.array_equal(f1[k], f2[k]), f"Key {k} not deterministic!"


def test_dead_ball_duration_distribution():
    """Check that decoded dead-ball durations are physically plausible."""
    with tempfile.TemporaryDirectory() as tmp:
        raw, meta, n, truth = make(tmp, seed=42)
        labels, feats = build_labels(raw, meta, n, (L_, W_), [(0, n - 1)], FPS, calibrate=True)

        runs = []
        run_len = 0
        for l in labels:
            if l == "D":
                run_len += 1
            elif run_len > 0:
                runs.append(run_len / FPS)
                run_len = 0
        if run_len > 0:
            runs.append(run_len / FPS)

        assert len(runs) > 0, "No dead-ball spells detected"
        mean_dur = np.mean(runs)
        assert 2.0 <= mean_dur <= 18.0, f"Dead-ball mean duration {mean_dur:.2f}s outside realistic bounds"
        assert min(runs) >= 0.2, f"Micro-flicker detected in dead spells: min {min(runs)}s"


def test_restart_alignment():
    """Check alignment of P(D) with restart stoppage and resumption."""
    n = 200
    fps = 25
    pitch = (105.0, 68.0)
    cfg = Config()

    ball = np.full((n, 2), np.nan)
    # Pre-restart stoppage: ball out of bounds at [-2.0, 30.0] frames 20..80
    ball[20:80] = [-2.0, 30.0]
    # Restart thrown in at frame 80, moves at speed on pitch
    for t in range(80, 150):
        ball[t] = [10.0 + (t - 80) * 0.4, 30.0]

    players = np.full((n, 46, 2), 50.0)
    for t in range(80, 150):
        players[t, 0] = ball[t] + [0.5, 0.2]
        for s in range(1, 11):
            players[t, s] = [30.0 + (t - 80) * 0.1, 30.0 + s * 2.0]
    team = np.array([0] * 23 + [1] * 23)
    track_align = {"ball": ball, "players": players, "team": team}

    # Tag window covers 0.8s to 4.0s (frames 20 to 100)
    tag_inst = [{
        "code": "SET PIECES",
        "type": "Throw-in",
        "team": "N/A",
        "start_s": 0.8,
        "end_s": 4.0,
        "f_start": 20,
        "f_end": 100,
    }]

    feats = compute(track_align, pitch, fps, cfg, smart_tag_instances=tag_inst)
    p_dead = feats["p_dead"]

    # During stoppage (frames 30..70), P(D) must be high
    assert (p_dead[30:70] > 0.6).all(), f"P(D) during stoppage is too low: {p_dead[30:70].min()}"

    # After resumption (frames 100..140), P(D) must drop significantly
    assert (p_dead[100:140] < 0.25).all(), f"P(D) after restart resumption is too high: {p_dead[100:140].max()}"


def test_stability_under_parameter_perturbations():
    """Check P(D) remains continuous and stable under parameter perturbations."""
    n = 150
    fps = 25
    pitch = (105.0, 68.0)
    rng = np.random.default_rng(777)

    ball = rng.uniform(5, 60, (n, 2))
    ball[rng.random(n) < 0.3] = np.nan
    players = rng.uniform(5, 60, (n, 46, 2))
    team = np.array([0] * 23 + [1] * 23)
    track = {"ball": ball, "players": players, "team": team}

    cfg_base = Config()
    cfg_pert1 = Config(oob_margin_m=0.7, walk_speed=1.2, dead_gain=0.08)
    cfg_pert2 = Config(oob_margin_m=0.3, walk_speed=0.8, dead_gain=0.04)

    f_base = compute(track, pitch, fps, cfg_base)
    f_p1 = compute(track, pitch, fps, cfg_pert1)
    f_p2 = compute(track, pitch, fps, cfg_pert2)

    pd_base = f_base["p_dead"]
    pd_p1 = f_p1["p_dead"]
    pd_p2 = f_p2["p_dead"]

    for pd in (pd_base, pd_p1, pd_p2):
        assert (pd >= 0.0).all() and (pd <= 1.0).all()

    diff1 = np.abs(pd_base - pd_p1).mean()
    diff2 = np.abs(pd_base - pd_p2).mean()
    assert diff1 < 0.15, f"Parameter perturbation 1 caused excessive drift: {diff1}"
    assert diff2 < 0.15, f"Parameter perturbation 2 caused excessive drift: {diff2}"


def test_missing_ball_during_live_play_keeps_low_pd():
    """Verify missing ball during active running produces low P(D) and avoids artificial D."""
    n = 120
    fps = 25
    pitch = (105.0, 68.0)
    cfg = Config()

    ball = np.full((n, 2), np.nan)
    ball[0:15] = [52.5, 34.0]

    players = np.full((n, 46, 2), np.nan)
    for s in range(11):
        for t in range(n):
            players[t, s] = [25.0 + t * (2.2 / fps), 30.0 + s * 1.5]
    for s in range(23, 34):
        for t in range(n):
            players[t, s] = [65.0 + t * (2.0 / fps), 25.0 + (s - 23) * 1.5]

    team = np.array([0] * 23 + [1] * 23)
    track = {"ball": ball, "players": players, "team": team}

    feats = compute(track, pitch, fps, cfg)
    pd = feats["p_dead"]

    assert (pd[15:120] < 0.25).all(), f"P(D) elevated during live running missing ball: {pd[15:120].max()}"


def test_phase4_scenarios_comparison():
    """Evaluate duration-aware decoder vs baseline Viterbi across the 7 challenge scenarios:
    1. Short possessions (genuine 1.5s possession)
    2. Long possessions (30s unbroken sequence)
    3. Rapid turnovers (H -> A for 1.6s -> H)
    4. Loose balls (ball in open space for 2.0s without false handover)
    5. Aerial challenges / deflections (opposing player nearest for 0.4s without control)
    6. Stoppages (ball out of bounds, transition to D)
    7. Missing ball observations (unobserved ball for 3.0s during live play)
    """
    labels = []
    # 1. Long possession H (750 frames = 30s)
    labels.extend(["H"] * 750)
    # 2. Deflection: A nearest for 10 frames = 0.4s with fast ball, then H
    labels.extend(["H"] * 50 + ["H"] * 10 + ["H"] * 140)
    # 3. Loose ball: 50 frames open space, then H
    labels.extend(["H"] * 50 + ["H"] * 150)
    # 4. Rapid turnover: A tackles cleanly for 40 frames = 1.6s, then H
    labels.extend(["A"] * 40 + ["H"] * 200)
    # 5. Missing ball: 75 frames unobserved, then H
    labels.extend(["H"] * 75 + ["H"] * 125)
    # 6. Stoppage: 150 frames out of bounds D
    labels.extend(["D"] * 150)
    # 7. Restart: A takes restart and possesses for 200 frames
    labels.extend(["A"] * 200)

    n = len(labels)
    gt = np.array(labels)

    dH = np.full(n, 10.0, dtype=np.float32)
    dA = np.full(n, 10.0, dtype=np.float32)
    ball_speed = np.full(n, 2.0, dtype=np.float32)
    ball_ok = np.ones(n, dtype=bool)
    oob = np.zeros(n, dtype=bool)
    still = np.zeros(n, dtype=bool)
    activity = np.full(n, 1.8, dtype=np.float32)
    p_dead = np.full(n, 0.05, dtype=np.float32)
    ball_confidence = np.ones(n, dtype=np.float32)

    # 1. Long H:
    dH[0:750] = 0.8; dA[0:750] = 8.0
    # 2. Deflection:
    dH[750:950] = 0.8; dA[750:950] = 8.0
    dH[800:810] = 1.5; dA[800:810] = 0.4; ball_speed[800:810] = 12.0
    # 3. Loose ball:
    dH[950:1000] = 6.0; dA[950:1000] = 8.0; ball_speed[950:1000] = 8.0
    dH[1000:1150] = 0.6; dA[1000:1150] = 9.0
    # 4. Rapid turnover:
    dH[1150:1190] = 7.0; dA[1150:1190] = 0.7; ball_speed[1150:1190] = 1.8
    dH[1190:1390] = 0.8; dA[1190:1390] = 8.0
    # 5. Missing ball:
    dH[1390:1465] = np.nan; dA[1390:1465] = np.nan; ball_ok[1390:1465] = False
    ball_confidence[1390:1465] = 0.0; ball_speed[1390:1465] = np.nan; p_dead[1390:1465] = 0.10
    dH[1465:1590] = 0.7; dA[1465:1590] = 7.0
    # 6. Stoppage:
    dH[1590:1740] = np.nan; dA[1590:1740] = np.nan; ball_ok[1590:1740] = False
    oob[1590:1740] = True; p_dead[1590:1740] = 0.95; activity[1590:1740] = 0.6
    # 7. Restart A:
    dH[1740:1940] = 9.0; dA[1740:1940] = 0.8; ball_speed[1740:1940] = 2.0

    feats = {
        "dH": dH, "dA": dA, "ball_speed": ball_speed,
        "ball_ok": ball_ok, "oob": oob, "still": still,
        "activity": activity, "p_dead": p_dead,
        "ball_confidence": ball_confidence,
    }
    cfg = Config()
    L = log_emissions(feats, cfg)

    # 1. Test Duration-Aware Viterbi
    lab_da = decode_blocks(L, [(0, n - 1)], n, cfg, feats=feats, decoder="duration_aware", fps=FPS)
    acc_da = (lab_da == gt).mean()

    # 2. Test Baseline Viterbi
    lab_base = decode_blocks(L, [(0, n - 1)], n, cfg, feats=feats, decoder="viterbi", fps=FPS)
    acc_base = (lab_base == gt).mean()

    assert acc_da >= 0.99, f"Duration-aware accuracy too low: {acc_da:.3f}"
    # Verify Duration-Aware correctly captured the rapid turnover
    assert (lab_da[1150:1190] == "A").all(), "Duration-aware missed rapid turnover"
    # Verify baseline Viterbi is preserved and runnable
    assert acc_base > 0.90, f"Baseline Viterbi accuracy unexpected: {acc_base:.3f}"


def test_half_boundary_reset():
    """Verify that temporal state resets cleanly across half boundaries without state bleed."""
    n = 600
    fps = 25
    half1 = (0, 250)
    inter = (251, 349)
    half2 = (350, 599)
    blocks = [half1, half2]

    # Half 1 ends with strong H possession; Half 2 starts with strong A possession
    dH = np.full(n, 10.0, dtype=np.float32)
    dA = np.full(n, 10.0, dtype=np.float32)
    dH[0:251] = 0.5; dA[0:251] = 8.0   # H controls Half 1
    dH[350:600] = 8.0; dA[350:600] = 0.5  # A controls Half 2

    ball_speed = np.full(n, 2.0, dtype=np.float32)
    ball_ok = np.ones(n, dtype=bool)
    oob = np.zeros(n, dtype=bool)
    still = np.zeros(n, dtype=bool)
    activity = np.full(n, 1.8, dtype=np.float32)
    p_dead = np.full(n, 0.05, dtype=np.float32)
    ball_confidence = np.ones(n, dtype=np.float32)

    feats = {
        "dH": dH, "dA": dA, "ball_speed": ball_speed,
        "ball_ok": ball_ok, "oob": oob, "still": still,
        "activity": activity, "p_dead": p_dead,
        "ball_confidence": ball_confidence,
    }
    cfg = Config()
    L = log_emissions(feats, cfg)

    labels = decode_blocks(L, blocks, n, cfg, feats=feats, decoder="duration_aware", fps=fps)

    # 1. Halftime interval must be labeled 'X'
    assert (labels[251:350] == "X").all(), "Halftime interval was not labeled X"
    # 2. Half 1 must be H
    assert (labels[0:251] == "H").all(), "Half 1 was not labeled H"
    # 3. Half 2 must immediately start as A without lagging H inheritance
    assert (labels[350:375] == "A").all(), "Half 2 did not reset state cleanly at half kickoff"


def test_carrier_loss_mitigation():
    """Verify that carrier continuity prevents spurious opponent handovers when carrier disappears."""
    n = 250
    fps = 25
    gt = np.full(n, "H", dtype="<U1")
    dH = np.full(n, 0.5, dtype=np.float64)
    dA = np.full(n, 3.5, dtype=np.float64)

    # Carrier drops for 1.5s (frames 75..112 = 37 frames), opponent is pressing at 1.3m
    dH[75:112] = 5.0
    dA[75:112] = 1.3

    feats = {
        "dH": dH, "dA": dA, "ball_speed": np.full(n, 2.0),
        "ball_ok": np.ones(n, bool), "oob": np.zeros(n, bool), "still": np.zeros(n, bool),
        "activity": np.full(n, 1.8), "p_dead": np.full(n, 0.05),
        "ball_confidence": np.ones(n), "valid_players": np.full(n, 22),
    }

    # 1. With carrier continuity enabled:
    cfg_mit = Config(switch_team=2.5, switch_dead=2.0, carrier_continuity=True)
    from mtah_possession.carrier_continuity import compute_carrier_continuity
    cH, cA, dH_eff, dA_eff, _, _ = compute_carrier_continuity(
        dH, dA, feats["ball_speed"], feats["ball_ok"], feats["ball_confidence"], fps=fps
    )
    feats_mit = dict(feats)
    feats_mit["dH"] = dH_eff
    feats_mit["dA"] = dA_eff
    feats_mit["carrier_cH"] = cH
    feats_mit["carrier_cA"] = cA

    L_mit = log_emissions(feats_mit, cfg_mit)
    lab_mit = decode_blocks(L_mit, [(0, n - 1)], n, cfg_mit, feats=feats_mit, decoder="duration_aware", fps=fps)
    assert (lab_mit == "H").all(), "Carrier continuity failed to prevent false handover during carrier loss"
    assert describe(lab_mit, fps)["possessions"] == 1

    # 2. Baseline without carrier continuity: flips to Away
    cfg_base = Config(switch_team=2.5, switch_dead=2.0, carrier_continuity=False)
    L_base = log_emissions(feats, cfg_base)
    lab_base = decode_blocks(L_base, [(0, n - 1)], n, cfg_base, feats=feats, decoder="duration_aware", fps=fps)
    assert describe(lab_base, fps)["possessions"] > 1, "Baseline was expected to fracture without continuity"


def test_genuine_turnover_under_carrier_loss():
    """Verify that a genuine tackle is promptly recognized even if previous carrier disappears."""
    n = 250
    fps = 25
    gt = np.concatenate([np.full(100, "H"), np.full(150, "A")])
    dH = np.full(n, 0.5, dtype=np.float64)
    dA = np.full(n, 3.5, dtype=np.float64)

    # Frame 100: Away tackles cleanly (dA = 0.5m), Home carrier is lost (dH = 5.0m)
    dH[100:250] = 5.0
    dA[100:250] = 0.5

    feats = {
        "dH": dH, "dA": dA, "ball_speed": np.full(n, 2.0),
        "ball_ok": np.ones(n, bool), "oob": np.zeros(n, bool), "still": np.zeros(n, bool),
        "activity": np.full(n, 1.8), "p_dead": np.full(n, 0.05),
        "ball_confidence": np.ones(n), "valid_players": np.full(n, 22),
    }

    cfg = Config(switch_team=2.5, switch_dead=2.0, carrier_continuity=True)
    from mtah_possession.carrier_continuity import compute_carrier_continuity
    cH, cA, dH_eff, dA_eff, _, _ = compute_carrier_continuity(
        dH, dA, feats["ball_speed"], feats["ball_ok"], feats["ball_confidence"], fps=fps
    )
    feats["dH"] = dH_eff
    feats["dA"] = dA_eff
    feats["carrier_cH"] = cH
    feats["carrier_cA"] = cA

    L = log_emissions(feats, cfg)
    lab = decode_blocks(L, [(0, n - 1)], n, cfg, feats=feats, decoder="duration_aware", fps=fps)

    assert (lab == gt).all(), "Genuine turnover was not detected accurately under carrier continuity"
    assert describe(lab, fps)["possessions"] == 2


def test_carrier_continuity_decay():
    """Verify that carrier continuity decays smoothly and does not preserve state indefinitely."""
    n = 200  # 8 seconds
    fps = 25
    dH = np.full(n, 5.0, dtype=np.float64)
    dA = np.full(n, 5.0, dtype=np.float64)
    # Frame 0..20 Home has ball, then vanishes for frames 20..200 (7.2s)
    dH[0:20] = 0.5

    from mtah_possession.carrier_continuity import compute_carrier_continuity
    cH, cA, dH_eff, dA_eff, _, _ = compute_carrier_continuity(
        dH, dA, np.full(n, 2.0), np.ones(n, bool), np.ones(n), fps=fps, tau_carrier=2.0
    )

    # Initial control
    assert cH[19] == 1.0, "Initial carrier control was not 1.0"
    # Decaying monotonically
    assert (np.diff(cH[20:150]) <= 1e-6).all(), "Continuity is not monotonically decaying"
    # Over 2 seconds (50 frames), it decays to approx 0.36
    assert 0.25 <= cH[70] <= 0.45, f"Continuity after 2s unexpected: {cH[70]}"
    # Over 6 seconds, it decays to < 0.1
    assert cH[170] < 0.10, f"Continuity did not decay sufficiently over long absence: {cH[170]}"


def test_dead_ball_flight_counter_evidence():
    """Verify that inbounds aerial ball flight does not trigger false dead ball."""
    n = 100
    fps = 25
    pitch = (105.0, 68.0)
    ball = np.full((n, 2), np.nan)
    # Ball traveling at 14 m/s in midfield (frames 20..70)
    for t in range(n):
        ball[t] = [30.0 + t * (14.0 / fps), 34.0]

    players = np.full((n, 46, 2), 50.0)
    # Players calm and standing still
    team = np.array([0] * 23 + [1] * 23)
    track = {"ball": ball, "players": players, "team": team}

    # 1. Refined V2: flight counter-evidence suppresses P(D)
    cfg_v2 = Config(dead_ball_v2=True)
    f_v2 = compute(track, pitch, fps, cfg_v2)
    p_dead_v2 = f_v2["p_dead"]
    assert (p_dead_v2 < 0.05).all(), f"Refined V2 elevated P(D) during inbounds flight: {p_dead_v2.max()}"

    # 2. Baseline V1: elevated P(D) during flight due to missing live flight suppression
    cfg_v1 = Config(dead_ball_v2=False)
    f_v1 = compute(track, pitch, fps, cfg_v1)
    p_dead_v1 = f_v1["p_dead"]
    assert p_dead_v1.max() > 0.40, f"Baseline V1 did not exhibit expected flight vulnerability: {p_dead_v1.max()}"
    assert p_dead_v1.mean() > 10 * p_dead_v2.mean(), f"Flight counter-evidence did not strongly suppress P(D): v1={p_dead_v1.mean()}, v2={p_dead_v2.mean()}"


def test_carrier_holding_does_not_trigger_dead():
    """Verify that a goalkeeper or defender holding a stationary ball remains live."""
    n = 150  # 6 seconds
    fps = 25
    pitch = (105.0, 68.0)
    ball = np.full((n, 2), [15.0, 34.0])  # Inside penalty box
    players = np.full((n, 46, 2), 50.0)
    # Goalkeeper standing right at the ball (0.3m away)
    players[:, 0] = [15.3, 34.0]

    team = np.array([0] * 23 + [1] * 23)
    track = {"ball": ball, "players": players, "team": team}

    cfg = Config(dead_ball_v2=True)
    f = compute(track, pitch, fps, cfg)
    p_dead = f["p_dead"]
    assert (p_dead < 0.25).all(), f"Holding ball falsely generated dead-ball evidence: {p_dead.max()}"


def test_unpossessed_stationary_dead_ball():
    """Verify that an abandoned stationary ball correctly triggers dead-ball evidence."""
    n = 150  # 6 seconds
    fps = 25
    pitch = (105.0, 68.0)
    ball = np.full((n, 2), [52.5, 34.0])  # Stationary in center circle
    players = np.full((n, 46, 2), 50.0)
    # Nearest player is 8 metres away
    players[:, :] = [30.0, 30.0]

    team = np.array([0] * 23 + [1] * 23)
    track = {"ball": ball, "players": players, "team": team}

    cfg = Config(dead_ball_v2=True)
    f = compute(track, pitch, fps, cfg)
    p_dead = f["p_dead"]
    # After ~2.5s of stillness with no carrier nearby, P(D) must rise significantly (>0.70)
    assert p_dead[100:].mean() > 0.70, f"Unpossessed stationary ball failed to trigger dead ball: {p_dead[100:].mean()}"


if __name__ == "__main__":
    test_short_and_long_gaps()
    test_missing_ball_does_not_generate_artificial_dead_ball()
    test_determinism()
    test_simulated_ball_dropout()
    test_dead_ball_duration_distribution()
    test_restart_alignment()
    test_stability_under_parameter_perturbations()
    test_missing_ball_during_live_play_keeps_low_pd()
    test_phase4_scenarios_comparison()
    test_half_boundary_reset()
    test_carrier_loss_mitigation()
    test_genuine_turnover_under_carrier_loss()
    test_carrier_continuity_decay()
    test_dead_ball_flight_counter_evidence()
    test_carrier_holding_does_not_trigger_dead()
    test_unpossessed_stationary_dead_ball()
    for seed in range(4):
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta, n, truth = make(tmp, seed)
            lab, _ = build_labels(raw, meta, n, (L_, W_), [(0, n - 1)], FPS)
            print("seed", seed, "truth", describe(truth)["possessions"],
                  "pred", describe(lab)["possessions"], "acc %.3f" % float((lab == truth).mean()))
    test_pipeline()
    print("ok")
