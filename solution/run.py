"""Regenerate the possession file from the provided data.

    python solution/run.py --out my_team.csv          # data in ./data or $MTAH_DATA_DIR
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames, asset_paths  # noqa: E402
from mtah_possession.pipeline import build_labels, describe, write_csv  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--target-possession-s", type=float, default=15.0,
                    help="target mean possession duration in seconds (default: 15.0)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="use unscaled switch penalties from Config without calibration")
    ap.add_argument("--decoder", choices=["semi_markov", "duration_aware", "viterbi"], default="semi_markov",
                    help="decoder algorithm: 'semi_markov' (Candidate B default), 'duration_aware' (Phase 8), or 'viterbi' (baseline)")
    args = ap.parse_args()

    meta, raw = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    tag_paths = asset_paths(GAME, args.data_dir).get("smart_tagging")
    smart_tag_path = tag_paths if (tag_paths and os.path.exists(tag_paths)) else None

    n = source_frames("tactical", GAME)
    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    print("reading tracking ...")
    labels, feats = build_labels(
        raw, meta, n, GAMES[GAME]["pitch"], blocks, GAMES[GAME]["fps"],
        target_possession_s=args.target_possession_s,
        calibrate=not args.no_calibrate,
        smart_tag_path=smart_tag_path,
        decoder=args.decoder,
    )
    multiplier = feats.get("multiplier", 1.0)
    print("chosen multiplier: {:.4f}".format(multiplier))
    write_csv(args.out, labels)
    d = describe(labels, GAMES[GAME]["fps"])
    print("wrote {}: {} possessions, mean {:.1f} s | H {H} A {A} D {D} X {X}".format(
        args.out, d["possessions"], d["mean_s"], **{k: d[k] for k in "HADX"}))
    print("next: python evaluation/validate_submission.py {}".format(args.out))
