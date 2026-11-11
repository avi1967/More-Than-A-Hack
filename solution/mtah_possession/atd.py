"""Read the raw ATD text file into numpy arrays, in metres, indexed by video frame.

Raw line:  "<frame_id>:<x,y,s>;<x,y,s>;...(one group per track slot):<x,y,s>(ball)"
Missing observations are the literal text NaN. frame_id counts from 1; video frames from 0.
"""
import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """0 for slots on the home side, 1 for away. The first <Team> in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _parse_chunk(lines, n_slots, bad):
    width = 1 + 3 * n_slots + 3
    text = ",".join(l.strip().replace(":", ",").replace(";", ",") for l in lines)
    arr = np.fromstring(text, sep=",", dtype=np.float64)
    if arr.size != width * len(lines):  # a malformed line: fall back to line-by-line
        rows = []
        for l in lines:
            row = np.fromstring(l.strip().replace(":", ",").replace(";", ","),
                                sep=",", dtype=np.float64)
            if row.size != width:
                bad.append(1)
                row = np.full(width, np.nan)
            rows.append(row)
        arr = np.concatenate(rows)
    return arr.reshape(len(lines), width)


def load_tracking(raw_path, meta_path, n_frames, pitch, chunk=4000):
    """Returns dict(ball=(N,2), players=(N,S,2), team=(S,)), positions in metres, NaN = unseen."""
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

    bad, total = [], [0]

    def flush(lines):
        total[0] += len(lines)
        rows = _parse_chunk(lines, n_slots, bad)
        ids = rows[:, 0]
        ok = np.isfinite(ids)
        vid = ids[ok].astype(np.int64) - 1  # frame_id counts from 1
        rows = rows[ok]
        keep = (vid >= 0) & (vid < n_frames)
        vid, rows = vid[keep], rows[keep]
        grid = rows[:, 1:1 + 3 * n_slots].reshape(len(rows), n_slots, 3)
        players[vid, :, 0] = grid[:, :, 0] * length
        players[vid, :, 1] = grid[:, :, 1] * width
        ball[vid, 0] = rows[:, -3] * length
        ball[vid, 1] = rows[:, -2] * width

    buf = []
    with open(raw_path) as handle:
        for line in handle:
            if line.strip():
                buf.append(line)
            if len(buf) >= chunk:
                flush(buf)
                buf = []
    if buf:
        flush(buf)
    if len(bad) > 0.01 * max(1, total[0]):
        raise ValueError("{} of {} ATD lines do not have {} numbers each: the file layout is not "
                         "what this loader expects (1 id + {} slots x 3 + ball x 3)".format(
                             len(bad), total[0], 1 + 3 * n_slots + 3, n_slots))
    return {"ball": ball, "players": players, "team": team}
