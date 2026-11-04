###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Shot boundary detection and recurring scenes
##
## Rogério Soares (73803)
## Guilherme Simões (74514)
##
###############################################################################

import glob
import json
import os

import cv2
import numpy as np

import grouping   # group_shots: union-find grouping of shots (Part 3, Task 7)
import metrics
import scenes
import signals
import verification

# Part 1: HSV colour histogram signal (chi-square) and its best threshold on dev_01 and dev_02.
HIST_BINS = 8
HISTOGRAM_THRESHOLD = 0.08061058771   # paste the full value of best["histogram"][0] (the notebook check cell shows it)
MERGE_GAP = 2                  # detections at most this many frames apart count as one cut
# Part 2: a candidate is rejected when it has at least this many SIFT matches (best dev F1).
MIN_SIFT_MATCHES = 33  
VERIFY_GAP = 4
LOWE_RATIO = 0.75
# Best scene grouping on dev_01 and dev_02 (Part 3 sweep).
SCENE_FEATURE = "histogram"
SCENE_KEYFRAMES = 3
SCENE_RULE = "max"
SCENE_THRESHOLD = 0.950492495

INPUT_DIR = os.path.join("..", "input")
OUTPUT_DIR = os.path.join("..", "output")


def read_frames(path, size=(320, 180)):
    """Read every frame of a video, resized to `size`. Returns a list of BGR images."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def merge_nearby(indices, score, gap=MERGE_GAP):
    """Merge detections at most `gap` frames apart, keeping the frame with the highest score.

    indices: frames where a detector fired; score: array indexed by frame (the signal itself, or a combination).
    """
    groups = []
    for t in indices:
        if groups and gap > 0 and t - groups[-1][-1] <= gap:
            groups[-1].append(t)
        else:
            groups.append([t])
    return np.array([max(group, key=lambda t: score[t]) for group in groups], dtype=int)


def process_video(frames, gt_shots):
    """Return (transitions, scenes) for one video.

    transitions: list of {"type": "cut", "start": t, "end": t}, t = first frame of the new shot (Parts 1 and 2)
    scenes:      dict {str(shot id): label} for the ground-truth shots (Part 3)
    """
    signal = signals.histogram_signal(frames, HIST_BINS)
    candidates = merge_nearby(np.flatnonzero(signal > HISTOGRAM_THRESHOLD), signal)
    kept = verification.verified_cuts(
        frames,
        candidates,
        min_matches=MIN_SIFT_MATCHES,
        gap=VERIFY_GAP,
        ratio=LOWE_RATIO,
    )
    transitions = [
        {"type": "cut", "start": int(t), "end": int(t)} for t in kept
    ]
    sim = scenes.similarity_matrix(
        frames,
        gt_shots,
        SCENE_FEATURE,
        SCENE_KEYFRAMES,
        SCENE_RULE,
    )
    labels = grouping.group_shots(sim, SCENE_THRESHOLD)
    scene_dict = scenes.scene_labels(gt_shots, labels)
    return transitions, scene_dict


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}                                    # video name -> counts from metrics.evaluate_video
    for video_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4"))):
        name = os.path.splitext(os.path.basename(video_path))[0]
        gt = json.load(open(os.path.join(INPUT_DIR, name + ".json")))
        frames = read_frames(video_path)
        transitions, scene_dict = process_video(frames, gt["shots"])
        pred = {"video": name + ".mp4", "transitions": transitions, "scenes": scene_dict}
        with open(os.path.join(OUTPUT_DIR, name + ".json"), "w") as f:
            json.dump(pred, f, indent=1, default=int)    # default=int: accepts NumPy integers
        results[name] = metrics.evaluate_video(gt, pred)

    # summary table (Section 8): one row per video, then the dev / test totals
    dev = [results[n] for n in sorted(results) if n.startswith("dev")]
    test = [results[n] for n in sorted(results) if n.startswith("test")]
    metrics.print_table([(n, results[n]) for n in sorted(results)]
                        + [("DEV", metrics.sum_counts(dev)), ("TEST", metrics.sum_counts(test))])