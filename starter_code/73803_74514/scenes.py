import numpy as np
import cv2 as cv

import grouping
import metrics
import signals

def keyframe_indices(shot, count):
    """Frame indices inside one ground-truth shot, inclusive of start and end."""
    if count < 1:
        raise ValueError("count must be at least 1")
    start = int(shot["start"])
    end = int(shot["end"])
    if count == 1:
        return [(start + end) // 2]
    indices = []
    for position in np.linspace(start, end, count):
        index = int(round(position))
        index = min(max(index, start), end)
        if index not in indices:
            indices.append(index)
    return indices

def color_similarity(h1, h2):
    return 1 - signals.histogram_distance(h1, h2)

def sift_features(frame):
    sift = cv.SIFT_create()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    _, desc = sift.detectAndCompute(gray, None)
    return desc

def sift_match_count(desc_a, desc_b, ratio=0.75):
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) < 2:
        return 0
    bf = cv.BFMatcher(cv.NORM_L2)
    pairs = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)
    return len(good)

def combine_keyframe_scores(scores, rule):
    if rule == "mean":
        return (sum(scores) / len(scores))
    elif rule == "max":
        return max(scores)
    else:
        raise ValueError("Unsupported rule: " + rule)

def similarity_matrix(frames, shots, feature, n_keyframes, rule):
    """Shot x shot similarity, in the order of 'shots'.

    feature is "histogram" or "sift". rule is "max" or "mean".
    The diagonal stays 0; group_shots only reads pairs a < b.
    """
    if feature not in ("histogram", "sift"):
        raise ValueError("feature must be 'histogram' or 'sift'")
    n = len(shots)
    matrix = np.zeros((n, n), dtype=np.float64)
    representations = []
    for shot in shots:
        shot_reps = []
        for index in keyframe_indices(shot, n_keyframes):
            frame = frames[index]
            if feature == "histogram":
                shot_reps.append(signals.color_histogram(frame))
            else:
                shot_reps.append(sift_features(frame))
        representations.append(shot_reps)

    for a in range(n):
        for b in range(a + 1, n):
            scores = []
            for rep_a in representations[a]:
                for rep_b in representations[b]:
                    if feature == "histogram":
                        scores.append(color_similarity(rep_a, rep_b))
                    else:
                        scores.append(float(sift_match_count(rep_a, rep_b)))
            value = combine_keyframe_scores(scores, rule) if scores else 0.0
            matrix[a, b] = matrix[b, a] = value
    return matrix


def ground_truth_matrix(shots):
    n = len(shots)
    matrix = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            matrix[a, b] = float(shots[a]["scene"] == shots[b]["scene"])
    return matrix


def scene_labels(shots, labels):
    return {str(shot["id"]): int(label) for shot, label in zip(shots, labels)}


def pair_counts(shots, sim, threshold):
    labels = grouping.group_shots(sim, threshold)
    pred = {str(shot["id"]): str(label) for shot, label in zip(shots, labels)}
    gt = {str(shot["id"]): shot["scene"] for shot in shots}
    return metrics.pairwise_scene_counts(gt, pred)


def best_threshold(video_sims, video_shots, n_steps=40):
    off = np.concatenate([
        sim[np.triu_indices(len(sim), k=1)] for sim in video_sims
    ])
    if off.size == 0 or off.min() == off.max():
        thresholds = np.array([0.0 if off.size == 0 else float(off.max())])
    else:
        thresholds = np.linspace(float(off.min()), float(off.max()), n_steps)
    best = None
    for threshold in thresholds:
        counts = [
            pair_counts(shots, sim, float(threshold))
            for sim, shots in zip(video_sims, video_shots)
        ]
        tp, fp, fn = (sum(count[k] for count in counts) for k in range(3))
        precision, recall, f1 = metrics.precision_recall_f1(tp, fp, fn)
        if best is None or f1 > best[3]:
            best = (float(threshold), precision, recall, f1)
    return best