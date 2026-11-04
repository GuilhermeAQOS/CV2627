import numpy as np
import cv2 as cv

def compute_candidate_match_num(
    frames: list,
    candidates: np.ndarray,
    gap: int = 4,
    ratio: float = 0.75,
) -> np.ndarray:
    match_list = []
    for candidate in candidates:
        cand_start_idx = candidate - gap
        cand_end_idx = candidate + gap
        if cand_start_idx < 0 or cand_end_idx >= len(frames):
            match_list.append(0)
            continue
        cand_start = frames[cand_start_idx]
        cand_end = frames[cand_end_idx]
        matches, _, _ = sift_match(cand_start, cand_end, ratio)
        match_list.append(len(matches))
    return np.asarray(match_list)


def verified_cuts(
    frames: list,
    candidates: np.ndarray,
    min_matches: int = 15,
    gap: int = 4,
    ratio: float = 0.75,
) -> np.ndarray:
    """Keep candidates that do not match across the gap.

    A large match count means frames t-gap and t+gap show the same content,
    so the candidate is a false alarm inside one shot.
    """
    candidates = np.asarray(candidates)
    if len(candidates) == 0:
        return candidates
    match_num = compute_candidate_match_num(frames, candidates, gap=gap, ratio=ratio)
    return candidates[match_num < min_matches]


def sift_match(frame_a: np.ndarray, frame_b: np.ndarray, ratio: float = 0.75) -> tuple[list, list, list]:
    sift = cv.SIFT_create()
    gray_a, gray_b = cv.cvtColor(frame_a,cv.COLOR_BGR2GRAY), cv.cvtColor(frame_b,cv.COLOR_BGR2GRAY)
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    kp_a = list(kp_a or [])
    kp_b = list(kp_b or [])
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) < 2:
        return [], kp_a, kp_b
    bf = cv.BFMatcher(cv.NORM_L2)
    pairs = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)
    return (good, kp_a, kp_b)