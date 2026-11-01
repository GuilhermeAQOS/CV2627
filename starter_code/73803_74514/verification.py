import numpy as np
import cv2 as cv

def verify_cuts(
    frames: list,
    candidates: np.ndarray,
    gap: int = 4,
    min_matches: int = 20,
    ratio: float = 0.75,
) -> np.ndarray:
    kept = []
    for candidate in candidates:
        cand_start_idx = candidate - gap
        cand_end_idx = candidate + gap
        if cand_start_idx < 0 or cand_end_idx >= len(frames):
            kept.append(candidate)
            continue
        cand_start = frames[cand_start_idx]
        cand_end = frames[cand_end_idx]
        matches, _, _ = sift_match(cand_start, cand_end, ratio)
        if len(matches) < min_matches:
            kept.append(candidate)
    return np.asarray(kept)

def sift_match(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    ratio: float = 0.75
) -> tuple[list, list, list]:
    sift = cv.SIFT_create()
    gray_a, gray_b = cv.cvtColor(frame_a,cv.COLOR_BGR2GRAY), cv.cvtColor(frame_b,cv.COLOR_BGR2GRAY)
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or desc_b.shape[0] < 2:
        return []
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