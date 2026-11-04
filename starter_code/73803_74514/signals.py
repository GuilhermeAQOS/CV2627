import cv2
import numpy as np


# ---------- 1. Pixel difference (baseline) ----------
def pixel_difference(f1, f2):
    """Mean absolute grey-level difference between two BGR frames."""
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.int16)  # int16: uint8 would wrap around
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return np.mean(np.abs(g1 - g2))


# ---------- 2. Colour histogram + chi-square distance ----------
def color_histogram(frame, bins=16):
    """HSV histogram of a BGR frame: `bins` per channel, each channel normalised to sum 1."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ranges = [180, 256, 256]                      # OpenCV value range of H, S, V
    hist = [np.histogram(hsv[:, :, c], bins=bins, range=(0, ranges[c]))[0] for c in range(3)]
    hist = np.concatenate([h / h.sum() for h in hist])
    return hist / 3                               # whole histogram sums to 1


def histogram_distance(h1, h2):
    """Chi-square distance between two histograms (0 = identical)."""
    return 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))


# ---------- 3. Edge Change Ratio ----------
def edge_change_ratio(f1, f2, kernel_size=5):
    """ECR between two BGR frames, in [0, 1]."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    e1 = cv2.Canny(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), 100, 200) > 0   # edges of frame t-1
    e2 = cv2.Canny(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), 100, 200) > 0   # edges of frame t
    d1 = cv2.dilate(e1.astype(np.uint8), kernel) > 0
    d2 = cv2.dilate(e2.astype(np.uint8), kernel) > 0

    x_in = np.sum(e2 & ~d1)       # entering edges
    x_out = np.sum(e1 & ~d2)      # exiting edges
    # no edges at all (e.g. black frame): ratio 0, so we never divide by zero
    r_in = x_in / np.sum(e2) if np.sum(e2) > 0 else 0.0
    r_out = x_out / np.sum(e1) if np.sum(e1) > 0 else 0.0
    return max(r_in, r_out)


def compute_signals(frames, bins=16):
    n = len(frames)
    pixel, hist, ecr = np.zeros(n), np.zeros(n), np.zeros(n)
    prev_hist = color_histogram(frames[0], bins)
    for t in range(1, n):
        cur_hist = color_histogram(frames[t], bins)
        pixel[t] = pixel_difference(frames[t - 1], frames[t])
        hist[t] = histogram_distance(prev_hist, cur_hist)
        ecr[t] = edge_change_ratio(frames[t - 1], frames[t])
        prev_hist = cur_hist
    return {"pixel": pixel, "histogram": hist, "ecr": ecr}