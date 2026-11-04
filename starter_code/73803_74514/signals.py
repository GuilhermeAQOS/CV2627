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


def grayscale_histogram_signal(frames: list, num_bins: int = 256) -> np.ndarray:
    """One value per frame. d(0) = 0; d(t) is the L1 distance of grey histograms."""
    if len(frames) < 2:
        return np.zeros(len(frames), dtype=np.float64)
    histograms = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = np.histogram(gray, bins=num_bins, range=(0, 256))[0].astype(np.float64)
        histograms.append(hist / hist.sum())
    histograms = np.stack(histograms)
    score = np.sum(np.abs(np.diff(histograms, axis=0)), axis=1)
    return np.concatenate(([0.0], score))


#Histogram Distance measurement
def histogram_chi_sqrd_distances(frames: np.ndarray, color_space: str = 'RGB') -> np.ndarray:
    if color_space == 'RGB':
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        red_frames = [frame[:, :, 0] for frame in rgb_frames]
        green_frames = [frame[:, :, 1] for frame in rgb_frames]
        blue_frames = [frame[:, :, 2] for frame in rgb_frames]
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