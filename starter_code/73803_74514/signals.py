import numpy as np
import matplotlib.pyplot as plt
import cv2
import tp1
import os

"""Helper functions"""

def normalize_histogram(histogram: np.ndarray) -> np.ndarray:
    return histogram / np.sum(histogram, axis=1, keepdims=True)


"""Dissimilarity functions"""

def absolute_histogram_differences(frames: list, num_bins: int = 256, color_space='RGB') -> np.ndarray:
    """
    This Function only computes the absolute histogram diff between consecutive frames, created as an initial exploratory tool to look at discrepancies between frames. It is useful for understanding the data and the problem.
    """
    if color_space == 'RGB':

        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        red_frames = [frame[:, :, 0] for frame in rgb_frames]
        green_frames = [frame[:, :, 1] for frame in rgb_frames]
        blue_frames = [frame[:, :, 2] for frame in rgb_frames]


        red_histogram = np.array([np.histogram(frame, bins=num_bins, range=(0, 256))[0] for frame in red_frames])
        green_histogram = np.array([np.histogram(frame, bins=num_bins, range=(0, 256))[0] for frame in green_frames])
        blue_histogram = np.array([np.histogram(frame, bins=num_bins, range=(0, 256))[0] for frame in blue_frames])

        red_histogram = normalize_histogram(red_histogram)
        green_histogram = normalize_histogram(green_histogram)
        blue_histogram = normalize_histogram(blue_histogram)

        red_histogram_diff = np.diff(red_histogram, axis=0)
        green_histogram_diff = np.diff(green_histogram, axis=0)
        blue_histogram_diff = np.diff(blue_histogram, axis=0)

        return red_histogram_diff, green_histogram_diff, blue_histogram_diff
    
    elif color_space == 'HSV':

        hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames]
        h_frames = [frame[:, :, 0] for frame in hsv_frames]
        s_frames = [frame[:, :, 1] for frame in hsv_frames]
        v_frames = [frame[:, :, 2] for frame in hsv_frames]

        h = np.array([np.histogram(frame, bins=180, range=(0, 180))[0] for frame in h_frames])
        s = np.array([np.histogram(frame, bins=256, range=(0, 256))[0] for frame in s_frames])
        v = np.array([np.histogram(frame, bins=256, range=(0, 256))[0] for frame in v_frames])

        h = normalize_histogram(h)
        s = normalize_histogram(s)
        v = normalize_histogram(v)

        h_hist_diff = np.diff(h, axis=0)
        s_hist_diff = np.diff(s, axis=0)
        v_hist_diff = np.diff(v, axis=0)

        return h_hist_diff, s_hist_diff, v_hist_diff

    elif color_space == 'GRAY':

        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        gray_histogram = np.array([np.histogram(frame, bins=num_bins, range=(0, 256))[0] for frame in gray_frames])
        gray_histogram = normalize_histogram(gray_histogram)
        gray_histogram_diff = np.diff(gray_histogram, axis=0)

        return gray_histogram_diff

#Histogram Distance measurement
def histogram_chi_sqrd_distances(frames: np.ndarray, color_space: str = 'RGB') -> np.ndarray:
    if color_space == 'RGB':
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        red_frames = [frame[:, :, 0] for frame in rgb_frames]
        green_frames = [frame[:, :, 1] for frame in rgb_frames]
        blue_frames = [frame[:, :, 2] for frame in rgb_frames]

        red_histogram = np.array([np.histogram(frame, bins=256, range=(0, 256))[0] for frame in red_frames])
        green_histogram = np.array([np.histogram(frame, bins=256, range=(0, 256))[0] for frame in green_frames])
        blue_histogram = np.array([np.histogram(frame, bins=256, range=(0, 256))[0] for frame in blue_frames])

        red_histogram = normalize_histogram(red_histogram)
        green_histogram = normalize_histogram(green_histogram)
        blue_histogram = normalize_histogram(blue_histogram)

        redchi_sqrd_distances = np.array([0.5 * np.sum(((red_histogram[i] - red_histogram[i - 1]) ** 2) / (red_histogram[i] + red_histogram[i - 1] + 1e-10)) for i in range(1, len(red_histogram))])
        greenchi_sqrd_distances = np.array([0.5 * np.sum(((green_histogram[i] - green_histogram[i - 1]) ** 2) / (green_histogram[i] + green_histogram[i - 1] + 1e-10)) for i in range(1, len(green_histogram))])
        bluechi_sqrd_distances = np.array([0.5 * np.sum(((blue_histogram[i] - blue_histogram[i - 1]) ** 2) / (blue_histogram[i] + blue_histogram[i - 1] + 1e-10)) for i in range(1, len(blue_histogram))])

        return redchi_sqrd_distances, greenchi_sqrd_distances, bluechi_sqrd_distances

    if color_space == 'HSV':
        hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames]
        h_frames = [frame[:, :, 0] for frame in hsv_frames]
        s_frames = [frame[:, :, 1] for frame in hsv_frames]
        v_frames = [frame[:, :, 2] for frame in hsv_frames]

        h_histogram = np.array([np.histogram(frame, bins=180, range=(0, 180))[0] for frame in h_frames])
        s_histogram = np.array([np.histogram(frame, bins=256, range=(0, 256))[0] for frame in s_frames])
        v_histogram = np.array([np.histogram(frame, bins=256, range=(0, 256))[0] for frame in v_frames])

        h_histogram = normalize_histogram(h_histogram)
        s_histogram = normalize_histogram(s_histogram)
        v_histogram = normalize_histogram(v_histogram)

        hchi_sqrd_distances = np.array([0.5 * np.sum(((h_histogram[i] - h_histogram[i - 1]) ** 2) / (h_histogram[i] + h_histogram[i - 1] + 1e-10)) for i in range(1, len(h_histogram))])
        schi_sqrd_distances = np.array([0.5 * np.sum(((s_histogram[i] - s_histogram[i - 1]) ** 2) / (s_histogram[i] + s_histogram[i - 1] + 1e-10)) for i in range(1, len(s_histogram))])
        vchi_sqrd_distances = np.array([0.5 * np.sum(((v_histogram[i] - v_histogram[i - 1]) ** 2) / (v_histogram[i] + v_histogram[i - 1] + 1e-10)) for i in range(1, len(v_histogram))])

        return hchi_sqrd_distances, schi_sqrd_distances, vchi_sqrd_distances

#Baseline
def absolute_mean_difference_between_frames(frames: list, color_space: str = 'GRAY') -> np.ndarray:
    if color_space == 'GRAY':
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        mean_diff = np.array([np.mean(np.abs(gray_frames[i] - gray_frames[i - 1])) for i in range(1, len(gray_frames))])
        return mean_diff
    elif color_space == 'RGB':
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        mean_diff = np.array([np.mean(np.abs(rgb_frames[i] - rgb_frames[i - 1])) for i in range(1, len(rgb_frames))])
        return mean_diff
    elif color_space == 'HSV':
        hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames]
        mean_diff = np.array([np.mean(np.abs(hsv_frames[i] - hsv_frames[i - 1])) for i in range(1, len(hsv_frames))])
        return mean_diff

#Edge Change Ratio
def edge_change_ratio(
    frames: list,
    kernel_size: int = 5
) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    if len(frames) < 2:
        return np.empty(0, dtype=np.float64)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    gray_frames = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for frame in frames
    ]
    edge_maps = [
        cv2.Canny(frame, threshold1=100, threshold2=200).astype(bool)
        for frame in gray_frames
    ]

    edge_change_ratios = []

    for previous_edges, current_edges in zip(edge_maps, edge_maps[1:]):
        previous_dilated = cv2.dilate(
            previous_edges.astype(np.uint8),
            kernel
        ).astype(bool)
        current_dilated = cv2.dilate(
            current_edges.astype(np.uint8),
            kernel
        ).astype(bool)

        previous_count = np.sum(previous_edges)
        current_count = np.sum(current_edges)

        entering_count = np.sum(current_edges & ~previous_dilated)
        exiting_count = np.sum(previous_edges & ~current_dilated)

        entering_ratio = (
            entering_count / current_count
            if current_count > 0
            else 0.0
        )
        exiting_ratio = (
            exiting_count / previous_count
            if previous_count > 0
            else 0.0
        )

        edge_change_ratios.append(
            max(entering_ratio, exiting_ratio)
        )

    return np.asarray(edge_change_ratios, dtype=np.float64)
    