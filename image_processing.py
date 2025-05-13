#!/usr/bin/env python3

import cv2
import numpy as np
import scipy.ndimage


def get_video_metadata(filename):
    cap = cv2.VideoCapture(filename)

    ret, frame = cap.read()
    channels = frame.shape[2] if ret else 0

    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "channels": channels,
    }

    cap.release()
    return metadata


def load_video(
    filename,
    crop_region=None,
    frame_start=0,
    frame_end=None,
    frame_factor=1,
    rgb=True,
):
    """
    Load a specific rectangular region from a video file.

    Args:
        filename: Path to video file
        x_start, x_end: Horizontal pixel range to extract
        y_start, y_end: Vertical pixel range to extract
        frame_start: Starting frame index
        frame_end: Ending frame index (None means entire video)
        frame_factor: Only keep every frame_factor-th frame
        rgb: Convert to RGB if True, else grayscale

    Returns:
        Numpy array with shape (height, width, channel, time)
    """
    cap = cv2.VideoCapture(filename)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_end = frame_end or total_frames

    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame_count >= frame_end - frame_start:
            break

        # decimate
        if frame_count % frame_factor == 0:
            if crop_region:
                x_start, x_end, y_start, y_end = crop_region
                frame = frame[y_start:y_end, x_start:x_end]

            # should drop this soon. we should always use color data when available
            # and greyscale only when the source is greyscale, so no conversions
            if not rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cropped_frame[:, :, np.newaxis]

            frames.append(frame)

        frame_count += 1

    cap.release()

    return np.stack(frames, axis=-1) if frames else None


def normalize_and_save(image_array, output_filename):
    """
    Normalizes an image to the 0-65535 range and saves it as a 16-bit image.

    Args:
        image_array: Input image array
        output_filename: Path to save the normalized image
    """
    # Normalize to 0-65535 (16-bit)
    image_array = image_array - image_array.min()  # Shift min to 0
    image_array = (image_array / image_array.max()) * 2**16

    # Convert to uint16
    image_array = image_array.astype(np.uint16)

    # Save using OpenCV
    cv2.imwrite(output_filename, image_array)


def remove_tilt(max_indices):
    """
    Remove planar tilt from 3D data by fitting and subtracting a plane.

    Args:
        max_indices: 3D array with shape (height, width, channels)

    Returns:
        Untilted 3D array with same shape
    """
    h, w, c = max_indices.shape

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    A = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)

    coeffs_sum = np.zeros(3)
    for i in range(c):
        z = max_indices[:, :, i].reshape(-1)
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        coeffs_sum += coeffs

    avg_coeffs = coeffs_sum / c
    plane = avg_coeffs[0] * x + avg_coeffs[1] * y + avg_coeffs[2]

    untilted = max_indices - plane[..., None]

    return untilted


def remove_tilt_grayscale(img):
    """
    Remove planar tilt from 2D grayscale data by fitting and subtracting a plane.

    Args:
        img: 2D array

    Returns:
        Untilted 2D array
    """
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    A = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    z = img.reshape(-1)

    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]

    return img - plane


def smooth_outliers(image, threshold=1.0, filter_size=10):
    """
    Smooth outliers in an image by replacing them with local mean values.

    Args:
        image: Input image array
        threshold: Threshold for outlier detection (in standard deviations)
        filter_size: Size of the local neighborhood for mean calculation

    Returns:
        Smoothed image array
    """
    local_mean = scipy.ndimage.uniform_filter(image, size=filter_size)
    local_var = scipy.ndimage.uniform_filter(
        (image - local_mean) ** 2, size=filter_size
    )
    local_std = np.sqrt(np.clip(local_var, 1e-8, None))

    outliers = np.abs(image - local_mean) > (threshold * local_std)

    smoothed = image.copy()
    smoothed[outliers] = local_mean[outliers]

    return smoothed
