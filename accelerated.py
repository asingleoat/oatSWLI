#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import time
import torch

# Import our refactored modules
from tensor_utils import (
    cross_correlate,
    make_reference,
    make_reference_avg,
    process_chunks_fixed_size,
    fps,
    frame_factor,
    nmps,
)
from image_processing import (
    load_video,
    normalize_and_save,
    remove_tilt_grayscale,
    smooth_outliers,
    get_video_metadata,
)
from visualization import create_3d_surface_plot, setup_interactive_plots


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Load a video file and process its FFT/IFT."
    )
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU acceleration if available"
    )
    parser.add_argument(
        "--plot2d", action="store_true", help="Show interactive 2D plots"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=32, help="Chunk size for processing"
    )
    args = parser.parse_args()

    metadata = get_video_metadata(args.video_file)
    print(metadata)
    shape = metadata["shape"]

    if shape[2] == 3:
        shape = shape[0], shape[1], 2, shape[3]
        # drop blue, TODO, deal with noisy blue in a more principled manner

    crop = False
    # crop = True
    crop_region = (0, 200, 0, 200) if crop else None

    # Create reference chirp
    h, w, c, t = shape
    # point to extract reference chirp from
    x, y = (
        ((crop_region[0] + crop_region[1]) // 2, (crop_region[2] + crop_region[3]) // 2)
        if crop_region
        else (h // 2, w // 2)
    )
    chirp_window = 50  # tweak me, see TODO, not currently centered :O
    print(x, y, "check")
    video_array_chirp = load_video(
        args.video_file,
        (x, x + chirp_window, y, y + chirp_window),
        rgb=True,
        frame_factor=frame_factor,
    )
    video_array_chirp = video_array_chirp[:, :, 1:, :]  # drop blue channel

    reference_chirp = make_reference_avg(
        video_array_chirp, window=chirp_window, use_gpu=args.gpu
    )

    # Process video in chunks
    chunk_h = chunk_w = args.chunk_size
    (max_indices, ift_result) = process_chunks_fixed_size(
        shape,
        lambda v: cross_correlate(reference_chirp, v, use_gpu=args.gpu),
        lambda x_start, x_end, y_start, y_end: (
            load_video(
                args.video_file,
                (x_start, x_end, y_start, y_end),
                rgb=True,
                frame_factor=frame_factor,
            )
        )[
            :, :, 1:, :
        ],  # drop blue channel
        chunk_h=chunk_h,
        chunk_w=chunk_w,
    )

    # Post-process height map
    max_indices = remove_tilt_grayscale(max_indices)

    # Convert to physical units
    to_nm_factor = frame_factor * nmps / fps
    max_indices *= to_nm_factor

    # Save results
    np.save("array.npy", max_indices)
    normalize_and_save(max_indices, "out.png")

    # Report timing
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    # Create and display 3D visualization
    fig = create_3d_surface_plot(max_indices)
    fig.show()

    # Optionally show interactive 2D plots
    if args.plot2d:
        setup_interactive_plots(video_array_BGR, ift_result, reference_chirp)
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
