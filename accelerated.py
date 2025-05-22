#!/usr/bin/env python3

import asyncio
import sys
import argparse
import numpy as np
import time
import torch
import os
import psutil

# Import our refactored modules
from tensor_utils import (
    cross_correlate,
    make_reference,
    make_reference_avg,
    process_chunks_fixed_size,
    fps,
    frame_factor,
    nmps,
    get_available_vram,
    gpu_worker,
    estimate_chunk_vram_usage,
)
from image_processing import (
    load_video,
    normalize_and_save,
    remove_tilt_grayscale,
    smooth_outliers,
    get_video_metadata,
)
from visualization import create_3d_surface_plot, setup_interactive_plots


async def main():
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
        "--plot3d", action="store_true", help="Show interactive 3D plots"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=64, help="Chunk size for processing"
    )
    parser.add_argument(
        "--peak-method", 
        choices=["quadratic", "wavelet"], 
        default="quadratic",
        help="Method for subpixel peak detection"
    )
    args = parser.parse_args()

    metadata = get_video_metadata(args.video_file)
    print(metadata)
    shape = metadata["shape"]

    # if shape[2] == 3:
    # drop blue, TODO, deal with noisy blue in a more principled manner
    # shape = shape[0], shape[1], 2, shape[3]

    h, w, c, t = shape
    # crop = False
    crop = True
    # TODO: take in crop params from the CLI
    # may be the whole field, the trivial crop
    crop_region = (0, 128, 0, 128) if crop else (0, h, 0, w)
    shape = (crop_region[1], crop_region[3], c, t)
    chunk_h = chunk_w = args.chunk_size

    max_concurrent = None
    if args.gpu:
        available_vram = get_available_vram()
        chunk_bytes = estimate_chunk_vram_usage((chunk_h, chunk_w, c, t))
        max_concurrent = max(1, available_vram // chunk_bytes)
        chunks_to_process = ((h + chunk_h - 1) // chunk_h) * (
            (w + chunk_w - 1) // chunk_w
        )
        print(available_vram, chunk_bytes, max_concurrent, chunks_to_process)
    else:
        available_sysram = psutil.virtual_memory().available
        chunk_bytes = estimate_chunk_vram_usage((chunk_h, chunk_w, c, t))
        max_concurrent = max(1, available_sysram // chunk_bytes) // 4
        chunks_to_process = ((h + chunk_h - 1) // chunk_h) * (
            (w + chunk_w - 1) // chunk_w
        )
        print(
            available_sysram / (1024**2),
            chunk_bytes / (1024**2),
            max_concurrent,
            chunks_to_process,
        )

    # Create reference chirp
    # point to extract reference chirp from
    x, y = (crop_region[0] + crop_region[1]) // 2, (
        crop_region[2] + crop_region[3]
    ) // 2

    chirp_window = 50  # see TODO, not currently centered :O
    video_array_chirp = load_video(
        args.video_file,
        (x, x + chirp_window, y, y + chirp_window),
        rgb=True,
        frame_factor=frame_factor,
    )
    # video_array_chirp = video_array_chirp[:, :, 1:, :]  # drop blue channel

    reference_chirp = make_reference_avg(
        video_array_chirp, window=chirp_window, use_gpu=args.gpu
    )
    print("Shape of reference chirp: ", reference_chirp.shape)
    # # Process video in chunks
    # if args.gpu:
    #     max_concurrent = 4  # max(1, available_vram // chunk_bytes)
    # else:
    #     max_concurrent = os.cpu_count()  # Default for CPU processing
    max_concurrent = 1
    result_queue = await process_chunks_fixed_size(
        shape,
        reference_chirp,
        args.video_file,
        chunk_h=chunk_h,
        chunk_w=chunk_w,
        max_concurrent=max_concurrent,
        max_workers=max_concurrent,  # Use all available CPU cores
        use_gpu=args.gpu,
        raw_data=args.plot2d,
        peak_method=args.peak_method,
    )

    # Reassemble results
    max_indices = np.zeros((shape[0], shape[1]))
    if args.plot2d:
        ift_result = np.zeros((shape[0], shape[1], shape[2], shape[3]))

    while True:
        chunk = await result_queue.get()
        if chunk is None:
            break

        max_indices[
            chunk["y_start"] : chunk["y_end"], chunk["x_start"] : chunk["x_end"]
        ] = chunk["result_left"]

        if args.plot2d:
            ift_result[
                chunk["y_start"] : chunk["y_end"],
                chunk["x_start"] : chunk["x_end"],
                :,
                :,
            ] = chunk["result_right"]

        result_queue.task_done()

    # Initialize frequencies variable
    frequencies = None
    
    # Reassemble results
    max_indices = np.zeros((shape[0], shape[1]))
    if args.plot2d:
        ift_result = np.zeros((shape[0], shape[1], shape[2], shape[3]))

    while True:
        chunk = await result_queue.get()
        if chunk is None:
            break

        max_indices[
            chunk["y_start"] : chunk["y_end"], chunk["x_start"] : chunk["x_end"]
        ] = chunk["result_left"]

        # Capture frequencies if using wavelet method
        if args.peak_method == "wavelet" and "frequencies" in chunk and chunk["frequencies"] is not None:
            frequencies = chunk["frequencies"]

        if args.plot2d:
            ift_result[
                chunk["y_start"] : chunk["y_end"],
                chunk["x_start"] : chunk["x_end"],
                :,
                :,
            ] = chunk["result_right"]

        result_queue.task_done()

    # Post-process height map
    max_indices = remove_tilt_grayscale(max_indices)

    # Convert to physical units
    to_nm_factor = frame_factor * nmps / fps
    max_indices *= to_nm_factor

    # Save results
    np.save("array.npy", max_indices)
    normalize_and_save(max_indices, "out.png")
    
    # Save frequency information if using wavelet method
    if args.peak_method == "wavelet" and frequencies is not None:
        np.save("estimated_frequencies.npy", frequencies.cpu().numpy() if hasattr(frequencies, 'cpu') else frequencies)

    # Report timing
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    # Create and display 3D visualization
    if args.plot3d:
        fig = create_3d_surface_plot(max_indices)
        fig.show()

    # Optionally show interactive 2D plots
    if args.plot2d:
        setup_interactive_plots(metadata, ift_result, reference_chirp, crop_region)
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    asyncio.run(main())
