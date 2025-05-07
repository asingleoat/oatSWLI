#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import time
import torch

# Import our refactored modules
from tensor_utils import (
    cross_correlate, make_reference, make_reference_avg, 
    process_chunks_fixed_size, fps, frame_factor, nmps
)
from image_processing import (
    load_video, normalize_and_save, remove_tilt_grayscale, 
    smooth_outliers
)
from visualization import create_3d_surface_plot, setup_interactive_plots

def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Load a video file and process its FFT/IFT.")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--plot2d", action="store_true", help="Show interactive 2D plots")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size for processing")
    args = parser.parse_args()

    # Load and prepare video
    crop_region = (0, 200, 0, 200) if False else None
    video_array_BGR = load_video(args.video_file, crop_region, rgb=True, frame_factor=frame_factor)
    video_array = video_array_BGR[:,:,1:,:] # drop blue channel
    print(f"Video shape: {video_array.shape}")

    # Create reference chirp
    h, w, c, t = video_array.shape
    x, y = h // 2, w // 2  # point to extract reference chirp from
    reference_chirp = make_reference_avg(video_array, x, y, window=50, use_gpu=args.gpu)

    # Process video in chunks
    chunk_h = chunk_w = args.chunk_size
    (max_indices, ift_result) = process_chunks_fixed_size(
        video_array, 
        lambda v: cross_correlate(reference_chirp, v, use_gpu=args.gpu), 
        chunk_h=chunk_h, 
        chunk_w=chunk_w
    )

    # Post-process height map
    max_indices = remove_tilt_grayscale(max_indices)
    
    # Convert to physical units
    to_nm_factor = frame_factor * nmps / fps
    max_indices *= to_nm_factor

    # Save results
    np.save('array.npy', max_indices)    
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
