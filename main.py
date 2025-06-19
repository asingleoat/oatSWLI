#!/usr/bin/env python3

import sys
import argparse
import os
import time
import ffmpeg
import numpy as np

from primitives import (
    align_multi,
    align_multi_p,
    normalize,
)


from image_processing import (
    # load_video,
    # normalize_and_save,
    remove_tilt_grayscale,
    # smooth_outliers,
    # get_video_metadata,
)

from visualization import create_3d_surface_plot, setup_interactive_plots


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="SWLI video processor"
    )
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument(
        "--plot3d", action="store_true", help="Show interactive 3D plots"
    )

    args = parser.parse_args()
    video = load_video_frames(args.video_file, gray=True);
    print(video.shape)

    max_indices = remove_tilt_grayscale(align_multi_p(video[1,1,0,:], video[:,:,0,:], max_dev=400, band=(0.005, 0.25)))

    # Create and display 3D visualization
    if args.plot3d:
        print("Will render")
        fig = create_3d_surface_plot(max_indices) # test render first frame
        fig.show()
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


    
def load_video_frames(path, resize=None, limit=None, gray=False):
    """
    Load video into NumPy array.
    Returns: shape [T, H, W, C] (RGB) or [T, H, W] (gray)
    - pix_fmt: 'rgb24', 'gray', etc.
    - resize: (width, height) or None
    - limit: max number of frames
    """
    pix_fmt = 'gray' if gray else 'rgb24'

    probe = ffmpeg.probe(path)
    width = int(probe['streams'][0]['width'])
    height = int(probe['streams'][0]['height'])
    if resize:
        width, height = resize

    cmd = (
        ffmpeg
        .input(path)
        .filter('scale', width, height) if resize else ffmpeg.input(path)
    )
    if limit:
        cmd = cmd.output('pipe:', format='rawvideo', pix_fmt=pix_fmt, vframes=limit)
    else:
        cmd = cmd.output('pipe:', format='rawvideo', pix_fmt=pix_fmt)

    out, _ = cmd.run(capture_stdout=True, capture_stderr=True)
    num_channels = {'rgb24': 3, 'gray': 1}[pix_fmt]
    frame_size = width * height * num_channels
    total_frames = len(out) // frame_size
    arr = np.frombuffer(out, np.uint8).reshape((total_frames, height, width, num_channels))
    arr = np.transpose(arr, (2, 1, 3, 0))  # [T, H, W, C] → [W, H, C, T]
    return arr.astype(np.float32) # TODO: do I need this cast?

if __name__ == "__main__":
    main()

    
