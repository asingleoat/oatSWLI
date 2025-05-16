#!/usr/bin/env python3

import asyncio
import torch
import torch.nn.functional as F
import numpy as np
import math

# async def gpu_worker(chunk, semaphore, output_queue):
#     async with semaphore:
#         result = await process_on_gpu(chunk)
#         await output_queue.put(result)


async def gpu_worker(input_queue, output_queue, semaphore):
    while True:
        chunk = await input_queue.get()
        if chunk is None:
            input_queue.task_done()
            break

        async with semaphore:
            try:
                result = await process_on_gpu(chunk)
                await output_queue.put(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("OOM — requeuing chunk")
                    await input_queue.put(chunk)
                    torch.cuda.empty_cache()
                    await asyncio.sleep(0.1)  # backoff
                else:
                    raise  # propagate other errors

        input_queue.task_done()


def get_available_vram(device=0):
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    return total - reserved


def estimate_chunk_vram_usage(chunk_shape, dtype=torch.float64):
    return math.prod(chunk_shape) * dtype.itemsize


def gaussian_smooth_time(array, sigma, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    if radius <= 0:
        return array

    kernel_size = 2 * radius + 1
    t = torch.arange(-radius, radius + 1, dtype=array.dtype, device=array.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    kernel = kernel.to(array.dtype)
    kernel1d = kernel.view(1, 1, kernel_size)

    h, w, c, t = array.shape
    flat = array.view(-1, 1, t)  # shape: (h*w*c, 1, t)

    padded = F.pad(flat, (radius, radius), mode="reflect")

    smoothed_flat = F.conv1d(padded, kernel1d)

    smoothed = smoothed_flat.view(h, w, c, t)
    return smoothed


def cross_correlate(reference_chirp, video_array, use_gpu=True):
    """
    Cross-correlate reference chirp with video array using FFT.

    Uses the convolution theorem to compute cross-correlations in O(n log n) time
    instead of O(n^2) time by transforming to frequency space.
    """
    with torch.no_grad():
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        video_torch = torch.tensor(video_array, device=device, dtype=torch.complex64)
        # print(f"Tensor is on device: {video_torch.device}")

        fft_result = torch.fft.fftshift(torch.fft.fft(video_torch, dim=-1), dim=-1)

        # Remove DC component for better numerics
        fft_result = remove_dc(fft_result)

        fft_conjugate = torch.conj(fft_result)

        # Pointwise product in frequency domain
        products = reference_chirp[None, None, :] * fft_conjugate

        ift_result = torch.fft.ifftshift(
            torch.fft.ifft(torch.fft.ifftshift(products, dim=-1), dim=-1).real, dim=-1
        )

        # Find subpixel peaks for precise alignment
        max_indices = quadratic_subpixel_peak(ift_result)

        return (
            max_indices.cpu().numpy() if use_gpu else max_indices.numpy(),
            ift_result.cpu().numpy() if use_gpu else ift_result.numpy(),
        )


def remove_dc(array):
    """Remove DC component (zero frequency) from FFT result."""
    t = array.shape[-1]
    # symmetric fft has the dc component in the center
    center_idx = t // 2

    # mask across dc component of all points
    array[..., center_idx] = 0

    return array


def quadratic_subpixel_peak(array):
    """
    Find subpixel peak locations by fitting quadratics to local maxima.
    """
    # Smooth the array to reduce noise
    smoothed = gaussian_smooth_time(array, 20 * (fps / nmps) / frame_factor)

    # Take product over color channels of cross correlations
    smoothed = smoothed.prod(axis=-2)

    h, w, t = smoothed.shape

    flat = smoothed.view(-1, t)  # (h*w*c, t)
    idx = torch.argmax(flat, dim=1)  # (h*w*c,)

    # Clamp indices to avoid edge cases
    idx_clamped = idx.clamp(1, t - 2)

    i = torch.arange(flat.size(0), device=array.device)
    y0 = flat[i, idx_clamped - 1]
    y1 = flat[i, idx_clamped]
    y2 = flat[i, idx_clamped + 1]

    # Quadratic fit coefficients
    a = 0.5 * (y0 + y2 - 2 * y1)
    b = 0.5 * (y2 - y0)

    # Calculate subpixel offset
    subpixel_offset = torch.where(a != 0, -b / (2 * a), torch.zeros_like(a))
    subpixel_argmax = idx_clamped.float() + subpixel_offset

    return subpixel_argmax.view(h, w)


def make_reference(video_array, x, y, use_gpu=True):
    """Create reference chirp from a single point in the video."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    video_torch = torch.tensor(
        video_array[x, y, :], device=device, dtype=torch.complex64
    )
    print(f"Reference tensor is on device: {video_torch.device}")
    fft_result = torch.fft.fft(video_torch, dim=-1)
    return fft_result


def make_reference_avg(video_array, window=50, use_gpu=True):
    """Create reference chirp by averaging over a window of points."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    h, w, c, t = video_array.shape
    result = torch.tensor(np.zeros((c, t)), device=device, dtype=torch.complex64)
    with torch.no_grad():
        for i in range(window):
            for j in range(window):
                for channel in range(c):
                    video_torch = torch.tensor(
                        video_array[i, j, channel, :],
                        device=device,
                        dtype=torch.complex64,
                    )
                    fft_result = torch.fft.fft(video_torch, dim=-1)

                    result[channel] = fft_result + result[channel]

    for channel in range(c):
        result[channel] = torch.fft.fftshift(result[channel], dim=-1)
        result[channel] = remove_dc(result[channel])
        result[channel] = result[channel] / torch.norm(result[channel], p=2)
    return result


def process_chunks_fixed_size(
    shape, process_func, get_chunk_data, chunk_h=128, chunk_w=128
):
    """
    Process array in chunks to reduce memory usage.

    This allows processing large arrays that wouldn't fit in GPU memory all at once.
    Each pixel alignment can be computed independently (embarrassingly parallel).
    Args:
        shape: Full array dimensions (h, w, color, depth)
        process_func: Function to process each chunk
        get_chunk_data: Function to retrieve chunk data given chunk coordinates
        chunk_h: Height of processing chunks
        chunk_w: Width of processing chunks

    Returns:
        Processed results covering entire shape
    """
    h, w, color, depth = shape
    result_left = np.zeros((h, w))
    result_right = np.zeros((h, w, color, depth))

    chunk_counter = 0
    for y_start in range(0, h, chunk_h):
        for x_start in range(0, w, chunk_w):
            y_end = min(y_start + chunk_h, h)
            x_end = min(x_start + chunk_w, w)

            chunk_counter += 1
            print(f"Processing chunk: {chunk_counter}")
            # Retrieve chunk data using get_chunk_data
            chunk = get_chunk_data(x_start, x_end, y_start, y_end)

            # Process chunk using original process_func
            (processed_chunk_left, processed_chunk_right) = process_func(chunk)

            result_left[y_start:y_end, x_start:x_end] = processed_chunk_left
            result_right[y_start:y_end, x_start:x_end, :, :] = processed_chunk_right

    return (result_left, result_right)


# Global parameters used by functions
fps = 100
frame_factor = 1
nmps = 500
