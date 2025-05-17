#!/usr/bin/env python3

import asyncio
import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import concurrent.futures
from image_processing import (
    load_video,
    )
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


def cross_correlate(reference_chirp, video_array, use_gpu=True, raw_data=True):
    """
    Cross-correlate reference chirp with video array using FFT.

    Uses the convolution theorem to compute cross-correlations in O(n log n) time
    instead of O(n^2) time by transforming to frequency space.
    """
    with torch.no_grad():
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        video_torch = torch.tensor(video_array, device=device, dtype=torch.complex64)
        print(f"Tensor is on device: {video_torch.device}")

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

        if raw_data:
            return (
                max_indices.cpu().numpy() if use_gpu else max_indices.numpy(),
                ift_result.cpu().numpy() if use_gpu else ift_result.numpy(),
            )
        else:
            return (
                max_indices.cpu().numpy() if use_gpu else max_indices.numpy(),
                None,
            )


def remove_dc(array):
    """Remove DC component (zero frequency) from FFT result."""
    t = array.shape[-1]
    # symmetric fft has the dc component in the center
    center_idx = t // 2

    # mask across dc component of all points
    array[..., center_idx] = 0

    return array


def quadratic_subpixel_peak(array, use_gpu=True):
    """
    Find subpixel peak locations by fitting quadratics to local maxima.
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Smooth the array to reduce noise
    # smoothed = gaussian_smooth_time(array, 20 * (fps / nmps) / frame_factor)

    # Take sum of cubes over color channels of cross correlations
    cubed = (array**3)
    weights = torch.tensor([0.3, 1, 1], device=device)
    weights = weights.view(1, 1, -1, 1)  # reshape for broadcasting
    smoothed = (cubed * weights).sum(dim=2)
        
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

def estimate_cosine_peak_torch(array, device='cuda'):
    """
    Fit a cosine curve to x, y using PyTorch and return peak location (-phi / omega).
    
    Parameters:
        x (np.ndarray or torch.Tensor): 1D array of x values
        y (np.ndarray or torch.Tensor): 1D array of y values
        device (str): 'cuda' or 'cpu'
    
    Returns:
        peak_x (float): x-position of cosine peak
        fit_params (dict): dict with keys A, omega, phi, c
    """
    x = torch.as_tensor(array, dtype=torch.float32, device=device)
    y = torch.as_tensor(np.arange(array.shape[0]), dtype=torch.float32, device=device)

    # Init parameters as torch tensors with gradients
    A     = torch.tensor((y.max() - y.min()) / 2, device=device, requires_grad=True)
    omega = torch.tensor(2 * torch.pi / (x[-1] - x[0]), device=device, requires_grad=True)
    phi   = torch.tensor(0.0, device=device, requires_grad=True)
    c     = torch.tensor(y.mean(), device=device, requires_grad=True)

    params = [A, omega, phi, c]

    # Define model
    def model(x):
        return A * torch.cos(omega * x + phi) + c

    # Define closure for L-BFGS
    optimizer = torch.optim.LBFGS(params, max_iter=1000, tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((model(x) - y)**2)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Final values
    with torch.no_grad():
        peak_x = (-phi / omega).item()
        fit_params = {
            "A": A.item(),
            "omega": omega.item(),
            "phi": phi.item(),
            "c": c.item()
        }

    return peak_x, fit_params


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


async def process_chunks_fixed_size(
    shape,
    reference_chirp,
    filename,
    chunk_h=128,
    chunk_w=128,
    max_concurrent=4,
    max_workers=4,
    use_gpu=True,
    raw_data=True,
):
    """
    Asynchronously process array in chunks with bounded parallelism.

    Args:
        shape: Full array dimensions (h, w, color, depth)
        process_func: Function to process each chunk
        get_chunk_data: Function to retrieve chunk data given chunk coordinates
        chunk_h: Height of processing chunks
        chunk_w: Width of processing chunks
        max_concurrent: Maximum number of concurrent chunk processing tasks
        max_workers: Maximum number of CPU workers for chunk loading

    Returns:
        asyncio.Queue containing chunk results with positional metadata
    """
    h, w, color, depth = shape
    result_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Use all available CPU cores if not specified
    if max_workers is None:
        max_workers = os.cpu_count()

    # Create ProcessPoolExecutor outside of a context manager
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    try:
        async def process_chunk(x_start, x_end, y_start, y_end):
            async with semaphore:
                try:
                    print(f"Running chunk: {x_start, x_end, y_start, y_end}")
                    # Parallelize chunk data loading
                    chunk_future = executor.submit(load_video, filename, (x_start, x_end, y_start, y_end))
                    chunk = await asyncio.wrap_future(chunk_future)

                    # Process the chunk
                    # processed_result = cross_correlate(reference_chirp, chunk[:, :, 1:, :], use_gpu)  # drop blue channel
                    processed_result = cross_correlate(reference_chirp, chunk, use_gpu, raw_data)  # drop blue channel

                    await result_queue.put(
                        {
                            "x_start": x_start,
                            "x_end": x_end,
                            "y_start": y_start,
                            "y_end": y_end,
                            "result_left": processed_result[0],
                            "result_right": processed_result[1],
                        }
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("OOM — requeuing chunk")
                        # await input_queue.put(chunk)
                        torch.cuda.empty_cache()
                        await asyncio.sleep(0.1)  # backoff
                    else:
                        raise  # propagate other errors

        # Create tasks for all chunks
        tasks = []
        chunk_counter = 0
        for y_start in range(0, h, chunk_h):
            for x_start in range(0, w, chunk_w):
                y_end = min(y_start + chunk_h, h)
                x_end = min(x_start + chunk_w, w)

                chunk_counter += 1
                print(f"Queuing chunk: {chunk_counter}")

                task = asyncio.create_task(process_chunk(x_start, x_end, y_start, y_end))
                tasks.append(task)

        # Wait for all chunks to be processed
        await asyncio.gather(*tasks)

        # Signal end of processing
        await result_queue.put(None)

    finally:
        # Ensure executor is shut down properly
        executor.shutdown(wait=True)

    return result_queue


# Global parameters used by functions
fps = 100
frame_factor = 1
nmps = 500
