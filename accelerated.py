#!/usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy
import time
import torch


start_time = time.perf_counter()

# print("CUDA available:", torch.cuda.is_available())
# print("Number of CUDA devices:", torch.cuda.device_count())
# print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
# print("PyTorch built with CUDA:", torch.backends.cuda.is_built())

def load_video(filename, crop_region=None):
    cap = cv2.VideoCapture(filename)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # convert to grayscale, TODO: don't do this
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if crop_region:
            x_start, x_end, y_start, y_end = crop_region
            gray_frame = gray_frame[x_start:x_end, y_start:y_end]  

        frames.append(gray_frame)
    
    cap.release()
    
    # shape: (height, width, time)
    return np.stack(frames, axis=-1)

def cross_correlate(reference_chirp, video_array, use_gpu=True):
    """
    we use a corollary of the convolution theorem (https://en.wikipedia.org/wiki/Convolution_theorem):
    {\mathcal {F}}\left\{f\star g\right\}={\overline {{\mathcal {F}}\left\{f\right\}}}\cdot {\mathcal {F}}\left\{g\right\}
    which states that the Fourier transform of the cross-correlation of two sequences is equal to the pointwise product
    if the conjugate of the Fourier transform of one of the sequences with the Fourier transform of the other.
  
    the upshot of this is that we can compute cross-correlations in O(n log n) time instead of
    O(n^2) time, we just have to shove everything into frequency space first (take ffts) and then
    yank it back out (take iffts)
    """
    # supposedly no_grad reduces VRAM usage, haven't benched
    with torch.no_grad():
      device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
  
      video_torch = torch.tensor(video_array, device=device, dtype=torch.complex64)
      print(f"Tensor is on device: {video_torch.device}") 
  
      fft_result = torch.fft.fftshift(torch.fft.fft(video_torch, dim=-1), dim=-1)
  
      # drop the absolute offset, we compute relative depths for better numerics, absolute frame
      # number is nonsense anyway so the depths are already relative
      fft_result = remove_dc(fft_result)
  
      fft_conjugate = torch.conj(fft_result)
  
      reference_chirp = reference_chirp
  
      # here's pointwise product
      products = reference_chirp[None, None, :] * fft_conjugate
  
      ift_result = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(products, dim=-1), dim=-1).real, dim=-1)
  
      # argmax gives us the index of best alignment, we can do a naive integer valued argmax, or try
      # to fit a curve for subpixel alignment
      # max_indices = torch.argmax(ift_result, dim=-1)
      # max_indices = subpixel_argmax_3d_torch(ift_result, dim=-1)
      max_indices = quadratic_subpixel_peak_3d_torch(ift_result, dim=-1)
      
      return (max_indices.cpu().numpy() if use_gpu else max_indices.numpy(), ift_result.cpu().numpy() if use_gpu else ift_result.numpy())

def remove_dc(array):
    t = array.shape[-1]
    # symmetric fft has the dc component in the center
    center_idx = t // 2

    # mask across dc component of all points
    array[..., center_idx] = 0

    return array

def quadratic_subpixel_peak_torch(arr):
    # the cross-correlation is in general a complicated chirp similar to the intensity curve of the
    # pixel. so we don't want to try to curve fit broadly. instead we find the general location of
    # the subpixel maximum by finding the pixel maximum, then curve fit only locally to find the
    # subpixel max

    argmax_idx = torch.argmax(arr)
    
    # in the literal edge case, we can't fit a quadratic
    if argmax_idx == 0 or argmax_idx == arr.shape[0] - 1:
        return argmax_idx.float()

    # we currently only look one frame before and after, it's likely that the true maximum is not within
    # one frame of the integer argmax and we should look further out, maybe a dozen frames in each direction
    # TODO: that ^
    x = torch.tensor([argmax_idx - 1, argmax_idx, argmax_idx + 1], dtype=torch.float32, device=arr.device)
    y = arr[x.long()]

    # fit a quadratic: y = ax^2 + bx + c
    # this is overkill at the moment with just three points per curve to fit, but supports taking more depth
    X = torch.stack([x**2, x, torch.ones_like(x)], dim=-1)
    # coefficients of the quadratic
    a, b, c = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze()

    # find subpixel peak as: x = -b / (2a)
    subpixel_max = -b / (2 * a)

    return subpixel_max

def quadratic_subpixel_peak_3d_torch(array, dim=-1):
    device = array.device
    h, w, t = array.shape
    
    # the cross-correlation is in general a complicated chirp similar to the intensity curve of the
    # pixel. so we don't want to try to curve fit broadly. instead we find the general location of
    # the subpixel maximum by finding the pixel maximum, then curve fit only locally to find the
    # subpixel max

    discrete_argmax = torch.argmax(array, dim=dim)  # Shape: (h, w)

    # in the literal edge case, we can't fit a quadratic so nudge those points in by one
    interior_argmax = torch.clamp(discrete_argmax, 1, t - 2)

    # all neighbours as arrays in parallel
    idx_left = interior_argmax - 1
    idx_right = interior_argmax + 1

    batch_indices = torch.arange(h, device=device).view(-1, 1).expand(-1, w)
    batch_indices_x = torch.arange(w, device=device).expand(h, -1)

    max_values = torch.abs(array[batch_indices, batch_indices_x, discrete_argmax])

    y0 = array[batch_indices, batch_indices_x, idx_left]
    y1 = array[batch_indices, batch_indices_x, interior_argmax]
    y2 = array[batch_indices, batch_indices_x, idx_right]

    # fit quadratic in closed form from three points
    a = 0.5 * (y0 + y2 - 2 * y1)
    b = 0.5 * (y2 - y0)


    # find subpixel peak as: x = -b / (2a)
    subpixel_max = interior_argmax + torch.where(a != 0, -b / (2 * a), torch.zeros_like(b))

    # underdeveloped thoughts about rejecting poorly fit pixels
    # threshold = 100
    # argmax_2d_filtered = torch.where(max_values <= threshold, subpixel_max, torch.zeros_like(subpixel_max))
    # return argmax_2d_filtered

    return subpixel_max

def normalize_and_save(image_array, output_filename):
    """Normalizes an image to the 0-255 range and saves it as a grayscale image."""
    
    # Normalize to 0-255
    image_array = image_array - image_array.min()  # Shift min to 0
    image_array = (image_array / image_array.max()) * 2**16
    
    # Convert to uint8
    image_array = image_array.astype(np.uint16)

    # Save using OpenCV
    cv2.imwrite(output_filename, image_array)

def remove_tilt(max_indices):
    h, w = max_indices.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    A = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    z = max_indices.reshape(-1)

    # fit plane
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)  # Solve for [a, b, c]

    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]

    untilted_max_indices = max_indices - plane

    return untilted_max_indices
    
def process_chunks_fixed_size(array, process_func, chunk_h=128, chunk_w=128):
    """
    our problem is embarrassingly parallel
    (https://en.wikipedia.org/wiki/Embarrassingly_parallel), each pixel alignment can be computed
    independently of each other one. we could dump the whole problem onto the GPU in one go, but
    that has very high GPU VRAM requirements, instead we push smaller chunks.

    we currently push one chunk at a time even if more than one would fit in VRAM, optimization TODO.
    """
    h, w, depth = array.shape
    result_left = np.zeros((h, w))
    result_right = np.zeros((h, w, depth))

    for y_start in range(0, h, chunk_h):
        for x_start in range(0, w, chunk_w):

            y_end = min(y_start + chunk_h, h)
            x_end = min(x_start + chunk_w, w)


            chunk = array[y_start:y_end, x_start:x_end, :]

            # the implementation isn't very flexible, we assume that each computation returns two
            # components: in practise these are the frame indicies (depths) and the pointwise
            # cross-correlation curves for inspection
            (processed_chunk_left, processed_chunk_right) = process_func(chunk)

            result_left[y_start:y_end, x_start:x_end] = processed_chunk_left
            result_right[y_start:y_end, x_start:x_end, :] = processed_chunk_right

    return (result_left, result_right)

def make_reference(video_array, x, y, use_gpu=True):
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    video_torch = torch.tensor(video_array[x,y,:], device=device, dtype=torch.complex64)
    print(video_torch.shape)
    print(f"Reference tensor is on device: {video_torch.device}") 
    fft_result = torch.fft.fft(video_torch, dim=-1)
    return fft_result

def make_reference_avg(video_array, x, y, window=100, use_gpu=True):
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    h,w,t = video_array.shape
    result = torch.tensor(np.zeros((t)), device=device, dtype=torch.complex64)
    with torch.no_grad():
        for i in range(window):
            for j in range(window):
                video_torch = torch.tensor(video_array[x+i,y+j,:], device=device, dtype=torch.complex64)
                fft_result = torch.fft.fft(video_torch, dim=-1)
                
                result = fft_result + result

    result = torch.fft.fftshift(result, dim=-1)
    result = remove_dc(result)
    result = result / torch.norm(result, p=2)
    return result


def smooth_outliers(image, threshold=1.0, filter_size=10):
    """
    somewhat unprincipled smoothing
    """
    local_mean = scipy.ndimage.uniform_filter(image, size=filter_size)
    local_var = scipy.ndimage.uniform_filter((image - local_mean) ** 2, size=filter_size)
    local_std = np.sqrt(np.clip(local_var, 1e-8, None))

    outliers = np.abs(image - local_mean) > (threshold * local_std)

    smoothed = image.copy()
    smoothed[outliers] = local_mean[outliers]

    return smoothed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a video file and process its FFT/IFT.")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    args = parser.parse_args()


    crop = False
    # crop = True
    crop_region = (0, 250, 0, 250) if crop else None

    video_array = load_video(args.video_file, crop_region)
    
    h, w, t = video_array.shape
    # point to extract reference chirp from
    x = h // 2
    y = w // 2
    # reference_chirp = make_reference(video_array, x, y, use_gpu=args.gpu)
    reference_chirp = make_reference_avg(video_array, x, y, window=100, use_gpu=args.gpu)

    # max_indices = cross_correlate(video_array, use_gpu=args.gpu) # one biig chunk
    (max_indices, ift_result) = process_chunks_fixed_size(video_array, lambda v: cross_correlate(reference_chirp, v, use_gpu=args.gpu)) # many medium chunks

    # three questionable data cleaning steps:
    max_indices = remove_tilt(max_indices)
    max_indices = smooth_outliers(max_indices)
    # max_indices = np.clip(max_indices, a_min=-50, a_max=50)

    normalize_and_save(max_indices, "out.png")

    # lazy benchmark
    end_time = time.perf_counter()
    print(f"Execution time upto image saving: {end_time - start_time:.6f} seconds")

    # matplotlib's 3d surface visualization is painfully slow, so we use plotly which is WebGL accelerated
    fig = go.Figure(data=[go.Surface(z=max_indices)])
    fig.update_layout(title=dict(text='Heightmap'), autosize=True,
                      # width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

    # still use matplotlib for interactive 2d plots because I wrote it first and haven't bother
    # replacing it with plotly plots yet
    # plot2d = False;
    plot2d = True;
    if plot2d:

        t_fixed = 0  # arbitrary choice of frame to display

        fig, ax = plt.subplots()
        img_display = ax.imshow(video_array[..., t_fixed], cmap='plasma')
        ax.set_title(f"Frame no:{t_fixed}")
        plt.colorbar(img_display)

        # Create a second figure for plotting the intensity over time
        fig_sequence, ax_sequence = plt.subplots()
        ax_sequence.set_title("Pixel Intensity")
        ax_sequence.set_xlabel("Time Frame")
        ax_sequence.set_ylabel("Intensity")

        reference_curve = torch.abs(torch.fft.ifft(reference_chirp, axis=-1)).cpu()
        reference_plot, = ax_sequence.plot(np.arange(reference_curve.shape[0]), reference_curve, label="Reference Curve", linestyle="dashed", color="red",)
        sequence_plot, = ax_sequence.plot([], [], label="Selected Pixel", color="blue")

        # click on a point in the 2d image to view the cross-correlation curve for that point and the reference chirp
        def on_click(event):
            # ignore clicks outside the image
            if event.xdata is None or event.ydata is None:
                return
            
            x, y = int(event.ydata), int(event.xdata)
        
            sequence = ift_result[x, y, :].real / 1000

            # Update plots
            sequence_plot.set_xdata(np.arange(sequence.shape[0]))
            sequence_plot.set_ydata(sequence)
            
            ax_sequence.relim()
            ax_sequence.autoscale_view()
            
            fig_sequence.canvas.draw()        
        
        # wire in click event listener
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.legend()
        plt.show()
