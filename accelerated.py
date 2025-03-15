#!/usr/bin/env python3

# from vispy import app, scene
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

start_time = time.perf_counter()

# print("CUDA available:", torch.cuda.is_available())
# print("Number of CUDA devices:", torch.cuda.device_count())
# print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
# print("PyTorch built with CUDA:", torch.backends.cuda.is_built())


def load_video_as_array(filename, crop_region=None):
    cap = cv2.VideoCapture(filename)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
        # Apply cropping only if a crop region is specified
        if crop_region:
            x_start, x_end, y_start, y_end = crop_region
            gray_frame = gray_frame[x_start:x_end, y_start:y_end]  

        frames.append(gray_frame)
    
    cap.release()
    
    return np.stack(frames, axis=-1)  # Shape: (height, width, time)

def compute_fft_ifft_torch(reference_chirp, video_array, use_gpu=True):
  with torch.no_grad():
    """Computes FFT, conjugate multiplication, and IFT using PyTorch (GPU/CPU)."""

    # Select device: CUDA (GPU) or CPU
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Move data to GPU if needed
    video_torch = torch.tensor(video_array, device=device, dtype=torch.complex64)
    print(f"Tensor is on device: {video_torch.device}") 
    # Compute FFT
    fft_result = torch.fft.fftshift(torch.fft.fft(video_torch, dim=-1), dim=-1)
    fft_result = remove_dc(fft_result)

    # n = reference_chirp.shape[0]  # Length of the signal
    # step_size = 750
    # freq = np.fft.fftshift(np.fft.fftfreq(n, d=1/step_size), axes=-1)  # Compute frequency bins
    # cutoff_freq = 750
    # # Zero out frequencies beyond the cutoff
    # fft_result[:,:,np.abs(freq) > cutoff_freq] = 0

    fft_conjugate = torch.conj(fft_result)


    # Extract the full frequency spectrum of the selected spatial point
    selected_spectrum = reference_chirp

    # Perform element-wise multiplication
    products = selected_spectrum[None, None, :] * fft_conjugate

    # Compute IFT
    ift_result = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(products, dim=-1), dim=-1).real, dim=-1)

    # Move result back to CPU if needed
    # return ift_result.cpu().numpy() if use_gpu else ift_result.numpy()

    # Compute argmax along the time axis (dimension -1)
    # max_indices = torch.argmax(ift_result, dim=-1)
    # max_indices = subpixel_argmax_3d_torch(ift_result, dim=-1)
    max_indices = quadratic_subpixel_peak_3d_torch(ift_result, dim=-1)
    
    return (max_indices.cpu().numpy() if use_gpu else max_indices.numpy(), ift_result.cpu().numpy() if use_gpu else ift_result.numpy())

def remove_dc(array):
    """
    Removes the DC component from each FFT in a 3D tensor (already fftshifted).

    Parameters:
        fft_3d (torch.Tensor): Input 3D tensor (height, width, time) with already fftshifted FFTs.

    Returns:
        torch.Tensor: FFT 3D tensor with DC components removed.
    """
    t = array.shape[-1]  # Last dimension is time (frequency space)
    center_idx = t // 2  # Compute center index for DC component

    # Set DC component (center frequency) to zero along the last axis
    array[..., center_idx] = 0

    return array

def quadratic_subpixel_peak_torch(arr):
    """
    Computes a more accurate peak index by fitting a quadratic function 
    around the discrete argmax using PyTorch.

    Parameters:
        arr (torch.Tensor): 1D tensor representing the signal along one axis.

    Returns:
        torch.Tensor: Estimated sub-index of the maximum (continuous).
    """
    argmax_idx = torch.argmax(arr)  # Get discrete maximum index
    
    # Handle edge cases where the max is at the boundary
    if argmax_idx == 0 or argmax_idx == arr.shape[0] - 1:
        return argmax_idx.float()  # Cannot fit a parabola at the edge

    # Get three neighboring points (argmax and its two neighbors)
    x = torch.tensor([argmax_idx - 1, argmax_idx, argmax_idx + 1], dtype=torch.float32, device=arr.device)
    y = arr[x.long()]  # Corresponding values

    # Fit a quadratic function: y = ax^2 + bx + c
    X = torch.stack([x**2, x, torch.ones_like(x)], dim=-1)  # Design matrix
    coeffs = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze()  # Solve for [a, b, c]

    # Compute subpixel max index using vertex formula: x = -b / (2a)
    a, b, _ = coeffs
    subpixel_max = -b / (2 * a)

    return subpixel_max

def subpixel_argmax_3d_torch(array, dim=-1):
    """
    Applies subpixel argmax estimation across a given axis of a 3D PyTorch tensor.

    Parameters:
        array (torch.Tensor): Input 3D tensor (height, width, time).
        axis (int): Axis along which to compute subpixel argmax.

    Returns:
        torch.Tensor: 2D tensor of subpixel argmax indices.
    """
    device = array.device
    discrete_argmax = torch.argmax(array, dim=dim)  # Get argmax along the time axis
    subpixel_max = torch.zeros_like(discrete_argmax, dtype=torch.float32, device=device)  # Output tensor

    # Iterate over each pixel and apply quadratic fitting
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            subpixel_max[i, j] = quadratic_subpixel_peak_torch(array[i, j, :])

    return subpixel_max

def quadratic_subpixel_peak_3d_torch(array, dim=-1):
    """
    Computes a subpixel-accurate argmax along a given axis using quadratic fitting.
    Fully vectorized and GPU-accelerated.

    Parameters:
        array (torch.Tensor): 3D tensor (height, width, time).
        axis (int): Axis along which to compute subpixel argmax.

    Returns:
        torch.Tensor: 2D tensor of subpixel argmax indices.
    """
    device = array.device
    height, width, time = array.shape

    # Compute discrete argmax
    discrete_argmax = torch.argmax(array, dim=dim)  # Shape: (height, width)

    # Handle edge cases where argmax is at the boundary (return discrete index)
    safe_argmax = torch.clamp(discrete_argmax, 1, time - 2)  # Keep in bounds

    
    # Create index tensors for neighboring points
    idx_left = safe_argmax - 1
    idx_right = safe_argmax + 1

    # Gather values at argmax and neighbors
    batch_indices = torch.arange(height, device=device).view(-1, 1).expand(-1, width)
    batch_indices_x = torch.arange(width, device=device).expand(height, -1)

    max_values = torch.abs(array[batch_indices, batch_indices_x, discrete_argmax])

    y0 = array[batch_indices, batch_indices_x, idx_left]  # Left neighbor
    y1 = array[batch_indices, batch_indices_x, safe_argmax]  # Peak value
    y2 = array[batch_indices, batch_indices_x, idx_right]  # Right neighbor

    # Quadratic fit: Solve for peak x = -b / (2a)
    a = 0.5 * (y0 + y2 - 2 * y1)
    b = 0.5 * (y2 - y0)

    # Compute the refined subpixel peak index
    subpixel_max = safe_argmax + torch.where(a != 0, -b / (2 * a), torch.zeros_like(b))  # Avoid division by zero
    threshold = 1000000000
    argmax_2d_filtered = torch.where(max_values <= threshold, subpixel_max, torch.zeros_like(subpixel_max))

    # return subpixel_max  # Shape: (height, width)
    return argmax_2d_filtered

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
    """Removes the average tilt (linear trend) from the max_indices 2D array using least squares."""

    height, width = max_indices.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))  # Coordinate grid

    # Construct the design matrix for least squares fitting
    A = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)  # Shape: (height * width, 3)
    z = max_indices.reshape(-1)  # Flatten max_indices for solving

    # Solve for plane coefficients using least squares: Ax = b
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)  # Solve for [a, b, c]

    # Compute the fitted plane
    fitted_plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]

    # Subtract the plane from max_indices
    detrended_max_indices = max_indices - fitted_plane

    return detrended_max_indices
    
def process_chunks_2d(array, process_func, n_chunks=4, m_chunks=4):
    """
    Splits a 2D array into (n_chunks, m_chunks) regions, applies `process_func` to each region,
    and reassembles the processed chunks into the full-size result array.

    Parameters:
        array (numpy.ndarray): The input 2D array.
        process_func (function): The function to apply to each chunk.
        n_chunks (int): Number of vertical splits.
        m_chunks (int): Number of horizontal splits.

    Returns:
        numpy.ndarray: The full-size reassembled result after processing each chunk.
    """

    height, width, time = array.shape
    chunk_height = height // n_chunks
    chunk_width = width // m_chunks

    result = np.zeros((height, width))

    for i in range(n_chunks):
        for j in range(m_chunks):
            # Compute chunk boundaries
            y_start = i * chunk_height
            y_end = (i + 1) * chunk_height if i < n_chunks - 1 else height  # Handle edge case
            x_start = j * chunk_width
            x_end = (j + 1) * chunk_width if j < m_chunks - 1 else width  # Handle edge case

            # Extract the chunk
            chunk = array[y_start:y_end, x_start:x_end, :]

            # Process the chunk
            processed_chunk = process_func(chunk)

            # Store back in the result array
            result[y_start:y_end, x_start:x_end] = processed_chunk

    return result

def process_chunks_fixed_size(array, process_func, chunk_height=128, chunk_width=128):
    """
    Splits a 3D array (height, width, time) into fixed-size chunks,
    applies `process_func` to reduce each chunk to 2D, and reassembles the result.

    Parameters:
        array (numpy.ndarray): The input 3D array (height, width, time).
        process_func (function): The function to apply to each chunk (must return a 2D result).
        chunk_height (int): Desired height of each chunk.
        chunk_width (int): Desired width of each chunk.

    Returns:
        numpy.ndarray: The full-size 2D result array.
    """

    height, width, depth = array.shape
    result_left = np.zeros((height, width))  # ✅ 2D placeholder
    result_right = np.zeros((height, width, depth))  # ✅ 2D placeholder

    for y_start in range(0, height, chunk_height):
        for x_start in range(0, width, chunk_width):
            # Compute chunk boundaries (handling edges)
            y_end = min(y_start + chunk_height, height)
            x_end = min(x_start + chunk_width, width)

            # Extract the 3D chunk
            chunk = array[y_start:y_end, x_start:x_end, :]

            # Process the chunk (output must be 2D)
            (processed_chunk_left, processed_chunk_right) = process_func(chunk)

            # Store back in the result array
            result_left[y_start:y_end, x_start:x_end] = processed_chunk_left
            result_right[y_start:y_end, x_start:x_end, :] = processed_chunk_right

    return (result_left, result_right)

def make_reference(video_array, x, y, use_gpu=True):
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Move data to GPU if needed
    video_torch = torch.tensor(video_array[x,y,:], device=device, dtype=torch.complex64)
    print(video_torch.shape)
    print(f"Reference tensor is on device: {video_torch.device}") 
    # Compute FFT
    # fft_result = torch.fft.fftshift(torch.fft.fft(video_torch, dim=-1), dim=-1)
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


def replace_outliers(data, threshold=3.0):
    """
    Detects outliers based on absolute difference from neighbors and replaces them with the neighbor average.

    Parameters:
        data (np.ndarray): 2D input array.
        threshold (float): Outlier threshold (higher = less sensitive).

    Returns:
        np.ndarray: Outlier-filtered array.
    """
    data = data.astype(np.float32)  # Ensure float type

    # Pad array to avoid index issues at borders
    padded_data = np.pad(data, pad_width=1, mode="edge")

    # Compute neighbor average (excluding center)
    neighbor_avg = (
        padded_data[:-2, 1:-1] + padded_data[2:, 1:-1] +  # Vertical neighbors
        padded_data[1:-1, :-2] + padded_data[1:-1, 2:]   # Horizontal neighbors
    ) / 4

    # Compute absolute difference from neighbor average
    abs_diff = np.abs(data - neighbor_avg)

    # Identify outliers using the standard deviation of differences
    std_dev = np.std(abs_diff)
    outliers = abs_diff > (threshold * std_dev)

    # Replace outliers with the neighbor average
    cleaned_data = data.copy()
    cleaned_data[outliers] = neighbor_avg[outliers]

    return cleaned_data

def smooth_outliers(image, threshold=1.0, filter_size=10):
    """
    Detects and smooths outliers in an image using a local mean filter.

    Parameters:
        image (np.ndarray): 2D grayscale image.
        threshold (float): Number of standard deviations above which a pixel is considered an outlier.
        filter_size (int): Size of the smoothing filter.

    Returns:
        np.ndarray: Image with outliers smoothed.
    """
    # Compute local mean and standard deviation using a moving window
    local_mean = scipy.ndimage.uniform_filter(image, size=filter_size)
    local_var = scipy.ndimage.uniform_filter((image - local_mean) ** 2, size=filter_size)
    local_std = np.sqrt(np.clip(local_var, 1e-8, None))  # Prevent NaNs by clamping small values

    # Identify outliers
    outliers = np.abs(image - local_mean) > (threshold * local_std)

    # Replace outliers with local mean
    smoothed_image = image.copy()
    smoothed_image[outliers] = local_mean[outliers]

    return smoothed_image

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load a video file and process its FFT/IFT.")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    args = parser.parse_args()

    # ==== Configuration ====
    crop = False  # Set to False to keep the original size
    # crop = True  # Set to False to keep the original size
    crop_region = (0, 250, 0, 250) if crop else None  # Define crop region if enabled

    # Load video
    # video_array = load_video_as_array("./capprobeBBS_90fps_750nmps.AVI", crop_region)
    video_array = load_video_as_array(args.video_file, crop_region)

    # Choose the (x, y) coordinate of interest
    h, w, t = video_array.shape
    x = h // 2
    y = w // 2
    # x, y = 0, 0
    # Adjust as needed
    # Extract the full frequency spectrum of the selected spatial point
    # reference_chirp = make_reference(video_array, x, y, use_gpu=args.gpu)
    reference_chirp = make_reference_avg(video_array, x, y, window=100, use_gpu=args.gpu)


 
    # Run computation
    # max_indices = compute_fft_ifft_torch(video_array, use_gpu=args.gpu)
    (max_indices, ift_result) = process_chunks_fixed_size(video_array, lambda v: compute_fft_ifft_torch(reference_chirp, v, use_gpu=args.gpu))
    # max_indices = replace_outliers(max_indices, 0.1)
    max_indices = remove_tilt(max_indices)
    max_indices = smooth_outliers(max_indices)
    # max_indices = np.clip(max_indices, a_min=-50, a_max=50)
    normalize_and_save(max_indices, "out.png")

    end_time = time.perf_counter()
    print(f"Execution time upto image saving: {end_time - start_time:.6f} seconds")

    # canvas = scene.SceneCanvas(keys="interactive", show=True)
    # view = canvas.central_widget.add_view()

    # # Generate a large 3D heightmap
    # x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    # z = np.sin(x**2 + y**2)

    # # Convert colormap to a color array
    # cmap = color.get_colormap("viridis")
    # colors = cmap.map((Z - Z.min()) / (Z.max() - Z.min()))  # Normalize Z to [0, 1]

    # # Create the surface plot without passing `cmap` directly
    # surface = scene.visuals.SurfacePlot(
    #     x=X, y=Y, z=Z, colors=colors, parent=view.scene
    # )    
    # view.camera = scene.TurntableCamera()  # Enable 3D rotation
    # app.run()
    plot = False;
    # plot = True;
    if plot:
        height, width = max_indices.shape
        # Generate x, y coordinate grid
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        # Create a 3D figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface (height map)
        ax.plot_surface(X, Y, max_indices, rstride=2,cstride=2, cmap="plasma", edgecolor="none")

        # Labels and title
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Max Frame Index (Height)")
        ax.set_title("3D Height Map of Max Indices")
        
        # Choose a fixed time index to display an image frame
        t_fixed = 10  # Adjust this to select a frame in time
        # Create interactive figure
        fig, ax = plt.subplots()
        img_display = ax.imshow(video_array[..., t_fixed], cmap='gray')  # Show frame at fixed t
        ax.set_title(f"Frame at t={t_fixed}")
        plt.colorbar(img_display)

        # Create a second figure for plotting the intensity over time
        fig_time_series, ax_time_series = plt.subplots()
        ax_time_series.set_title("Pixel Intensity Over Time")
        ax_time_series.set_xlabel("Time Frame")
        ax_time_series.set_ylabel("Intensity")

        reference_curve = torch.abs(torch.fft.ifft(reference_chirp, axis=-1)).cpu()
        # reference_curve = reference_chirp.cpu()
        reference_plot, = ax_time_series.plot(np.arange(reference_curve.shape[0]), reference_curve, label="Reference Curve", linestyle="dashed", color="red",)
        time_series_plot, = ax_time_series.plot([], [], label="Selected Pixel", color="blue")  # Pixel intensity plot
    
        # Click event function
        def on_click(event):
            if event.xdata is None or event.ydata is None:
                return  # Ignore clicks outside the image
            
            x, y = int(event.ydata), int(event.xdata)  # Convert to integer coordinates
        
            # Extract time-series data for the clicked pixel
            time_series = ift_result[x, y, :].real / 1000

    
            # Update plots
            time_series_plot.set_xdata(np.arange(time_series.shape[0]))
            time_series_plot.set_ydata(time_series)
            
            ax_time_series.relim()
            ax_time_series.autoscale_view()
            
            fig_time_series.canvas.draw()        
        
        # Connect the click event to the function
        fig.canvas.mpl_connect('button_press_event', on_click)

        plt.show()

    
    # print(f"IFT result shape: {max_indices.shape}")
    # print(f"Using {'GPU (PyTorch)' if use_gpu else 'CPU (PyTorch)'}")
