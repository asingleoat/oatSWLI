import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# ==== Configuration ====
crop = True  # Set to False to keep the original size
crop_region = (0, 200, 00, 200) if crop else None  # Define crop region if enabled

# def load_video_as_array(filename, crop_region):
#     cap = cv2.VideoCapture(filename)
#     frames = []
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#         x_start, x_end, y_start, y_end = crop_region
#         cropped_frame = gray_frame[x_start:x_end, y_start:y_end]  # Crop the frame
#         frames.append(cropped_frame)
    
#     cap.release()
    
#     return np.stack(frames, axis=-1)  # Shape: (cropped_height, cropped_width, time)

# Choose the (x, y) coordinate to use as template chirp. Any clean pixel will work, this could be iteratively improved by aligning chirps using the final height map and averaging.
x, y = 50, 50  # Adjust as needed

# Load video
video_array = load_video_as_array("./DTbrass_100fps_500nmps.AVI", crop_region)

# gpu = True
# # Select backend: NumPy for CPU, CuPy for GPU
# xp = cp if use_gpu else np  

# # Move to GPU
# video_gpu = cp.array(video_array)
# # Compute FFT on GPU
# fft_gpu = cp.fft.fft(video_gpu, axis=-1)
# # Compute IFT on GPU
# ift_gpu = cp.fft.ifft(fft_gpu, axis=-1).real  # Extract real part
# # Move result back to CPU
# ift_result = np.fft.ifftshift(cp.asnumpy(ift_gpu)

fft_result = np.fft.fftshift(np.fft.fft(video_array, axis=-1), axes=-1)  # FFT over time dimension
t = video_array.shape[-1]  # Last dimension is time (frequency space)
center_idx = t // 2  # Compute center index for DC component

# Set DC component (center frequency) to zero along the last axis
fft_result[..., center_idx] = 0


fft_conjugate = np.conj(fft_result)

# Extract the full frequency spectrum of the selected spatial point
selected_spectrum = fft_result[x, y, :]  # Shape: (time,)

products = selected_spectrum[None, None, :] * fft_conjugate
ift_result = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(products, axes=-1), axis=-1).real, axes=-1)  # Keep only real part
# ift_result = np.fft.ifft(products, axis=-1).real  # Shape: (height, width, time)


max_indices = np.argmax(ift_result, axis=-1)  # Shape: (height, width)

height, width = max_indices.shape
# Generate x, y coordinate grid
x = np.arange(width)
y = np.arange(height)
X, Y = np.meshgrid(x, y)

# Create a 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface (height map)
ax.plot_surface(X, Y, max_indices, cmap="viridis", edgecolor="none")

# Labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Max Frame Index (Height)")
ax.set_title("3D Height Map of Max Indices")



print(fft_result.shape)  # Should be (height, width, time)


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
time_series_plot, = ax_time_series.plot([], [])  # Empty plot to be updated

# Click event function
def on_click(event):
    if event.xdata is None or event.ydata is None:
        return  # Ignore clicks outside the image
    
    x, y = int(event.ydata), int(event.xdata)  # Convert to integer coordinates

    # Extract time-series data for the clicked pixel
    time_series = ift_result[x, y, :]

    # Update the second plot
    time_series_plot.set_xdata(np.arange(time_series.shape[0]))
    time_series_plot.set_ydata(time_series)

    ax_time_series.relim()
    ax_time_series.autoscale_view()
    
    fig_time_series.canvas.draw()

# Connect the click event to the function
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
