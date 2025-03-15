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
crop = False  # Set to False to keep the original size
crop_region = (50, 150, 30, 130) if crop else None  # Define crop region if enabled

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


# Load cropped video

def compute_fft(video_array):
    return np.fft.fftshift(np.fft.fft(video_array, axis=-1), axes=-1)  # FFT over time dimension

# Load video and compute FFT
video_array = load_video_as_array("./capprobeBBS_90fps_750nmps.AVI", crop_region)
fft_result = compute_fft(video_array)
fft_conjugate = np.conj(fft_result)

# Choose the (x, y) coordinate of interest
x, y = 50, 50  # Adjust as needed

# Extract the full frequency spectrum of the selected spatial point
selected_spectrum = fft_result[x, y, :]  # Shape: (time,)

products = selected_spectrum[None, None, :] * fft_conjugate
ift_result = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(products, axes=-1), axis=-1).real, axes=-1)  # Keep only real part


max_indices = np.argmax(ift_result, axis=-1)  # Shape: (height, width)

# Open a new Matplotlib figure for displaying max indices
fig_max, ax_max = plt.subplots()
img_max = ax_max.imshow(max_indices, cmap="viridis")  # Display using a perceptually uniform colormap
ax_max.set_title("Max Index per Pixel")
plt.colorbar(img_max, ax=ax_max, label="Frame Index of Max Value")

plt.show()

# ift_result = np.fft.ifft(products, axis=-1).real  # Shape: (height, width, time)

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
