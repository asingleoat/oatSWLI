#!/usr/bin/env python3

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import torch


def create_3d_surface_plot(heightmap, title="Heightmap"):
    """
    Create an interactive 3D surface plot using Plotly.

    Args:
        heightmap: 2D array of height values
        title: Plot title

    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[go.Surface(z=heightmap, colorscale="earth")])
    fig.update_layout(
        title=dict(text=title),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90),
        scene={"aspectratio": {"x": 1, "y": 1, "z": 0.2}},
    )
    return fig


def normalize_max_abs(arr):
    max_abs = np.max(np.abs(arr))
    return arr / max_abs if max_abs != 0 else arr


def clip_negative(arr):
    return np.maximum(arr, 0)


def setup_interactive_plots(metadata, ift_result, reference_chirp, crop_region):
    """
    Set up interactive 2D plots for exploring video data.

    Args:

        ift_result: Cross-correlation results
        reference_chirp: Reference chirp used for correlation

    Returns:
        Tuple of (figure, axes) for the plots
    """
    h_start, h_end, w_start, w_end = crop_region
    # Create figure for the video frame
    fig, ax = plt.subplots()
    img_display = ax.imshow(
        metadata["frame"][h_start:h_end, w_start:w_end, :]
    )  # , cmap="plasma")
    ax.set_title(f"First Frame")
    plt.colorbar(img_display)

    # Create a second figure for plotting the intensity over time
    fig_sequence, ax_sequence = plt.subplots()
    ax_sequence.set_title("Pixel Intensity")
    ax_sequence.set_xlabel("Time Frame")
    ax_sequence.set_ylabel("Intensity")

    reference_curve = torch.abs(torch.fft.ifft(reference_chirp, axis=-1)).cpu()
    (reference_plot,) = ax_sequence.plot(
        np.arange(reference_curve.shape[1]),
        reference_curve[0],
        label="Reference Curve",
        linestyle="dashed",
        color="black",
    )

    (sequence_plot_r,) = ax_sequence.plot([], [], label="Selected Pixel_r", color="red")
    (sequence_plot_g,) = ax_sequence.plot(
        [], [], label="Selected Pixel_g", color="green"
    )
    (sequence_plot_b,) = ax_sequence.plot(
        [], [], label="Selected Pixel_b", color="blue"
    )

    (sequence_plot_p,) = ax_sequence.plot([], [], label="product", color="orange")

    (sequence_plot_cube,) = ax_sequence.plot([], [], label="sum of cubes", color="pink")

    # Define click handler
    def on_click(event):
        # ignore clicks outside the image
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.ydata), int(event.xdata)

        sequence = ift_result[x, y, :].real / 1000
        cubed_0 = np.power(sequence[0], 3)
        cubed_1 = np.power(sequence[1], 3)
        cubed_2 = np.power(sequence[2], 3)
        # Update plots
        sequence_plot_r.set_xdata(np.arange(sequence.shape[1]))
        sequence_plot_r.set_ydata(normalize_max_abs(sequence[-1]))
        sequence_plot_g.set_xdata(np.arange(sequence.shape[1]))
        sequence_plot_g.set_ydata(normalize_max_abs(sequence[-2]))
        sequence_plot_b.set_xdata(np.arange(sequence.shape[1]))
        sequence_plot_b.set_ydata(normalize_max_abs(sequence[0]))
        sequence_plot_p.set_xdata(np.arange(sequence.shape[1]))
        sequence_plot_p.set_ydata(
            normalize_max_abs(clip_negative(sequence[0]))
            * normalize_max_abs(clip_negative(sequence[-1]))
        )
        sequence_plot_cube.set_xdata(np.arange(sequence.shape[1]))
        sequence_plot_cube.set_ydata(2 * normalize_max_abs(cubed_0 + cubed_1 + cubed_2))

        ax_sequence.relim()
        ax_sequence.autoscale_view()

        fig_sequence.canvas.draw()

    # Connect click event listener
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.legend()

    return (fig, ax), (fig_sequence, ax_sequence)
