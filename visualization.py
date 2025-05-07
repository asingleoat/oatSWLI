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
        scene={"aspectratio": {"x": 1, "y": 1, "z": 2}},
    )
    return fig


def setup_interactive_plots(video_array_BGR, ift_result, reference_chirp):
    """
    Set up interactive 2D plots for exploring video data.

    Args:
        video_array_BGR: Video array with BGR channels
        ift_result: Cross-correlation results
        reference_chirp: Reference chirp used for correlation

    Returns:
        Tuple of (figure, axes) for the plots
    """
    t_fixed = 0  # arbitrary choice of frame to display

    # Create figure for the video frame
    fig, ax = plt.subplots()
    img_display = ax.imshow(video_array_BGR[..., t_fixed], cmap="plasma")
    ax.set_title(f"Frame no:{t_fixed}")
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

    # Define click handler
    def on_click(event):
        # ignore clicks outside the image
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.ydata), int(event.xdata)

        sequence = ift_result[x, y, :].real / 1000

        # Update plots
        sequence_plot_r.set_xdata(np.arange(sequence.shape[1]))
        sequence_plot_r.set_ydata(sequence[-1])
        sequence_plot_g.set_xdata(np.arange(sequence.shape[1]))
        sequence_plot_g.set_ydata(sequence[-2])
        sequence_plot_b.set_xdata(np.arange(sequence.shape[1]))
        sequence_plot_b.set_ydata(sequence[0])

        ax_sequence.relim()
        ax_sequence.autoscale_view()

        fig_sequence.canvas.draw()

    # Connect click event listener
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.legend()

    return (fig, ax), (fig_sequence, ax_sequence)
