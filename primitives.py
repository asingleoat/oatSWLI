#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import os
from visualization import create_3d_surface_plot, setup_interactive_plots, plot_lines

try:
    profile
except NameError:

    def profile(func):
        return func


sigma = 10
bias = 2.0
a = 1

def normalize(x):
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x  # avoid divide-by-zero
    return x / max_abs


def subtract_mean(arr):
    mean = np.mean(arr)
    return arr - mean


def unwrap_shift(i, N):
    """Unwrap circular index i to signed shift in range [-N/2, N/2)"""
    return np.where(i < N // 2, i, i - N)


def rewrap_shift(i, N):
    return np.where(i > 0, i, i + N)


def average_if_defined(x, y):
    if x is not None and y is not None:
        return 0.5 * (x + y)
    elif x is not None:
        return x
    else:
        return y


def align_multi(b_host, a_host, max_dev=1.0):
    reference = np.conj(np.fft.rfft(b_host.astype(np.float32), axis=-1))
    N = a_host.shape[-1]  # Length of sequences

    num_cores = os.cpu_count()

    # Create chunks for 2D processing (flatten first two dimensions)
    original_shape = a_host.shape[:2]  # Store original 2D shape
    a_flat = a_host.reshape(
        -1, a_host.shape[-1]
    )  # Flatten to (total_sequences, sequence_length)

    chunk_size = a_flat.shape[0] // num_cores
    chunks = []

    for i in range(num_cores):
        start_idx = i * chunk_size
        if i == num_cores - 1:  # Last chunk gets remainder
            end_idx = a_flat.shape[0]
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append((start_idx, end_idx))

    # Pre-allocate output array for alignment offsets
    offsets = np.empty(a_flat.shape[0], dtype=np.float32)

    def process_chunk(start_idx, end_idx):
        for seq_idx in range(start_idx, end_idx):
            a_seq = a_flat[seq_idx]

            # Compute cross-spectrum
            A = np.fft.rfft(a_seq.astype(np.float32))
            cross_spec = A * reference

            # Step 1: Get coarse integer shift via cross-correlation
            corr = np.fft.irfft(cross_spec, n=N)
            peak = np.argmax(np.abs(corr))
            shift_int = unwrap_shift(peak, N)

            # Step 2: Refine with phase slope analysis
            freqs = np.fft.rfftfreq(N)
            valid = freqs > 0
            f = freqs[valid]
            cross_spec_valid = cross_spec[valid]

            if len(f) < 3:
                offsets[seq_idx] = float(shift_int)
                continue

            # Subtract expected phase from integer shift
            phase = np.unwrap(np.angle(cross_spec_valid))
            expected_phase = -2 * np.pi * f * shift_int
            residual_phase = (phase - expected_phase + np.pi) % (2 * np.pi) - np.pi

            # Estimate residual delay per bin
            delta = -residual_phase / (2 * np.pi * f)

            # Mask plausible bins and compute weighted average
            mag = np.abs(cross_spec_valid)
            mag /= mag.max() + 1e-12
            mask = np.abs(delta) <= max_dev

            if np.sum(mask) < 3:
                offsets[seq_idx] = float(shift_int)  # Fallback
            else:
                delta_refined = np.sum(mag[mask] * delta[mask]) / np.sum(mag[mask])
                offsets[seq_idx] = -(shift_int + delta_refined)

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(process_chunk, start_idx, end_idx)
            for start_idx, end_idx in chunks
        ]

        for future in futures:
            future.result()

    # Reshape offsets back to original 2D shape
    offsets_2d = offsets.reshape(original_shape)

    return offsets_2d


def align_multi_p(b_host, a_host, max_dev=1.0, band=(0.05, 0.4), tile_size=(100, 100)):
    reference = np.conj(np.fft.rfft(b_host.astype(np.float32), axis=-1))
    N = a_host.shape[-1]
    H, W = a_host.shape[:2]

    num_cores = os.cpu_count()

    tile_H, tile_W = tile_size
    num_tiles_h = (H + tile_H - 1) // tile_H
    num_tiles_w = (W + tile_W - 1) // tile_W

    offsets = np.empty((H, W), dtype=np.float32)

    phase_offsets = np.empty((H, W), dtype=np.float32)

    coherence_offsets = np.empty((H, W), dtype=np.float32)

    global_fallback_count = 0
    total_tweaks = 0
    counter = 0

    # Tile processing in diagonals: stage = i + j
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for stage in range(num_tiles_h + num_tiles_w - 1):
            futures = []

            for i in range(num_tiles_h):
                j = stage - i
                if j < 0 or j >= num_tiles_w:
                    continue

                row_start = i * tile_H
                col_start = j * tile_W
                row_end = min((i + 1) * tile_H, H)
                col_end = min((j + 1) * tile_W, W)

                tile_data = a_host[row_start:row_end, col_start:col_end, :]

                left_bias = None
                if j > 0:
                    left_bias = offsets[row_start:row_end, col_start - 1]
                bottom_bias = None
                if i > 0:
                    bottom_bias = offsets[
                        row_start - 1, col_start : col_start + col_end
                    ]

                futures.append(
                    executor.submit(
                        process_chunk,
                        tile_data,
                        reference,
                        N,
                        max_dev,
                        band,
                        (row_start, col_start),
                        left_bias,
                        bottom_bias,
                    )
                )

            for fut in futures:
                (
                    (i0, j0),
                    tile_offsets,
                    fallback_count,
                    avg_tweak,
                    max_tweak,
                    tile_phase,
                    tile_coherence,
                ) = fut.result()
                h, w = tile_offsets.shape
                offsets[i0 : i0 + h, j0 : j0 + w] = tile_offsets
                phase_offsets[i0 : i0 + h, j0 : j0 + w] = tile_phase
                coherence_offsets[i0 : i0 + h, j0 : j0 + w] = tile_coherence
                global_fallback_count += fallback_count
                total_tweaks += avg_tweak
                counter += 1

    print("Avg Phase Refinement:", total_tweaks / counter)
    print("Max Phase Refinement:", max_tweak)
    print("Fallbacks:", global_fallback_count)
    print("Fallback proportion:", global_fallback_count / (H * W))
    return (-offsets, -phase_offsets, -coherence_offsets)

@profile
def align_single(b_host, a_host, max_dev=1.0, band=(0.05, 0.4), tile_size=(1000, 1000)):
    reference = np.conj(np.fft.rfft(b_host.astype(np.float32), axis=-1))
    N = a_host.shape[-1]
    H, W = a_host.shape[:2]


    tile_H, tile_W = tile_size
    num_tiles_h = (H + tile_H - 1) // tile_H
    num_tiles_w = (W + tile_W - 1) // tile_W

    offsets = np.empty((H, W), dtype=np.float32)

    phase_offsets = np.empty((H, W), dtype=np.float32)

    coherence_offsets = np.empty((H, W), dtype=np.float32)

    correlations = np.empty((H, W, N), dtype=np.float32)

    global_fallback_count = 0
    total_tweaks = 0
    counter = 0

    # Tile processing in diagonals: stage = i + j
    for stage in range(num_tiles_h + num_tiles_w - 1):
        results = []
        for i in range(num_tiles_h):
            j = stage - i
            if j < 0 or j >= num_tiles_w:
                continue

            row_start = i * tile_H
            col_start = j * tile_W
            row_end = min((i + 1) * tile_H, H)
            col_end = min((j + 1) * tile_W, W)

            tile_data = a_host[row_start:row_end, col_start:col_end, :]

            left_bias = None
            if j > 0:
                left_bias = offsets[row_start:row_end, col_start - 1]
            bottom_bias = None
            if i > 0:
                bottom_bias = offsets[
                    row_start - 1, col_start : col_start + col_end
                ]

            results.append(
                        process_chunk(
                        tile_data,
                        reference,
                        N,
                        max_dev,
                        band,
                        (row_start, col_start),
                        left_bias,
                        bottom_bias,
                    )
                )

        for result in results:
            (
                (i0, j0),
                tile_offsets,
                fallback_count,
                avg_tweak,
                max_tweak,
                tile_phase,
                tile_coherence,
                tile_correlations,
            ) = result
            h, w = tile_offsets.shape
            offsets[i0 : i0 + h, j0 : j0 + w] = tile_offsets
            phase_offsets[i0 : i0 + h, j0 : j0 + w] = tile_phase
            coherence_offsets[i0 : i0 + h, j0 : j0 + w] = tile_coherence
            correlations[i0 : i0 + h, j0 : j0 + w] = tile_correlations
            global_fallback_count += fallback_count
            total_tweaks += avg_tweak
            counter += 1

    print("Avg Phase Refinement:", total_tweaks / counter)
    print("Max Phase Refinement:", max_tweak)
    print("Fallbacks:", global_fallback_count)
    print("Fallback proportion:", global_fallback_count / (H * W))
    return (-offsets, -phase_offsets, -coherence_offsets, correlations)

@profile
def process_chunk(
    tile_data, reference, N, max_dev, band, tile_origin, left_bias, bottom_bias
):
    H, W, _ = tile_data.shape
    local_offsets = np.empty((H, W), dtype=np.float32)
    phase_offsets = np.empty((H, W), dtype=np.float32)  # debug only
    correlations =  np.empty((H,W,N), dtype=np.float32)  # debug only
    coherence_offsets = np.empty((H, W), dtype=np.float32)  # debug only
    fallback_count = 0
    tweaks = 0
    max_tweak = 0
    pixels = 0
    th, tw = tile_origin
    lines = []
    labels = []

    prev_shift_h = None
    prev_shift_w = None
    for i in range(H):
        for j in range(W):

            if i == 0 and bottom_bias is not None:
                prev_shift_h = bottom_bias[j]
            elif i > 0:
                prev_shift_h = local_offsets[i - 1, j]

            if j == 0 and left_bias is not None:
                prev_shift_w = left_bias[i]
            elif j > 0:
                prev_shift_w = local_offsets[i, j - 1]

            prev_shift = average_if_defined(prev_shift_w, prev_shift_h)

            a_seq = tile_data[i, j, :]
            A = np.fft.rfft(a_seq.astype(np.float32))
            cross_spec = A * reference

            freqs = np.fft.rfftfreq(N)
            band_mask = (freqs >= band[0]) & (freqs <= band[1])
            cross_spec_band = np.zeros_like(cross_spec)
            cross_spec_band[band_mask] = cross_spec[band_mask]
            cross_spec_band[0] *= 8 / N

            corr = np.fft.irfft(cross_spec_band, n=N)
            # corr *= -1
            peak = argmax_with_prior(corr, prev_shift, sigma=sigma)
            shift_int = unwrap_shift(peak, N)
            # corr *= -1

            valid = freqs > 0
            f = freqs[valid]
            cross_spec_valid = cross_spec[valid]

            phase = np.unwrap(np.angle(cross_spec_valid))
            expected_phase = -2 * np.pi * f * shift_int
            residual_phase = (phase - expected_phase + np.pi) % (2 * np.pi) - np.pi
            delta = -residual_phase / (2 * np.pi * f)

            mag = np.abs(cross_spec_valid)
            mag /= mag.max() + 1e-12
            mask = np.abs(delta) <= max_dev

            correlations[i,j] = corr
            if np.sum(mask) < 3:
                local_offsets[i, j] = float(shift_int)
                phase_offsets[i, j] = 0
                coherence_offsets[i, j] = float(shift_int)
                fallback_count += 1
            else:
                delta_refined = np.sum(mag[mask] * delta[mask]) / np.sum(mag[mask])
                local_offsets[i, j] = shift_int + delta_refined
                phase_offsets[i, j] = delta_refined
                coherence_offsets[i, j] = float(shift_int)
                tweaks += np.abs(delta_refined)
                max_tweak = max(max_tweak, np.abs(delta_refined))
                pixels += 1

            # if (th + i, tw + j) == (71, 83) or (th + i, tw + j) == (70, 83):
            #     print(cross_spec_band[0])
            #     print(cross_spec_band[100])
            #     new_shift, prior, score = argmax_with_prior_debug(
            #         corr, prev_shift, sigma=sigma
            #     )
            #     print((th + i, tw + j), new_shift, shift_int, delta_refined)
            #     lines += [corr, score, prior]
            #     labels += ["corr", "score", "prior"]
    if lines:
        print(len(lines))
        plot_lines(lines, labels=labels, prev_shift=None, new_shift=None)
    avg_tweak = tweaks / pixels if pixels > 0 else 0
    return (
        tile_origin,
        local_offsets,
        fallback_count,
        avg_tweak,
        max_tweak,
        phase_offsets,
        coherence_offsets,
        correlations,
    )


def unwrap_around(i, N, center):
    # Wrap i ∈ [0, N) to be as close as possible to center
    unwrapped = i.copy()
    delta = ((center - i + N // 2) % N) - N // 2
    return center - delta


def left_skewed_gaussian_manual(x, center, sigma, alpha=2.0):
    """
    Left-skewed Gaussian by warping the input. `alpha` controls skew severity.
    """
    x_shifted = x - center
    x_skewed = np.where(x_shifted < 0, x_shifted * alpha, x_shifted)
    return np.exp(-0.5 * (x_skewed / sigma) ** 2)


def argmax_with_prior(corr, prev_shift, sigma):
    corr = subtract_mean(corr)
    corr = np.diff(corr)
    corr *= -1
    corr = np.append(corr, corr[-1])
    N = len(corr)

    if prev_shift is not None:
        indices = unwrap_around(np.arange(len(corr)), N, prev_shift)
        prior = left_skewed_gaussian_manual(indices, prev_shift, sigma, alpha=a)
        # prior = np.exp(-0.5 * ((indices - prev_shift) / sigma) ** 2)
        prior /= prior.max()  # normalize
        prior += bias  # non-zero chance of abrupt step in ground truth
        prior /= prior.max()  # normalize
        score = corr * prior
    else:
        score = corr
    idx = np.argmax(score)

    idx = unwrap_shift(idx, N)
    return idx


def argmax_with_prior_debug(corr, prev_shift, sigma):
    N = len(corr)
    indices = unwrap_around(np.arange(len(corr)), N, prev_shift)

    prior = left_skewed_gaussian_manual(indices, prev_shift, sigma, alpha=a)
    # prior = np.exp(-0.5 * ((indices - prev_shift) / sigma) ** 2)
    prior /= prior.max()  # normalize
    prior += bias  # non-zero chance of abrupt step in ground truth
    prior /= prior.max()  # normalize

    score = corr * prior
    idx = np.argmax(score)

    idx = unwrap_shift(idx, N)
    return idx, prior, score


def local_residual_phase_delay(a, b, max_dev=1.0):
    N = len(a)
    A = np.fft.rfft(a)
    B = np.fft.rfft(b)
    cross_spec = A * np.conj(B)

    # Step 1: Get coarse integer shift
    corr = np.fft.irfft(cross_spec, n=N)
    peak = np.argmax(np.abs(corr))
    shift_int = unwrap_shift(peak, N)

    freqs = np.fft.rfftfreq(N)
    valid = freqs > 0
    f = freqs[valid]
    cross_spec = cross_spec[valid]

    # Step 2: Subtract expected phase from integer shift
    phase = np.unwrap(np.angle(cross_spec))
    expected_phase = -2 * np.pi * f * shift_int
    residual_phase = (phase - expected_phase + np.pi) % (2 * np.pi) - np.pi

    # Step 3: Estimate residual delay per bin
    delta = -residual_phase / (2 * np.pi * f)

    # Step 4: Mask plausible bins
    mag = np.abs(cross_spec)
    mag /= mag.max() + 1e-12
    mask = np.abs(delta) <= max_dev

    if np.sum(mask) < 3:
        return float(shift_int)  # fallback
    delta_refined = np.sum(mag[mask] * delta[mask]) / np.sum(mag[mask])

    return -(shift_int + delta_refined)


def estimate_subsample_shift_rfft(a, b, max_dev=1.0):
    N = len(a)
    A = np.fft.rfft(a)
    B = np.fft.rfft(b)
    cross_spectrum = A * np.conj(B)

    # Integer alignment via cross-correlation
    corr = np.fft.irfft(cross_spectrum, n=N)
    i = np.argmax(np.abs(corr))
    integer_shift = -unwrap_shift(i, N)

    # Phase slope refinement
    phase = np.angle(cross_spectrum)
    phase_unwrapped = np.unwrap(phase)
    freqs = np.fft.rfftfreq(N)

    # Predict phase slope from integer shift
    expected_phase = -2 * np.pi * freqs * integer_shift

    # Compute residual
    residual = (phase_unwrapped - expected_phase + np.pi) % (2 * np.pi) - np.pi
    # residual = np.abs(phase_unwrapped - expected_phase)
    # residual = (residual + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

    # Step 3: Mask bins consistent with ≤±1 sample subshift
    max_phase_dev = 2 * np.pi * freqs * max_dev
    good = np.abs(residual) <= max_phase_dev
    weight = np.abs(cross_spectrum)
    weight /= weight.max() + 1e-12

    # Weight only bins close to the expected slope
    # mag = np.abs(cross_spectrum)
    # mag /= mag.max() + 1e-12
    # threshold = 0.1 * np.max(mag)
    # good = residual < 0.1  # radians; adjust this threshold as needed
    # weight = mag * good

    # Least-squares phase slope
    x = freqs[good]
    y = phase_unwrapped[good]
    w = weight[good]
    if len(x) < 3:
        return float(shift_int)  # fallback: too few good bins

    slope = np.sum(w * x * y) / np.sum(w * x**2)
    refined_shift = -slope * N  # final sub-sample estimate
    shift = slope / (2 * np.pi)  # in samples
    return shift


def estimate_subsample_shift_rfft_old(a, b):
    N = len(a)
    A = np.fft.rfft(a)
    B = np.fft.rfft(b)
    cross_spectrum = A * np.conj(B)

    phase_diff = np.angle(cross_spectrum)
    phase_diff_unwrapped = np.unwrap(phase_diff)

    freqs = np.fft.rfftfreq(N)  # in cycles/sample
    mag = np.abs(cross_spectrum)
    weight = mag / (mag.max() + 1e-12)

    # weighted least squares: w * x and w * y
    x = freqs
    y = phase_diff_unwrapped
    w = weight

    # slope = sum(w x y) / sum(w x²)
    slope = np.sum(w * x * y) / np.sum(w * x**2)
    shift = slope / (2 * np.pi)  # in samples
    return shift


def add_noise(x, snr_db):
    """Add Gaussian noise to signal x to achieve a target SNR in dB."""
    signal_power = np.mean(np.abs(x) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*x.shape)
    return x + noise


def sinc_pulse(length=256, width=10, offset=0):
    x = np.arange(length)
    center = length // 2 + offset
    return np.sinc((abs(x - center) + 1e-12) ** 2.4 / width)


def gaussian_cosine(length=256, offset=0, width=10, freq=0.01):
    x = np.arange(length)
    center = length // 2 + offset
    gauss = np.exp(-0.5 * ((x - center) / width) ** 2)
    carrier = np.cos(2 * np.pi * freq * (x - center))
    return gauss * carrier


def gaussian_cosine_batch(
    batch_shape, length=256, width=10, freq=0.01, offsets=None, snr=60
):
    """
    Generate a [B, C, N] batch of Gaussian-modulated cosines.
    - offsets: either scalar, list of length B, or [B, C] array of offsets
    """
    if offsets is None:
        offsets = np.zeros(batch_shape)
    else:
        offsets = np.asarray(offsets)
        if offsets.ndim == 0:
            offsets = np.full(batch_shape, offsets)

    x = np.arange(length)
    pulses = np.empty((*batch_shape, length), dtype=np.float32)

    for b in range(batch_shape[0]):
        for c in range(batch_shape[1]):
            center = length // 2 + offsets[b, c]
            gauss = np.exp(-0.5 * ((x - center) / width) ** 2)
            carrier = np.cos(2 * np.pi * freq * (x - center))
            pulses[b, c, :] = add_noise(gauss * carrier, snr)

    return pulses


def run_shift_trials(ref, shifted, estimate_fn, snr_db, shift_amt, n_trials=100):
    errors = []
    estimates = []

    for _ in range(n_trials):
        noisy_ref = add_noise(ref, snr_db)
        noisy_shifted = add_noise(shifted, snr_db)
        # ref_band = bandlimit(noisy_ref)
        # shifted_band = bandlimit(noisy_shifted)
        est = estimate_fn(noisy_ref, noisy_shifted)
        estimates.append(est)
        errors.append(abs(est - shift_amt))

    estimates = np.array(estimates)
    errors = np.array(errors)

    print(f"True shift: {shift_amt:.4f}")
    print(f"Mean estimate: {np.mean(estimates):.4f}")
    print(f"Mean error: {np.mean(errors):.4f}")
    print(f"Std dev of error: {np.std(errors):.4f}")
    print(f"Min/Max estimate: {np.min(estimates):.4f} / {np.max(estimates):.4f}")


def run_shift_trials_3d(ref, shifted, estimate_fn, snr_db, shift_amt, n_trials=100):
    errors = []
    estimates = []
    times = []

    for _ in range(n_trials):
        noisy_ref = add_noise(ref, snr_db)
        est, duration = estimate_fn(noisy_ref, shifted)
        estimates.append(est)
        times.append(duration)
        errors.append(abs(est - shift_amt))

    estimates = np.array(estimates)
    errors = np.array(errors)
    times = np.array(times)
    print(f"True shift: {shift_amt:.4f}")
    print(f"Mean estimate: {np.mean(estimates):.4f}")
    print(f"Mean error: {np.mean(errors):.4f}")
    print(f"Std dev of error: {np.std(errors):.4f}")
    print(f"Min/Max estimate: {np.min(estimates):.4f} / {np.max(estimates):.4f}")
    print(f"Mean runtime: {np.mean(times):.4f}")


def main():
    # Test signal: sinc pulse
    N = 800
    x = np.arange(N)
    width = 50
    shift_amt = 10.7  # fractional offset

    shifted = gaussian_cosine(length=N, width=width, offset=shift_amt)
    ref = gaussian_cosine(length=N, width=width, offset=0)

    snr = 40
    shifted_batch = gaussian_cosine_batch(
        (192, 108), length=N, width=width, offsets=shift_amt, snr=snr
    )

    print(shifted_batch.shape)
    print(ref.shape)

    run_shift_trials(
        ref,
        shifted,
        local_residual_phase_delay,
        snr_db=snr,
        shift_amt=shift_amt,
        n_trials=100,
    )
    run_shift_trials_3d(
        ref, shifted_batch, align_multi, snr_db=snr, shift_amt=shift_amt, n_trials=1
    )

    # print(benchmark_cpu_multi_threaded(shifted[np.newaxis, np.newaxis, :], ref))
    # # Plot phase
    # plt.plot(freqs, raw_phase, label='raw phase')
    # plt.plot(freqs, unwrapped_phase, label='unwrapped phase')
    plt.plot(add_noise(shifted, snr), label="noise")
    plt.plot(add_noise(ref, snr), label="noise")
    plt.xlabel("frequency (cycles/sample)")
    plt.ylabel("phase difference (radians)")
    plt.title("Phase difference vs frequency")
    plt.legend()
    plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()
