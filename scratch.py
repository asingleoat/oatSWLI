import matplotlib.pyplot as plt
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import os

def normalize(x):
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x  # avoid divide-by-zero
    return x / max_abs

def unwrap_shift(i, N):
    """Unwrap circular index i to signed shift in range [-N/2, N/2)"""
    return i if i < N // 2 else i - N

def benchmark_cpu_multi_threaded(b_host, a_host, max_dev=1.0):
    # Pre-compute reference FFT (conjugated for cross-correlation)
    reference = np.conj(np.fft.rfft(b_host.astype(np.float32), axis=-1))
    N = a_host.shape[-1]  # Length of sequences
    
    num_cores = os.cpu_count()
    
    # Create chunks for 2D processing (flatten first two dimensions)
    original_shape = a_host.shape[:2]  # Store original 2D shape
    a_flat = a_host.reshape(-1, a_host.shape[-1])  # Flatten to (total_sequences, sequence_length)
    
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
    
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_chunk, start_idx, end_idx) 
                  for start_idx, end_idx in chunks]
        
        for future in futures:
            future.result()
    
    end = time.time()
    duration = end - start
    
    # Reshape offsets back to original 2D shape
    offsets_2d = offsets.reshape(original_shape)
    
    return -offsets_2d, duration


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
    signal_power = np.mean(np.abs(x)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*x.shape)
    return x + noise

def sinc_pulse(length=256, width=10, offset=0):
    x = np.arange(length)
    center = length // 2 + offset
    return np.sinc((abs(x - center)+1e-12)**2.4 / width)

def gaussian_cosine(length=256, offset=0, width=10, freq=0.01):
    x = np.arange(length)
    center = length // 2 + offset    
    gauss = np.exp(-0.5 * ((x - center) / width) ** 2)
    carrier = np.cos(2 * np.pi * freq * (x - center))
    return gauss * carrier

def gaussian_cosine_batch(batch_shape, length=256, width=10, freq=0.01, offsets=None, snr=60):
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


# Test signal: sinc pulse
N = 800
x = np.arange(N)
width = 50
shift_amt = 10.7  # fractional offset

shifted = gaussian_cosine(length=N, width=width, offset=shift_amt)
ref = gaussian_cosine(length=N, width=width, offset=0)

snr = 40
shifted_batch = gaussian_cosine_batch((1920,1080), length=N, width=width, offsets=shift_amt, snr=snr)

# shifted = sinc_pulse(length=N, width=width, offset=shift_amt)
# ref = sinc_pulse(length=N, width=width, offset=0)

def run_shift_trials(ref, shifted, estimate_fn, snr_db, n_trials=100):
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

def run_shift_trials_3d(ref, shifted, estimate_fn, snr_db, n_trials=100):
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

    
run_shift_trials(ref, shifted, local_residual_phase_delay, snr_db=snr, n_trials=100)
run_shift_trials_3d(ref, shifted_batch, benchmark_cpu_multi_threaded, snr_db=snr, n_trials=1)


# print(benchmark_cpu_multi_threaded(shifted[np.newaxis, np.newaxis, :], ref))
# # Plot phase
# plt.plot(freqs, raw_phase, label='raw phase')
# plt.plot(freqs, unwrapped_phase, label='unwrapped phase')
plt.plot(add_noise(shifted, snr), label='noise')
plt.plot(add_noise(ref, snr), label='noise')
plt.xlabel("frequency (cycles/sample)")
plt.ylabel("phase difference (radians)")
plt.title("Phase difference vs frequency")
plt.legend()
plt.grid(True)
# plt.show()

