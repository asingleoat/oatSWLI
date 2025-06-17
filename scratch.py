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


def benchmark_cpu_multi_threaded(a_host, b_host):
    N = len(a_host)
    reference = np.conj(np.fft.rfft(b_host.astype(np.float32), axis=-1))

    # TODO: may want to limit num_cores to control memory pressure
    num_cores = os.cpu_count()
    
    chunk_size = a_host.shape[0] // num_cores
    chunks = []
    
    for i in range(num_cores):
        start_idx = i * chunk_size
        if i == num_cores - 1:  # last chunk gets remainder of values
            end_idx = a_host.shape[0]
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append((start_idx, end_idx))
    
    # pre-allocate output arrays
    rfft_shape = list(a_host.shape)
    # note shape for rfft which exploits hermitian symmetry
    rfft_shape[-1] = a_host.shape[-1] // 2 + 1
    
    # final result is real
    result = np.empty(a_host.shape, dtype=np.float32)
    # intermediate frequency domain result is complex for phase retrieval
    fc = np.empty(rfft_shape, dtype=np.complex64)
    
    def process_chunk(start_idx, end_idx):
        a_chunk = a_host[start_idx:end_idx]
        
        fa_chunk = np.fft.rfft(a_chunk, axis=-1)
        
        fc_chunk = fa_chunk * fb_vector
        
        fc[start_idx:end_idx] = fc_chunk
        result[start_idx:end_idx] = np.fft.irfft(fc_chunk, n=a_host.shape[-1], axis=-1).astype(np.float32)
        
        return None
    
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_chunk, start_idx, end_idx) 
                  for start_idx, end_idx in chunks]
        
        for future in futures:
            future.result()
    
    end = time.time()
    duration = end - start
    return result, fc, duration


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

# Test signal: sinc pulse
N = 800
x = np.arange(N)
width = 50
shift_amt = 10.7  # fractional offset

shifted = gaussian_cosine(length=N, width=width, offset=shift_amt)
ref = gaussian_cosine(length=N, width=width, offset=0)

# shifted = sinc_pulse(length=N, width=width, offset=shift_amt)
# ref = sinc_pulse(length=N, width=width, offset=0)

snr = 26
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

run_shift_trials(ref, shifted, local_residual_phase_delay, snr_db=snr, n_trials=100)

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

