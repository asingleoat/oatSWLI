import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import os

shape = (1920, 1080, 60*1)

# Data to convolve
np.random.seed(1)
a_host = np.random.rand(*shape).astype(np.complex64)
b_host = np.random.rand(*shape).astype(np.complex64)
b_host_vec = np.random.rand(shape[2]).astype(np.complex64)
fb_vector = np.fft.fft(b_host_vec, axis=-1)

# -------------------- CUDA --------------------
from pycuda.elementwise import ElementwiseKernel

import pycuda.driver as drv
import pycuda.gpuarray as cua
from pyvkfft.fft import fftn, ifftn
import pyvkfft.cuda as vkfft_cuda


class CUDAConvolver:
    def __init__(self, reference, dtype=np.complex64):
        drv.init()
        self.ctx = drv.Device(0).make_context()
        self.ctx.push()
        self.dtype = dtype

        shape = reference.shape
        self.plan = vkfft_cuda.VkFFTApp(
            shape,
            inplace=False,
            norm=2,
            dtype=dtype,
            enable_tuning=True,
            axes=[-1],
            ndim=1,
            backend="cuda",
        )
        self.plan_inplace = vkfft_cuda.VkFFTApp(
            shape,
            inplace=True,
            norm=2,
            dtype=dtype,
            enable_tuning=True,
            axes=[-1],
            ndim=1,
            backend="cuda",
        )

        self.cuda_mul_kernel = ElementwiseKernel(
            "pycuda::complex<float> *x, pycuda::complex<float> *y, pycuda::complex<float> *out",
            "out[i] = x[i] * y[i]",
            "cuda_pointwise_mul",
        )

        self.b_dev = cua.to_gpu(reference)
        self.a_ift = cua.empty_like(self.b_dev)  # preallocated
        self.a_fft = cua.empty_like(self.b_dev)  # preallocated
        self.a_result = cua.empty_like(self.b_dev)  # preallocated
        self.plan_inplace.fft(self.b_dev)

    def run(self, a_host):
        a_dev = cua.to_gpu(a_host)
        start = time.time()
        self.plan.fft(a_dev, self.a_fft)
        self.cuda_mul_kernel(self.a_fft, self.b_dev, self.a_result)
        self.plan.ifft(self.a_result, self.a_ift)
        r0 = self.a_ift.get()
        r1 = self.a_result.get()
        end = time.time()
        return (r0, r1, end - start)

    def close(self):
        self.ctx.pop()
        self.ctx.detach()


# -------------------- OpenCL --------------------
import pyopencl as cl
import pyopencl.array as cl_array
import pyvkfft.opencl as vkfft_opencl


class OpenCLConvolver:
    def __init__(self, reference, dtype=np.complex64):
        self.dtype = dtype

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        shape = reference.shape
        self.plan = vkfft_opencl.VkFFTApp(
            shape=shape,
            inplace=False,
            norm=2,
            dtype=dtype,
            queue=self.queue,
            axes=[-1],
            ndim=1,
            backend="opencl",
        )
        self.plan_inplace = vkfft_opencl.VkFFTApp(
            shape=shape,
            inplace=True,
            norm=2,
            dtype=dtype,
            queue=self.queue,
            axes=[-1],
            ndim=1,
            backend="opencl",
        )

        self.mul_kernel = cl.elementwise.ElementwiseKernel(
            self.ctx,
            """__global const float2 *a,
               __global const float2 *b,
               __global float2 *out""",
            """out[i].x = a[i].x * b[i].x - a[i].y * b[i].y;
               out[i].y = a[i].x * b[i].y + a[i].y * b[i].x;""",
            "complex_mul",
        )
        self.b_dev = cl_array.to_device(self.queue, reference)
        self.a_ift = cl_array.empty_like(self.b_dev)  # preallocated
        self.a_result = cl_array.empty_like(self.b_dev)  # preallocated
        self.plan_inplace.fft(self.b_dev)

    def run(self, a_host):
        a_dev = cl_array.to_device(self.queue, a_host)
        start = time.time()
        self.plan_inplace.fft(a_dev)
        self.mul_kernel(a_dev, self.b_dev, self.a_result)
        self.plan.ifft(self.a_result, self.a_ift)
        r0, r1 = (self.a_ift.get(), self.a_result.get())
        end = time.time()
        return (r0, r1, end - start)


import torch


class TorchConvolver:
    def __init__(self, reference, dtype=torch.complex64):
        self.dtype = dtype
        self.b = torch.fft.fft(
            torch.tensor(reference, device="cuda", dtype=self.dtype), dim=-1
        )

    def run(self, a_host):
        a = torch.tensor(a_host, device="cuda", dtype=self.dtype)
        start = time.time()
        fft_result = torch.fft.fft(a, dim=-1) * self.b
        result = torch.fft.ifft(fft_result, dim=-1)
        r0, r1 = (result.cpu().numpy(), fft_result.cpu().numpy())
        end = time.time()
        return (r0, r1, end - start)


def benchmark_torch(a_host, b_host):
    start = time.time()

    a = torch.from_numpy(a_host).to(device="cuda", dtype=torch.complex64)
    b = torch.from_numpy(b_host).to(device="cuda", dtype=torch.complex64)

    torch.cuda.synchronize()
    fft_result = torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1)
    result = torch.fft.ifft(fft_result)
    torch.cuda.synchronize()
    r0, r1 = result.cpu().numpy(), fft_result.cpu().numpy()
    end = time.time()
    return (r0, r1, end - start)


def benchmark_cpu_single_threaded(a_host, b_host):
    fb = b_host # np.fft.fft(b_host, axis=-1)  # this one can be precomputed

    start = time.time()
    fa = np.fft.fft(a_host, axis=-1)
    fc = fa * fb
    result = np.fft.ifft(fc, axis=-1).real
    end = time.time()
    duration = end - start
    return result, fc, duration

def benchmark_cpu_multi_threaded(a_host, b_host):
    num_cores = os.cpu_count()
    
    chunk_size = a_host.shape[0] // num_cores
    chunks = []
    
    for i in range(num_cores):
        start_idx = i * chunk_size
        if i == num_cores - 1:  # Last chunk gets remainder
            end_idx = a_host.shape[0]
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append((start_idx, end_idx))
    
    def process_chunk(start_idx, end_idx):
        a_chunk = a_host[start_idx:end_idx]
        
        fa_chunk = np.fft.fft(a_chunk, axis=-1)
        fc_chunk = fa_chunk * fb_vector
        result_chunk = np.fft.ifft(fc_chunk, axis=-1).real
        return result_chunk, fc_chunk
    
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_chunk, start_idx, end_idx) 
                  for start_idx, end_idx in chunks]
        
        chunk_results = []
        chunk_ffts = []
        for future in futures:
            result_chunk, fc_chunk = future.result()
            chunk_results.append(result_chunk)
            chunk_ffts.append(fc_chunk)
    
    # PARALLELISM OPPORTUNITY: Could pre-allocate output arrays and write chunks directly
    # to avoid concatenation step entirely
    result = np.concatenate(chunk_results, axis=0)
    fc = np.concatenate(chunk_ffts, axis=0)
    
    end = time.time()
    duration = end - start
    return result, fc, duration



if __name__ == "__main__":
    # (result_torch, fft_result, runtime) = benchmark_torch(a_host, b_host)
    # print(f"PyTorch convolution: {runtime:.4f}s")
    # print(f"Torch checksum: {np.sum(np.abs(result_torch))}\n")

    (result_cpu, fft_cpu, runtime) = benchmark_cpu_single_threaded(a_host, fb_vector)
    print(f"CPU convolution: {runtime:.4f}s")
    print(f"CPU checksum: {np.sum(np.abs(result_cpu))}")
    print(f"CPU fft checksum: {np.sum(np.abs(fft_cpu))}\n")

    (result_cpu, fft_cpu, runtime) = benchmark_cpu_multi_threaded(a_host, fb_vector)
    print(f"CPU convolution: {runtime:.4f}s")
    print(f"CPU checksum: {np.sum(np.abs(result_cpu))}")
    print(f"CPU fft checksum: {np.sum(np.abs(fft_cpu))}\n")

    
    # print("Context reuse benchmarks:")
    # opencl_conv = OpenCLConvolver(b_host, dtype=np.complex64)
    # (result_opencl, fft_opencl, runtime) = opencl_conv.run(a_host)
    # print(f"Cold OpenCL convolution: {runtime:.4f}s")
    # (result_opencl, fft_opencl, runtime) = opencl_conv.run(a_host)
    # print(f"Warm OpenCL convolution: {runtime*1000:.4f}ms")
    # print(f"OpenCL checksum: {np.sum(np.abs(result_opencl))}")
    # print(f"OpenCL fft checksum: {np.sum(np.abs(fft_opencl))}\n")

    # cuda_conv = CUDAConvolver(b_host, dtype=np.complex64)
    # (result_cuda, fft_cuda, runtime) = cuda_conv.run(a_host)
    # print(f"Cold CUDA convolution: {runtime:.4f}s")
    # (result_cuda, fft_cuda, runtime) = cuda_conv.run(a_host)
    # print(f"Warm CUDA convolution: {runtime*1000:.4f}ms")
    # print(f"CUDA checksum: {np.sum(np.abs(result_cuda))}")
    # print(f"CUDA fft checksum: {np.sum(np.abs(fft_cuda))}\n")
    # cuda_conv.close()

    # torch_conv = TorchConvolver(b_host, dtype=torch.complex64)
    # (result_torch, fft_torch, runtime) = torch_conv.run(a_host)
    # print(f"Cold TORCH convolution: {runtime:.4f}s")
    # (result_torch, fft_torch, runtime) = torch_conv.run(a_host)
    # print(f"Warm TORCH convolution: {runtime*1000:.4f}ms")
    # print(f"TORCH checksum: {np.sum(np.abs(result_torch))}")
    # print(f"TORCH fft checksum: {np.sum(np.abs(fft_torch))}\n")
