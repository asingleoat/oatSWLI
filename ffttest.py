import numpy as np
import time

shape = (2**10, 2**6, 2**6)

# Data to convolve
np.random.seed(1)
a_host = np.random.rand(*shape).astype(np.complex64)
b_host = np.random.rand(*shape).astype(np.complex64)


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


def benchmark_cpu(a_host, b_host):
    fb = np.fft.fft(b_host, axis=-1)  # this one can be precomputed

    start = time.time()
    fa = np.fft.fft(a_host, axis=-1)
    fc = fa * fb
    result = np.fft.ifft(fc, axis=-1).real
    end = time.time()
    duration = end - start
    return result, fc, duration


if __name__ == "__main__":
    # (result_torch, fft_result, runtime) = benchmark_torch(a_host, b_host)
    # print(f"PyTorch convolution: {runtime:.4f}s")
    # print(f"Torch checksum: {np.sum(np.abs(result_torch))}\n")

    # (result_cpu, fft_cpu, runtime) = benchmark_cpu(a_host, b_host)
    # print(f"CPU convolution: {runtime:.4f}s")
    # print(f"CPU checksum: {np.sum(np.abs(result_cpu))}")
    # print(f"CPU fft checksum: {np.sum(np.abs(fft_cpu))}\n")

    print("Context reuse benchmarks:")
    opencl_conv = OpenCLConvolver(b_host, dtype=np.complex64)
    (result_opencl, fft_opencl, runtime) = opencl_conv.run(a_host)
    print(f"Cold OpenCL convolution: {runtime:.4f}s")
    (result_opencl, fft_opencl, runtime) = opencl_conv.run(a_host)
    print(f"Warm OpenCL convolution: {runtime*1000:.4f}ms")
    print(f"OpenCL checksum: {np.sum(np.abs(result_opencl))}")
    print(f"OpenCL fft checksum: {np.sum(np.abs(fft_opencl))}\n")

    cuda_conv = CUDAConvolver(b_host, dtype=np.complex64)
    (result_cuda, fft_cuda, runtime) = cuda_conv.run(a_host)
    print(f"Cold CUDA convolution: {runtime:.4f}s")
    (result_cuda, fft_cuda, runtime) = cuda_conv.run(a_host)
    print(f"Warm CUDA convolution: {runtime*1000:.4f}ms")
    print(f"CUDA checksum: {np.sum(np.abs(result_cuda))}")
    print(f"CUDA fft checksum: {np.sum(np.abs(fft_cuda))}\n")
    cuda_conv.close()

    # torch_conv = TorchConvolver(b_host, dtype=torch.complex64)
    # (result_torch, fft_torch, runtime) = torch_conv.run(a_host)
    # print(f"Cold TORCH convolution: {runtime:.4f}s")
    # (result_torch, fft_torch, runtime) = torch_conv.run(a_host)
    # print(f"Warm TORCH convolution: {runtime*1000:.4f}ms")
    # print(f"TORCH checksum: {np.sum(np.abs(result_torch))}")
    # print(f"TORCH fft checksum: {np.sum(np.abs(fft_torch))}\n")
