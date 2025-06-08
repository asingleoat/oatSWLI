import numpy as np
import time

shape = (2**10, 2**10)

# Data to convolve
np.random.seed(1)
a_host = np.random.rand(*shape).astype(np.complex64)
b_host = np.random.rand(*shape).astype(np.complex64)


# -------------------- CUDA --------------------
from pycuda.elementwise import ElementwiseKernel

cuda_pointwise_mul = ElementwiseKernel(
    "pycuda::complex<float> *x, pycuda::complex<float> *y, pycuda::complex<float> *out",
    "out[i] = x[i] * y[i]",
    "cuda_pointwise_mul",
)


def benchmark_cuda(a_host, b_host):
    import pycuda.driver as drv
    import pycuda.gpuarray as cua
    from pyvkfft.fft import fftn, ifftn
    from pyvkfft.cuda import VkFFTApp

    drv.init()
    ctx = drv.Device(0).make_context()
    ctx.push()

    try:
        a_gpu = cua.to_gpu(a_host)
        b_gpu = cua.to_gpu(b_host)
        plan = VkFFTApp(
            a_gpu.shape, inplace=True, norm=2, dtype=np.complex64, enable_tuning=True
        )
        drv.Context.synchronize()

        start = time.time()
        A = plan.fft(a_gpu)
        B = plan.fft(b_gpu)

        prod = cua.empty_like(A)
        cuda_pointwise_mul(A, B, prod)
        C = prod

        c_gpu = plan.ifft(C)
        drv.Context.synchronize()
        end = time.time()

        c_result = c_gpu.get()
        print(f"CUDA convolution: {end - start:.4f}s")

    finally:
        ctx.pop()
        ctx.detach()
        return c_result


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
            shape, inplace=False, norm=2, dtype=dtype, enable_tuning=True
        )
        self.plan_inplace = vkfft_cuda.VkFFTApp(
            shape, inplace=True, norm=2, dtype=dtype, enable_tuning=True
        )

        self.cuda_mul_kernel = ElementwiseKernel(
            "pycuda::complex<float> *x, pycuda::complex<float> *y, pycuda::complex<float> *out",
            "out[i] = x[i] * y[i]",
            "cuda_pointwise_mul",
        )
        self.b_dev = cua.to_gpu(reference)
        self.a_ift = cua.empty_like(self.b_dev)  # preallocated
        self.a_result = cua.empty_like(self.b_dev)  # preallocated
        self.plan_inplace.fft(self.b_dev)

    def run(self, a_host):
        a_dev = cua.to_gpu(a_host)
        start = time.time()
        self.plan_inplace.fft(a_dev)
        self.cuda_mul_kernel(a_dev, self.b_dev, self.a_result)
        self.plan.ifft(self.a_result, self.a_ift)
        end = time.time()
        return (self.a_ift.get(), self.a_result.get(), end - start)

    def close(self):
        self.ctx.pop()
        self.ctx.detach()


# -------------------- OpenCL --------------------
def benchmark_opencl(a_host, b_host):
    import pyopencl as cl
    import pyopencl.array as cla
    from pyvkfft.fft import fftn, ifftn

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    # Create OpenCL context and queue
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    from pyopencl.elementwise import ElementwiseKernel as ClElementwiseKernel

    context = queue.context
    cl_pointwise_mul = ClElementwiseKernel(
        context,
        """
        __global const float2 *x,
        __global const float2 *y,
        __global float2 *out
        """,
        """
        out[i].x = x[i].x * y[i].x - x[i].y * y[i].y;
        out[i].y = x[i].x * y[i].y + x[i].y * y[i].x;
        """,
        "cl_pointwise_mul",
    )

    a_cl = cla.to_device(queue, a_host)
    b_cl = cla.to_device(queue, b_host)

    queue.finish()
    start = time.time()
    A = fftn(a_cl)
    B = fftn(b_cl)
    prod = cl.array.empty_like(A)
    cl_pointwise_mul(A, B, prod)
    C = prod

    c_cl = ifftn(C)
    queue.finish()
    end = time.time()

    c_result = c_cl.get()
    print(f"OpenCL convolution: {end - start:.4f}s")
    return c_result


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
            shape=shape, inplace=False, norm=2, dtype=dtype, queue=self.queue
        )
        self.plan_inplace = vkfft_opencl.VkFFTApp(
            shape=shape, inplace=True, norm=2, dtype=dtype, queue=self.queue
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
        end = time.time()
        return (self.a_ift.get(), self.a_result.get(), end - start)

import torch
class TorchConvolver:
    def __init__(self, dtype=torch.complex64):
        self.dtype = dtype

    def run(self, a_host, b_host):
        a = torch.tensor(a_host, device="cuda", dtype=self.dtype)
        b = torch.tensor(b_host, device="cuda", dtype=self.dtype)
        start = time.time()
        result = torch.fft.ifftn(torch.fft.fftn(a) * torch.fft.fftn(b))
        end = time.time()
        return (result.cpu().numpy(), end - start)


def benchmark_torch(a_host, b_host):
    a = torch.from_numpy(a_host).to(device="cuda", dtype=torch.complex64)
    b = torch.from_numpy(b_host).to(device="cuda", dtype=torch.complex64)

    torch.cuda.synchronize()
    start = time.time()
    result = torch.fft.ifft2(torch.fft.fft2(a) * torch.fft.fft2(b))
    torch.cuda.synchronize()
    end = time.time()

    print(f"PyTorch convolution: {end - start:.4f}s")
    return result


def benchmark_cpu(a_host, b_host):
    start = time.time()
    fa = np.fft.fft2(a_host)
    fb = np.fft.fft2(b_host)
    fc = fa * fb
    result = np.fft.ifft2(fc).real
    end = time.time()
    duration = end - start

    return result, fc, duration


if __name__ == "__main__":
    print("One off benchmarks:")
    result_opencl = benchmark_opencl(a_host, b_host)
    print(f"OpenCL checksum: {np.sum(np.abs(result_opencl))}\n")
    result_torch = benchmark_torch(a_host, b_host)
    print(f"Torch checksum: {result_torch.abs().sum().item()}\n")
    result_cuda = benchmark_cuda(a_host, b_host)
    print(f"CUDA checksum: {np.sum(np.abs(result_cuda))}\n")

    (result_cpu, fft_cpu, runtime) = benchmark_cpu(a_host, b_host)
    print(f"CPU convolution: {runtime:.4f}s")
    print(f"CPU checksum: {np.sum(np.abs(result_cpu))}")
    print(f"CPU fft checksum: {np.sum(np.abs(fft_cpu))}\n")

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

    torch_conv = TorchConvolver(dtype=torch.complex64)
    (result_torch, runtime) = torch_conv.run(a_host, b_host)
    print(f"Cold TORCH convolution: {runtime:.4f}s")
    (result_torch, runtime) = torch_conv.run(a_host, b_host)
    print(f"Warm TORCH convolution: {runtime*1000:.4f}ms")
    print(f"TORCH checksum: {np.sum(np.abs(result_torch))}\n")
