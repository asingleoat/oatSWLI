import numpy as np
import time

shape = (2**11, 2**11)

# Data to convolve
a_host = np.random.rand(*shape).astype(np.complex64)
b_host = np.random.rand(*shape).astype(np.complex64)

# -------------------- CUDA --------------------
from pycuda.elementwise import ElementwiseKernel

cuda_pointwise_mul = ElementwiseKernel(
    "pycuda::complex<float> *x, pycuda::complex<float> *y, pycuda::complex<float> *out",
    "out[i] = x[i] * y[i]",
    "cuda_pointwise_mul"
)


def benchmark_cuda(a_host,b_host):
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
        plan = VkFFTApp(a_gpu.shape, inplace=True, norm=1, dtype=np.complex64, enable_tuning=True)
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
    def __init__(self, shape, dtype=np.complex64):
        drv.init()
        self.ctx = drv.Device(0).make_context()
        self.ctx.push()
        self.dtype = dtype
        self.plan = vkfft_cuda.VkFFTApp(shape, inplace=True, norm=1, dtype=dtype, enable_tuning=True)
        drv.Context.synchronize()

        self.mul_kernel = ElementwiseKernel(
            "pycuda::complex<float> *x, pycuda::complex<float> *y, pycuda::complex<float> *out",
            "out[i] = x[i] * y[i]",
            "cuda_pointwise_mul"
        )

        
    def run(self, a_host, b_host):
        a_dev = cua.to_gpu(a_host)
        b_dev = cua.to_gpu(b_host)
        start = time.time()
        a_dev = self.plan.fft(a_dev)
        b_dev = self.plan.fft(b_dev)
        self.mul_kernel(a_dev, b_dev, a_dev)

        a_dev = self.plan.ifft(a_dev)
        end = time.time()
        return (a_dev.get(), end - start)

    def close(self):
        self.ctx.pop()
        self.ctx.detach()
    
# -------------------- OpenCL --------------------
def benchmark_opencl(a_host,b_host):
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
        "cl_pointwise_mul"
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
    def __init__(self, shape, dtype=np.complex64):
        self.dtype = dtype

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        
        self.plan = vkfft_opencl.VkFFTApp(shape=shape, inplace=True, dtype=dtype, queue=self.queue)
        self.tmp = cl_array.empty(self.queue, shape, dtype)

        self.mul_kernel = cl.elementwise.ElementwiseKernel(
            self.ctx,
            """__global const float2 *a,
               __global const float2 *b,
               __global float2 *out""",
            """out[i].x = a[i].x * b[i].x - a[i].y * b[i].y;
               out[i].y = a[i].x * b[i].y + a[i].y * b[i].x;""",
            "complex_mul"
        )

    def run(self, a_host, b_host):
        a_dev = cl_array.to_device(self.queue, a_host)
        b_dev = cl_array.to_device(self.queue, b_host)
        start = time.time()
        a_dev = self.plan.fft(a_dev)
        b_dev = self.plan.fft(b_dev)
        self.mul_kernel(a_dev, b_dev, a_dev)
        a_dev = self.plan.ifft(a_dev)
        end = time.time()     
        return (a_dev.get(), end - start)



import torch

class TorchConvolver:
    def __init__(self, dtype=torch.complex64):
        self.dtype = dtype

    def run(self, a_host, b_host):
        a = torch.tensor(a_host, device='cuda', dtype=self.dtype)
        b = torch.tensor(b_host, device='cuda', dtype=self.dtype)
        start = time.time()
        result = torch.fft.ifftn(torch.fft.fftn(a) * torch.fft.fftn(b))
        end = time.time()
        return (result.cpu().numpy(), end - start)


def benchmark_torch(a_host,b_host):
    a = torch.from_numpy(a_host).to(device='cuda',dtype=torch.complex64)
    b = torch.from_numpy(b_host).to(device='cuda',dtype=torch.complex64)    

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

    return result, duration

if __name__ == "__main__":
    print("One off benchmarks:")
    result_opencl = benchmark_opencl(a_host,b_host)
    print(f"OpenCL checksum: {np.sum(np.abs(result_opencl))}\n")
    result_torch = benchmark_torch(a_host,b_host)
    print(f"Torch checksum: {result_torch.abs().sum().item()}\n")
    result_cuda = benchmark_cuda(a_host,b_host)
    print(f"CUDA checksum: {np.sum(np.abs(result_cuda))}\n")

    (result_cpu, runtime) = benchmark_cpu(a_host,b_host)
    print(f"CPU convolution: {runtime:.4f}s")
    print(f"CPU checksum: {np.sum(np.abs(result_cpu))}\n")

    
    print("Context reuse benchmarks:")
    opencl_conv = OpenCLConvolver(shape, dtype=np.complex64)
    (result_opencl, runtime) = opencl_conv.run(a_host, b_host)
    print(f"Cold OpenCL convolution: {runtime:.4f}s")
    (result_opencl, runtime) = opencl_conv.run(a_host, b_host)
    print(f"Warm OpenCL convolution: {runtime*1000:.4f}ms")
    print(f"OpenCL checksum: {np.sum(np.abs(result_opencl))}\n")


    cuda_conv = CUDAConvolver(shape, dtype=np.complex64)
    (result_cuda, runtime) = cuda_conv.run(a_host, b_host)
    print(f"Cold CUDA convolution: {runtime:.4f}s")
    (result_cuda, runtime) = cuda_conv.run(a_host, b_host)
    print(f"Warm CUDA convolution: {runtime*1000:.4f}ms")
    print(f"CUDA checksum: {np.sum(np.abs(result_cuda))}\n")

    cuda_conv.close()

    torch_conv = TorchConvolver(dtype=torch.complex64)
    (result_torch, runtime) = torch_conv.run(a_host, b_host)
    print(f"Cold TORCH convolution: {runtime:.4f}s")
    (result_torch, runtime) = torch_conv.run(a_host, b_host)
    print(f"Warm TORCH convolution: {runtime*1000:.4f}ms")
    print(f"TORCH checksum: {np.sum(np.abs(result_torch))}\n")    
