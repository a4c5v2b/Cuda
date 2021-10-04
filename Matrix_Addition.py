from numba import cuda
import numpy as np
import math
import time

"""
一個二維配置，某個線程在矩陣中的位置可以表示為：
col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x


如何將二維Block映射到自己的數據上並沒有固定的映射方法，一般情況將映射為矩陣的行，將映射為矩陣的列。 Numba提供了一個更簡單的方法幫我們計算線程的編號：.x.y

row, col = cuda.grid(2)
其中，參數2表示這是一個2維的執行配置。 1維或3維的時候，可以將參數改為1或3。

對應的執行設定也要改為二維：

threads_per_block = (16, 16)
blocks_per_grid = (32, 32)
gpu_kernel[blocks_per_grid, threads_per_block]
(16, 16)的二維Block是一個常用的配置，共256個線程。 之前也曾提到過，每個Block的Thread個數最好是128、256或512，這與GPU的硬體架構高度相關。


"""



# GPU Processing
@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    if idx < n:
        result[idx] = a[idx] + b[idx]

# CPU Processing
def cpu_add(a, b, result, n):

    for i in range(n):
        result[i] = a[i] + b[i]
    return result

def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2*x


    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    cpu_result = np.zeros(n)
    gpu_result = cpu_result
    gpu_result_device = cuda.device_array(n)

    gpu_start_time_device = time.time()
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    gpu_add[blocks_per_grid, threads_per_block](x, y, gpu_result_device, n)
    cuda.synchronize()
    gpu_end_time_device = time.time()
    gpu_time_used_device = gpu_end_time_device - gpu_start_time_device

    gpu_start_time = time.time()
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n/threads_per_block)
    gpu_add[blocks_per_grid,threads_per_block](x,y,gpu_result,n)
    cuda.synchronize()
    gpu_end_time = time.time()
    gpu_time_used = gpu_end_time - gpu_start_time



    cpu_start_time = time.time()
    cpu_result = cpu_add(x,y, cpu_result, n)
    cpu_end_time = time.time()
    cpu_time_used = cpu_end_time - cpu_start_time

    if(np.array_equal(cpu_result, gpu_result, gpu_result_device)):
        print("The results of cpu and gpu are equal!")

    print("GPU processing time (using host array) = "+ str(gpu_time_used))
    print("GPU processing time (using device array) = " + str(gpu_time_used_device))
    print("CPU processing time = "+ str(cpu_time_used))


if __name__ == "__main__":
    main()
