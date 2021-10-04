from numba import cuda
import numpy as np
import math
from time import time



@cuda.jit
def vector_add(a,b,result,n):
    idx = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    if idx < n:
        result[idx] = a[idx] + b[idx]

def main():
    n = 20000000
    x = np.random.uniform(10,20,n)
    y = np.random.uniform(10,20,n)

    start = time()
    """使用默認流"""
    # Host To device
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    result_device = cuda.device_array(n)
    result_streams_device = cuda.device_array(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n/threads_per_block)

    vector_add[blocks_per_grid,threads_per_block](x_device,y_device,result_device,n) # Kernel

    # Device To Host
    default_stream_result = result_device.copy_to_host()
    cuda.synchronize()
    print("gpu vector add time " + str(time()-start))

    start = time()

    # 使用5個stream
    number_of_streams = 5
    #每個流處理的data為原本的1/5
    segment_size = n//number_of_streams

    # Built 5個 cuda stream
    stream_list = list()
    for i in range(0, number_of_streams):
        stream = cuda.stream()
        stream_list.append(stream)


    threads_per_block = 1024
    blocks_per_grid = math.ceil(segment_size/threads_per_block)
    streams_result = np.empty(n)

    #啟動多個stream
    for i in range(0, number_of_streams):

        # Host To Device
        x_i_device = cuda.to_device(x[i*segment_size:(i+1)*segment_size],stream=stream_list[i])
        y_i_device = cuda.to_device(y[i * segment_size:(i + 1) * segment_size], stream=stream_list[i])

        vector_add[blocks_per_grid,threads_per_block,stream_list[i]](
            x_i_device,
            y_i_device,
            result_streams_device[i*segment_size:(i+1)*segment_size],
            segment_size)


        # Device To Host
        streams_result[i*segment_size:(i+1)*segment_size] = result_streams_device[i*segment_size:(i+1)*segment_size].copy_to_host()
    cuda.synchronize()

    print("gpu streams vector add time " + str(time()-start))

    if (np.array_equal(default_stream_result, streams_result)):
        print("result correct")


if __name__ == "__main__":
    main()


