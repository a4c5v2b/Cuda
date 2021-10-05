from numba import cuda,float32,complex128
import numpy as np
import time
import math

def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N,1))
    M = np.exp(-2j*np.pi*k*n/N)
    return np.dot(M,x)


def FFT_recursive(x):
     """A recursive implementation of the 1D Cooley-Tukey FFT"""
     x = np.asarray(x, dtype=float)
     N = x.shape[0]

     if N % 2 > 0:
         raise ValueError("size of x must be a power of 2")
     elif N <= 32:  # this cutoff should be optimized
         return DFT_slow(x)
     else:
         X_even = FFT_recursive(x[::2]) #0,2,4,6 -> 0,4 & 2,6 ->
         X_odd = FFT_recursive(x[1::2]) #1,3,5,7 -> 1,5 & 3,7 ->
         factor = np.exp(-2j * np.pi * np.arange(N) / N)
         return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
                                X_even + factor[int(N / 2):] * X_odd])

def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)] # First half columns
        X_odd = X[:, int(X.shape[1] / 2):]  # Second half columns
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                         / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                    X_even - factor * X_odd])

    return X.ravel()

def FFT_Invoke_cuda(x):
    N = x.shape[0]
    N_min = min(N, 32)
    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")
    x = np.asarray(x, dtype=np.float32).reshape(N_min, -1)
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    M = np.asarray(M, dtype=np.complex64)
    threadsperblock = (16,16)
    blockspergrid_x = int(math.ceil(x.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(x.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    x_cuda = cuda.to_device(x)
    M_cuda = cuda.to_device(M)
    result_cuda = cuda.device_array((M.shape[0],x.shape[1]),dtype=np.complex64)
    FFT_cuda[blockspergrid, threadsperblock](M_cuda,x_cuda,result_cuda)
    cuda.synchronize()
    result_cuda_host = result_cuda.copy_to_host()
    # build-up each level of the recursive calculation all at once
    while result_cuda_host.shape[0] < N:
        result_cuda_host_even = result_cuda_host[:, :int(result_cuda_host.shape[1] / 2)]  # First half columns
        result_cuda_host_odd = result_cuda_host[:, int(result_cuda_host.shape[1] / 2):]  # Second half columns
        factor = np.exp(-1j * np.pi * np.arange(result_cuda_host.shape[0])/ result_cuda_host.shape[0])[:, None]
        result_cuda_host = np.vstack([result_cuda_host_even + factor * result_cuda_host_odd,
                                      result_cuda_host_even - factor * result_cuda_host_odd])

    return result_cuda_host.ravel()

@cuda.jit("(complex64[:,:],float32[:,:],complex64[:,:])")
def FFT_cuda(M,x,result):
    threadsperblock = 16
    sA = cuda.shared.array(shape=(threadsperblock, threadsperblock), dtype=complex128)
    sB = cuda.shared.array(shape=(threadsperblock, threadsperblock), dtype=float32)
    tx = cuda.threadIdx.x  # Index in the block
    ty = cuda.threadIdx.y

    row, col = cuda.grid(2)


    gridStride_x = cuda.gridDim.x * cuda.blockDim.x
    gridStride_y = cuda.gridDim.y * cuda.blockDim.y

    #if row < M.shape[0] and col < x.shape[1]:
    for i in range(row, M.shape[0], gridStride_x):
        for m in range(col, M.shape[1], gridStride_y):
            tmp = 0
            # for k in range(M.shape[1]):
            #     tmp += M[row, k] * x[k, col]
            # result[row, col] = tmp
            for s in range(int(M.shape[1] / threadsperblock)):
                sA[tx, ty] = M[row, ty + s * threadsperblock]  # +i*TPB = stride    A[x,ty+i*TPB] = A[0,ty+i*TPB] ... A[5,ty+i*TPB]
                sB[tx, ty] = x[tx + s * threadsperblock, col]  # B[tx+i*TPB, y] = B[tx+i*TPB, 0]...B[tx+i*TPB, 5]
                cuda.syncthreads()  # synchronize all threads, 線程同步等待block所有thread loading結束, 所有thread執行完先行下一步
                # 已將 A和B的submatrix copy到sA和sB

                for j in range(threadsperblock):
                    tmp += sA[tx, j] * sB[j, ty]  # 從shared memory讀取data較global memory快
                cuda.syncthreads()

                result[row,col] = tmp



#
# #@cuda.jit("(uint32[:,:],uint32[:,:],uint32[:,:])")
# @cuda.jit("(complex128[:,:],complex128[:,:],complex128[:,:])")
# def dot_trial(a,b,c):
#     row,col = cuda.grid(2)
#
#     if row < c.shape[0] and col < c.shape[1]:
#
#         tmp = 0
#         for k in range(a.shape[1]):
#             tmp += a[row, k] * b[k, col]
#         c[row, col] = tmp
#
#
#
#
# # a = np.arange(6).reshape((3,2))
# # b = a.T
#
# a = np.random.random((32,32)) + np.random.random((32,32)) * 1j
# #a = np.asarray(a, dtype=np.complex64)
# b = (np.random.random((16,32)) + np.random.random((16,32)) * 1j).T
# #b = np.asarray(b, dtype=np.complex64)
#
# np_dot = np.dot(a,b)
#
# #a = cuda.to_device(a)
# #b = cuda.to_device(b)
# c = cuda.device_array((a.shape[0],b.shape[1]),dtype=np.complex128)
# cc = c.copy_to_host()
# threadsperblock = [16,16]
# blockspergrid_x = math.ceil(a.shape[0] / threadsperblock[0])
# blockspergrid_y = math.ceil(b.shape[1] / threadsperblock[1])
# blockspergrid = (blockspergrid_x, blockspergrid_y)
#
# dot_trial[blockspergrid,threadsperblock](a,b,c)
# cuda_dot = c.copy_to_host()

#np.allclose(np_dot, cuda_dot)

x = np.random.random(32768)
#x = np.arange(8)
#%timeit DFT_slow(x)
#%timeit np.fft.fft(x)



# DFT_slow_start_time = time.time()
# DFT_slow_result = DFT_slow(x)
# DFT_slow_end_time = time.time()
# print("The time used for the slow DFT is: " + str(DFT_slow_end_time-DFT_slow_start_time))


numpy_fft_start_time = time.time()
np_result = np.fft.fft(x)
numpy_fft_end_time = time.time()
print("The time used for the numpy DFT is: " + str(numpy_fft_end_time-numpy_fft_start_time))

FFT_recursive_start_time = time.time()
FFT_recursive_result = FFT_recursive(x)
FFT_recursive_end_time = time.time()
print("The time used for the FFT_recursive is: " + str(FFT_recursive_end_time-FFT_recursive_start_time))

FFT_vectorized_start_time = time.time()
FFT_vectorized_result = FFT_vectorized(x)
FFT_vectorized_end_time = time.time()
print("The time used for the FFT_vectorized is: " + str(FFT_vectorized_end_time-FFT_vectorized_start_time))

FFT_Invoke_cuda_start_time = time.time()
FFT_Invoke_cuda_result = FFT_Invoke_cuda(x)
FFT_Invoke_cuda_end_time = time.time()
print("The time used for the FFT_cuda is: " + str(FFT_Invoke_cuda_end_time-FFT_Invoke_cuda_start_time))


#if(np.allclose(FFT_recursive_result, DFT_slow_result) == True)

if (np.allclose(FFT_recursive_result, np_result) == True and np.allclose(FFT_recursive_result, FFT_vectorized_result) == True):
    print("Results of all approaches are equal!")