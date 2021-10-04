from numba import cuda, float32
import numba
import numpy
import math
import time

TPB = 16  # Thread per block



@cuda.jit
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2)  # Return the index of thread
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp+=A[row,k]*B[k,col]
        C[row,col] = tmp


@numba.jit(nopython=True)  # Speed up CPU processing
def matmul_cpu(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            tmp = 0
            for k in range(A.shape[1]):
                tmp += A[x,k]*B[k,y]
            C[x,y] = tmp
    #return C

# @numba.jit(nopython=True)  # Speed up CPU processing
# def matmul_cpu_2(A, B, C):
#     for x in range(A.shape[0]):
#         for y in range(B.shape[1]):
#             tmp = 0
#             for k in range(A.shape[1]):
#                 tmp += A[x,k]*B[k,y]
#             C[x,y] = tmp
#     return C

@cuda.jit
def matmul_shared_mem(A, B, C):
    # 在shared memory中定義vector大小和類型
    # Vector可被整個block的所有thread share
    sA = cuda.shared.array(shape=(TPB,TPB),dtype=float32)
    sB = cuda.shared.array(shape=(TPB,TPB),dtype=float32)
    x, y = cuda.grid(2) # Index in the whole grid x = row, y = col

    tx = cuda.threadIdx.x # Index in the block
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        return
    tmp = 0
    for i in range(int(A.shape[1]/TPB)):
        sA[tx,ty] = A[x, ty+i*TPB] # +i*TPB = stride    A[x,ty+i*TPB] = A[0,ty+i*TPB] ... A[5,ty+i*TPB]
        sB[tx,ty] = B[tx+i*TPB, y] #B[tx+i*TPB, y] = B[tx+i*TPB, 0]...B[tx+i*TPB, 5]
        cuda.syncthreads()  # synchronize all threads, 線程同步等待block所有thread loading結束, 所有thread執行完先行下一步
        # 已將 A和B的submatrix copy到sA和sB

        for j in range(TPB):
            tmp += sA[tx,j]*sB[j,ty]  #從shared memory讀取data較global memory快
        cuda.syncthreads()
    C[x,y] = tmp



A = numpy.full((TPB*200, TPB*200), 3, numpy.float32)
B = numpy.full((TPB*200, TPB*200 ), 4, numpy.float32)
C_cpu = numpy.full((A.shape[0], B.shape[1]), 0, numpy.float32)

# Start in CPU
print("Start processing in CPU")
start_cpu = time.time()
matmul_cpu(A,B,C_cpu)
end_cpu = time.time()
time_cpu = (end_cpu - start_cpu)
print("CPU time: "+ str(time_cpu))


# Start in GPU
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)
#C_global_mem = cuda.to_device(C_cpu)
C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
C_shared_mem = cuda.device_array((A.shape[0], B.shape[1]))

threadsperblock = (TPB,TPB)
blockspergrid_x = int(math.ceil(A.shape[0]/threadsperblock[0]))
blockspergrid_y = int(math.ceil(A.shape[1]/threadsperblock[1]))
blockspergrid = (blockspergrid_x,blockspergrid_y)

print("Start processing in GPU")
start_gpu = time.time()
matmul_gpu[blockspergrid,threadsperblock](A_global_mem,B_global_mem,C_global_mem)
cuda.synchronize()
end_gpu = time.time()
time_gpu = (end_gpu - start_gpu)
print("GPU time(global memory): "+ str(time_gpu))
C_global_gpu = C_global_mem.copy_to_host()

start_gpu_shared = time.time()
matmul_gpu[blockspergrid,threadsperblock](A_global_mem,B_global_mem,C_shared_mem)
cuda.synchronize()
end_gpu_shared = time.time()
time_gpu = (end_gpu_shared - start_gpu_shared)
print("GPU time(shared memory): "+ str(time_gpu))
C_shared_gpu = C_shared_mem.copy_to_host()

# print("C_CPU:{}".format(C_cpu[:10]))
# print("GPU(global memory):{}".format(C_global_gpu[:10]))
# print("GPU(shared memory):{}".format(C_shared_gpu[:10]))

if (numpy.array_equal(C_cpu, C_global_gpu)) and (numpy.array_equal(C_cpu, C_shared_gpu)):
    print("The results of cpu and gpu are equal!")