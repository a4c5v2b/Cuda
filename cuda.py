import cv2
import numpy as np
from numba import cuda
import time
import math



# GPU function
@cuda.jit
def process_gpu(img,channels):
    tx = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    ty = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y
    for c in range(channels):
        color = img[tx,ty][c]*2.0+30
        if color > 255:
            img[tx, ty][c] = 255
        elif color < 0:
            img[tx, ty][c] = 0
        else:
            img[tx, ty][c] = color

# CPU function
def process_cpu(img,dst):
    rows,cols,channels = img.shape
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = img[i,j][c]*2.0+30
                if color > 255:
                    dst[i,j][c] = 255
                elif color < 0:
                    dst[i,j][c] = 0
                else:
                    dst[i,j][c] = color

if __name__ == "__main__":
    # Create an image.
    img = cv2.imread('5k_image.jpg')
    rows,cols,channels=img.shape
    dst_cpu = img.copy()
    dst_gpu = img.copy()
    start_cpu = time.time()
    process_cpu(img,dst_cpu)
    end_cpu = time.time()
    print("CPU process time: {}".format(end_cpu-start_cpu))

    #GPU Function
    dImg = cuda.to_device(img) # Transfer to the GPU Memory
    threadsperblock = (16,16)
    blockspergrid_x = int(math.ceil(rows/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cols / threadsperblock[1]))
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    cuda.synchronize() #Synchronize cpu and gpu 告知CPU等待GPU執行完核函數後，再進行CPU端後續計算。這個過程被稱為同步
    start_gpu = time.time()
    process_gpu[blockspergrid,threadsperblock](dImg,channels)
    cuda.synchronize()
    dst_gpu = dImg.copy_to_host()
    end_gpu = time.time()
    print("GPU process time: {}".format(end_gpu - start_gpu))

    #Save
    cv2.imwrite("result_cpu.jpg", dst_cpu)
    cv2.imwrite("result_gpu.jpg", dst_gpu)
    print("Done")


