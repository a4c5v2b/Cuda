from numba import cuda


"""
例如，我們想並行啟動1000個Thread，可以將blockDim設置為128，，向上取整為8。 
使用時，執行配置可以寫成，CUDA共啟動個Thread，實際計算時只使用前1000個Thread，多餘的24個Thread不進行計算。1000 ÷ 128 = 7.8gpuWork[8, 128]()
"""

def cpu_print(N):
    for i in range(0,N):
        print(i)


@cuda.jit
def gpu_print(N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if (idx < N):
        #print("Hello. I'm a thread "+str(cuda.threadIdx.x)+" in block "+str(cuda.blockIdx.x)+". My idx is "+str(idx))
        #print(cuda.threadIdx.x)
        #print(cuda.blockIdx.x)
        print(idx)





def main():
    print("gpu print: ")
    gpu_print[2,4](5)
    cuda.synchronize()
    print("cpu print: ")
    cpu_print(5)

if __name__ == "__main__":
    main()




