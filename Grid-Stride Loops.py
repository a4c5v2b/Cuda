from numba import cuda

@cuda.jit
def gpu_print(N):
    idxwithinGrid = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    gridStride = cuda.gridDim.x*cuda.blockDim.x

    for i in range(idxwithinGrid, N, gridStride):
        print(i)


"""
跨步大小為網格中線程總數，用來計算。 迴圈的步長是網格中線程總數，這也是為什麼將這種方式稱為網格跨步。 
如果網格總線程數為1024，那麼0號線程將計算第0、1024、2048... 號的數據。 
這裏我們也不用再明確使用來判斷是否越界，因為迴圈也有這個判斷
"""

"""
使用網格跨步的優勢主要有：

擴展性：可以解決數據量比線程數大的問題
線程複用：CUDA線程啟動和銷毀都有開銷，主要是線程記憶體空間初始化的開銷;不使用網格跨步，CUDA需要啟動大於計算數的線程，每個線程內只做一件事情，做完就要被銷毀;
使用網格跨步，線程內有迴圈，每個線程可以幹更多事情，所有線程的啟動銷毀開銷更少。
方便調試：我們可以把核函數的執行配置寫為，如下所示，那麼核函數的跨步大小就成為了1，
核函數里的迴圈與CPU函數中順序執行的迴圈的邏輯一樣，
非常方便驗證CUDA並行計算與原來的CPU函數計算邏輯是否一致。[1, 1]forfor

"""

def main():
    gpu_print[2,4](32)
    cuda.synchronize()


if __name__ == "__main__":
    main()