from numba import cuda

def cpu_print():
    print("print by cpu.")

@cuda.jit
def gpu_print():
    # GPU核函数
    print("print by gpu.")

def main():
    gpu_print[1,2]()  # 出兩次print by gpu 因為有兩個thread run 同一個function
    cuda.synchronize() # 唔奇呢句會print cpu 先因為無等埋gpu
    cpu_print()

if __name__ == "__main__":
    main()