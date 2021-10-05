#from numba import jit, float32
from numba import cuda
import numpy as np
import numba
import time

size = 1000
x = np.random.rand(size,size).astype(np.float64)
c = np.zeros(x.shape[1],np.float64)


def tan_sum_cpu(x,c):

    tan_sum = 0
    for i in range(size):
        for j in range(size):
            tan_sum += np.tanh(x[i,j])

        c[i] = tan_sum


@numba.jit("(float64[:,:],float64[:])")  # Complexity = n*n
def tan_sum(x,c):

    tan_sum = 0
    for i in range(size):
        for j in range(size):
            tan_sum += np.tanh(x[i,j])

        c[i] = tan_sum


# Start in CPU
#print("The original c = {}".format(c))
print("Start processing in CPU")
start_cpu = time.time()
tan_sum_cpu(x,c)
end_cpu = time.time()
time_cpu = (end_cpu - start_cpu)
print("CPU time: "+ str(time_cpu))

start_cpu = time.time()
tan_sum(x,c)
#print("The c after the summation = {}".format(c))
end_cpu = time.time()
time_cpu = (end_cpu - start_cpu)
print("CPU time (numba: 1st compilation): "+ str(time_cpu))

start_cpu = time.time()
tan_sum(x,c)
#print("The c after the second summation = {}".format(c))
end_cpu = time.time()
time_cpu = (end_cpu - start_cpu)
print("CPU time (numba: after compilation): "+ str(time_cpu))




