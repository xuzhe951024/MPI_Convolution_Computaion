from mpi4py import MPI
import numpy as np


def psum(a):
    locsum = np.sum(a)
    rcvBuf = np.array(0.0, 'd')
    MPI.COMM_WORLD.Allreduce([locsum, MPI.DOUBLE],
                             [rcvBuf, MPI.DOUBLE],
                             op=MPI.SUM)
    return rcvBuf


def convolve_func(main, kernel):
    DIMx, DIMy = main.shape
    convDimX = DIMx - (kernel.shape[0] - 1)
    convDimY = DIMy - (kernel.shape[1] - 1)
    conv = np.empty([convDimX, convDimY], dtype='int64')
    conv.fill(0)
    for i in range(convDimX):
        for j in range(convDimY):
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    conv[i, j] += main[i + k, j + l] * kernel[k, l]
    return conv


paddingFirst = np.zeros(4)

inputData = np.array(
    [[0, 0, 0, 0], [3, 9, 5, 9], [1, 7, 4, 3], [2, 1, 6, 5], [3, 9, 5, 9], [1, 7, 4, 3], [2, 1, 6, 5], [3, 9, 5, 9],
     [1, 7, 4, 3],
     [2, 1, 6, 5], [0, 0, 0, 0]])
kernel = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
conv = convolve_func(inputData, kernel)
conv1 = np.reshape(conv, (-1, 9))
print(conv1)
