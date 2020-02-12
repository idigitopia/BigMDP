from deeprmax.vi.VI_helper_funcs import hAsh, unhAsh

def hm_dist(l1: list, l2: list):
    return sum(abs(le1 - le2) for le1, le2 in zip(l1, l2))

def get_nearest_neighbor(pi, hs:str,unhash_dict_:dict):
    """
    takes hashed state as input
    """
    if hs in pi:
        nearest_neighbor = hs
    else:
        unhashed_hs = unhAsh(hs)
        hm_dist_dict = {s_hat: hm_dist(unhashed_hs, unhash_dict_[s_hat]) for s_hat in pi}
        nearest_neighbor = min(hm_dist_dict.keys(), key=(lambda k: hm_dist_dict[k]))

    return nearest_neighbor

import math as mth

NN_kernel_code_template = """
__global__ void NNKernel(float *InVector, float *outVector, float *searchVector)
{
    // 2D Thread ID (assuming that only *one* block will be executed)
    int tx = %(MATRIX_SIZE)s * (blockDim.y*blockIdx.y + threadIdx.y) +  blockDim.x*blockIdx.x + threadIdx.x;

    if(%(POP_COUNT)s > tx){
    float dist = 0;
    for (int i = 0; i <%(VEC_SIZE)s; ++i) {
        dist += abs(searchVector[%(VEC_SIZE)s*tx + i] - InVector[i]);
    }
    outVector[tx] = dist;
    }
}
"""
import numpy

def get_nearest_neighbors_gpu(searchMatrix, queryVector):
    # Define kernel code template
    import pycuda.autoinit

    POP_COUNT, VEC_SIZE = searchMatrix.shape
    InVector = gpuarray.to_gpu(queryVector)
    OutVector = gpuarray.to_gpu(np.zeros((POP_COUNT,), dtype=numpy.float32))
    SearcVector = gpuarray.to_gpu(searchMatrix)

    MATRIX_SIZE = mth.ceil(mth.sqrt(POP_COUNT))
    BLOCK_SIZE = 16

    if MATRIX_SIZE % BLOCK_SIZE != 0:
        grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
    else:
        grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

    # Define actual kernel
    kernel_code = NN_kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE,
        'VEC_SIZE': VEC_SIZE,
        'POP_COUNT': POP_COUNT
    }
    # Compile and get function
    mod = compiler.SourceModule(kernel_code)
    get_nearest_fxn = mod.get_function("NNKernel")

    # call the function
    get_nearest_fxn(  # inputs
        InVector, OutVector, SearcVector,
        # grid
        grid=grid,
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block=(BLOCK_SIZE, BLOCK_SIZE, 1))

    OutVector_cpu = OutVector.get()
    SearcVector.gpudata.free()
    InVector.gpudata.free()
    OutVector.gpudata.free()
    # return out vector
    return OutVector_cpu
