complex_vi_kernel_code_template = """
__global__ void MatrixMulKernel(float *TPM, float *TIM, float *RM,  float *V,  float *VNEW, float *QNEW, float *Error)
{
    // TPM   : Transition Probability Matrix
    // TIM   : Transition Index Matrix
    // RM    : Reward Matrix
    // V     : Seed Value Vector
    // VNEW  : Target Value Vector
    // QNEW  : Target Q (state,action) Vector
    // PNEW  : Target Policy Vector [Not Used]
    // Error : Target Bellman Backup Error Vector


    // 2D Thread ID (assuming that only *one* block will be executed) / One thread per state
    // continue only if thread id is less than total number of states.  
    int tx = %(MATRIX_SIZE)s * (blockDim.y*blockIdx.y + threadIdx.y) +  blockDim.x*blockIdx.x + threadIdx.x;

    if(tx < %(ROW_COUNT)s){ 

    //
    //       [----A3----]       |    <--Action offset-->     |                            |
    //     [----A2----]-]       v                            v                            v
    //   [----A1----]-]-]    => [--A1--]  [--B1--]  [--C1--] [--A2--]  [--B2--]  [--C2--] [--A3--]  [--B3--]  [--C3--]
    //   [----B1----]-]         ^         ^
    //   [----C1----]           |<-state->|
    //                             Offset

    // Each thread is responsible for bellman backup of one state
    // Pvalue is used to store the value of one state that is computed by the thread 
    // Since the Tensor is fully serialized , we manually need to keep track of ids

    int tx = %(MATRIX_SIZE)s * (blockDim.y*blockIdx.y + threadIdx.y) +  blockDim.x*blockIdx.x + threadIdx.x;
    int action_offset = (int) %(COL_COUNT)s* %(ROW_COUNT)s;
    int state_offset =  (int) %(COL_COUNT)s;


    // Initializing temporary variables (shared per state)
    float MaxPvalue = 0;
    float newVal = 0;
    float sum_of_all_st_axn_pairs = 0;



    for (int i = 0; i < %(ACTION_COUNT)s; ++i) {
        float Pvalue = 0;
        // Get expected value for all probable next states
        for (int k = 0; k < %(COL_COUNT)s; ++k) {
            int cell_id = (int) tx * state_offset + i * action_offset + k;
            int ns_idx = (int) TIM[cell_id];
            Pvalue +=  TPM[cell_id]* (RM[cell_id] + %(GAMMA)s * V[ns_idx]);
        } 

        // Keep track of Maximum Q(state,action) value
        if(i==0){MaxPvalue = Pvalue;}else{if(MaxPvalue < Pvalue){MaxPvalue = Pvalue;}} 

        // update Q(state,action) value
        int q_cell_id = (int) tx * %(ACTION_COUNT)s + i;
        QNEW[q_cell_id] = Pvalue;

        // keep track of the sum for slip probability
        sum_of_all_st_axn_pairs += Pvalue;
    }

    
    newVal = (1-%(SLIP_ACTION_PROB)s) * MaxPvalue + (%(SLIP_ACTION_PROB)s / %(ACTION_COUNT)s ) * sum_of_all_st_axn_pairs;

    // Write the matrix to device memory;
    // each thread writes one element
    VNEW[tx] = newVal;
    Error[tx] = VNEW[tx] - V[tx];
    //Error[tx] = (1-%(SLIP_ACTION_PROB)s);
    }
}
"""






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
