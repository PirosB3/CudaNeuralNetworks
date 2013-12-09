#include <math.h>

#define WARP_SIZE 16
#define DEBUG true


__global__ void updateWeightsCUDA(float *weights, float *changes, float *delta_outputs, float *inputs, int n_inputs, int n_outputs) {
    int width = n_outputs;
    int height = n_inputs;
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;

    if ((threadX < width) && (threadY < height)) {
        int idx = width * threadY + threadX;
        float change = delta_outputs[threadX] * inputs[threadY];
        
        weights[idx] += 0.5 * change + 0.5 * changes[idx];
        changes[idx] = change;
    }

}

__global__ void mapStepCUDA(float *inputs, float *matrix, float *buffer, int width, int height) {
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;

    if ((threadX < width) && (threadY < height)) {
        int idx = width * threadY + threadX;
        buffer[idx] = inputs[threadY] * matrix[idx];
    }
}

__global__ void reduceStepCUDA(float *input, float *output, int width, int height) {

    __shared__ float sharedMemory[WARP_SIZE * WARP_SIZE];

    // STEP 1: exclude all threads that do not depend from problem
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;

    if ((threadX < width) && (threadY < height)) {

        // STEP 2: Move to shared memory
        int gridId = threadY * width + threadX;
        int blockId = threadIdx.y * blockDim.x + threadIdx.x;
        sharedMemory[blockId] = input[gridId];
        __syncthreads();

        int n = (int)ceil((float)blockDim.y/2);
        while(n >= 1) {
            if (threadIdx.y < n) {

                if ((threadY + n) < height) {
                    int firstIndex = blockId;
                    int secondIndex = blockDim.x * (threadIdx.y + n) + threadIdx.x;
                    sharedMemory[firstIndex] += sharedMemory[secondIndex];
                }
            }
            __syncthreads();
            if (n == 1) {
                break;
            } else {
                n = (int)ceil((float)n/2);
            }
        }
        __syncthreads();

        // STEP 3: Write back results
        if (threadIdx.y == 1) {
            output[blockIdx.y * width + threadX] = sharedMemory[threadIdx.x];
        }
    }
}

void setWeightsForLayers(float *weights, float *changes, float *delta_outputs, float *inputs, int n_inputs, int n_outputs) {

    // Copy to device memory
    int grid_size = n_inputs * n_outputs;
    float *weights_d = _copyHostDevice(weights, grid_size);
    float *changes_d = _copyHostDevice(changes, grid_size);
    float *delta_outputs_d = _copyHostDevice(weights, grid_size);
    float *inputs_d = _copyHostDevice(inputs, n_inputs);

    // Define block structure
    dim3 block(WARP_SIZE, WARP_SIZE);
    dim3 grid = getGridBasedOnBlockSize(n_outputs, n_inputs, WARP_SIZE);

    // RUN RUN RUN!
    updateWeightsCUDA<<<grid, block>>>(weights_d, changes_d, delta_outputs_d, inputs_d, n_inputs, n_outputs);

    // Copy back weights and momenutm
    weights = _copyDeviceHost(weights_d, grid_size, weights);
    changes = _copyDeviceHost(changes_d, grid_size, changes);
}

void update_layer(float *src_layer, float *dst_layer, int src_n, int dst_n, float *weights) {
    dim3 block(WARP_SIZE, WARP_SIZE);

    float *src_layer_d, *weights_d, *buffer_d;
    int total = src_n * dst_n;
 
    // Allocate input in global memory
    src_layer_d = _copyHostDevice(src_layer, src_n);
    weights_d = _copyHostDevice(weights, total);
    cudaMalloc((void**)&buffer_d, sizeof(float) * total);
 
    // Create block dimensions and run parallel update layer
    int gridX = (int)ceil((float)dst_n/WARP_SIZE);
    int gridY = (int)ceil((float)src_n/WARP_SIZE);
    dim3 grid(gridX, gridY);

    // RUN RUN RUN!
    if (DEBUG) {
        printf("\n***** Updating layer *****\n");

        printf("\nFrom\n");
        drawMatrix(src_layer, src_n, 1);

        printf("\nTo\n");
        drawMatrix(weights, dst_n, src_n);
    }
    mapStepCUDA<<<grid, block>>>(src_layer_d, weights_d, buffer_d, dst_n, src_n);

    // Set the current target to the input
    float *currentTarget = buffer_d;
    int currentHeight = src_n;

    while (currentHeight > 1) {

        // Calculate grid size
        int gridX = (int)ceil((float)dst_n/WARP_SIZE);
        int gridY = (int)ceil((float)currentHeight/WARP_SIZE);
        dim3 grid(gridX, gridY);

        // Allocate new buffer
        float *buffer_d;
        cudaMalloc((void**)&buffer_d, sizeof(float) * (dst_n * gridY));
 
        // RUN RUN RUN!
        reduceStepCUDA<<<grid, block>>>(currentTarget, buffer_d, dst_n, src_n);

        // Free old memory and keep track of the new one
        cudaFree(currentTarget);
        currentHeight = grid.y;
        currentTarget = buffer_d;
    }

    dst_layer =_copyDeviceHost(currentTarget, dst_n, dst_layer);
    for (int i=0; i < dst_n; i++) {
        dst_layer[i] = tanh(dst_layer[i]);
    }

    if (DEBUG) {
        printf("\nResult is\n");
        drawMatrix(dst_layer, dst_n, 1);
        printf("\n***** ENDED UPDATING LAYER *****\n");
        _sleep(1);
    }
}
