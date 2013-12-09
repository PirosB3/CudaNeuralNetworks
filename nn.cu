#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <windows.h>

#define WARP_SIZE 32
#define DEBUG false

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

float *_copyHostDevice(float *src, int src_size) {
    float *src_d;
    cudaMalloc((void**)&src_d, sizeof(float) * src_size);
    cudaMemcpy(src_d, src, sizeof(float) * src_size, cudaMemcpyHostToDevice);
    return src_d;
}

float *_copyDeviceHost(float *src, int src_size, float *dst=NULL) {
    float *target;
    if (dst == NULL) {
        target = (float*)malloc(sizeof(float) * src_size);
    } else {
        target = dst;
    }
    
    cudaMemcpy(target, src, sizeof(float) * src_size, cudaMemcpyDeviceToHost);
    return target;
}

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


typedef struct {
    int n_inputs;
    int n_hidden;
    int n_outputs;
    
    float *out_input;
    float *out_hidden;
    float *out_output;

    float *changes_input_hidden;
    float *changes_hidden_output;
    
    float *w_input_hidden;
    float *w_hidden_output;
} NeuralNet;

typedef struct {
    int *result;
    int *data;
} Pattern;

void drawMatrix(float *m, int width, int height) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            printf("%f ", m[i * width + j]);
        }
        printf("\n");
    }
}


void buildLayer(float *arr, int n, float initial) {
    int i=0;
    while(i < n){
        *arr = initial;
        arr++;
        i++;
    }
}

float* buildWeightsLayer(int outer_n, int inner_n, float seed) {
    int total = outer_n * inner_n;
    float *w = (float *)malloc(sizeof(float) * total);
    for(int i=0; i < total; i++) {
        if (seed == -1) {
          w[i] = ((float)rand()/(float)RAND_MAX);
        } else {
          w[i] = seed;
        }
    }
    return w;
}

dim3 getGridBasedOnBlockSize(int width, int height, int block_size) {
    int gridX = (int)ceil((float)width / block_size);
    int gridY = (int)ceil((float)height / block_size);
    return dim3(gridX, gridY);
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

NeuralNet buildNeuralNet(int n_inputs, int n_outputs, int n_hidden) {

    float *out_input = (float *)malloc(sizeof(float) * (n_inputs + 1));
    float *out_hidden = (float *)malloc(sizeof(float) * n_hidden);
    float *out_output = (float *)malloc(sizeof(float) * n_outputs);

    buildLayer(out_input, n_inputs + 1, 1.0f);
    buildLayer(out_hidden, n_hidden, 1.0f);
    buildLayer(out_output, n_outputs, 1.0f);
    
    // Build changes layer
    float *changes_input_hidden = buildWeightsLayer(n_inputs + 1, n_hidden, 0.0f);
    float *changes_hidden_output = buildWeightsLayer(n_hidden, n_outputs, 0.0f);
    
    // Build weight matrix
    float *w_input_hidden = buildWeightsLayer(n_inputs + 1, n_hidden, -1.0f);
    float *w_hidden_output = buildWeightsLayer(n_hidden, n_outputs, -1.0f);

    NeuralNet nn;

    nn.n_inputs = n_inputs + 1;
    nn.n_outputs = n_outputs;
    nn.n_hidden = n_hidden;

    nn.out_input = out_input;
    nn.out_hidden = out_hidden;
    nn.out_output = out_output;

    nn.changes_input_hidden = changes_input_hidden;
    nn.changes_hidden_output = changes_hidden_output;

    nn.w_input_hidden = w_input_hidden;
    nn.w_hidden_output = w_hidden_output;

    return nn;
}

float dsigmoid(float y) {
    return 1.0 - pow(y,2.0f);
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
        Sleep(1000);
    }

}

void update_pattern(Pattern pattern, NeuralNet nn) {
    // Write inputs
    int i;
    for(i=0; i < nn.n_inputs -1; i++) {
        nn.out_input[i] = pattern.data[i];
    }

    // Run parallel update
    update_layer(nn.out_input, nn.out_hidden, nn.n_inputs, nn.n_hidden, nn.w_input_hidden);
    update_layer(nn.out_hidden, nn.out_output, nn.n_hidden, nn.n_outputs, nn.w_hidden_output);
}

float back_propagate_network(Pattern p, NeuralNet n) {
    /*
     * This is the backpropagation process, where error is calculated and
     * propagated back through the network in order to adjust the weights
     * between neurons.
     * NOTE: This section will also be parallelised. Unfortunately, the hidden delta
     * needs to be calculated after the output delta. So we can only parallelize
     * part of the process (this is what I think currently, I might be wrong!).
    */

    int i, j;
    float *output_delta = (float*)malloc(sizeof(float) * n.n_outputs);
    float *hidden_delta = (float*)malloc(sizeof(float) * n.n_hidden);

    
    // Calculate output delta
    for (i=0; i < n.n_outputs; i++) {
        float error = p.result[i] - n.out_output[i];
        output_delta[i] = dsigmoid(n.out_output[i]) * error;
    }
    
    
    // Calculate hidden delta
    for(i=0; i < n.n_hidden; i++) {
        float error = 0.0f;
        for (j=0; j < n.n_outputs; j++) {
            error += output_delta[j] * n.w_hidden_output[i * n.n_outputs + j];
        }
        hidden_delta[i] = dsigmoid(n.out_hidden[i]) * error;
    }
    
    /*
     * NOTE: Once the deltas have been calculated, we can update ALL the weights
     * in once. This section fits perfectly with the CUDA programming model of
     * grids and blocks.
    */

    // Set hidden-output weights
    for(i=0; i < n.n_hidden; i++) {
      for (j=0; j < n.n_outputs; j++) {
        float change = output_delta[j] * n.out_hidden[i];
        n.w_hidden_output[i * n.n_outputs + j] += 0.5 * change + 0.5 * n.changes_hidden_output[i * n.n_outputs + j];
        n.changes_hidden_output[i * n.n_outputs + j] = change;
      }
    }

    
    // Set input-hidden weights
    for(i=0; i < n.n_inputs; i++) {
      for(j=0; j < n.n_hidden; j++) {
        float change = hidden_delta[j] * n.out_input[i];
        n.w_input_hidden[i * n.n_hidden + j] += 0.5 * change + 0.5 * n.changes_input_hidden[i * n.n_hidden + j];
        n.changes_input_hidden[i * n.n_hidden + j] = change;
      }
    }

    // Calculate error
    float error = 0.0f;
    for (i=0; i < n.n_outputs; i++) {
        error = error + 0.5f * pow(p.result[i] - n.out_output[i], 2);
    }

    return error;
}



float xback_propagate_network(Pattern p, NeuralNet n) {
    /*
     * This is the backpropagation process, where error is calculated and
     * propagated back through the network in order to adjust the weights
     * between neurons.
     * NOTE: This section will also be parallelised. Unfortunately, the hidden delta
     * needs to be calculated after the output delta. So we can only parallelize
     * part of the process (this is what I think currently, I might be wrong!).
    */

    if (DEBUG) {
        printf("\n ***** BACK PROPAGATE *****\n");
    }

    int i, j;
    float *output_delta = (float*)malloc(sizeof(float) * n.n_outputs);
    float *hidden_delta = (float*)malloc(sizeof(float) * n.n_hidden);

    
    // Calculate output delta
    for (i=0; i < n.n_outputs; i++) {
        float error = p.result[i] - n.out_output[i];
        output_delta[i] = dsigmoid(n.out_output[i]) * error;
    }
    
    
    // Calculate hidden delta
    for(i=0; i < n.n_hidden; i++) {
        float error = 0.0f;
        for (j=0; j < n.n_outputs; j++) {
            error += output_delta[j] * n.w_hidden_output[i * n.n_outputs + j];
        }
        hidden_delta[i] = dsigmoid(n.out_hidden[i]) * error;
    }

    // Set hidden-output weights
    setWeightsForLayers(n.w_hidden_output, n.changes_hidden_output, output_delta, n.out_hidden, n.n_hidden, n.n_outputs);
    if (DEBUG) {
        printf("\nHidden-Output weights\n");
        drawMatrix(n.w_hidden_output, n.n_outputs, n.n_hidden);
        Sleep(500);
    }
   
    setWeightsForLayers(n.w_input_hidden, n.changes_input_hidden, hidden_delta, n.out_input, n.n_inputs, n.n_hidden);
    if (DEBUG) {
        printf("\nInput-Hidden weights\n");
        drawMatrix(n.w_input_hidden, n.n_hidden, n.n_inputs);
        Sleep(500);
    }
   
    // Calculate error
    float error = 0.0f;
    for (i=0; i < n.n_outputs; i++) {
        error = error + 0.5f * pow(p.result[i] - n.out_output[i], 2);
    }
    if (DEBUG) {
        printf("\n ***** Error for this pattern is: %f *****\n", error); 
        Sleep(2000);
    }
    return error;
}


void train_network(Pattern *patterns, int n_patterns, int n_iterations, NeuralNet nn) {
  int i, j;
  for (i=0; i < n_iterations; i++) {
    float error = 0;
    for (j=0; j < n_patterns; j++) {
       update_pattern(patterns[j], nn);
       error += back_propagate_network(patterns[j], nn);
    }
    if (i % 10 == 0) {
       printf("Error is: %-.5f\n", error);
       if (DEBUG) Sleep(2000);
    }
  }
}

Pattern makePatternSingleOutput(int *data, int result) {
    Pattern p;
    p.data = data;

    p.result = (int *)malloc(sizeof(int));
    p.result[0] = result;

    return p;
}

int main() {
    srand((unsigned)time(NULL));
    
    int n_inputs = 2;
    int n_hidden = 9;
    int n_outputs = 1;
    
    // Build output layer
    NeuralNet nn = buildNeuralNet(n_inputs, n_outputs, n_hidden);
    
    // Build training samples
    int _p1[] = {0,0};
    Pattern p1 = makePatternSingleOutput(_p1, 0);
    int _p2[] = {0,1};
    Pattern p2 = makePatternSingleOutput(_p2, 1);
    int _p3[] = {1,1};
    Pattern p3 = makePatternSingleOutput(_p3, 1);
    int _p4[] = {1,0};
    Pattern p4 = makePatternSingleOutput(_p4, 1);
    
    Pattern patterns[] = {p3, p2, p1, p4};
    
    // Train the network
    train_network(patterns, 4, 10000, nn);

    printf("\n\nTesting the network\n");
    update_pattern(p2, nn);
    for (int i=0; i < nn.n_outputs; i++) {
        printf("Output: %f, expected: %i\n", nn.out_output[i], p2.result[i]);
    }
    cudaDeviceReset();
    return 0;
}