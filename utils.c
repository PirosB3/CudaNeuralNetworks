#define WARP_SIZE 16
#define DEBUG true

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

dim3 getGridBasedOnBlockSize(int width, int height, int block_size) {
    int gridX = (int)ceil((float)width / block_size);
    int gridY = (int)ceil((float)height / block_size);
    return dim3(gridX, gridY);
}

void _sleep(int n) {
    #ifdef __APPLE__
        sleep(n);
    #else _WIN32
        Sleep(n * 1000);
    #endif
}

void drawMatrix(float *m, int width, int height) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            printf("%f ", m[i * width + j]);
        }
        printf("\n");
    }
}
