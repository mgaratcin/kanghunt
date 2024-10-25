// detect_collisions.cu

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to compare dp1 and dp2
__global__ void check_collision_kernel(const char* dp1, const char* dp2, int length, int* collision_flag) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < length && dp1[idx] != dp2[idx]) {
        // If any character differs, set collision_flag to 0
        atomicExch(collision_flag, 0);
    }
}

// Function to detect collision
extern "C" void detect_collision(const char* dp1, const char* dp2, int length) {
    // Allocate device memory
    char* d_dp1;
    char* d_dp2;
    int* d_collision_flag;
    int h_collision_flag = 1; // Assume collision exists initially

    cudaMalloc((void**)&d_dp1, length * sizeof(char));
    cudaMalloc((void**)&d_dp2, length * sizeof(char));
    cudaMalloc((void**)&d_collision_flag, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_dp1, dp1, length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dp2, dp2, length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_collision_flag, &h_collision_flag, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;
    check_collision_kernel<<<gridSize, blockSize>>>(d_dp1, d_dp2, length, d_collision_flag);

    // Copy result back to host
    cudaMemcpy(&h_collision_flag, d_collision_flag, sizeof(int), cudaMemcpyDeviceToHost);

    // Check collision flag
    if (h_collision_flag == 1) {
        std::cout << "\n[+] Collision found between dp1 and dp2!" << std::endl;
    }

    // Free device memory
    cudaFree(d_dp1);
    cudaFree(d_dp2);
    cudaFree(d_collision_flag);
}
