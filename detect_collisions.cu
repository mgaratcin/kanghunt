//Copyright 2024 MGaratcin//  
//All rights reserved.//
//This code is proprietary and confidential. Unauthorized copying, distribution,//
//modification, or any other use of this code, in whole or in part, is strictly//
//prohibited. The use of this code without explicit written permission from the//
//copyright holder is not permitted under any circumstances.//

#include <iostream>
#include <cuda_runtime.h>
#include <vector> // Add this line

#define CHECK_CUDA_ERROR(call)                                                     \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess) {                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "  \
                      << cudaGetErrorString(err) << std::endl;                     \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    }

// CUDA kernel to compare dp1 and dp2 batches
__global__ void check_collision_kernel(const uint8_t* dp1_batch, const uint8_t* dp2_batch, int length, int batch_size, int* collision_flags) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = idx / length;
    int data_idx = idx % length;

    if (batch_idx < batch_size && data_idx < length) {
        if (dp1_batch[batch_idx * length + data_idx] != dp2_batch[batch_idx * length + data_idx]) {
            // If any byte differs, set collision_flag to 0
            collision_flags[batch_idx] = 0;
        }
    }
}

extern "C" void detect_collision_batch(const uint8_t* dp1_batch, const uint8_t* dp2_batch, int length, int batch_size) {
    // Allocate device memory
    uint8_t* d_dp1_batch;
    uint8_t* d_dp2_batch;
    int* d_collision_flags;
    std::vector<int> h_collision_flags(batch_size, 1); // Assume collision exists initially

    size_t data_size = length * batch_size * sizeof(uint8_t);
    size_t flags_size = batch_size * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dp1_batch, data_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dp2_batch, data_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_collision_flags, flags_size));

    // Copy data from the host to the device
    CHECK_CUDA_ERROR(cudaMemcpy(d_dp1_batch, dp1_batch, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dp2_batch, dp2_batch, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_collision_flags, h_collision_flags.data(), flags_size, cudaMemcpyHostToDevice));

    // Launch the kernel
    int blockSize = 256;
    int gridSize = (batch_size * length + blockSize - 1) / blockSize;
    check_collision_kernel<<<gridSize, blockSize>>>(d_dp1_batch, d_dp2_batch, length, batch_size, d_collision_flags);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(h_collision_flags.data(), d_collision_flags, flags_size, cudaMemcpyDeviceToHost));

    // Check for collisions
    for (int i = 0; i < batch_size; ++i) {
        if (h_collision_flags[i] == 1) {
            std::cout << "\n[+] Collision found between dp1 and dp2 in batch " << i << "!" << std::endl;
        }
    }

    // Free the allocated GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_dp1_batch));
    CHECK_CUDA_ERROR(cudaFree(d_dp2_batch));
    CHECK_CUDA_ERROR(cudaFree(d_collision_flags));
}
