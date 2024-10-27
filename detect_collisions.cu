// Copyright 2024 MGaratcin
// All rights reserved.
// This code is proprietary and confidential. Unauthorized copying, distribution,
// modification, or any other use of this code, in whole or in part, is strictly
// prohibited. The use of this code without explicit written permission from the
// copyright holder is not permitted under any circumstances.

#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA_ERROR(call)                                                     \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess) {                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "  \
                      << cudaGetErrorString(err) << std::endl;                     \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    }

// CUDA kernel to perform condition checks for dp1 keys
__global__ void dp1_condition_kernel(const uint8_t* keys_bytes, int key_size_bytes, int num_keys, int* condition_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        const uint8_t* key = keys_bytes + idx * key_size_bytes;

        // Condition for dp1: binary length is 136 bits and last 20 bits are zero

        // Since keys are 256 bits, we need to check the last 20 bits

        int byte_index = key_size_bytes - 1;
        int bits_to_check = 50;
        int full_bytes = bits_to_check / 8; // 2 bytes
        int remaining_bits = bits_to_check % 8; // 4 bits

        bool condition_met = true;

        // Check the full bytes
        for (int i = 0; i < full_bytes; ++i) {
            if (key[byte_index - i] != 0) {
                condition_met = false;
                break;
            }
        }

        // Check the remaining bits
        if (condition_met && remaining_bits > 0) {
            uint8_t mask = (1 << remaining_bits) - 1;
            if ((key[byte_index - full_bytes] & mask) != 0) {
                condition_met = false;
            }
        }

        if (condition_met) {
            condition_flags[idx] = 1;
        } else {
            condition_flags[idx] = 0;
        }
    }
}

// CUDA kernel to perform condition checks for dp2 keys
__global__ void dp2_condition_kernel(const uint8_t* keys_bytes, int key_size_bytes, int num_keys, int* condition_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        const uint8_t* key = keys_bytes + idx * key_size_bytes;

        // Condition for dp2: binary length is 136 bits and last 34 bits are zero

        int byte_index = key_size_bytes - 1;
        int bits_to_check = 50;
        int full_bytes = bits_to_check / 8; // 4 bytes
        int remaining_bits = bits_to_check % 8; // 2 bits

        bool condition_met = true;

        // Check the full bytes
        for (int i = 0; i < full_bytes; ++i) {
            if (key[byte_index - i] != 0) {
                condition_met = false;
                break;
            }
        }

        // Check the remaining bits
        if (condition_met && remaining_bits > 0) {
            uint8_t mask = (1 << remaining_bits) - 1;
            if ((key[byte_index - full_bytes] & mask) != 0) {
                condition_met = false;
            }
        }

        if (condition_met) {
            condition_flags[idx] = 1;
        } else {
            condition_flags[idx] = 0;
        }
    }
}

// CUDA kernel to perform collision detection between dp1 and dp2 keys
__global__ void collision_detection_kernel(const uint8_t* dp1_keys, int dp1_count, const uint8_t* dp2_keys, int dp2_count, int key_size_bytes, int* collision_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_comparisons = dp1_count * dp2_count;

    if (idx < total_comparisons) {
        int dp1_idx = idx / dp2_count;
        int dp2_idx = idx % dp2_count;

        const uint8_t* dp1_key = dp1_keys + dp1_idx * key_size_bytes;
        const uint8_t* dp2_key = dp2_keys + dp2_idx * key_size_bytes;

        bool collision = (key_size_bytes == 17);
        for (int k = 0; k < key_size_bytes; ++k) {
            if (dp1_key[k] != dp2_key[k]) {
                collision = false;
                break;
            }
        }

        if (collision) {
            collision_flags[idx] = 1;
        } else {
            collision_flags[idx] = 0;
        }
    }
}

extern "C" void process_keys_on_gpu(
    const uint8_t* dp1_keys_bytes, int dp1_count,
    const uint8_t* dp2_keys_bytes, int dp2_count,
    int key_size_bytes) {

    // Allocate device memory for keys and flags
    uint8_t* d_dp1_keys;
    uint8_t* d_dp2_keys;
    int* d_dp1_flags;
    int* d_dp2_flags;

    size_t dp1_size = dp1_count * key_size_bytes * sizeof(uint8_t);
    size_t dp2_size = dp2_count * key_size_bytes * sizeof(uint8_t);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dp1_keys, dp1_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dp2_keys, dp2_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dp1_flags, dp1_count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dp2_flags, dp2_count * sizeof(int)));

    // Copy keys to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_dp1_keys, dp1_keys_bytes, dp1_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dp2_keys, dp2_keys_bytes, dp2_size, cudaMemcpyHostToDevice));

    // Initialize flags to zero
    CHECK_CUDA_ERROR(cudaMemset(d_dp1_flags, 0, dp1_count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_dp2_flags, 0, dp2_count * sizeof(int)));

    // Launch kernels to perform condition checks
    int blockSize = 256;
    int gridSize;

    gridSize = (dp1_count + blockSize - 1) / blockSize;
    dp1_condition_kernel<<<gridSize, blockSize>>>(d_dp1_keys, key_size_bytes, dp1_count, d_dp1_flags);
    CHECK_CUDA_ERROR(cudaGetLastError());

    gridSize = (dp2_count + blockSize - 1) / blockSize;
    dp2_condition_kernel<<<gridSize, blockSize>>>(d_dp2_keys, key_size_bytes, dp2_count, d_dp2_flags);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy flags back to host
    std::vector<int> dp1_flags(dp1_count);
    std::vector<int> dp2_flags(dp2_count);

    CHECK_CUDA_ERROR(cudaMemcpy(dp1_flags.data(), d_dp1_flags, dp1_count * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(dp2_flags.data(), d_dp2_flags, dp2_count * sizeof(int), cudaMemcpyDeviceToHost));

    // Collect keys that meet the conditions
    std::vector<uint8_t> dp1_keys_filtered;
    std::vector<uint8_t> dp2_keys_filtered;

    std::vector<int> dp1_indices;
    std::vector<int> dp2_indices;

    for (int i = 0; i < dp1_count; ++i) {
        if (dp1_flags[i]) {
            dp1_keys_filtered.insert(dp1_keys_filtered.end(), &dp1_keys_bytes[i * key_size_bytes], &dp1_keys_bytes[(i + 1) * key_size_bytes]);
            dp1_indices.push_back(i);
        }
    }

    for (int i = 0; i < dp2_count; ++i) {
        if (dp2_flags[i]) {
            dp2_keys_filtered.insert(dp2_keys_filtered.end(), &dp2_keys_bytes[i * key_size_bytes], &dp2_keys_bytes[(i + 1) * key_size_bytes]);
            dp2_indices.push_back(i);
        }
    }

    int dp1_filtered_count = dp1_indices.size();
    int dp2_filtered_count = dp2_indices.size();

    if (dp1_filtered_count > 0 && dp2_filtered_count > 0) {
        // Allocate device memory for filtered keys
        uint8_t* d_dp1_keys_filtered;
        uint8_t* d_dp2_keys_filtered;
        int* d_collision_flags;

        size_t dp1_filtered_size = dp1_filtered_count * key_size_bytes * sizeof(uint8_t);
        size_t dp2_filtered_size = dp2_filtered_count * key_size_bytes * sizeof(uint8_t);
        size_t collision_flags_size = dp1_filtered_count * dp2_filtered_count * sizeof(int);

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dp1_keys_filtered, dp1_filtered_size));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dp2_keys_filtered, dp2_filtered_size));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_collision_flags, collision_flags_size));

        // Copy filtered keys to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_dp1_keys_filtered, dp1_keys_filtered.data(), dp1_filtered_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_dp2_keys_filtered, dp2_keys_filtered.data(), dp2_filtered_size, cudaMemcpyHostToDevice));

        // Initialize collision flags to zero
        CHECK_CUDA_ERROR(cudaMemset(d_collision_flags, 0, collision_flags_size));

        // Launch collision detection kernel
        int total_comparisons = dp1_filtered_count * dp2_filtered_count;
        gridSize = (total_comparisons + blockSize - 1) / blockSize;

        collision_detection_kernel<<<gridSize, blockSize>>>(d_dp1_keys_filtered, dp1_filtered_count, d_dp2_keys_filtered, dp2_filtered_count, key_size_bytes, d_collision_flags);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Copy collision flags back to host
        std::vector<int> collision_flags(dp1_filtered_count * dp2_filtered_count);
        CHECK_CUDA_ERROR(cudaMemcpy(collision_flags.data(), d_collision_flags, collision_flags_size, cudaMemcpyDeviceToHost));

        // Check for collisions
        for (int i = 0; i < dp1_filtered_count; ++i) {
            for (int j = 0; j < dp2_filtered_count; ++j) {
                if (collision_flags[i * dp2_filtered_count + j]) {
                    // Collision found between dp1_keys_filtered[i] and dp2_keys_filtered[j]
                    std::cout << "\n[+] Collision found between dp1 index " << dp1_indices[i] << " and dp2 index " << dp2_indices[j] << "!" << std::endl;
                    // Handle collision as needed (e.g., pass data back to CPU)
                }
            }
        }

        // Free filtered keys and collision flags
        CHECK_CUDA_ERROR(cudaFree(d_dp1_keys_filtered));
        CHECK_CUDA_ERROR(cudaFree(d_dp2_keys_filtered));
        CHECK_CUDA_ERROR(cudaFree(d_collision_flags));
    }

    // Free device memory for keys and flags
    CHECK_CUDA_ERROR(cudaFree(d_dp1_keys));
    CHECK_CUDA_ERROR(cudaFree(d_dp2_keys));
    CHECK_CUDA_ERROR(cudaFree(d_dp1_flags));
    CHECK_CUDA_ERROR(cudaFree(d_dp2_flags));
}
