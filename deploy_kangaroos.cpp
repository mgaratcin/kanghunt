#include "deploy_kangaroos.h"
#include <iostream>
#include "secp256k1/SECP256K1.h"
#include "secp256k1/Point.h"
#include "secp256k1/Int.h"
#include <random>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <cuda_runtime.h>

#ifndef KANGAROO_BATCH_SIZE
#define KANGAROO_BATCH_SIZE 1024
#endif

#define TARGET_KEY "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

static std::atomic<uint64_t> kangaroo_counter{0};
static std::mutex gpu_mutex; // Mutex for thread-safe GPU operations
static std::mutex output_mutex; // Mutex for thread-safe output

// CUDA kernel declaration
extern "C" void detect_collisions_cuda(Int* base_keys, Int* current_keys, int batch_size);

void deploy_kangaroos(const std::vector<Int>& kangaroo_batch) {
    Secp256K1 secp; // Initialize the SECP256K1 context using the default constructor
    Point target_key;

    // Allocate memory on GPU for base_key and current_key arrays as Int
    Int* d_base_keys;
    Int* d_current_keys;

    // Lock the GPU operations to a single thread at a time
    std::lock_guard<std::mutex> lock(gpu_mutex);
    cudaMalloc((void**)&d_base_keys, kangaroo_batch.size() * sizeof(Int));
    cudaMalloc((void**)&d_current_keys, kangaroo_batch.size() * sizeof(Int));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(1, 10000000000);

    for (const auto& base_key : kangaroo_batch) {
        Int current_key = base_key;

        // Copy base_key and current_key directly to device as Int
        cudaMemcpy(d_base_keys, &base_key, sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_current_keys, &current_key, sizeof(Int), cudaMemcpyHostToDevice);

        // Launch CUDA kernel within the GPU lock
        detect_collisions_cuda(d_base_keys, d_current_keys, KANGAROO_BATCH_SIZE);

        // Synchronize CUDA device to ensure kernel execution completion
        cudaDeviceSynchronize();
    }

    // Free GPU memory
    cudaFree(d_base_keys);
    cudaFree(d_current_keys);
}
