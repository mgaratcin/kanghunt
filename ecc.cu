// ecc.cu

#include "secp256k1.cuh"
#include "ecc.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cstring> // For memcpy

// Error checking macro
#define cudaRunOrAbort(ans)                                                    \
    { cudaAssert((ans), __FILE__, __LINE__); }

// Error handling function
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// Kernel definition (no extern "C" needed)
__global__ void getPublicKeyByPrivateKeyKernel(Point *output, const u64 *privateKeys, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        secp256k1PublicKey(&output[idx], &privateKeys[idx * 4]);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// Host function definition with C linkage
void getPublicKeyByPrivateKey(Point output[], const u64 flattenedPrivateKeys[][4], int n) {
    // Define block and grid sizes
    const int BLOCK_SIZE = 256;
    // Using a fixed grid size instead of dynamic for better performance
    const int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Define the number of streams to use (adjust based on your GPU's capabilities)
    const int NUM_STREAMS = 4;

    // Allocate pinned host memory for input and output
    Point *pinnedPoints;
    u64 *pinnedPrivateKeys;
    cudaRunOrAbort(cudaMallocHost(&pinnedPoints, n * sizeof(Point)));
    cudaRunOrAbort(cudaMallocHost(&pinnedPrivateKeys, n * 4 * sizeof(u64)));

    // Flatten the private keys using a single memcpy (assuming they're already contiguous)
    std::memcpy(pinnedPrivateKeys, flattenedPrivateKeys, n * 4 * sizeof(u64));

    // Allocate device memory
    Point *devicePoints;
    u64 *devicePrivateKeys;
    cudaRunOrAbort(cudaMalloc(&devicePoints, n * sizeof(Point)));
    cudaRunOrAbort(cudaMalloc(&devicePrivateKeys, n * 4 * sizeof(u64)));

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaRunOrAbort(cudaStreamCreate(&streams[i]));
    }

    // Determine the number of elements per stream
    int elementsPerStream = (n + NUM_STREAMS - 1) / NUM_STREAMS;

    // Launch operations in each stream
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * elementsPerStream;
        int currentSize = (offset + elementsPerStream > n) ? (n - offset) : elementsPerStream;

        if (currentSize <= 0) {
            // No more data to process
            break;
        }

        // Asynchronously copy a chunk of private keys to device
        cudaRunOrAbort(cudaMemcpyAsync(
            devicePrivateKeys + offset * 4,
            pinnedPrivateKeys + offset * 4,
            currentSize * 4 * sizeof(u64),
            cudaMemcpyHostToDevice,
            streams[i]
        ));

        // Calculate grid size for the current chunk
        int currentGridSize = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Launch kernel for the current chunk
        getPublicKeyByPrivateKeyKernel<<<currentGridSize, BLOCK_SIZE, 0, streams[i]>>>(
            devicePoints + offset,
            devicePrivateKeys + offset * 4,
            currentSize
        );

        // Check for kernel launch errors
        cudaRunOrAbort(cudaGetLastError());

        // Asynchronously copy the results back to host
        cudaRunOrAbort(cudaMemcpyAsync(
            pinnedPoints + offset,
            devicePoints + offset,
            currentSize * sizeof(Point),
            cudaMemcpyDeviceToHost,
            streams[i]
        ));
    }

    // Synchronize all streams to ensure completion
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaRunOrAbort(cudaStreamSynchronize(streams[i]));
    }

    // Destroy all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaRunOrAbort(cudaStreamDestroy(streams[i]));
    }

    // Copy the results from pinned memory to the output array
    std::memcpy(output, pinnedPoints, n * sizeof(Point));

    // Free allocated memory
    cudaRunOrAbort(cudaFreeHost(pinnedPoints));
    cudaRunOrAbort(cudaFreeHost(pinnedPrivateKeys));
    cudaRunOrAbort(cudaFree(devicePoints));
    cudaRunOrAbort(cudaFree(devicePrivateKeys));
}

#ifdef __cplusplus
}
#endif
