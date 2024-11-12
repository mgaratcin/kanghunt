#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "point.cuh"
#include "ecc.cuh"
#include "u64.cuh"

#define BATCH_SIZE 2048
#define NUM_STREAMS 192

// Function to convert a hexadecimal private key to u64 array
void hexToPrivateKey(const char* hex, u64 privateKey[4]) {
    for (int i = 0; i < 4; ++i) {
        sscanf(hex + 16 * i, "%16llx", &privateKey[3 - i]); // Parse in reverse for endian compatibility
    }
}

// Function to convert u64 array back to hexadecimal string
void privateKeyToHex(const u64 privateKey[4], char* hex) {
    for (int i = 0; i < 4; ++i) {
        sprintf(hex + 16 * i, "%016llx", privateKey[3 - i]);
    }
    hex[64] = '\0';
}

// Function to increment the private key by 1
int incrementPrivateKey(u64 privateKey[4]) {
    for (int i = 0; i < 4; ++i) {
        if (++privateKey[i] != 0) {
            return 1; // Successfully incremented without overflow
        }
    }
    return 0; // Overflow occurred
}

// Function to validate a hexadecimal string
bool isValidHex(const char* hex, int length) {
    for (int i = 0; i < length; ++i) {
        if (!isxdigit(hex[i])) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <start_private_key_in_hex> <end_private_key_in_hex>\n", argv[0]);
        return 1;
    }

    // Validate input lengths
    if (strlen(argv[1]) != 64 || strlen(argv[2]) != 64) {
        printf("Error: Private keys must be exactly 64 hexadecimal characters.\n");
        return 1;
    }

    // Validate hexadecimal characters
    if (!isValidHex(argv[1], 64) || !isValidHex(argv[2], 64)) {
        printf("Error: Private keys must contain only valid hexadecimal characters (0-9, a-f, A-F).\n");
        return 1;
    }

    // Parse the start and end private keys
    u64 startKey[4];
    u64 endKey[4];
    hexToPrivateKey(argv[1], startKey);
    hexToPrivateKey(argv[2], endKey);

    // Lambda function to compare two keys
    auto isLessOrEqual = [&](const u64 a[4], const u64 b[4]) -> bool {
        for (int i = 3; i >= 0; --i) {
            if (a[i] < b[i]) return true;
            if (a[i] > b[i]) return false;
        }
        return true;
    };

    // Initialize currentKey with startKey
    u64 currentKey[4];
    memcpy(currentKey, startKey, sizeof(u64) * 4);

    // Allocate pinned memory for batches
    u64* h_batchKeys;
    Point* h_batchPublicKeys;
    cudaMallocHost(&h_batchKeys, sizeof(u64) * 4 * BATCH_SIZE * NUM_STREAMS);
    cudaMallocHost(&h_batchPublicKeys, sizeof(Point) * BATCH_SIZE * NUM_STREAMS);

    // Allocate device memory for keys and results
    u64* d_batchKeys;
    Point* d_batchPublicKeys;
    cudaMalloc(&d_batchKeys, sizeof(u64) * 4 * BATCH_SIZE * NUM_STREAMS);
    cudaMalloc(&d_batchPublicKeys, sizeof(Point) * BATCH_SIZE * NUM_STREAMS);

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    while (isLessOrEqual(currentKey, endKey)) {
        // Split batches across streams
        for (int s = 0; s < NUM_STREAMS; ++s) {
            int offset = s * BATCH_SIZE * 4;
            int streamBatchSize = 0;

            // Prepare a batch for the current stream
            while (streamBatchSize < BATCH_SIZE && isLessOrEqual(currentKey, endKey)) {
                // Copy currentKey to host batch
                for (int i = 0; i < 4; ++i) {
                    h_batchKeys[streamBatchSize * 4 + i + offset] = currentKey[i];
                }
                streamBatchSize++;

                if (!incrementPrivateKey(currentKey)) break;
            }

            // Asynchronous copy to device
            cudaMemcpyAsync(&d_batchKeys[offset], &h_batchKeys[offset],
                            sizeof(u64) * 4 * streamBatchSize, cudaMemcpyHostToDevice, streams[s]);

            // Define grid and block sizes
            int threadsPerBlock = 1024;
            int blocks = (streamBatchSize + threadsPerBlock - 1) / threadsPerBlock;

            // Launch the kernel on each stream
            getPublicKeyByPrivateKeyKernel<<<blocks, threadsPerBlock, 0, streams[s]>>>(
                &d_batchPublicKeys[s * BATCH_SIZE], &d_batchKeys[s * BATCH_SIZE * 4], streamBatchSize);

            // Asynchronously copy results back to host
            cudaMemcpyAsync(&h_batchPublicKeys[s * BATCH_SIZE], &d_batchPublicKeys[s * BATCH_SIZE],
                            sizeof(Point) * streamBatchSize, cudaMemcpyDeviceToHost, streams[s]);
        }

        // Wait for all streams to complete
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamSynchronize(streams[i]);
        }

        // Output the results
        for (int s = 0; s < NUM_STREAMS; ++s) {
            for (int i = 0; i < BATCH_SIZE; ++i) {
                char privateKeyHex[65];
                privateKeyToHex(&h_batchKeys[i * 4 + s * BATCH_SIZE * 4], privateKeyHex);
                printf("Private Key: %s\n", privateKeyHex);
                printf("Public Key:\n");
                for (int j = 0; j < 4; ++j) printf("%016llx", h_batchPublicKeys[i + s * BATCH_SIZE].x[3 - j]);
                printf(":");
                for (int j = 0; j < 4; ++j) printf("%016llx", h_batchPublicKeys[i + s * BATCH_SIZE].y[3 - j]);
                printf("\n\n");
            }
        }
    }

    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // Free memory
    cudaFree(d_batchKeys);
    cudaFree(d_batchPublicKeys);
    cudaFreeHost(h_batchKeys);
    cudaFreeHost(h_batchPublicKeys);

    return 0;
}
