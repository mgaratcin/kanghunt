//Copyright 2024 MGaratcin//  

//All rights reserved.//

//This code is proprietary and confidential. Unauthorized copying, distribution,//
//modification, or any other use of this code, in whole or in part, is strictly//
//prohibited. The use of this code without explicit written permission from the//
//copyright holder is not permitted under any circumstances.//

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>

// CUDA Kernel to check for collisions between two binary arrays
__global__ void checkCollision(const char* dp1, const char* dp2, int length, bool* collisionFlag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx < length) {
        // Compare the values of dp1 and dp2 at index idx
        if (dp1[idx] == dp2[idx] && dp1[idx] == '1') {
            *collisionFlag = true;
        }
    }
}

// Function to check for collisions between dp1 and dp2 on the GPU
bool detectCollision(const std::string& binaryStr1, const std::string& binaryStr2) {
    int length = binaryStr1.size();

    // Allocate device memory
    char* d_dp1;
    char* d_dp2;
    bool* d_collisionFlag;
    bool collisionFlag = false;

    cudaMalloc((void**)&d_dp1, length * sizeof(char));
    cudaMalloc((void**)&d_dp2, length * sizeof(char));
    cudaMalloc((void**)&d_collisionFlag, sizeof(bool));

    // Copy data from host to device
    cudaMemcpy(d_dp1, binaryStr1.c_str(), length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dp2, binaryStr2.c_str(), length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_collisionFlag, &collisionFlag, sizeof(bool), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;

    // Launch the collision detection kernel
    checkCollision<<<gridSize, blockSize>>>(d_dp1, d_dp2, length, d_collisionFlag);

    // Copy the collision flag back to host
    cudaMemcpy(&collisionFlag, d_collisionFlag, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_dp1);
    cudaFree(d_dp2);
    cudaFree(d_collisionFlag);

    return collisionFlag;
}

int main() {
    // Example binary strings for dp1 and dp2
    std::string dp1 = "10101010";  // Replace with actual data
    std::string dp2 = "01010101";  // Replace with actual data

    // Check for collision
    bool collision = detectCollision(dp1, dp2);

    // Output result
    if (collision) {
        std::cout << "Collision detected!" << std::endl;
    } else {
        std::cout << "No collision detected." << std::endl;
    }

    return 0;
}
