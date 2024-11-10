// =======================
// File: keygen_cuda.cu
// =======================
// Filename: keygen_cuda.cu

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "secp256k1.cuh"
#include "ptx.cuh"

// Converts a hexadecimal string to a byte array
bool hexToBytes(const std::string& hex, unsigned char* bytes, size_t length) {
    if (hex.length() != length * 2)
        return false;

    for (size_t i = 0; i < length; ++i) {
        unsigned int byte;
        std::stringstream ss;
        ss << std::hex << hex.substr(i * 2, 2);
        if (!(ss >> byte))
            return false;
        bytes[i] = static_cast<unsigned char>(byte);
    }
    return true;
}

// Function to increment a big-endian 256-bit integer represented as a byte array
void incrementPrivKey(unsigned char* privKey) {
    for (int i = 31; i >= 0; --i) {
        if (++privKey[i] != 0)
            break;
    }
}

// Function to compute the total number of keys between start and end
size_t computeTotalKeys(const unsigned char* startKey, const unsigned char* endKey) {
    size_t total = 0;
    bool carry = false;
    for (int i = 0; i < 32; ++i) {
        size_t diff = static_cast<size_t>(endKey[i]) - static_cast<size_t>(startKey[i]) + (carry ? 1 : 0);
        total = (total << 8) + (diff & 0xFF);
        carry = diff > 0xFF;
    }
    total += 1; // Inclusive of endKey
    return total;
}

// Kernel function to compute public keys from private keys
__global__ void computePublicKeys(unsigned int* privKeys, unsigned int* pubKeysX, unsigned int* pubKeysY, size_t numKeys) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < numKeys) {
        // Load the private key
        unsigned int privKey[8];
        for (int i = 0; i < 8; ++i) {
            privKey[i] = privKeys[idx + i * numKeys];
        }

        // Initialize the point (G)
        unsigned int x[8], y[8];
        copyBigInt(_GX, x);
        copyBigInt(_GY, y);

        // Compute public key: [privKey] * G
        // Implement scalar multiplication
        pointMultiply(privKey, x, y);

        // Store the public key
        for (int i = 0; i < 8; ++i) {
            pubKeysX[idx + i * numKeys] = x[i];
            pubKeysY[idx + i * numKeys] = y[i];
        }
    }
}

// Implement scalar multiplication
__device__ void pointMultiply(const unsigned int scalar[8], unsigned int x[8], unsigned int y[8]) {
    // Implement scalar multiplication using the double-and-add algorithm
    // Using Jacobian coordinates for efficiency

    unsigned int Rx[8], Ry[8], Rz[8]; // Result point in Jacobian coordinates
    unsigned int Px[8], Py[8], Pz[8]; // Point G in Jacobian coordinates

    // Initialize R to the point at infinity (represented by Z = 0)
    for (int i = 0; i < 8; ++i) {
        Rx[i] = 0;
        Ry[i] = 0;
        Rz[i] = 0;
    }

    // Initialize P to G (the generator point)
    copyBigInt(_GX, Px);
    copyBigInt(_GY, Py);
    for (int i = 0; i < 8; ++i) {
        Pz[i] = 0;
    }
    Pz[7] = 1; // Z = 1 for affine point converted to Jacobian coordinates

    // Loop over bits of scalar from most significant bit to least significant bit
    for (int bit = 255; bit >= 0; --bit) {
        // R = 2R
        pointDouble(Rx, Ry, Rz, Rx, Ry, Rz);

        // Check if the bit is set
        int word = bit / 32;
        int bitPos = bit % 32;
        if ((scalar[word] >> bitPos) & 1) {
            // R = R + P
            pointAdd(Rx, Ry, Rz, Px, Py, Pz, Rx, Ry, Rz);
        }
    }

    // Convert R back to affine coordinates
    jacobianToAffine(Rx, Ry, Rz, x, y);
}

// Implement point addition in Jacobian coordinates
__device__ void pointAdd(const unsigned int X1[8], const unsigned int Y1[8], const unsigned int Z1[8],
                         const unsigned int X2[8], const unsigned int Y2[8], const unsigned int Z2[8],
                         unsigned int X3[8], unsigned int Y3[8], unsigned int Z3[8]) {
    // Temporary variables
    unsigned int Z1Z1[8], Z2Z2[8];
    unsigned int U1[8], U2[8];
    unsigned int S1[8], S2[8];
    unsigned int H[8], R[8];
    unsigned int H2[8], H3[8];
    unsigned int U1H2[8];
    unsigned int temp1[8], temp2[8], temp3[8];

    // Compute Z1Z1 = Z1^2 mod p
    squareModP(Z1, Z1Z1);

    // Compute Z2Z2 = Z2^2 mod p
    squareModP(Z2, Z2Z2);

    // Compute U1 = X1 * Z2Z2 mod p
    mulModP(X1, Z2Z2, U1);

    // Compute U2 = X2 * Z1Z1 mod p
    mulModP(X2, Z1Z1, U2);

    // Compute Z1^3 and Z2^3
    mulModP(Z1, Z1Z1, temp1); // temp1 = Z1^3
    mulModP(Z2, Z2Z2, temp2); // temp2 = Z2^3

    // Compute S1 = Y1 * Z2^3 mod p
    mulModP(Y1, temp2, S1);

    // Compute S2 = Y2 * Z1^3 mod p
    mulModP(Y2, temp1, S2);

    // Compute H = U2 - U1 mod p
    subModP(U2, U1, H);

    // Compute R = S2 - S1 mod p
    subModP(S2, S1, R);

    // Check if H is zero
    bool H_is_zero = true;
    for (int i = 0; i < 8; i++) {
        if (H[i] != 0) {
            H_is_zero = false;
            break;
        }
    }

    if (H_is_zero) {
        // Check if R is zero
        bool R_is_zero = true;
        for (int i = 0; i < 8; i++) {
            if (R[i] != 0) {
                R_is_zero = false;
                break;
            }
        }

        if (R_is_zero) {
            // P1 == P2, perform point doubling
            pointDouble(X1, Y1, Z1, X3, Y3, Z3);
            return;
        } else {
            // Result is point at infinity
            for (int i = 0; i < 8; i++) {
                X3[i] = 0;
                Y3[i] = 0;
                Z3[i] = 0;
            }
            return;
        }
    }

    // Compute H2 = H^2 mod p
    squareModP(H, H2);

    // Compute H3 = H * H2 mod p
    mulModP(H, H2, H3);

    // Compute U1H2 = U1 * H2 mod p
    mulModP(U1, H2, U1H2);

    // Compute R^2
    squareModP(R, temp1); // temp1 = R^2

    // Compute 2 * U1H2
    doubleModP(U1H2, temp2); // temp2 = 2 * U1H2

    // Compute X3 = R^2 - H3 - 2 * U1H2 mod p
    subModP(temp1, H3, temp3); // temp3 = R^2 - H3
    subModP(temp3, temp2, X3); // X3 = temp3 - temp2

    // Compute Y3 = R * (U1H2 - X3) - S1 * H3 mod p
    subModP(U1H2, X3, temp1); // temp1 = U1H2 - X3
    mulModP(R, temp1, temp2); // temp2 = R * (U1H2 - X3)
    mulModP(S1, H3, temp3);   // temp3 = S1 * H3
    subModP(temp2, temp3, Y3); // Y3 = temp2 - temp3

    // Compute Z3 = H * Z1 * Z2 mod p
    mulModP(Z1, Z2, temp1);   // temp1 = Z1 * Z2
    mulModP(temp1, H, Z3);    // Z3 = temp1 * H
}

// Implement point doubling in Jacobian coordinates
__device__ void pointDouble(const unsigned int X1[8], const unsigned int Y1[8], const unsigned int Z1[8],
                            unsigned int X3[8], unsigned int Y3[8], unsigned int Z3[8]) {
    // Temporary variables
    unsigned int S1[8]; // S1 = Y1^2 mod p
    unsigned int S2[8]; // S2 = X1 * S1 mod p
    unsigned int S[8];  // S = 4 * S2 mod p
    unsigned int A[8];  // A = X1^2 mod p
    unsigned int M[8];  // M = 3 * A mod p
    unsigned int X3_temp[8]; // X3_temp = M^2 mod p
    unsigned int two_S[8];  // two_S = 2 * S mod p
    unsigned int T[8]; // T = S - X3 mod p
    unsigned int Y3_temp[8]; // Y3_temp = M * T mod p
    unsigned int S1_squared[8];  // S1_squared = S1^2 mod p
    unsigned int eight_S1_squared[8]; // eight_S1_squared = 8 * S1_squared mod p
    unsigned int temp1[8], temp2[8]; // temporary variables

    // Compute S1 = Y1^2 mod p
    squareModP(Y1, S1);

    // Compute S2 = X1 * S1 mod p
    mulModP(X1, S1, S2);

    // Compute S = 4 * S2 mod p
    doubleModP(S2, temp1); // temp1 = 2 * S2 mod p
    doubleModP(temp1, S);  // S = 4 * S2 mod p

    // Compute A = X1^2 mod p
    squareModP(X1, A);

    // Compute M = 3 * A mod p
    addModP(A, A, temp1); // temp1 = 2 * A mod p
    addModP(temp1, A, M); // M = 3 * A mod p

    // Compute X3_temp = M^2 mod p
    squareModP(M, X3_temp);

    // Compute two_S = 2 * S mod p
    doubleModP(S, two_S);

    // Compute X3 = X3_temp - two_S mod p
    subModP(X3_temp, two_S, X3);

    // Compute T = S - X3 mod p
    subModP(S, X3, T);

    // Compute Y3_temp = M * T mod p
    mulModP(M, T, Y3_temp);

    // Compute S1_squared = S1^2 mod p
    squareModP(S1, S1_squared);

    // Compute eight_S1_squared = 8 * S1_squared mod p
    doubleModP(S1_squared, temp1); // temp1 = 2 * S1_squared
    doubleModP(temp1, temp2); // temp2 = 4 * S1_squared
    doubleModP(temp2, eight_S1_squared); // eight_S1_squared = 8 * S1_squared

    // Compute Y3 = Y3_temp - eight_S1_squared mod p
    subModP(Y3_temp, eight_S1_squared, Y3);

    // Compute Z3 = 2 * Y1 * Z1 mod p
    mulModP(Y1, Z1, temp1); // temp1 = Y1 * Z1 mod p
    doubleModP(temp1, Z3);  // Z3 = 2 * Y1 * Z1 mod p
}


// Convert point from Jacobian to affine coordinates
__device__ void jacobianToAffine(const unsigned int X[8], const unsigned int Y[8], const unsigned int Z[8],
                                 unsigned int x[8], unsigned int y[8]) {
    // Compute Zinv = inverse of Z modulo p
    unsigned int Zinv[8];
    copyBigInt(Z, Zinv);
    invModP(Zinv);

    // Compute Zinv2 = Zinv^2
    unsigned int Zinv2[8];
    mulModP(Zinv, Zinv, Zinv2);

    // Compute Zinv3 = Zinv^2 * Zinv = Zinv^3
    unsigned int Zinv3[8];
    mulModP(Zinv2, Zinv, Zinv3);

    // x = X * Zinv2 mod p
    mulModP(X, Zinv2, x);

    // y = Y * Zinv3 mod p
    mulModP(Y, Zinv3, y);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: ./keygen_cuda <start_privkey_hex> <end_privkey_hex>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string startKeyHex = argv[1];
    std::string endKeyHex = argv[2];

    unsigned char startKey[32];
    unsigned char endKey[32];

    if (!hexToBytes(startKeyHex, startKey, 32)) {
        std::cerr << "Invalid start private key format." << std::endl;
        return EXIT_FAILURE;
    }

    if (!hexToBytes(endKeyHex, endKey, 32)) {
        std::cerr << "Invalid end private key format." << std::endl;
        return EXIT_FAILURE;
    }

    // Validate that startKey <= endKey
    bool validRange = false;
    for (int i = 0; i < 32; ++i) {
        if (startKey[i] < endKey[i]) {
            validRange = true;
            break;
        } else if (startKey[i] > endKey[i]) {
            validRange = false;
            break;
        }
    }

    if (!validRange) {
        std::cerr << "Error: Start private key is greater than end private key." << std::endl;
        return EXIT_FAILURE;
    }

    // Compute total number of keys
    size_t totalKeys = computeTotalKeys(startKey, endKey);
    std::cout << "Total keys to process: " << totalKeys << std::endl;

    // Prepare private keys
    std::vector<unsigned int> h_privKeys(totalKeys * 8); // 8 words per key

    unsigned char currentKey[32];
    memcpy(currentKey, startKey, 32);

    for (size_t i = 0; i < totalKeys; ++i) {
        // Convert currentKey (byte array) to unsigned int array
        for (int j = 0; j < 8; ++j) {
            h_privKeys[i + j * totalKeys] =
                (currentKey[4 * j] << 24) | (currentKey[4 * j + 1] << 16) |
                (currentKey[4 * j + 2] << 8) | (currentKey[4 * j + 3]);
        }
        incrementPrivKey(currentKey);
    }

    // Allocate device memory
    unsigned int* d_privKeys;
    unsigned int* d_pubKeysX;
    unsigned int* d_pubKeysY;

    cudaMalloc(&d_privKeys, totalKeys * 8 * sizeof(unsigned int));
    cudaMalloc(&d_pubKeysX, totalKeys * 8 * sizeof(unsigned int));
    cudaMalloc(&d_pubKeysY, totalKeys * 8 * sizeof(unsigned int));

    // Copy private keys to device
    cudaMemcpy(d_privKeys, h_privKeys.data(), totalKeys * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalKeys + threadsPerBlock - 1) / threadsPerBlock;

    computePublicKeys<<<blocksPerGrid, threadsPerBlock>>>(d_privKeys, d_pubKeysX, d_pubKeysY, totalKeys);
    cudaDeviceSynchronize();

    // Copy public keys back to host (if needed)
    std::vector<unsigned int> h_pubKeysX(totalKeys * 8);
    std::vector<unsigned int> h_pubKeysY(totalKeys * 8);

    cudaMemcpy(h_pubKeysX.data(), d_pubKeysX, totalKeys * 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pubKeysY.data(), d_pubKeysY, totalKeys * 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // For demonstration, print the first public key
    std::cout << "First public key X: ";
    for (int i = 0; i < 8; ++i) {
        printf("%08x", h_pubKeysX[i]);
    }
    std::cout << std::endl;

    std::cout << "First public key Y: ";
    for (int i = 0; i < 8; ++i) {
        printf("%08x", h_pubKeysY[i]);
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_privKeys);
    cudaFree(d_pubKeysX);
    cudaFree(d_pubKeysY);

    return EXIT_SUCCESS;
}