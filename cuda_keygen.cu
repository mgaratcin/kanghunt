#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "point.cuh"
#include "ecc.cuh"
#include "u64.cuh" // Ensure this is included if u64 is defined here

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

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <start_private_key_in_hex> <end_private_key_in_hex>\n", argv[0]);
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

    while (isLessOrEqual(currentKey, endKey)) {
        // Prepare the output for the public key
        Point publicKey;

        // Run the CUDA function to get the public key
        // Pass &currentKey to match the expected type const u64 (*)[4]
        getPublicKeyByPrivateKey(&publicKey, (const u64 (*)[4]) &currentKey, 1);

        // Output the private key and its corresponding public key in hexadecimal format
        char privateKeyHex[65];
        privateKeyToHex(currentKey, privateKeyHex);
        printf("Private Key: %s\n", privateKeyHex);
        printf("Public Key:\n");
        for (int i = 0; i < 4; ++i) printf("%016llx", publicKey.x[3 - i]);
        printf(":");
        for (int i = 0; i < 4; ++i) printf("%016llx", publicKey.y[3 - i]);
        printf("\n\n");

        // Increment the private key
        if (!incrementPrivateKey(currentKey)) {
            printf("Reached the maximum private key value.\n");
            break;
        }
    }

    return 0;
}
