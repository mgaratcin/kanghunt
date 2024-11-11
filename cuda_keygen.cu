#include <stdio.h>
#include "point.cuh"
#include "ecc.cuh"

// Function to convert a hexadecimal private key to u64 array
void hexToPrivateKey(const char* hex, u64 privateKey[4]) {
    for (int i = 0; i < 4; ++i) {
        sscanf(hex + 16 * i, "%16llx", &privateKey[3 - i]); // Parse in reverse for endian compatibility
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <private_key_in_hex>\n", argv[0]);
        return 1;
    }

    // Parse the input private key
    u64 privateKey[4];
    hexToPrivateKey(argv[1], privateKey);

    // Prepare the output for the public key
    Point publicKey;

    // Run the CUDA function to get the public key
    getPublicKeyByPrivateKey(&publicKey, &privateKey, 1);

    // Output the public key in hexadecimal format
    printf("Public Key:\n");
    for (int i = 0; i < 4; ++i) printf("%016llx", publicKey.x[3 - i]);
    printf(":");
    for (int i = 0; i < 4; ++i) printf("%016llx", publicKey.y[3 - i]);
    printf("\n");

    return 0;
}
