// detect_collisions.cu

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define KEY_SIZE_BYTES 17 // 135 bits = 17 bytes

// Define a structure to hold 135-bit keys
struct uint135_t {
    unsigned char data[KEY_SIZE_BYTES]; // 17 bytes to hold 135 bits
};

// Function to print uint135_t in hex format
void print_uint135_t_hex(const uint135_t& num) {
    for (int i = 0; i < KEY_SIZE_BYTES; ++i) {
        printf("%02x", num.data[i]);
    }
    printf("\n");
}

extern "C" {

// Kernel function (not used in this context)
__global__ void dummy_kernel() {
    // No operation needed
}

// Host function to print the 1,000th base_key and current_key
void detect_collisions_cuda(uint135_t* d_base_keys, uint135_t* d_current_keys, int batch_size) {
    // Allocate host memory for the 1,000th keys
    uint135_t h_base_key;
    uint135_t h_current_key;

    // Calculate the offset to the 10,000th element (index 9999)
    size_t offset = 9999 * sizeof(uint135_t);

    // Copy the 10,000th keys from device to host
    cudaMemcpy(&h_base_key, (unsigned char*)d_base_keys + offset, sizeof(uint135_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_current_key, (unsigned char*)d_current_keys + offset, sizeof(uint135_t), cudaMemcpyDeviceToHost);

    // Print the keys
    printf("10,000th base_key: ");
    print_uint135_t_hex(h_base_key);

    printf("10,000th current_key: ");
    print_uint135_t_hex(h_current_key);
}

}
