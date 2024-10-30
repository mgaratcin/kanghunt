#include <stdio.h>
#include <curand_kernel.h>
#include <string.h>

#define BINARY_LENGTH 135
#define INITIAL_VALUE "0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
#define THREADS_PER_BLOCK 256 // Increase thread count to maximize occupancy
#define BLOCKS 256 // Increase block count to maximize GPU utilization and achieve faster execution
#define TARGET_TAIL_BITS 20
#define PRINT_INTERVAL 1000000000ULL
#define BATCH_SIZE 100000000ULL // Reduced from 10000000ULL for testing
// #define MAX_STORE_VALUES 10240000 // Removed since storage functionality is disabled
#define SEED 1234 // Seed for random number generator

// Device function to check if a value ends with 20 zeros
__device__ __forceinline__ bool ends_with_20_zeros(unsigned long long value_low)
{
    return (value_low & ((1ULL << TARGET_TAIL_BITS) - 1)) == 0;
}

// Device function to print 135-bit value in binary format
__device__ void print_135_bit_value_device(unsigned long long high, unsigned long long mid, unsigned long long low)
{
    for (int j = 6; j >= 0; j--) printf("%d", (int)((high >> j) & 1));
    for (int j = 63; j >= 0; j--) printf("%d", (int)((mid >> j) & 1));
    for (int j = 63; j >= 0; j--) printf("%d", (int)((low >> j) & 1));
    printf("\n");
}

// Helper function to initialize 135-bit value from a binary string
__host__ void initialize_135_bit_value(const char *binary_str, unsigned long long &high, unsigned long long &mid, unsigned long long &low)
{
    high = 0;
    mid = 0;
    low = 0;
    int length = strlen(binary_str);

    for (int i = 0; i < length; i++)
    {
        if (binary_str[i] == '1')
        {
            if (i < 7) high |= (1ULL << (6 - i));
            else if (i < 71) mid |= (1ULL << (70 - i));
            else if (i < 135) low |= (1ULL << (134 - i));
        }
    }
}

// Kernel to initialize curand states
__global__ void init_curand_states(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize each state with a unique seed based on thread index
    curand_init(seed, idx, 0, &state[idx]);
}

// Generate paths kernel without storage functionality
__global__ void generate_paths(curandState *state, unsigned long long tame_high, unsigned long long tame_mid, unsigned long long tame_low)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];

    unsigned long long wild_high = tame_high;
    unsigned long long wild_mid = tame_mid;
    unsigned long long wild_low = tame_low;

    unsigned long long steps_tame = 0;
    unsigned long long steps_wild = 0;

    // Using a larger increment for tame_low to speed up tame path updates
    const unsigned long long tame_increment = 16384;

    while (true) // Infinite loop
    {
        for (unsigned long long batch = 0; batch < BATCH_SIZE; ++batch)
        {
            // Increment tame path by a larger value to increase speed
            tame_low += tame_increment;
            if (tame_low < tame_increment)
            {
                tame_mid++;
                if (tame_mid == 0) tame_high = (tame_high + 1) & 0x7FULL;
            }
            steps_tame++;

            // Increment wild path by a pseudo-random value less than 135 bits
            unsigned long long random_increment = ((unsigned long long)(curand(&localState) & 0xFFFFFFFFFFFFULL) << 12) | ((unsigned long long)(curand(&localState) & 0xFFFULL));
            wild_low += random_increment;
            if (wild_low < random_increment)
            {
                wild_mid++;
                if (wild_mid == 0) wild_high = (wild_high + 1) & 0x7FULL;
            }
            steps_wild++;

            // Check for collision
            if (tame_high == wild_high && tame_mid == wild_mid && tame_low == wild_low)
            {
                // Limit the number of collision messages to prevent flooding the output
                if (idx == 0) {
                    printf("Collision detected! Steps Tame: %llu, Steps Wild: %llu\n", steps_tame, steps_wild);
                }
            }
        }

        // Print progress at each batch
        if (idx == 0)
        {
            printf("Batch completed: Steps Tame: %llu, Steps Wild: %llu\n", steps_tame * tame_increment, steps_wild);
        }

        // Periodically update the global state to prevent corruption
        if (steps_tame % 100000 == 0) {
            state[idx] = localState;
        }
    }

    // The following code will never be reached due to the infinite loop
    /*
    if (idx == 0) {
        printf("Max steps reached without collision.\n");
        printf("Final Tame Value: ");
        print_135_bit_value_device(tame_high, tame_mid, tame_low);
        printf("Final Wild Value: ");
        print_135_bit_value_device(wild_high, wild_mid, wild_low);
        printf("Final Steps Tame: %llu, Final Steps Wild: %llu\n", steps_tame * tame_increment, steps_wild);
    }
    */
}

int main()
{
    cudaError_t err;

    // Allocate memory for curand states
    curandState *d_state;
    err = cudaMalloc(&d_state, THREADS_PER_BLOCK * BLOCKS * sizeof(curandState));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device state: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // (Storage functionality removed)

    // Initialize the 135-bit tame value
    unsigned long long tame_high, tame_mid, tame_low;
    initialize_135_bit_value(INITIAL_VALUE, tame_high, tame_mid, tame_low);

    // Initialize curand states
    init_curand_states<<<BLOCKS, THREADS_PER_BLOCK>>>(d_state, SEED);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch init_curand_states kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch the generate_paths kernel without storage parameters
    generate_paths<<<BLOCKS, THREADS_PER_BLOCK>>>(d_state, tame_high, tame_mid, tame_low);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch generate_paths kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Since the kernel runs indefinitely, we use cudaDeviceSynchronize() to wait for it.
    // You can stop the program manually when desired.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Device Synchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Free allocated memory (This part will never be reached due to the infinite loop)
    cudaFree(d_state);
    /*
    cudaFree(stored_tame_high);
    cudaFree(stored_tame_mid);
    cudaFree(stored_tame_low);
    cudaFree(stored_wild_high);
    cudaFree(stored_wild_mid);
    cudaFree(stored_wild_low);
    */

    return 0;
}
