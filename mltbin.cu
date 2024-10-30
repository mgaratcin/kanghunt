#include <stdio.h>
#include <curand_kernel.h>
#include <string.h>
#include <math.h> // Include math library for log2f
#include <cuda_runtime.h>
#include <thread>
#include <vector>

#define BINARY_LENGTH 135
#define INITIAL_VALUE "0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
#define THREADS_PER_BLOCK 256 // Number of threads per block
#define BLOCKS_PER_GPU 256    // Number of blocks per GPU
#define TARGET_TAIL_BITS 20
#define PRINT_INTERVAL 1000000000ULL
#define BATCH_SIZE 100000000ULL // Adjusted for testing
#define SEED 1234 // Base seed for random number generator

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
    const unsigned long long tame_increment = 65536;

    while (true) // Infinite loop
    {
        for (unsigned long long batch = 0; batch < BATCH_SIZE; ++batch)
        {
            // Increment tame path by a larger value to increase speed
            unsigned long long new_tame_low = tame_low + tame_increment;
            if (new_tame_low < tame_low) // Handle overflow
            {
                tame_mid++;
                if (tame_mid == 0) tame_high = (tame_high + 1) & 0x7FULL;
            }
            tame_low = new_tame_low;
            steps_tame++;

            // Increment wild path by a pseudo-random value less than 135 bits
            unsigned long long random_increment = ((unsigned long long)(curand(&localState) & 0xFFFFFFFFFFFFULL) << 12) | ((unsigned long long)(curand(&localState) & 0xFFFULL));
            unsigned long long new_wild_low = wild_low + random_increment;
            if (new_wild_low < wild_low) // Handle overflow
            {
                wild_mid++;
                if (wild_mid == 0) wild_high = (wild_high + 1) & 0x7FULL;
            }
            wild_low = new_wild_low;
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
            // Calculate Total Operations = (steps_tame * tame_increment) + steps_wild
            double total_operations = (double)(steps_tame * tame_increment) + (double)(steps_wild);
            double n = log2(total_operations);

            printf("Batch completed: Total operations: 2^%.2lf\n", n);
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

// Function to run on each GPU
void run_on_device(int device_id, const char* initial_value, unsigned long long seed_offset)
{
    cudaError_t err;

    // Set the current device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to set device: %s\n", device_id, cudaGetErrorString(err));
        return;
    }

    // Get device properties for logging
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    printf("Running on GPU %d: %s\n", device_id, deviceProp.name);

    // Allocate memory for curand states
    curandState *d_state;
    err = cudaMalloc(&d_state, THREADS_PER_BLOCK * BLOCKS_PER_GPU * sizeof(curandState));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to allocate device state: %s\n", device_id, cudaGetErrorString(err));
        return;
    }

    // Initialize the 135-bit tame value
    unsigned long long tame_high, tame_mid, tame_low;
    initialize_135_bit_value(initial_value, tame_high, tame_mid, tame_low);

    // Initialize curand states with unique seed per GPU
    unsigned long long seed = SEED + seed_offset;
    init_curand_states<<<BLOCKS_PER_GPU, THREADS_PER_BLOCK>>>(d_state, seed);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to launch init_curand_states kernel: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        return;
    }

    // Launch the generate_paths kernel
    generate_paths<<<BLOCKS_PER_GPU, THREADS_PER_BLOCK>>>(d_state, tame_high, tame_mid, tame_low);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to launch generate_paths kernel: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        return;
    }

    // Synchronize the device (will block indefinitely due to the infinite loop in the kernel)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: CUDA Device Synchronize failed: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        return;
    }

    // Free allocated memory (Unreachable due to infinite loop)
    cudaFree(d_state);
}

int main()
{
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (device_count < 1) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    printf("Found %d CUDA device(s).\n", device_count);

    // Initialize the 135-bit tame value on host
    unsigned long long tame_high, tame_mid, tame_low;
    initialize_135_bit_value(INITIAL_VALUE, tame_high, tame_mid, tame_low);

    // Create threads for each GPU
    std::vector<std::thread> threads;
    for (int device_id = 0; device_id < device_count; ++device_id)
    {
        // Each GPU gets a unique seed offset to ensure different random sequences
        unsigned long long seed_offset = device_id * 1000;
        threads.emplace_back(run_on_device, device_id, INITIAL_VALUE, seed_offset);
    }

    // Wait for all threads to finish (they won't, due to the infinite loop in the kernel)
    for (auto &t : threads)
    {
        t.join();
    }

    return 0;
}
