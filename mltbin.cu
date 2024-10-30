#include <stdio.h>
#include <curand_kernel.h>
#include <string.h>
#include <math.h> // Include math library for log2f
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <unordered_map> // For storing distinguished points
#include <mutex>

#define BINARY_LENGTH 135
#define INITIAL_VALUE "0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
#define THREADS_PER_BLOCK 256 // Number of threads per block
#define BLOCKS_PER_GPU 256    // Number of blocks per GPU
#define TARGET_TAIL_BITS 25   // Distinguished points with at least 25 trailing zeros
#define PRINT_INTERVAL 1000000000ULL
#define BATCH_SIZE 100000000ULL // Adjusted for testing
#define SEED 1234 // Base seed for random number generator
#define MAX_DISTINGUISHED_POINTS 1000000000 // Maximum number of distinguished points to store

struct Counter128 {
    unsigned long long low;
    unsigned long long high;
};

struct Point {
    unsigned long long high;
    unsigned long long mid;
    unsigned long long low;
    unsigned long long steps;
    bool is_tame;
};

std::unordered_map<unsigned long long, Point> distinguished_points;
std::mutex dp_mutex;

// Device function to check if a value ends with TARGET_TAIL_BITS zeros
__device__ __forceinline__ bool ends_with_25_zeros(unsigned long long value_low)
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

// Device function to add a value to a 128-bit counter
__device__ void atomicAdd128(Counter128 *counter, unsigned long long value)
{
    unsigned long long old_low = atomicAdd(&(counter->low), value);
    if (old_low + value < old_low) // Handle overflow of the lower part
    {
        atomicAdd(&(counter->high), 1ULL);
    }
}

// Generate paths kernel with distinguished points storage functionality
__global__ void generate_paths(curandState *state, unsigned long long tame_high, unsigned long long tame_mid, unsigned long long tame_low, Counter128 *global_counter, Point *dp_points, int *dp_count)
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

            // Check if tame point is distinguished
            if (ends_with_25_zeros(tame_low))
            {
                int dp_idx = atomicAdd(dp_count, 1);
                if (dp_idx < MAX_DISTINGUISHED_POINTS) {
                    dp_points[dp_idx] = {tame_high, tame_mid, tame_low, steps_tame, true};
                } else {
                    // Prevent out-of-bounds memory access
                    atomicSub(dp_count, 1);
                }
            }

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

            // Check if wild point is distinguished
            if (ends_with_25_zeros(wild_low))
            {
                int dp_idx = atomicAdd(dp_count, 1);
                if (dp_idx < MAX_DISTINGUISHED_POINTS) {
                    dp_points[dp_idx] = {wild_high, wild_mid, wild_low, steps_wild, false};
                } else {
                    // Prevent out-of-bounds memory access
                    atomicSub(dp_count, 1);
                }
            }
        }

        // Use atomic operation to update the global 128-bit counter
        atomicAdd128(global_counter, steps_tame * tame_increment + steps_wild);

        // Print progress periodically
        if (idx == 0 && global_counter->low % PRINT_INTERVAL < tame_increment * BATCH_SIZE) {
            unsigned long long total_operations_low = global_counter->low;
            unsigned long long total_operations_high = global_counter->high;
            double total_operations = (double)total_operations_high * pow(2.0, 64) + (double)total_operations_low;
            double n = log2(total_operations);
            printf("Batch completed: Total operations: 2^%.2lf\n", n);
            printf("Total distinguished points found so far: %d\n", *dp_count);
        }

        // Periodically update the global state to prevent corruption
        if (steps_tame % 100000 == 0) {
            state[idx] = localState;
        }
    }
}

// Function to run on each GPU
void run_on_device(int device_id, const char* initial_value, unsigned long long seed_offset, Counter128 *global_counter)
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

    // Allocate memory for distinguished points
    Point *d_dp_points;
    int *d_dp_count;
    err = cudaMalloc(&d_dp_points, MAX_DISTINGUISHED_POINTS * sizeof(Point));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to allocate distinguished points memory: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        return;
    }
    err = cudaMalloc(&d_dp_count, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to allocate distinguished points count memory: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        return;
    }
    cudaMemset(d_dp_count, 0, sizeof(int));

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
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
        return;
    }

    // Launch the generate_paths kernel
    generate_paths<<<BLOCKS_PER_GPU, THREADS_PER_BLOCK>>>(d_state, tame_high, tame_mid, tame_low, global_counter, d_dp_points, d_dp_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to launch generate_paths kernel: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
        return;
    }

    // Synchronize the device (will block indefinitely due to the infinite loop in the kernel)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: CUDA Device Synchronize failed: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
        return;
    }

    // Free allocated memory (Unreachable due to infinite loop)
    cudaFree(d_state);
    cudaFree(d_dp_points);
    cudaFree(d_dp_count);
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

    // Allocate unified memory for the global counter
    Counter128 *global_counter;
    err = cudaMallocManaged(&global_counter, sizeof(Counter128));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate global counter memory: %s\n", cudaGetErrorString(err));
        return 1;
    }
    global_counter->low = 0;
    global_counter->high = 0;

    // Create threads for each GPU
    std::vector<std::thread> threads;
    for (int device_id = 0; device_id < device_count; ++device_id)
    {
        // Each GPU gets a unique seed offset to ensure different random sequences
        unsigned long long seed_offset = device_id * 1000;
        threads.emplace_back(run_on_device, device_id, INITIAL_VALUE, seed_offset, global_counter);
    }

    // Wait for all threads to finish (they won't, due to the infinite loop in the kernel)
    for (auto &t : threads)
    {
        t.join();
    }

    // Free the global counter memory
    cudaFree(global_counter);

    return 0;
}
