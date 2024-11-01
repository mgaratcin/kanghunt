#include <stdio.h> 
#include <curand_kernel.h>
#include <string.h>
#include <math.h> // For log2
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <unistd.h>

// Constants and Macros
#define BINARY_LENGTH 135
#define INITIAL_VALUE "0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
#define THREADS_PER_BLOCK 256 // Number of threads per block
#define BLOCKS_PER_GPU 256    // Number of blocks per GPU
#define TARGET_TAIL_BITS 25   // Distinguished points with at least 25 trailing zeros
#define BATCH_SIZE 1000000ULL // Adjusted for initial debugging
#define SEED 1234 // Base seed for random number generator
#define MAX_DISTINGUISHED_POINTS_PER_KERNEL 1000000 // Adjusted for initial debugging

// Structure Definitions
struct Counter128 {
    unsigned long long low;
    unsigned long long high;
};

struct Point {
    unsigned long long high;
    unsigned long long mid;
    unsigned long long low;
    unsigned long long steps;
    unsigned char is_tame; // Changed from bool to unsigned char for alignment
    unsigned char padding[7]; // Padding to make the structure 40 bytes (aligned)
};

// Static assertion to ensure Point struct size is consistent
static_assert(sizeof(Point) == 40, "Point struct size must be 40 bytes");

// Host-side container for distinguished points
struct HostDPSContainer {
    std::vector<Point> points;
    std::mutex container_mutex;
    // Removed target_printed and associated logic

    void append(const std::vector<Point>& new_points, unsigned long long device_id) {
        std::lock_guard<std::mutex> lock(container_mutex);
        points.insert(points.end(), new_points.begin(), new_points.end());
        // Removed the check for TARGET_DP_INDEX and associated print and exit
    }
};

// Device function to check if a value ends with TARGET_TAIL_BITS zeros
__device__ __forceinline__ bool ends_with_target_zeros(unsigned long long value_low)
{
    return (value_low & ((1ULL << TARGET_TAIL_BITS) - 1)) == 0;
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

// Device function to atomically add to a 128-bit counter
__device__ void atomicAdd128(Counter128 *counter, unsigned long long value)
{
    unsigned long long old_low = atomicAdd(&(counter->low), value);
    if (old_low + value < old_low) // Handle overflow of the lower part
    {
        atomicAdd(&(counter->high), 1ULL);
    }
}

// Kernel to generate paths and collect distinguished points
__global__ void generate_paths(
    curandState *state,
    unsigned long long tame_high,
    unsigned long long tame_mid,
    unsigned long long tame_low,
    Counter128 *global_counter,
    Point *dp_points,
    unsigned int *dp_count,
    unsigned long long batch_size,
    unsigned int max_dp // Passed as a kernel parameter
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];

    // Initialize local tame and wild variables
    unsigned long long local_tame_high = tame_high;
    unsigned long long local_tame_mid = tame_mid;
    unsigned long long local_tame_low = tame_low;

    unsigned long long local_wild_high = tame_high;
    unsigned long long local_wild_mid = tame_mid;
    unsigned long long local_wild_low = tame_low;

    unsigned long long steps_tame = 0;
    unsigned long long steps_wild = 0;

    // Using a larger increment for tame_low to speed up tame path updates
    const unsigned long long tame_increment = 65536;

    for (unsigned long long batch = 0; batch < batch_size; ++batch)
    {
        // Increment tame path by a larger value to increase speed
        unsigned long long new_tame_low = local_tame_low + tame_increment;
        if (new_tame_low < local_tame_low) // Handle overflow
        {
            local_tame_mid++;
            if (local_tame_mid == 0) local_tame_high = (local_tame_high + 1) & 0x7FULL;
        }
        local_tame_low = new_tame_low;
        steps_tame++;

        // Check if tame point is distinguished
        if (ends_with_target_zeros(local_tame_low))
        {
            // Atomically fetch and add
            unsigned int dp_idx = atomicAdd(dp_count, 1);
            if (dp_idx < max_dp) {
                dp_points[dp_idx].high = local_tame_high;
                dp_points[dp_idx].mid = local_tame_mid;
                dp_points[dp_idx].low = local_tame_low;
                dp_points[dp_idx].steps = steps_tame;
                dp_points[dp_idx].is_tame = 1;
                // Set padding to 0
                for(int p=0; p<7; p++) dp_points[dp_idx].padding[p] = 0;
            }
            else {
                // Reached max_dp, exit early
                break;
            }
        }

        // Increment wild path by a pseudo-random value less than 135 bits
        unsigned long long random_increment = ((unsigned long long)(curand(&localState) & 0xFFFFFFFFFFFFULL) << 12) | ((unsigned long long)(curand(&localState) & 0xFFFULL));
        unsigned long long new_wild_low = local_wild_low + random_increment;
        if (new_wild_low < local_wild_low) // Handle overflow
        {
            local_wild_mid++;
            if (local_wild_mid == 0) local_wild_high = (local_wild_high + 1) & 0x7FULL;
        }
        local_wild_low = new_wild_low;
        steps_wild++;

        // Check if wild point is distinguished
        if (ends_with_target_zeros(local_wild_low))
        {
            // Atomically fetch and add
            unsigned int dp_idx = atomicAdd(dp_count, 1);
            if (dp_idx < max_dp) {
                dp_points[dp_idx].high = local_wild_high;
                dp_points[dp_idx].mid = local_wild_mid;
                dp_points[dp_idx].low = local_wild_low;
                dp_points[dp_idx].steps = steps_wild;
                dp_points[dp_idx].is_tame = 0;
                // Set padding to 0
                for(int p=0; p<7; p++) dp_points[dp_idx].padding[p] = 0;
            }
            else {
                // Reached max_dp, exit early
                break;
            }
        }

        // Optional: Exit the loop early if max_dp is reached to save computation
        if (*dp_count >= max_dp) {
            break;
        }
    }

    // Update the global 128-bit counter
    unsigned long long total_steps = steps_tame * tame_increment + steps_wild;
    atomicAdd128(global_counter, total_steps);

    // Save the updated state
    state[idx] = localState;
}

// Function to append dps from device to host
void append_dps_to_host(Point* h_dp_points, unsigned int dp_count_host, HostDPSContainer* host_dps, unsigned long long device_id)
{
    if (dp_count_host == 0) return;
    std::vector<Point> temp(h_dp_points, h_dp_points + dp_count_host);
    host_dps->append(temp, device_id);
}

// Function to run on each GPU
void run_on_device(
    int device_id,
    const char* initial_value,
    unsigned long long seed_offset,
    Counter128 *global_counter,
    HostDPSContainer* host_dps
)
{
    cudaError_t err;

    // Set the current device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to set device: %s\n", device_id, cudaGetErrorString(err));
        return;
    }

    // Allocate memory for curand states
    curandState *d_state;
    err = cudaMalloc(&d_state, THREADS_PER_BLOCK * BLOCKS_PER_GPU * sizeof(curandState));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to allocate device state: %s\n", device_id, cudaGetErrorString(err));
        return;
    }

    // Allocate device memory for distinguished points
    Point *d_dp_points;
    err = cudaMalloc(&d_dp_points, MAX_DISTINGUISHED_POINTS_PER_KERNEL * sizeof(Point));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to allocate distinguished points memory: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        return;
    }

    // Allocate device memory for dp_count
    unsigned int *d_dp_count;
    err = cudaMalloc(&d_dp_count, sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to allocate distinguished points count memory: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        return;
    }
    err = cudaMemset(d_dp_count, 0, sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to reset dp_count: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
        return;
    }

    // Allocate pinned host memory for dps
    Point *h_dp_points;
    err = cudaHostAlloc(&h_dp_points, MAX_DISTINGUISHED_POINTS_PER_KERNEL * sizeof(Point), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to allocate pinned host memory: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
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
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
        cudaFreeHost(h_dp_points);
        return;
    }

    // Synchronize to ensure curand states are initialized
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: CUDA Device Synchronize failed after init_curand_states: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
        cudaFreeHost(h_dp_points);
        return;
    }

    // Launch parameters
    unsigned long long iterations_per_launch = BATCH_SIZE;
    dim3 grid(BLOCKS_PER_GPU);
    dim3 block(THREADS_PER_BLOCK);

    // Main loop controlled by the host
    while (true)
    {
        // Reset dp_count on device
        err = cudaMemset(d_dp_count, 0, sizeof(unsigned int));
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU %d: Failed to reset dp_count: %s\n", device_id, cudaGetErrorString(err));
            break;
        }

        // Launch the generate_paths kernel with max_dp parameter
        generate_paths<<<grid, block>>>(
            d_state,
            tame_high,
            tame_mid,
            tame_low,
            global_counter,
            d_dp_points,
            d_dp_count,
            iterations_per_launch,
            MAX_DISTINGUISHED_POINTS_PER_KERNEL // Pass as a parameter
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU %d: Failed to launch generate_paths kernel: %s\n", device_id, cudaGetErrorString(err));
            break;
        }

        // Synchronize to wait for kernel completion
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU %d: CUDA Device Synchronize failed after generate_paths: %s\n", device_id, cudaGetErrorString(err));
            break;
        }

        // Copy dp_count from device to host
        unsigned int dp_count_host;
        err = cudaMemcpy(&dp_count_host, d_dp_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU %d: Failed to copy dp_count from device to host: %s\n", device_id, cudaGetErrorString(err));
            break;
        }

        // Clamp dp_count_host to max_dp
        if (dp_count_host > MAX_DISTINGUISHED_POINTS_PER_KERNEL) {
            dp_count_host = MAX_DISTINGUISHED_POINTS_PER_KERNEL;
        }

        if (dp_count_host > 0) {
            // Copy dps from device to host pinned memory
            err = cudaMemcpy(h_dp_points, d_dp_points, dp_count_host * sizeof(Point), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "GPU %d: Failed to copy distinguished points from device to host: %s\n", device_id, cudaGetErrorString(err));
                break;
            }

            // Append to the main host container
            append_dps_to_host(h_dp_points, dp_count_host, host_dps, device_id);
        }

        // Optional: Implement a termination condition here if needed
    }

    // Free allocated memory
    cudaFree(d_state);
    cudaFree(d_dp_points);
    cudaFree(d_dp_count);
    cudaFreeHost(h_dp_points);
}

// Structure to define step thresholds
struct StepThreshold {
    unsigned long long high;
    unsigned long long low;
    unsigned int exponent; // For printing purposes, e.g., 60 for 2^60
    bool printed;
};

// Step monitor function
void step_monitor(Counter128 *global_counter, std::vector<StepThreshold> &thresholds)
{
    bool all_printed = false;
    while (!all_printed)
    {
        // Read the current total steps
        unsigned long long high = global_counter->high;
        unsigned long long low = global_counter->low;

        // Iterate through thresholds
        for (auto &threshold : thresholds)
        {
            if (!threshold.printed)
            {
                if ( (high > threshold.high) || 
                     (high == threshold.high && low >= threshold.low) )
                {
                    // Calculate the actual step count for printing
                    // For 2^n, step count is 1ULL << n
                    // However, to display the total steps, we can compute it as high * 2^64 + low
                    // Since n can be up to 80, which is less than 128 bits

                    // To display 2^n, we can just print the exponent
                    printf("Total steps reached 2^%u.\n", threshold.exponent);
                    threshold.printed = true;
                }
            }
        }

        // Check if all thresholds have been printed
        all_printed = true;
        for (const auto &threshold : thresholds)
        {
            if (!threshold.printed)
            {
                all_printed = false;
                break;
            }
        }

        // Sleep for a short duration to avoid busy waiting
        sleep(1);
    }
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

    // Initialize the host container for dps
    HostDPSContainer host_dps;

    // Define the step thresholds
    std::vector<StepThreshold> thresholds = {
        {0, 1ULL << 60, 60, false},   // 2^60
        {1ULL << 1, 0, 65, false},    // 2^65
        {1ULL << 6, 0, 70, false},    // 2^70
        {1ULL << 11, 0, 75, false},   // 2^75
        {1ULL << 16, 0, 80, false}    // 2^80
    };

    // Create a thread for step monitoring
    std::thread monitor_thread(step_monitor, global_counter, std::ref(thresholds));

    // Create threads for each GPU
    std::vector<std::thread> threads;
    for (int device_id = 0; device_id < device_count; ++device_id)
    {
        // Each GPU gets a unique seed offset to ensure different random sequences
        unsigned long long seed_offset = device_id * 1000;
        threads.emplace_back(run_on_device, device_id, INITIAL_VALUE, seed_offset, global_counter, &host_dps);
    }

    // Wait for all device threads to finish
    for (auto &t : threads)
    {
        t.join();
    }

    // Wait for the monitor thread to finish
    monitor_thread.join();

    // Free the global counter memory
    cudaFree(global_counter);

    // Optionally, process or save the accumulated distinguished points here
    /*
    FILE *fp = fopen("distinguished_points.bin", "wb");
    if (fp) {
        fwrite(host_dps.points.data(), sizeof(Point), host_dps.points.size(), fp);
        fclose(fp);
    } else {
        fprintf(stderr, "Failed to open file for writing distinguished points.\n");
    }
    */

    return 0;
}
