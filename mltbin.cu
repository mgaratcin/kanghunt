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
#include <fstream>
#include <unordered_set>
#include <functional>
#include <iostream>
#include <queue>
#include <condition_variable>

// Include LZ4 headers
#include <lz4.h>
#include <lz4frame.h>

// Constants and Macros
#define BINARY_LENGTH 135
#define INITIAL_VALUE "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000" // Corrected to start with '1' followed by '0's
#define THREADS_PER_BLOCK 256 // Number of threads per block
#define BLOCKS_PER_GPU 256    // Number of blocks per GPU
#define TARGET_TAIL_BITS 35   // Distinguished points with at least 35 trailing zeros
#define BATCH_SIZE 1000000ULL // Number of iterations per kernel launch
#define SEED 1234 // Base seed for random number generator
#define MAX_DISTINGUISHED_POINTS_PER_KERNEL 1000000 // Maximum DPs per kernel
#define BUFFER_SIZE 100000 // Number of Points to buffer before writing to disk
#define COMPRESSION_BUFFER_SIZE (LZ4_compressBound(BUFFER_SIZE * sizeof(struct Point))) // Maximum compressed size

// Structure Definitions

// Packed Point structure to reduce size from 40 to 26 bytes
#pragma pack(push, 1)
struct Point {
    unsigned char high;        // 1 byte (7 bits used)
    unsigned long long mid;    // 8 bytes
    unsigned long long low;    // 8 bytes
    unsigned long long steps;  // 8 bytes
    unsigned char is_tame;     // 1 byte
};
#pragma pack(pop)

// Static assertion to ensure Point struct size is consistent
static_assert(sizeof(Point) == 26, "Point struct size must be 26 bytes");

// Device function to check if a value ends with TARGET_TAIL_BITS zeros
__device__ __forceinline__ bool ends_with_target_zeros(unsigned long long value_low)
{
    return (value_low & ((1ULL << TARGET_TAIL_BITS) - 1)) == 0;
}

// Helper function to initialize 135-bit value from a binary string
__host__ void initialize_135_bit_value(const char *binary_str, unsigned char &high, unsigned long long &mid, unsigned long long &low)
{
    high = 0;
    mid = 0;
    low = 0;
    int length = strlen(binary_str);

    for (int i = 0; i < length; i++)
    {
        if (binary_str[i] == '1')
        {
            if (i < 7) high |= (1 << (6 - i));
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

// Structure for 128-bit counter
struct Counter128 {
    unsigned long long low;
    unsigned long long high;
};

// Device function to atomically add to a 128-bit counter
__device__ void atomicAdd128(Counter128 *counter, unsigned long long value)
{
    unsigned long long old_low = atomicAdd(&(counter->low), value);
    if (old_low + value < old_low) // Handle overflow of the lower part
    {
        atomicAdd(&(counter->high), 1ULL);
    }
}

// Structure to define step thresholds
struct StepThreshold {
    unsigned long long high;
    unsigned long long low;
    unsigned int exponent; // For printing purposes, e.g., 60 for 2^60
    bool printed;
};

// Custom hash function for Point
struct PointHash {
    std::size_t operator()(const Point& p) const {
        return std::hash<unsigned long long>()(p.mid) ^ 
               std::hash<unsigned long long>()(p.low);
    }
};

// Custom equality function for Point
struct PointEqual {
    bool operator()(const Point& a, const Point& b) const {
        return (a.mid == b.mid) && (a.low == b.low);
    }
};

// Thread-safe queue for compression
template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
public:
    void enqueue(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cond_var_.notify_one();
    }

    bool dequeue(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty()) {
            cond_var_.wait(lock);
        }
        if (!queue_.empty()) {
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;
    }

    // To allow graceful shutdown
    bool try_dequeue(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!queue_.empty()) {
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;
    }
};

// Function to detect collisions by aggregating points from all GPUs
void detect_collisions(const std::vector<std::string>& filenames)
{
    // Create an unordered_set with custom hash and equality functions
    std::unordered_set<Point, PointHash, PointEqual> point_set;

    // To keep track of collisions
    std::vector<Point> collisions;

    // Compression buffers
    std::vector<char> compressed_buffer;
    std::vector<char> decompressed_buffer;

    // Initialize counter for total dps and tracking the 10,000th dp
    unsigned long long total_dps = 0;

    for (const auto& filename : filenames) {
        std::ifstream dp_file(filename, std::ios::binary | std::ios::in);
        if (!dp_file.is_open()) {
            fprintf(stderr, "Failed to open file %s for reading.\n", filename.c_str());
            continue;
        }

        while (dp_file.peek() != EOF) {
            // Read the size of the compressed block
            int compressed_size;
            dp_file.read(reinterpret_cast<char*>(&compressed_size), sizeof(int));
            if (dp_file.eof()) break;
            if (compressed_size <= 0) {
                fprintf(stderr, "Invalid compressed size in file %s.\n", filename.c_str());
                break;
            }

            // Read the compressed data
            compressed_buffer.resize(compressed_size);
            dp_file.read(compressed_buffer.data(), compressed_size);
            if (dp_file.eof() || dp_file.fail()) {
                fprintf(stderr, "Failed to read compressed data from file %s.\n", filename.c_str());
                break;
            }

            // Estimate decompressed size (assuming it's BUFFER_SIZE * sizeof(Point))
            size_t decompressed_size = BUFFER_SIZE * sizeof(Point);
            decompressed_buffer.resize(decompressed_size);

            // Decompress the data
            int actual_decompressed_size = LZ4_decompress_safe(
                compressed_buffer.data(),
                decompressed_buffer.data(),
                compressed_size,
                decompressed_size
            );

            if (actual_decompressed_size < 0) {
                fprintf(stderr, "Decompression failed for a block in file %s.\n", filename.c_str());
                break;
            }

            // Number of Points in the decompressed data
            size_t num_points = actual_decompressed_size / sizeof(Point);
            Point* points = reinterpret_cast<Point*>(decompressed_buffer.data());

            for (size_t i = 0; i < num_points; ++i) {
                total_dps++;

                // Detect and handle the 10,000th DP within run_on_device, so skip here

                auto result = point_set.insert(points[i]);
                if (!result.second) { // Duplicate found
                    collisions.push_back(points[i]);
                }
            }
        }

        dp_file.close();
    }

    // Print collision information
    if (!collisions.empty()) {
        printf("Collisions detected:\n");
        for (const auto& p : collisions) {
            printf("Collision at Point: mid=%llu, low=%llu\n", p.mid, p.low);
        }
    }
    else {
        printf("No collisions detected.\n");
    }

    // Notify if the 10,000th DP wasn't found
    if (total_dps < 10000) {
        printf("Less than 10,000 DPs were processed.\n");
    }
}

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
                    // Print the threshold reached
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

// Compression task structure
struct CompressionTask {
    std::vector<Point> buffer;
    std::string filename;
};

// Kernel to generate paths and collect distinguished points
__global__ void generate_paths(
    curandState *state,
    unsigned char tame_high,
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
    unsigned char local_tame_high = tame_high;
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
            if (local_tame_mid == 0) local_tame_high = (local_tame_high + 1) & 0x7F; // 7 bits
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
            if (local_wild_mid == 0) local_wild_high = (local_wild_high + 1) & 0x7F; // 7 bits
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

// Function to run on each GPU
void run_on_device(
    int device_id,
    const char* initial_value,
    unsigned long long seed_offset,
    Counter128 *global_counter,
    ThreadSafeQueue<CompressionTask> &compression_queue,
    std::atomic<unsigned long long> &global_dp_counter, // Reference to global DP counter
    std::mutex &print_mutex, // Mutex to protect print statements
    std::atomic<bool> &printed_10k_dp // Reference flag to indicate if 10k DP was printed
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
    unsigned char tame_high;
    unsigned long long tame_mid, tame_low;
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

    // Prepare filename
    char filename[256];
    snprintf(filename, sizeof(filename), "dp_device_%d.bin", device_id);

    // Initialize compression task buffer
    std::vector<Point> buffer;
    buffer.reserve(BUFFER_SIZE);

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

            // Buffer the points
            for (unsigned int i = 0; i < dp_count_host; ++i) {
                // Increment the global DP counter atomically
                unsigned long long current_count = global_dp_counter.fetch_add(1) + 1;

                // Check if this is the 10,000th DP
                if (current_count == 10000 && !printed_10k_dp.load()) {
                    // Serialize the Point into a binary buffer
                    unsigned char binary_dp[17]; // 135 bits = 17 bytes (136 bits)
                    memset(binary_dp, 0, sizeof(binary_dp));

                    // Copy 'high', 'mid', 'low' into binary_dp
                    binary_dp[0] = h_dp_points[i].high; // 1 byte
                    memcpy(&binary_dp[1], &h_dp_points[i].mid, sizeof(unsigned long long)); // 8 bytes
                    memcpy(&binary_dp[9], &h_dp_points[i].low, sizeof(unsigned long long)); // 8 bytes

                    // Convert binary_dp to hexadecimal string
                    std::string hex_dp = "";
                    for (int j = 0; j < 17; ++j) {
                        char buf[3];
                        snprintf(buf, sizeof(buf), "%02x", binary_dp[j]);
                        hex_dp += buf;
                    }

                    // Lock the mutex before printing to prevent race conditions
                    {
                        std::lock_guard<std::mutex> lock(print_mutex);
                        if (!printed_10k_dp.load()) { // Double-check within mutex
                            printf("10,000th DP Found:\n");
                            printf("Hexadecimal Format: %s\n", hex_dp.c_str());
                            printed_10k_dp.store(true);
                        }
                    }
                }

                // Add the DP to the buffer
                buffer.push_back(h_dp_points[i]);

                // If buffer is full, enqueue it for compression
                if (buffer.size() >= BUFFER_SIZE) {
                    // Create a compression task
                    CompressionTask task;
                    task.buffer = std::move(buffer);
                    task.filename = filename;
                    compression_queue.enqueue(std::move(task));
                    buffer.reserve(BUFFER_SIZE); // Reserve again for the next buffer
                }
            }
        }

        // Optional: Implement a termination condition here if needed
        // For example, based on global_counter or a user interrupt
        // For demonstration, we'll run indefinitely. You can add a condition to break the loop.
    }

    // Enqueue any remaining buffered points as a final task
    if (!buffer.empty()) {
        CompressionTask task;
        task.buffer = std::move(buffer);
        task.filename = filename;
        compression_queue.enqueue(std::move(task));
    }

    // Free allocated memory
    cudaFree(d_state);
    cudaFree(d_dp_points);
    cudaFree(d_dp_count);
    cudaFreeHost(h_dp_points);
}

// Compression worker function
void compression_worker(ThreadSafeQueue<CompressionTask> &compression_queue, std::atomic<bool> &done)
{
    // Initialize compression buffers
    std::vector<char> compressed_buffer(COMPRESSION_BUFFER_SIZE);

    while (!done || !compression_queue.try_dequeue(*(new CompressionTask()))) {
        CompressionTask task;
        if (compression_queue.dequeue(task)) {
            // Compress the buffer
            int compressed_size = LZ4_compress_default(
                reinterpret_cast<const char*>(task.buffer.data()),
                compressed_buffer.data(),
                task.buffer.size() * sizeof(Point),
                COMPRESSION_BUFFER_SIZE
            );

            if (compressed_size <= 0) {
                fprintf(stderr, "Compression failed.\n");
                continue;
            }

            // Open the file in append mode
            std::ofstream dp_file(task.filename, std::ios::binary | std::ios::app);
            if (!dp_file.is_open()) {
                fprintf(stderr, "Failed to open file %s for writing.\n", task.filename.c_str());
                continue;
            }

            // Write the size of the compressed data
            dp_file.write(reinterpret_cast<char*>(&compressed_size), sizeof(int));
            if (!dp_file) {
                fprintf(stderr, "Failed to write compressed size to file %s.\n", task.filename.c_str());
                dp_file.close();
                continue;
            }

            // Write the compressed data
            dp_file.write(compressed_buffer.data(), compressed_size);
            if (!dp_file) {
                fprintf(stderr, "Failed to write compressed data to file %s.\n", task.filename.c_str());
            }

            dp_file.close();
        }
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

    // Create a thread-safe queue for compression tasks
    ThreadSafeQueue<CompressionTask> compression_queue;

    // Create global counters for DPs
    std::atomic<unsigned long long> global_dp_counter(0);
    std::atomic<bool> printed_10k_dp(false);
    std::mutex print_mutex;

    // Create compression worker threads (one per CPU core or as needed)
    unsigned int compression_threads_count = std::thread::hardware_concurrency();
    if (compression_threads_count == 0) compression_threads_count = 4; // Fallback to 4 threads
    std::vector<std::thread> compression_workers;
    std::atomic<bool> compression_done(false);
    for (unsigned int i = 0; i < compression_threads_count; ++i) {
        compression_workers.emplace_back(compression_worker, std::ref(compression_queue), std::ref(compression_done));
    }

    // Create threads for each GPU
    std::vector<std::thread> device_threads;
    for (int device_id = 0; device_id < device_count; ++device_id)
    {
        // Each GPU gets a unique seed offset to ensure different random sequences
        unsigned long long seed_offset = device_id * 1000;
        device_threads.emplace_back(run_on_device, device_id, INITIAL_VALUE, seed_offset, global_counter, std::ref(compression_queue), std::ref(global_dp_counter), std::ref(print_mutex), std::ref(printed_10k_dp));
    }

    // Wait for all device threads to finish (they run indefinitely)
    for (auto &t : device_threads)
    {
        t.join();
    }

    // Signal compression workers to finish
    compression_done = true;
    // Notify all waiting compression workers
    for (unsigned int i = 0; i < compression_threads_count; ++i) {
        CompressionTask dummy_task;
        compression_queue.enqueue(std::move(dummy_task));
    }

    // Wait for all compression workers to finish
    for (auto &t : compression_workers)
    {
        t.join();
    }

    // Wait for the monitor thread to finish
    monitor_thread.join();

    // Collect all filenames
    std::vector<std::string> filenames;
    for (int device_id = 0; device_id < device_count; ++device_id) {
        char filename[256];
        snprintf(filename, sizeof(filename), "dp_device_%d.bin", device_id);
        filenames.emplace_back(filename);
    }

    // Detect collisions
    detect_collisions(filenames);

    // Free the global counter memory
    cudaFree(global_counter);

    // Optionally, you can perform additional processing on the saved files here

    return 0;
}
