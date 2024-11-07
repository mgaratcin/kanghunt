#include <stdio.h>
#include <curand_kernel.h>
#include <string.h>
#include <math.h> // For pow
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
#include <cstdlib> // For strtoull
#include <sys/stat.h>
#include <errno.h>
#include <algorithm> // For std::reverse
#include <cstdint>   // For fixed-size types

// Constants and Macros
#define BINARY_LENGTH 135
#define INITIAL_VALUE "4000000000000000000000000000000000" // Hex string representing 135-bit value
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GPU 256
#define TARGET_TAIL_BITS 25 // Number of trailing zero bits required
#define BATCH_SIZE 1000000ULL
#define SEED 1234 // Base seed for random number generator
#define MAX_DISTINGUISHED_POINTS_PER_KERNEL 1000000 // Maximum DPs per kernel
#define BUFFER_SIZE 100000 // Number of Points to buffer before writing to disk
#define NUM_STEP_FUNCTIONS 64 // Increased for finer granularity

// Structure Definitions

struct Steps128 {
    unsigned long long low;
    unsigned long long high;
} __attribute__((packed));

#pragma pack(push, 1)
struct Point {
    unsigned char high;        // 1 byte (bits 134–128)
    unsigned long long mid;    // 8 bytes (bits 127–64)
    unsigned long long low;    // 8 bytes (bits 63–0)
    Steps128 steps;            // 16 bytes
    unsigned char is_tame;     // 1 byte
} __attribute__((packed));
#pragma pack(pop)

// Static assertion to ensure Point struct size is consistent
static_assert(sizeof(Point) == 34, "Point struct size must be 34 bytes");

// Function to initialize 135-bit value from a hexadecimal string
__host__ void initialize_135_bit_value_from_hex(const char *hex_str, unsigned char &high, unsigned long long &mid, unsigned long long &low)
{
    // hex_str should be 34 hex digits (for 135 bits)
    if (strlen(hex_str) < 34)
    {
        fprintf(stderr, "Hex string too short: %s\n", hex_str);
        exit(1);
    }

    // Parse the 'high' part (first 2 hex digits, but only 7 bits are used)
    char high_str[3] = { hex_str[0], hex_str[1], '\0' };
    high = (unsigned char)(strtoul(high_str, NULL, 16) & 0x7F); // Only 7 bits

    // Parse the 'mid' part (next 16 hex digits)
    char mid_str[17];
    strncpy(mid_str, &hex_str[2], 16);
    mid_str[16] = '\0';
    mid = strtoull(mid_str, NULL, 16);

    // Parse the 'low' part (last 16 hex digits)
    char low_str[17];
    strncpy(low_str, &hex_str[18], 16);
    low_str[16] = '\0';
    low = strtoull(low_str, NULL, 16);
    
    // Since the hex string represents the number in big-endian order,
    // convert 'mid' and 'low' to little-endian format to match storage.
    // Swap bytes of 'mid' and 'low' to convert to little-endian.
    unsigned long long mid_le = 0;
    unsigned long long low_le = 0;
    for (int i = 0; i < 8; i++) {
        mid_le |= ((mid >> (i * 8)) & 0xFF) << ((7 - i) * 8);
        low_le |= ((low >> (i * 8)) & 0xFF) << ((7 - i) * 8);
    }
    mid = mid_le;
    low = low_le;
}

// Corrected FNV-1a hash function for points (little-endian order)
__host__ __device__ __forceinline__ unsigned long long fnv1a_hash_point(const unsigned char high, const unsigned long long mid, const unsigned long long low)
{
    const unsigned long long FNV_prime = 1099511628211ULL;
    unsigned long long hash = 14695981039346656037ULL; // FNV offset basis

    // Hash the 'high' byte
    hash ^= high;
    hash *= FNV_prime;

    // Hash the 'mid' 8 bytes in little-endian order
    for (int i = 0; i < 8; i++)
    {
        hash ^= (mid >> (i * 8)) & 0xFF;
        hash *= FNV_prime;
    }

    // Hash the 'low' 8 bytes in little-endian order
    for (int i = 0; i < 8; i++)
    {
        hash ^= (low >> (i * 8)) & 0xFF;
        hash *= FNV_prime;
    }

    return hash;
}

// Kernel to initialize curand states
__global__ void init_curand_states(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize each state with a unique seed based on thread index and device seed
    curand_init(seed + idx, 0, 0, &state[idx]);
}

// Structure for 128-bit counter
struct Counter128 {
    unsigned long long low;
    unsigned long long high;
} __attribute__((packed));

// Device function to atomically add to a 128-bit counter
__device__ void atomicAdd128(Counter128 *counter, Steps128 value)
{
    unsigned long long old_low = atomicAdd(&(counter->low), value.low);
    unsigned long long carry = (old_low + value.low < old_low) ? 1ULL : 0ULL;
    atomicAdd(&(counter->high), value.high + carry);
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
        // Improved hash combining using prime multipliers
        std::size_t res = 17;
        res = res * 31 + std::hash<unsigned char>()(p.high);
        res = res * 31 + std::hash<unsigned long long>()(p.mid);
        res = res * 31 + std::hash<unsigned long long>()(p.low);
        return res;
    }
};

// Custom equality function for Point
struct PointEqual {
    bool operator()(const Point& a, const Point& b) const {
        return (a.high == b.high) && (a.mid == b.mid) && (a.low == b.low);
    }
};

// Thread-safe queue for collision detection
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
            if (queue_.empty()) {
                return false;
            }
        }
        if (!queue_.empty()) {
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;
    }
};

// Function Prototypes
void step_monitor(Counter128 *global_counter, std::vector<StepThreshold> &thresholds, std::atomic<bool> &stop_flag, std::mutex &dp_mutex, Point &last_tame_dp, Point &last_wild_dp);
void collision_detection_thread(
    ThreadSafeQueue<Point> &collision_queue,
    std::unordered_set<Point, PointHash, PointEqual> &tame_points,
    std::mutex &collision_mutex,
    std::atomic<bool> &collision_found
);
void run_on_device(
    int device_id,
    Counter128 *global_counter,
    std::atomic<unsigned long long> &global_dp_counter,
    ThreadSafeQueue<Point> &collision_queue,
    std::atomic<bool> &collision_found,
    std::mutex &dp_mutex,
    Point &last_wild_dp
);
void compute_tame_path(
    const char* initial_value,
    std::unordered_set<Point, PointHash, PointEqual> &tame_points,
    std::atomic<bool> &collision_found,
    std::mutex &dp_mutex,
    Point &last_tame_dp
);

// Function to correctly add step size to tame value with overflow handling
__host__ __device__ void add_step_to_tame_value(
    unsigned char &tame_high,        // 7 bits
    unsigned long long &mid,         // 64 bits
    unsigned long long &low,         // 64 bits
    unsigned long long step_size     // 64 bits
)
{
    // Perform 128-bit addition: (mid:low) + step_size
    unsigned long long new_low = low + step_size;
    unsigned long long carry = (new_low < low) ? 1ULL : 0ULL;
    low = new_low;

    unsigned long long new_mid = mid + carry;
    carry = (new_mid < mid) ? 1ULL : 0ULL;
    mid = new_mid;

    // Update high with any carry from mid
    if (carry)
    {
        tame_high = (tame_high + 1) & 0x7F; // Ensure only 7 bits are used
    }
}

// Function to add a value to Steps128
__host__ __device__ void add_to_steps(Steps128 &steps, unsigned long long value)
{
    // Removed atomicAdd as Steps128 is a local variable in device code
    unsigned long long old_low = steps.low;
    steps.low += value;
    unsigned long long carry = (steps.low < old_low) ? 1ULL : 0ULL;
    steps.high += carry;
}

// Device function to compute step size based on the current point
__device__ unsigned long long compute_step_size(const unsigned char high, const unsigned long long mid, const unsigned long long low, const unsigned long long* step_sizes)
{
    unsigned long long hash = fnv1a_hash_point(high, mid, low);
    unsigned int index = hash % NUM_STEP_FUNCTIONS;
    return step_sizes[index];
}

// Host function to compute step size based on the current point
unsigned long long compute_step_size_host(const unsigned char high, const unsigned long long mid, const unsigned long long low, const std::vector<unsigned long long>& step_sizes)
{
    unsigned long long hash = fnv1a_hash_point(high, mid, low);
    unsigned int index = hash % NUM_STEP_FUNCTIONS;
    return step_sizes[index];
}

// Modified is_distinguished_point function
__host__ __device__ __forceinline__ bool is_distinguished_point(const unsigned char high, const unsigned long long mid, const unsigned long long low)
{
    // Check if the lower TARGET_TAIL_BITS bits of 'low' are zero
    return (low & ((1ULL << TARGET_TAIL_BITS) - 1)) == 0;
}

// Step monitor function
void step_monitor(Counter128 *global_counter, std::vector<StepThreshold> &thresholds, std::atomic<bool> &stop_flag, std::mutex &dp_mutex, Point &last_tame_dp, Point &last_wild_dp)
{
    bool all_printed = false;
    while (!all_printed && !stop_flag.load())
    {
        // Read the current total steps atomically
        unsigned long long high = global_counter->high;
        unsigned long long low = global_counter->low;

        // Iterate through thresholds
        for (auto &threshold : thresholds)
        {
            if (!threshold.printed)
            {
                if ((high > threshold.high) ||
                    (high == threshold.high && low >= threshold.low))
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

        // Sleep for one minute
        sleep(60);
    }
}

// Collision detection thread function
void collision_detection_thread(
    ThreadSafeQueue<Point> &collision_queue,
    std::unordered_set<Point, PointHash, PointEqual> &tame_points,
    std::mutex &collision_mutex,
    std::atomic<bool> &collision_found
)
{
    Point point;

    while (!collision_found.load()) {
        if (collision_queue.dequeue(point)) {
            std::lock_guard<std::mutex> lock(collision_mutex);

            // Check if the wild point matches any tame point
            if (tame_points.find(point) != tame_points.end()) {
                // Collision found
                collision_found.store(true);
                break;
            }
        }
    }
}

// Kernel to generate wild paths and collect distinguished points
__global__ void generate_wild_paths(
    curandState *state,
    Counter128 *global_counter,
    Point *dp_points,
    unsigned int *dp_count,
    unsigned long long batch_size,
    unsigned int max_dp,
    unsigned long long* d_step_sizes
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];

    // Initialize local wild variables
    unsigned char local_wild_high;
    unsigned long long local_wild_mid, local_wild_low;

    // Initialize wild point with high nibble '4' and random low nibble (0-F)
    local_wild_high = 0x40 | (curand(&localState) & 0x0F);

    // Generate random 'mid' and 'low' values
    local_wild_mid = ((unsigned long long)curand(&localState) << 32) | curand(&localState);
    local_wild_low = ((unsigned long long)curand(&localState) << 32) | curand(&localState);

    Steps128 steps_wild = {0ULL, 0ULL};

    for (unsigned long long batch = 0; batch < batch_size; ++batch)
    {
        // Compute step size based on the current point
        unsigned long long step_size = compute_step_size(local_wild_high, local_wild_mid, local_wild_low, d_step_sizes);

        // Update the wild point
        unsigned long long new_low = local_wild_low + step_size;
        unsigned long long carry = (new_low < local_wild_low) ? 1ULL : 0ULL;
        local_wild_low = new_low;

        unsigned long long new_mid = local_wild_mid + carry;
        carry = (new_mid < local_wild_mid) ? 1ULL : 0ULL;
        local_wild_mid = new_mid;

        if (carry)
        {
            local_wild_high = (local_wild_high + 1) & 0x7F; // Ensure only 7 bits are used
        }

        add_to_steps(steps_wild, step_size);

        // Check if wild point is distinguished
        if (is_distinguished_point(local_wild_high, local_wild_mid, local_wild_low))
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

        if (*dp_count >= max_dp) {
            break;
        }
    }

    // Update the global 128-bit counter
    atomicAdd128(global_counter, steps_wild);

    // Save the updated state
    state[idx] = localState;
}

// Function to get file size
unsigned long long get_file_size(const char* filename) {
    struct stat st;
    if (stat(filename, &st) != 0) {
        // File does not exist yet
        return 0;
    }
    return static_cast<unsigned long long>(st.st_size);
}

// Enhanced function to check if a Point has 25 trailing zero bits in the 'low' part
bool has_trailing_zeros(const Point &p) {
    bool compliant = (p.low & ((1ULL << TARGET_TAIL_BITS) - 1)) == 0;
    if (!compliant) {
        // Log non-compliant DP data to stderr for debugging
        fprintf(stderr, "Non-compliant DP detected: high=%02x mid=%016llx low=%016llx\n", p.high, p.mid, p.low);
    }
    return compliant;
}

// Function to run wild paths on each GPU
void run_on_device(
    int device_id,
    Counter128 *global_counter,
    std::atomic<unsigned long long> &global_dp_counter,
    ThreadSafeQueue<Point> &collision_queue,
    std::atomic<bool> &collision_found,
    std::mutex &dp_mutex,
    Point &last_wild_dp
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

    // Initialize curand states with unique seed per GPU and thread
    unsigned long long base_seed = SEED + (device_id * 1000000);

    // Initialize curand states
    init_curand_states<<<BLOCKS_PER_GPU, THREADS_PER_BLOCK>>>(d_state, base_seed);
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

    // Prepare step sizes array
    unsigned long long h_step_sizes[NUM_STEP_FUNCTIONS];
    unsigned int jumpBit = 60; // Reduced jumpBit for smaller step sizes

    // Calculate minAvg and maxAvg
    double minAvg = pow(2.0, (double)jumpBit - 1.05);
    double maxAvg = pow(2.0, (double)jumpBit - 0.95);

    // Generate step sizes between minAvg and maxAvg
    for (int i = 0; i < NUM_STEP_FUNCTIONS; ++i) {
        double fraction = (double)i / (NUM_STEP_FUNCTIONS - 1);
        double step_size = minAvg + fraction * (maxAvg - minAvg);
        unsigned long long step = (unsigned long long)step_size;

        h_step_sizes[i] = step;
    }

    // Copy step sizes to device
    unsigned long long* d_step_sizes;
    err = cudaMalloc(&d_step_sizes, NUM_STEP_FUNCTIONS * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to allocate device step sizes: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
        cudaFreeHost(h_dp_points);
        return;
    }
    err = cudaMemcpy(d_step_sizes, h_step_sizes, NUM_STEP_FUNCTIONS * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d: Failed to copy step sizes to device: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(d_state);
        cudaFree(d_dp_points);
        cudaFree(d_dp_count);
        cudaFreeHost(h_dp_points);
        cudaFree(d_step_sizes);
        return;
    }

    // Launch parameters
    unsigned long long iterations_per_launch = BATCH_SIZE;
    dim3 grid(BLOCKS_PER_GPU);
    dim3 block(THREADS_PER_BLOCK);

    // Define maximum file size (e.g., 2GB)
    const unsigned long long MAX_FILE_SIZE = 2ULL * 1024ULL * 1024ULL * 1024ULL; // 2GB

    // Variables to keep track of file size and part number
    unsigned long long current_file_size = 0;
    int file_part_number = 0;

    // Prepare initial filename for wild DPs
    char wild_filename[256];
    snprintf(wild_filename, sizeof(wild_filename), "wild_device_%d_part_%d.bin", device_id, file_part_number);

    // Get the current file size
    current_file_size = get_file_size(wild_filename);

    // Initialize buffer to store wild DPs before writing
    std::vector<Point> wild_buffer;
    wild_buffer.reserve(BUFFER_SIZE);

    // Main loop controlled by the host
    while (!collision_found.load())
    {
        // Reset dp_count on device
        err = cudaMemset(d_dp_count, 0, sizeof(unsigned int));
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU %d: Failed to reset dp_count: %s\n", device_id, cudaGetErrorString(err));
            break;
        }

        // Launch the generate_wild_paths kernel with max_dp parameter
        generate_wild_paths<<<grid, block>>>(
            d_state,
            global_counter,
            d_dp_points,
            d_dp_count,
            iterations_per_launch,
            MAX_DISTINGUISHED_POINTS_PER_KERNEL,
            d_step_sizes
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU %d: Failed to launch generate_wild_paths kernel: %s\n", device_id, cudaGetErrorString(err));
            break;
        }

        // Synchronize to wait for kernel completion
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU %d: CUDA Device Synchronize failed after generate_wild_paths: %s\n", device_id, cudaGetErrorString(err));
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
            // Copy DPs from device to host pinned memory
            err = cudaMemcpy(h_dp_points, d_dp_points, dp_count_host * sizeof(Point), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "GPU %d: Failed to copy distinguished points from device to host: %s\n", device_id, cudaGetErrorString(err));
                break;
            }

            // Buffer the wild DPs
            for (unsigned int i = 0; i < dp_count_host; ++i) {
                // Increment the global DP counter atomically
                unsigned long long current_count = global_dp_counter.fetch_add(1) + 1;

                // Add the DP to the wild buffer
                wild_buffer.push_back(h_dp_points[i]);

                // Update the last wild DP
                {
                    std::lock_guard<std::mutex> lock(dp_mutex);
                    last_wild_dp = h_dp_points[i];
                }

                // Enqueue the wild DP for collision detection
                collision_queue.enqueue(h_dp_points[i]);

                // If wild buffer is full, write it to disk as binary
                if (wild_buffer.size() >= BUFFER_SIZE) {
                    // Check if writing the buffer would exceed the max file size
                    unsigned long long data_size = wild_buffer.size() * sizeof(Point);
                    if (current_file_size + data_size > MAX_FILE_SIZE) {
                        // Start a new wild file
                        file_part_number++;
                        snprintf(wild_filename, sizeof(wild_filename), "wild_device_%d_part_%d.bin", device_id, file_part_number);
                        current_file_size = 0;
                    }

                    // Open the wild file in append mode
                    std::ofstream wild_file(wild_filename, std::ios::binary | std::ios::app);
                    if (!wild_file.is_open()) {
                        fprintf(stderr, "GPU %d: Failed to open wild file: %s\n", device_id, wild_filename);
                        wild_buffer.clear();
                        wild_buffer.reserve(BUFFER_SIZE);
                        continue;
                    }

                    // Write the buffered wild points directly as binary
                    for (const auto& p : wild_buffer) {
                        if (has_trailing_zeros(p)) {
                            wild_file.write(reinterpret_cast<const char*>(&p), sizeof(Point));
                        } else {
                            // Optionally, handle non-compliant Points if needed
                        }
                    }
                    if (wild_file) {
                        current_file_size += data_size;
                    } else {
                        fprintf(stderr, "GPU %d: Failed to write to wild file: %s\n", device_id, wild_filename);
                    }

                    wild_file.close();
                    wild_buffer.clear();
                    wild_buffer.reserve(BUFFER_SIZE);
                }

                // Check if a collision has been found to terminate early
                if (collision_found.load()) {
                    break;
                }
            }

            // If a collision has been found, terminate the loop
            if (collision_found.load()) {
                break;
            }
        }
    }

    // Write any remaining wild DPs in the buffer to disk
    if (!wild_buffer.empty()) {
        // Check if writing the buffer would exceed the max file size
        unsigned long long data_size = wild_buffer.size() * sizeof(Point);
        if (current_file_size + data_size > MAX_FILE_SIZE) {
            // Start a new wild file
            file_part_number++;
            snprintf(wild_filename, sizeof(wild_filename), "wild_device_%d_part_%d.bin", device_id, file_part_number);
            current_file_size = 0;
        }

        // Open the wild file in append mode
        std::ofstream wild_file(wild_filename, std::ios::binary | std::ios::app);
        if (wild_file.is_open()) {
            for (const auto& p : wild_buffer) {
                if (has_trailing_zeros(p)) {
                    wild_file.write(reinterpret_cast<const char*>(&p), sizeof(Point));
                }
            }
            if (wild_file) {
                current_file_size += data_size;
            } else {
                fprintf(stderr, "GPU %d: Failed to write remaining wild points to file: %s\n", device_id, wild_filename);
            }
            wild_file.close();
        } else {
            fprintf(stderr, "GPU %d: Failed to open wild file for remaining points: %s\n", device_id, wild_filename);
        }
        wild_buffer.clear();
    }

    // Free allocated memory
    cudaFree(d_state);
    cudaFree(d_dp_points);
    cudaFree(d_dp_count);
    cudaFreeHost(h_dp_points);
    cudaFree(d_step_sizes);
}

void write_tame_dp_to_file(const Point &p, int device_id, int &file_part_number, unsigned long long &current_file_size, std::vector<Point> &tame_buffer)
{
    // Define maximum file size (e.g., 2GB)
    const unsigned long long MAX_FILE_SIZE = 2ULL * 1024ULL * 1024ULL * 1024ULL; // 2GB

    // Prepare filename
    char tame_filename[256];
    snprintf(tame_filename, sizeof(tame_filename), "tame_device_%d_part_%d.bin", device_id, file_part_number);

    // Check if writing the buffer would exceed the max file size
    unsigned long long data_size = sizeof(Point);
    if (current_file_size + data_size > MAX_FILE_SIZE) {
        // Start a new tame file
        file_part_number++;
        snprintf(tame_filename, sizeof(tame_filename), "tame_device_%d_part_%d.bin", device_id, file_part_number);
        current_file_size = 0;
    }

    // Open the tame file in append mode
    std::ofstream tame_file(tame_filename, std::ios::binary | std::ios::app);
    if (!tame_file.is_open()) {
        fprintf(stderr, "Failed to open tame file: %s\n", tame_filename);
        return;
    }

    // Write the tame point directly as binary
    if (has_trailing_zeros(p)) {
        tame_file.write(reinterpret_cast<const char*>(&p), sizeof(Point));
    } else {
        // Optionally, handle non-compliant Points if needed
    }
    if (tame_file) {
        current_file_size += data_size;
    } else {
        fprintf(stderr, "Failed to write to tame file: %s\n", tame_filename);
    }

    tame_file.close();
}

// Function to compute the tame path on the host
void compute_tame_path(
    const char* initial_value,
    std::unordered_set<Point, PointHash, PointEqual> &tame_points,
    std::atomic<bool> &collision_found,
    std::mutex &dp_mutex,
    Point &last_tame_dp
)
{
    // Initialize the 135-bit tame value
    unsigned char tame_high;
    unsigned long long mid, low;
    initialize_135_bit_value_from_hex(initial_value, tame_high, mid, low);

    Steps128 steps_tame = {0ULL, 0ULL};

    // Prepare step sizes array
    std::vector<unsigned long long> step_sizes(NUM_STEP_FUNCTIONS);
    unsigned int jumpBit = 30;

    // Calculate minAvg and maxAvg
    double minAvg = pow(2.0, (double)jumpBit - 1.05);
    double maxAvg = pow(2.0, (double)jumpBit - 0.95);

    // Generate step sizes between minAvg and maxAvg
    for (int i = 0; i < NUM_STEP_FUNCTIONS; ++i) {
        double fraction = (double)i / (NUM_STEP_FUNCTIONS - 1);
        double step_size = minAvg + fraction * (maxAvg - minAvg);
        unsigned long long step = (unsigned long long)step_size;

        step_sizes[i] = step;
    }

    bool is_first_point = true; // Flag to skip initial DP detection if undesired

    // Variables for writing tame DPs
    int device_id = 0; // Assuming device_id=0 for tame DPs
    int file_part_number = 0;
    unsigned long long tame_current_file_size = 0;
    std::vector<Point> tame_buffer;
    tame_buffer.reserve(BUFFER_SIZE);

    // Iterate until collision is found
    while (!collision_found.load())
    {
        // Compute step size based on the current point
        unsigned long long step_size = compute_step_size_host(tame_high, mid, low, step_sizes);

        // Update the tame point by adding step_size
        add_step_to_tame_value(tame_high, mid, low, step_size);
        add_to_steps(steps_tame, step_size);

        // Update the last tame DP
        {
            std::lock_guard<std::mutex> lock(dp_mutex);
            last_tame_dp.high = tame_high;
            last_tame_dp.mid = mid;
            last_tame_dp.low = low;
            last_tame_dp.steps = steps_tame;
            last_tame_dp.is_tame = 1;
        }

        // Check if tame point is distinguished
        if (is_distinguished_point(tame_high, mid, low))
        {
            if (is_first_point) {
                is_first_point = false;
                // Optionally, skip the first point if undesired
            }

            Point p;
            p.high = tame_high;
            p.mid = mid;
            p.low = low;
            p.steps = steps_tame;
            p.is_tame = 1;

            // Add to tame_points set
            {
                std::lock_guard<std::mutex> lock(dp_mutex);
                tame_points.insert(p);
                last_tame_dp = p;
            }

            // Write the tame DP to binary file with secondary condition check
            write_tame_dp_to_file(p, device_id, file_part_number, tame_current_file_size, tame_buffer);
        }
    }
}

// Main Function
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

    // Define the step thresholds dynamically from 2^60 to 2^127
    std::vector<StepThreshold> thresholds;

    // Starting exponent
    unsigned int start_exponent = 60;
    // Maximum exponent (up to 127 for 128-bit counter)
    unsigned int max_exponent = 127;

    for (unsigned int exponent = start_exponent; exponent <= max_exponent; ++exponent)
    {
        StepThreshold threshold;
        threshold.exponent = exponent;
        threshold.printed = false;

        if (exponent < 64)
        {
            threshold.high = 0;
            threshold.low = 1ULL << exponent;
        }
        else
        {
            // For exponents >= 64, set the high part accordingly
            // Handle exponents up to 127 to prevent overflow
            if (exponent - 64 < 64)
            {
                threshold.high = 1ULL << (exponent - 64);
                threshold.low = 0;
            }
            else
            {
                threshold.high = 0;
                threshold.low = 0;
                fprintf(stderr, "Exponent %u is too large to represent in 128 bits.\n", exponent);
                continue;
            }
        }

        thresholds.push_back(threshold);
    }

    // Create a thread for step monitoring
    std::atomic<bool> stop_monitor(false);

    // Initialize last tame and wild DPs
    Point last_tame_dp = {};
    Point last_wild_dp = {};

    // Mutex to protect last DP variables
    std::mutex dp_mutex;

    std::thread monitor_thread(step_monitor, global_counter, std::ref(thresholds), std::ref(stop_monitor), std::ref(dp_mutex), std::ref(last_tame_dp), std::ref(last_wild_dp));

    // Create a thread-safe queue for collision detection
    ThreadSafeQueue<Point> collision_queue;

    // Create global counters for DPs
    std::atomic<unsigned long long> global_dp_counter(0);

    // Create a mutex for collision detection
    std::mutex collision_mutex;

    // Create an unordered_set for tame points
    std::unordered_set<Point, PointHash, PointEqual> tame_points;

    // Create an atomic flag to indicate if a collision was found
    std::atomic<bool> collision_found(false);

    // Create threads for each GPU to run wild paths
    std::vector<std::thread> device_threads;
    for (int device_id = 0; device_id < device_count; ++device_id)
    {
        device_threads.emplace_back(run_on_device, device_id, global_counter, std::ref(global_dp_counter), std::ref(collision_queue), std::ref(collision_found), std::ref(dp_mutex), std::ref(last_wild_dp));
    }

    // Start the tame path computation on a separate thread
    std::thread tame_thread(compute_tame_path, INITIAL_VALUE, std::ref(tame_points), std::ref(collision_found), std::ref(dp_mutex), std::ref(last_tame_dp));

    // Create a collision detection thread
    std::thread collision_thread(collision_detection_thread, std::ref(collision_queue), std::ref(tame_points), std::ref(collision_mutex), std::ref(collision_found));

    // Wait for the collision detection thread to finish (i.e., a collision is found)
    collision_thread.join();

    // Signal device threads and tame thread to stop by setting collision_found to true
    collision_found.store(true);

    // Wait for all device threads to finish
    for (auto &t : device_threads)
    {
        if (t.joinable()) {
            t.join();
        }
    }

    // Wait for the tame thread to finish
    if (tame_thread.joinable()) {
        tame_thread.join();
    }

    // Stop the monitor thread
    stop_monitor.store(true);
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }

    // Free the global counter memory
    cudaFree(global_counter);

    printf("Program terminated after detecting a collision.\n");

    return 0;
}
