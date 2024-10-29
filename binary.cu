#include <stdio.h>
#include <curand_kernel.h>
#include <string.h>

#define BINARY_LENGTH 135
#define INITIAL_VALUE "0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
#define THREADS_PER_BLOCK 1024 // Increase thread count to maximize occupancy
#define BLOCKS 4096 // Increase block count to maximize GPU utilization and achieve faster execution
#define TARGET_TAIL_BITS 20
#define PRINT_INTERVAL 10000000ULL
#define MAX_STEPS 1000000000ULL
#define MAX_STORE_VALUES 1024

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
void initialize_135_bit_value(const char *binary_str, unsigned long long &high, unsigned long long &mid, unsigned long long &low)
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

__global__ void generate_paths(curandState *state, unsigned long long tame_high, unsigned long long tame_mid, unsigned long long tame_low,
                               unsigned long long *stored_tame_high, unsigned long long *stored_tame_mid, unsigned long long *stored_tame_low,
                               unsigned long long *stored_wild_high, unsigned long long *stored_wild_mid, unsigned long long *stored_wild_low,
                               int *tame_store_idx, int *wild_store_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];

    unsigned long long wild_high = tame_high;
    unsigned long long wild_mid = tame_mid;
    unsigned long long wild_low = tame_low;

    unsigned long long steps_tame = 0;
    unsigned long long steps_wild = 0;

    // Using a larger increment for tame_low to speed up tame path updates
    const unsigned long long tame_increment = 1024;

    while (true)
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
        unsigned long long random_increment = ((curand(&localState) & 0xFFFFFFFFFFFFULL) << 12) | (curand(&localState) & 0xFFFULL);
        wild_low += random_increment;
        if (wild_low < random_increment)
        {
            wild_mid++;
            if (wild_mid == 0) wild_high = (wild_high + 1) & 0x7FULL;
        }
        steps_wild++;

        // Check if either value ends with 20 zeros and store if needed
        if (ends_with_20_zeros(tame_low))
        {
            int store_idx = atomicAdd(tame_store_idx, 1);
            if (store_idx < MAX_STORE_VALUES)
            {
                stored_tame_high[store_idx] = tame_high;
                stored_tame_mid[store_idx] = tame_mid;
                stored_tame_low[store_idx] = tame_low;
            }
        }
        if (ends_with_20_zeros(wild_low))
        {
            int store_idx = atomicAdd(wild_store_idx, 1);
            if (store_idx < MAX_STORE_VALUES)
            {
                stored_wild_high[store_idx] = wild_high;
                stored_wild_mid[store_idx] = wild_mid;
                stored_wild_low[store_idx] = wild_low;
            }
        }

        // Check for collision
        if (tame_high == wild_high && tame_mid == wild_mid && tame_low == wild_low)
        {
            printf("Collision detected! Steps Tame: %llu, Steps Wild: %llu\n", steps_tame, steps_wild);
            break;
        }

        // Break after MAX_STEPS to prevent infinite loop
        if (steps_tame >= MAX_STEPS / tame_increment || steps_wild >= MAX_STEPS)
        {
            if (idx == 0) {
                printf("Max steps reached without collision.\n");
                printf("Final Tame Value: ");
                print_135_bit_value_device(tame_high, tame_mid, tame_low);
                printf("Final Wild Value: ");
                print_135_bit_value_device(wild_high, wild_mid, wild_low);
            }
            break;
        }
    }

    state[idx] = localState;
}

int main()
{
    curandState *d_state;
    cudaMalloc(&d_state, THREADS_PER_BLOCK * BLOCKS * sizeof(curandState));

    unsigned long long *stored_tame_high, *stored_tame_mid, *stored_tame_low;
    unsigned long long *stored_wild_high, *stored_wild_mid, *stored_wild_low;
    int *tame_store_idx, *wild_store_idx;
    cudaMalloc(&stored_tame_high, MAX_STORE_VALUES * sizeof(unsigned long long));
    cudaMalloc(&stored_tame_mid, MAX_STORE_VALUES * sizeof(unsigned long long));
    cudaMalloc(&stored_tame_low, MAX_STORE_VALUES * sizeof(unsigned long long));
    cudaMalloc(&stored_wild_high, MAX_STORE_VALUES * sizeof(unsigned long long));
    cudaMalloc(&stored_wild_mid, MAX_STORE_VALUES * sizeof(unsigned long long));
    cudaMalloc(&stored_wild_low, MAX_STORE_VALUES * sizeof(unsigned long long));
    cudaMalloc(&tame_store_idx, sizeof(int));
    cudaMalloc(&wild_store_idx, sizeof(int));
    cudaMemset(tame_store_idx, 0, sizeof(int));
    cudaMemset(wild_store_idx, 0, sizeof(int));

    unsigned long long tame_high, tame_mid, tame_low;
    initialize_135_bit_value(INITIAL_VALUE, tame_high, tame_mid, tame_low);

    generate_paths<<<BLOCKS, THREADS_PER_BLOCK>>>(d_state, tame_high, tame_mid, tame_low, stored_tame_high, stored_tame_mid, stored_tame_low,
                                                  stored_wild_high, stored_wild_mid, stored_wild_low, tame_store_idx, wild_store_idx);

    cudaDeviceSynchronize();

    cudaFree(d_state);
    cudaFree(stored_tame_high);
    cudaFree(stored_tame_mid);
    cudaFree(stored_tame_low);
    cudaFree(stored_wild_high);
    cudaFree(stored_wild_mid);
    cudaFree(stored_wild_low);
    cudaFree(tame_store_idx);
    cudaFree(wild_store_idx);
    return 0;
}
