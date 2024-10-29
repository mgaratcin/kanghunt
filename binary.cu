#include <stdio.h>
#include <curand_kernel.h>
#include <string.h>

#define BINARY_LENGTH 135
#define INITIAL_VALUE "0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
#define THREADS_PER_BLOCK 256
#define BLOCKS 128
#define TARGET_TAIL_BITS 20
#define PRINT_INTERVAL 10000000ULL
#define MAX_STEPS 1000000000ULL
#define MAX_STORE_VALUES 1024

// Device function to check if a value ends with 20 zeros
__device__ bool ends_with_20_zeros(unsigned long long value_low)
{
    return (value_low & ((1ULL << TARGET_TAIL_BITS) - 1)) == 0;
}

// Device function to print 135-bit value in binary format
__device__ void print_135_bit_value_device(unsigned long long high, unsigned long long mid, unsigned long long low)
{
    // Print high part (7 bits to ensure we are representing 135 bits in total)
    for (int j = 6; j >= 0; j--)
    {
        printf("%d", (int)((high >> j) & 1));
    }
    // Print mid part (64 bits)
    for (int j = 63; j >= 0; j--)
    {
        printf("%d", (int)((mid >> j) & 1));
    }
    // Print low part (64 bits)
    for (int j = 63; j >= 0; j--)
    {
        printf("%d", (int)((low >> j) & 1));
    }
    printf("\n");
}

// Host function to print 135-bit value in binary format
void print_135_bit_value_host(unsigned long long high, unsigned long long mid, unsigned long long low)
{
    // Print high part (7 bits to ensure we are representing 135 bits in total)
    for (int j = 6; j >= 0; j--)
    {
        printf("%d", (int)((high >> j) & 1));
    }
    // Print mid part (64 bits)
    for (int j = 63; j >= 0; j--)
    {
        printf("%d", (int)((mid >> j) & 1));
    }
    // Print low part (64 bits)
    for (int j = 63; j >= 0; j--)
    {
        printf("%d", (int)((low >> j) & 1));
    }
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
            if (i < 7)
            {
                high |= (1ULL << (6 - i));
            }
            else if (i < 71)
            {
                mid |= (1ULL << (70 - i));
            }
            else if (i < 135)
            {
                low |= (1ULL << (134 - i));
            }
        }
    }
}

__global__ void generate_paths(curandState *state, unsigned long long tame_high, unsigned long long tame_mid, unsigned long long tame_low,
                               unsigned long long *stored_tame_high, unsigned long long *stored_tame_mid, unsigned long long *stored_tame_low,
                               unsigned long long *stored_wild_high, unsigned long long *stored_wild_mid, unsigned long long *stored_wild_low,
                               int *tame_store_idx, int *wild_store_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the starting value (135 bits split into high, mid, and low parts)
    unsigned long long wild_high = tame_high;
    unsigned long long wild_mid = tame_mid;
    unsigned long long wild_low = tame_low;

    if (idx == 0) {
        printf("Initial Tame Value: ");
        print_135_bit_value_device(tame_high, tame_mid, tame_low);
        printf("Initial Wild Value: ");
        print_135_bit_value_device(wild_high, wild_mid, wild_low);
    }

    // Seed the random state for wild path
    curand_init(1234, idx, 0, &state[idx]);

    unsigned long long int steps_tame = 0;
    unsigned long long int steps_wild = 0;

    while (true)
    {
        // Increment tame path by 1
        tame_low++;
        if (tame_low == 0) // Handle overflow from low to mid
        {
            tame_mid++;
            if (tame_mid == 0) // Handle overflow from mid to high
            {
                tame_high++;
                tame_high &= 0x7FULL; // Limit to 7 bits for a total of 135 bits
            }
        }
        tame_high &= 0x7FULL; // Ensure high part is always within 7 bits
        steps_tame++;

        // Increment wild path by a pseudo-random value less than 135 bits
        unsigned long long random_increment = ((curand(&state[idx]) & 0xFFFFFFFFFFFFULL) << 12) | (curand(&state[idx]) & 0xFFFULL);
        wild_low += random_increment;
        if (wild_low < random_increment) // Handle overflow from low to mid
        {
            wild_mid++;
            if (wild_mid == 0) // Handle overflow from mid to high
            {
                wild_high++;
                wild_high &= 0x7FULL; // Limit to 7 bits for a total of 135 bits
            }
        }
        wild_high &= 0x7FULL; // Ensure high part is always within 7 bits
        steps_wild++;

        // Debug print statement for each thread to verify values
        if (steps_tame % (PRINT_INTERVAL / 10) == 0 && idx == 0) {
            printf("Debug - Thread %d: Steps Tame: %llu, Steps Wild: %llu\n", idx, steps_tame, steps_wild);
            printf("Current Tame Value: ");
            print_135_bit_value_device(tame_high, tame_mid, tame_low);
            printf("Current Wild Value: ");
            print_135_bit_value_device(wild_high, wild_mid, wild_low);
        }

        // Check if either value ends with 20 zeros and store if needed
        if (ends_with_20_zeros(tame_low))
        {
            int store_idx = atomicAdd(tame_store_idx, 1);
            if (store_idx < MAX_STORE_VALUES) {
                stored_tame_high[store_idx] = tame_high;
                stored_tame_mid[store_idx] = tame_mid;
                stored_tame_low[store_idx] = tame_low;
            }
        }
        if (ends_with_20_zeros(wild_low))
        {
            int store_idx = atomicAdd(wild_store_idx, 1);
            if (store_idx < MAX_STORE_VALUES) {
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
        if (steps_tame >= MAX_STEPS || steps_wild >= MAX_STEPS)
        {
            if (idx == 0) {
                printf("Max steps reached without collision. Steps Tame: %llu, Steps Wild: %llu\n", steps_tame, steps_wild);
            }
            break;
        }
    }
}

int main()
{
    // Allocate space for random states
    curandState *d_state;
    cudaMalloc(&d_state, THREADS_PER_BLOCK * BLOCKS * sizeof(curandState));

    // Allocate space for stored values
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

    // Initialize the starting value (135 bits split into high, mid, and low parts)
    unsigned long long tame_high, tame_mid, tame_low;
    initialize_135_bit_value(INITIAL_VALUE, tame_high, tame_mid, tame_low);

    // Launch kernel
    generate_paths<<<BLOCKS, THREADS_PER_BLOCK>>>(d_state, tame_high, tame_mid, tame_low, stored_tame_high, stored_tame_mid, stored_tame_low,
                                                  stored_wild_high, stored_wild_mid, stored_wild_low, tame_store_idx, wild_store_idx);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy stored values back to host
    unsigned long long h_stored_tame_high[MAX_STORE_VALUES], h_stored_tame_mid[MAX_STORE_VALUES], h_stored_tame_low[MAX_STORE_VALUES];
    unsigned long long h_stored_wild_high[MAX_STORE_VALUES], h_stored_wild_mid[MAX_STORE_VALUES], h_stored_wild_low[MAX_STORE_VALUES];
    int h_tame_store_idx, h_wild_store_idx;
    cudaMemcpy(h_stored_tame_high, stored_tame_high, MAX_STORE_VALUES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stored_tame_mid, stored_tame_mid, MAX_STORE_VALUES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stored_tame_low, stored_tame_low, MAX_STORE_VALUES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stored_wild_high, stored_wild_high, MAX_STORE_VALUES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stored_wild_mid, stored_wild_mid, MAX_STORE_VALUES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stored_wild_low, stored_wild_low, MAX_STORE_VALUES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tame_store_idx, tame_store_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_wild_store_idx, wild_store_idx, sizeof(int), cudaMemcpyDeviceToHost);

    // Print stored values
    printf("\nStored Tame Values:\n");
    for (int i = 0; i < h_tame_store_idx && i < MAX_STORE_VALUES; i++)
    {
        print_135_bit_value_host(h_stored_tame_high[i], h_stored_tame_mid[i], h_stored_tame_low[i]);
    }

    printf("\nStored Wild Values:\n");
    for (int i = 0; i < h_wild_store_idx && i < MAX_STORE_VALUES; i++)
    {
        print_135_bit_value_host(h_stored_wild_high[i], h_stored_wild_mid[i], h_stored_wild_low[i]);
    }

    // Free memory
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
