#ifndef U64_CUH
#define U64_CUH

#include <stdint.h>

// Define u64 as a 64-bit unsigned integer for compatibility
typedef uint64_t u64;

// Addition with carry
inline __device__ bool u64Add(u64* output, u64 a, u64 b, u64 carry) {
    u64 sum = a + b + carry;
    *output = sum;
    // Overflow occurs if sum < a or sum < b
    return (sum < a) || (sum < b);
}

// Subtraction with borrow
inline __device__ bool u64Sub(u64* output, u64 a, u64 b, u64 borrow) {
    u64 diff = a - b - borrow;
    *output = diff;
    // Underflow occurs if a < b or (a == b and borrow is 1)
    return (a < b) || ((a == b) && borrow);
}

// Multiplication with carry
inline __device__ u64 u64Mul(u64* output, u64 a, u64 b, u64 carry) {
    u64 low = a * b + carry;
    u64 high = __umul64hi(a, b);

    // If adding carry causes an overflow in the low part, increment high
    if (low < carry) {
        high++;
    }

    *output = low;
    return high;
}

#endif // U64_CUH
