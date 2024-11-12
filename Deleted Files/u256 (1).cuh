#ifndef U256_CUH
#define U256_CUH

#include "u64.cuh"

// Define shift amounts for 64-bit words
constexpr int SHIFT_AMOUNT_LEFT = 1;
constexpr int SHIFT_AMOUNT_RIGHT = 1;

// Define bitmask for setting and clearing bits
constexpr u64 BITMASK_1 = 1ULL;
constexpr u64 BITMASK_CLEAR = ~BITMASK_1;

// Function to copy a 256-bit value
inline __device__ void u256Copy(u64 output[4], const u64 a[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        output[i] = a[i];
    }
}

// Function to copy a 512-bit value
inline __device__ void u512Copy(u64 output[8], const u64 a[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        output[i] = a[i];
    }
}

// Function to set a 256-bit value to zero
inline __device__ void u256SetZero(u64 a[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] = 0;
    }
}

// Function to set a 512-bit value to zero
inline __device__ void u512SetZero(u64 a[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = 0;
    }
}

// Function to extend a 256-bit value to 512 bits
inline __device__ void u256Extend(u64 output[8], const u64 a[4]) {
    u64 result[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        result[i] = a[i];
    }
#pragma unroll
    for (int i = 4; i < 8; ++i) {
        result[i] = 0;
    }
    u512Copy(output, result);
}

// Function to get a specific bit from a 256-bit value
inline __device__ bool u256GetBit(const u64 a[4], const int index) {
    return (a[index / 64] >> (index % 64)) & 1ULL;
}

// Function to get a specific bit from a 512-bit value
inline __device__ bool u512GetBit(const u64 a[8], const int index) {
    return (a[index / 64] >> (index % 64)) & 1ULL;
}

// Function to set a specific bit in a 256-bit value
inline __device__ void u256SetBit(u64 output[4], const int index, const bool value) {
    if (value) {
        output[index / 64] |= BITMASK_1 << (index % 64);
    } else {
        output[index / 64] &= ~(BITMASK_1 << (index % 64));
    }
}

// Function to set a specific bit in a 512-bit value
inline __device__ void u512SetBit(u64 a[8], const int index, const bool value) {
    if (value) {
        a[index / 64] |= BITMASK_1 << (index % 64);
    } else {
        a[index / 64] &= ~(BITMASK_1 << (index % 64));
    }
}

// Function to perform a left shift by 1 on a 256-bit value
inline __device__ void u256LShift1(u64 a[4]) {
#pragma unroll
    for (int i = 2; i >= 0; --i) {
        a[i + 1] = (a[i + 1] << SHIFT_AMOUNT_LEFT) | (a[i] >> 63);
    }
    a[0] <<= SHIFT_AMOUNT_LEFT;
}

// Function to perform a left shift by 1 on a 512-bit value
inline __device__ void u512LShift1(u64 a[8]) {
#pragma unroll
    for (int i = 6; i >= 0; --i) {
        a[i + 1] = (a[i + 1] << SHIFT_AMOUNT_LEFT) | (a[i] >> 63);
    }
    a[0] <<= SHIFT_AMOUNT_LEFT;
}

// Function to perform a right shift by 1 on a 256-bit value
inline __device__ void u256RShift1(u64 a[4]) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        a[i] = (a[i] >> SHIFT_AMOUNT_RIGHT) | (a[i + 1] << 63);
    }
    a[3] >>= SHIFT_AMOUNT_RIGHT;
}

// Function to check if a 256-bit value is odd
inline __device__ bool u256IsOdd(const u64 a[4]) { 
    return a[0] & BITMASK_1; 
}

// Function to check if a 256-bit value is zero
inline __device__ bool u256IsZero(const u64 a[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (a[i] != 0) {
            return false;
        }
    }
    return true;
}

// Function to compare two 256-bit values
inline __device__ int u256Compare(const u64 a[4], const u64 b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] > b[i]) {
            return 1;
        } else if (a[i] < b[i]) {
            return -1;
        }
    }
    return 0;
}

// Function to compare two 512-bit values
inline __device__ int u512Compare(const u64 a[8], const u64 b[8]) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) {
            return 1;
        } else if (a[i] < b[i]) {
            return -1;
        }
    }
    return 0;
}

// Function to add two 256-bit values
inline __device__ bool u256Add(u64 output[4], const u64 a[4], const u64 b[4]) {
    u64 result[4] = {0};
    bool carry = false;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        carry = u64Add(&result[i], a[i], b[i], carry);
    }

    u256Copy(output, result);
    return carry;
}

// Function to subtract two 256-bit values
inline __device__ bool u256Sub(u64 output[4], const u64 a[4], const u64 b[4]) {
    u64 result[4] = {0};
    bool borrow = false;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        borrow = u64Sub(&result[i], a[i], b[i], borrow);
    }

    u256Copy(output, result);
    return borrow;
}

// Function to subtract two 512-bit values
inline __device__ bool u512Sub(u64 output[8], const u64 a[8], const u64 b[8]) {
    bool borrow = false;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        borrow = u64Sub(&output[i], a[i], b[i], borrow);
    }

    return borrow;
}

// Function to multiply a 256-bit value with a 64-bit value
inline __device__ u64 u256MulWithU64(u64 output[4], const u64 a[4], const u64 b, u64 carry) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        carry = u64Mul(&output[i], a[i], b, carry);
    }
    return carry;
}

// Function to multiply two 256-bit values resulting in a 512-bit product
inline __device__ void u256Mul(u64 output[8], const u64 a[4], const u64 b[4]) {
    u64 result[8] = {0};

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u64 t0[4] = {0};
        u64 carry = 0;

        carry = u256MulWithU64(t0, a, b[i], carry);

        u64 t1[4];
        u256Copy(t1, result + i);

        carry += u256Add(result + i, t1, t0);

        result[i + 4] = carry;
    }

    u512Copy(output, result);
}

// Function to divide two 256-bit values, producing quotient and remainder
inline __device__ void u256Div(u64 quotient[4], u64 remainder[4],
                               const u64 dividend[4], const u64 divisor[4]) {
    u64 quotientAndRemainder[8] = {0};

    u256SetZero(quotientAndRemainder + 4); // Initialize remainder to zero
    u256Copy(quotientAndRemainder, dividend); // Set initial dividend

    for (int i = 255; i >= 0; --i) {
        u512LShift1(quotientAndRemainder); // Shift left by 1

        u64* q = quotientAndRemainder;
        u64* r = quotientAndRemainder + 4;

        if (u256Compare(r, divisor) >= 0) {
            u64 temp[4];
            u256Sub(temp, r, divisor);
            u256SetBit(q, i % 256, true);
            u256Copy(r, temp);
        }
    }

    u256Copy(quotient, quotientAndRemainder);
    u256Copy(remainder, quotientAndRemainder + 4);
}

// Function to divide two 512-bit values, producing quotient and remainder
inline __device__ void u512Div(u64 quotient[8], u64 remainder[8],
                               const u64 dividend[8], const u64 divisor[8]) {
    u512SetZero(quotient);
    u512SetZero(remainder);

    for (int i = 511; i >= 0; --i) {
        u512LShift1(remainder); // Shift remainder left by 1
        u512SetBit(remainder, 0, u512GetBit(dividend, i)); // Set LSB based on dividend

        if (u512Compare(remainder, divisor) >= 0) {
            u512Sub(remainder, remainder, divisor);
            u512SetBit(quotient, i, true);
        }
    }
}

// Function to compute a modulo p for a 256-bit value
inline __device__ void u256ModP(u64 output[4], const u64 a[4], const u64 p[4]) {
    u64 quotient[4] = {0};
    u64 remainder[4] = {0};

    u256Div(quotient, remainder, a, p);
    u256Copy(output, remainder);
}

// Function to compute a modulo p for a 512-bit value extended to 256 bits
inline __device__ void u512ModU256P(u64 output[4], const u64 a[8], const u64 p[4]) {
    u64 quotient[8] = {0};
    u64 outputExtended[8] = {0};
    u64 pExtended[8] = {0};

    u256Extend(pExtended, p);
    u512Div(quotient, outputExtended, a, pExtended);
    u256Copy(output, outputExtended);
}

// Function to perform modular addition: (a + b) mod p
inline __device__ void u256AddModP(u64 output[4], const u64 a[4],
                                   const u64 b[4], const u64 p[4]) {
    u64 added[4];
    bool carry = u256Add(added, a, b);

    u64 extended[8];
    u256Extend(extended, added);

    if (carry) {
        extended[4] += 1;
    }

    u512ModU256P(output, extended, p);
}

// Function to perform modular subtraction: (a - b) mod p
inline __device__ void u256SubModP(u64 output[4], const u64 a[4],
                                   const u64 b[4], const u64 p[4]) {
    u64 result[4] = {0};
    bool borrow = u256Sub(result, a, b);

    if (borrow) {
        u64 temp[4];
        u256Copy(temp, result);
        u256Add(result, temp, p);
    }

    u256ModP(result, result, p);
    u256Copy(output, result);
}

// Function to perform modular multiplication: (a * b) mod p
inline __device__ void u256MulModP(u64 output[4], const u64 a[4],
                                   const u64 b[4], const u64 p[4]) {
    u64 multiplied[8];
    u256Mul(multiplied, a, b);

    u64 result[4];
    u512ModU256P(result, multiplied, p);

    u256Copy(output, result);
}

// Function to perform modular exponentiation: (a^b) mod p
inline __device__ void u256PowModP(u64 output[4], const u64 a[4],
                                   const u64 b[4], const u64 p[4]) {
    u64 result[4] = {1, 0, 0, 0}; // Initialize result to 1

    u64 aModP[4];
    u256ModP(aModP, a, p);

    if (u256IsZero(aModP)) {
        u256SetZero(result);
    } else {
        u64 bCopy[4];
        u256Copy(bCopy, b);

        for (int i = 0; i < 256; ++i) {
            if (u256GetBit(bCopy, i)) {
                u256MulModP(result, result, aModP, p);
            }
            u256MulModP(aModP, aModP, aModP, p);
        }
    }

    u256Copy(output, result);
}

// Function to compute modular inverse: a^{-1} mod p
inline __device__ void u256InvModP(u64 output[4], const u64 a[4],
                                   const u64 p[4]) {
    u64 pMinus2[4] = {0};
    u256Sub(pMinus2, p, (u64[4]){2, 0, 0, 0});

    u256PowModP(output, a, pMinus2, p);
}

#endif // U256_CUH

