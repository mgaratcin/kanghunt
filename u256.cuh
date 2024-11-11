#ifndef _U256_CUH
#define _U256_CUH

#include "u64.cuh"

// 256-bit copy
__forceinline__ __device__ void u256Copy(u64 output[4], const u64 a[4]) {
    output[0] = a[0];
    output[1] = a[1];
    output[2] = a[2];
    output[3] = a[3];
}

// 512-bit copy
__forceinline__ __device__ void u512Copy(u64 output[8], const u64 a[8]) {
    output[0] = a[0];
    output[1] = a[1];
    output[2] = a[2];
    output[3] = a[3];
    output[4] = a[4];
    output[5] = a[5];
    output[6] = a[6];
    output[7] = a[7];
}

// Set 256-bit number to zero
__forceinline__ __device__ void u256SetZero(u64 a[4]) {
    a[0] = 0;
    a[1] = 0;
    a[2] = 0;
    a[3] = 0;
}

// Set 512-bit number to zero
__forceinline__ __device__ void u512SetZero(u64 a[8]) {
    a[0] = 0;
    a[1] = 0;
    a[2] = 0;
    a[3] = 0;
    a[4] = 0;
    a[5] = 0;
    a[6] = 0;
    a[7] = 0;
}

// Extend 256-bit to 512-bit by zeroing upper words
__forceinline__ __device__ void u256Extend(u64 output[8], const u64 a[4]) {
    output[0] = a[0];
    output[1] = a[1];
    output[2] = a[2];
    output[3] = a[3];
    output[4] = 0;
    output[5] = 0;
    output[6] = 0;
    output[7] = 0;
}

// Get a specific bit from a 256-bit number
__forceinline__ __device__ bool u256GetBit(const u64 a[4], int index) {
    int word = index >> 6;   // index / 64
    int bit = index & 63;    // index % 64
    return (a[word] >> bit) & 1;
}

// Get a specific bit from a 512-bit number
__forceinline__ __device__ bool u512GetBit(const u64 a[8], int index) {
    int word = index >> 6;   // index / 64
    int bit = index & 63;    // index % 64
    return (a[word] >> bit) & 1;
}

// Set or clear a specific bit in a 256-bit number
__forceinline__ __device__ void u256SetBit(u64 output[4], int index, bool value) {
    int word = index >> 6;   // index / 64
    int bit = index & 63;    // index % 64
    if (value)
        output[word] |= 1ULL << bit;
    else
        output[word] &= ~(1ULL << bit);
}

// Set or clear a specific bit in a 512-bit number
__forceinline__ __device__ void u512SetBit(u64 a[8], int index, bool value) {
    int word = index >> 6;   // index / 64
    int bit = index & 63;    // index % 64
    if (value)
        a[word] |= 1ULL << bit;
    else
        a[word] &= ~(1ULL << bit);
}

// Left shift a 256-bit number by 1
__forceinline__ __device__ void u256LShift1(u64 a[4]) {
    a[3] = (a[3] << 1) | (a[2] >> 63);
    a[2] = (a[2] << 1) | (a[1] >> 63);
    a[1] = (a[1] << 1) | (a[0] >> 63);
    a[0] <<= 1;
}

// Left shift a 512-bit number by 1
__forceinline__ __device__ void u512LShift1(u64 a[8]) {
    a[7] = (a[7] << 1) | (a[6] >> 63);
    a[6] = (a[6] << 1) | (a[5] >> 63);
    a[5] = (a[5] << 1) | (a[4] >> 63);
    a[4] = (a[4] << 1) | (a[3] >> 63);
    a[3] = (a[3] << 1) | (a[2] >> 63);
    a[2] = (a[2] << 1) | (a[1] >> 63);
    a[1] = (a[1] << 1) | (a[0] >> 63);
    a[0] <<= 1;
}

// Right shift a 256-bit number by 1
__forceinline__ __device__ void u256RShift1(u64 a[4]) {
    a[0] = (a[0] >> 1) | (a[1] << 63);
    a[1] = (a[1] >> 1) | (a[2] << 63);
    a[2] = (a[2] >> 1) | (a[3] << 63);
    a[3] >>= 1;
}

// Check if a 256-bit number is odd
__forceinline__ __device__ bool u256IsOdd(const u64 a[4]) {
    return a[0] & 1;
}

// Check if a 256-bit number is zero
__forceinline__ __device__ bool u256IsZero(const u64 a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

// Compare two 256-bit numbers
__forceinline__ __device__ int u256Compare(const u64 a[4], const u64 b[4]) {
    if (a[3] > b[3]) return 1;
    if (a[3] < b[3]) return -1;
    if (a[2] > b[2]) return 1;
    if (a[2] < b[2]) return -1;
    if (a[1] > b[1]) return 1;
    if (a[1] < b[1]) return -1;
    if (a[0] > b[0]) return 1;
    if (a[0] < b[0]) return -1;
    return 0;
}

// Compare two 512-bit numbers
__forceinline__ __device__ int u512Compare(const u64 a[8], const u64 b[8]) {
    if (a[7] > b[7]) return 1;
    if (a[7] < b[7]) return -1;
    if (a[6] > b[6]) return 1;
    if (a[6] < b[6]) return -1;
    if (a[5] > b[5]) return 1;
    if (a[5] < b[5]) return -1;
    if (a[4] > b[4]) return 1;
    if (a[4] < b[4]) return -1;
    if (a[3] > b[3]) return 1;
    if (a[3] < b[3]) return -1;
    if (a[2] > b[2]) return 1;
    if (a[2] < b[2]) return -1;
    if (a[1] > b[1]) return 1;
    if (a[1] < b[1]) return -1;
    if (a[0] > b[0]) return 1;
    if (a[0] < b[0]) return -1;
    return 0;
}

// Add two 256-bit numbers with carry
__forceinline__ __device__ bool u256Add(u64 output[4], const u64 a[4], const u64 b[4]) {
    bool carry = false;
    carry = u64Add(&output[0], a[0], b[0], carry);
    carry = u64Add(&output[1], a[1], b[1], carry);
    carry = u64Add(&output[2], a[2], b[2], carry);
    carry = u64Add(&output[3], a[3], b[3], carry);
    return carry;
}

// Subtract two 256-bit numbers with borrow
__forceinline__ __device__ bool u256Sub(u64 output[4], const u64 a[4], const u64 b[4]) {
    bool borrow = false;
    borrow = u64Sub(&output[0], a[0], b[0], borrow);
    borrow = u64Sub(&output[1], a[1], b[1], borrow);
    borrow = u64Sub(&output[2], a[2], b[2], borrow);
    borrow = u64Sub(&output[3], a[3], b[3], borrow);
    return borrow;
}

// Subtract two 512-bit numbers with borrow
__forceinline__ __device__ bool u512Sub(u64 output[8], const u64 a[8], const u64 b[8]) {
    bool borrow = false;
    borrow = u64Sub(&output[0], a[0], b[0], borrow);
    borrow = u64Sub(&output[1], a[1], b[1], borrow);
    borrow = u64Sub(&output[2], a[2], b[2], borrow);
    borrow = u64Sub(&output[3], a[3], b[3], borrow);
    borrow = u64Sub(&output[4], a[4], b[4], borrow);
    borrow = u64Sub(&output[5], a[5], b[5], borrow);
    borrow = u64Sub(&output[6], a[6], b[6], borrow);
    borrow = u64Sub(&output[7], a[7], b[7], borrow);
    return borrow;
}

// Multiply a 256-bit number with a 64-bit integer
__forceinline__ __device__ u64 u256MulWithU64(u64 output[4], const u64 a[4], const u64 b, u64 carry) {
    carry = u64Mul(&output[0], a[0], b, carry);
    carry = u64Mul(&output[1], a[1], b, carry);
    carry = u64Mul(&output[2], a[2], b, carry);
    carry = u64Mul(&output[3], a[3], b, carry);
    return carry;
}

// Multiply two 256-bit numbers to get a 512-bit result
__forceinline__ __device__ void u256Mul(u64 output[8], const u64 a[4], const u64 b[4]) {
    u64 temp[8] = {0};
    u64 carry;

    // Multiply a * b[0]
    carry = 0;
    carry = u256MulWithU64(&temp[0], a, b[0], carry);
    temp[4] += carry;

    // Multiply a * b[1] and add to temp shifted by 1 word
    carry = 0;
    carry = u256MulWithU64(&temp[1], a, b[1], carry);
    temp[5] += carry;

    // Multiply a * b[2] and add to temp shifted by 2 words
    carry = 0;
    carry = u256MulWithU64(&temp[2], a, b[2], carry);
    temp[6] += carry;

    // Multiply a * b[3] and add to temp shifted by 3 words
    carry = 0;
    carry = u256MulWithU64(&temp[3], a, b[3], carry);
    temp[7] += carry;

    // Copy the result to output
    u512Copy(output, temp);
}

// Divide two 256-bit numbers to get quotient and remainder
__forceinline__ __device__ void u256Div(u64 quotient[4], u64 remainder[4], const u64 dividend[4], const u64 divisor[4]) {
    u64 quotientAndRemainder[8] = {0};
    u256Copy(quotientAndRemainder, dividend);
    u256SetZero(quotientAndRemainder + 4);

    for (int i = 255; i >= 0; --i) {
        u512LShift1(quotientAndRemainder);

        u64* q = quotientAndRemainder;
        u64* r = quotientAndRemainder + 4;

        if (u256Compare(r, divisor) >= 0) {
            u256Sub(r, r, divisor);
            u256SetBit(q, 0, 1);
        }
    }

    u256Copy(quotient, quotientAndRemainder);
    u256Copy(remainder, quotientAndRemainder + 4);
}

// Divide two 512-bit numbers to get quotient and remainder
__forceinline__ __device__ void u512Div(u64 quotient[8], u64 remainder[8], const u64 dividend[8], const u64 divisor[8]) {
    u512SetZero(quotient);
    u512SetZero(remainder);

    for (int i = 511; i >= 0; --i) {
        u512LShift1(remainder);
        u512SetBit(remainder, 0, u512GetBit(dividend, i));

        if (u512Compare(remainder, divisor) >= 0) {
            u512Sub(remainder, remainder, divisor);
            u512SetBit(quotient, i, 1);
        }
    }
}

// Compute a mod p for 256-bit numbers
__forceinline__ __device__ void u256ModP(u64 output[4], const u64 a[4], const u64 p[4]) {
    u64 quotient[4];
    u64 remainder[4];
    u256Div(quotient, remainder, a, p);
    u256Copy(output, remainder);
}

// Compute a mod p for 512-bit a and 256-bit p
__forceinline__ __device__ void u512ModU256P(u64 output[4], const u64 a[8], const u64 p[4]) {
    u64 quotient[8];
    u64 outputExtended[8];
    u64 pExtended[8];

    u256Extend(pExtended, p);
    u512Div(quotient, outputExtended, a, pExtended);
    u256Copy(output, outputExtended);
}

// Add two 256-bit numbers modulo p
__forceinline__ __device__ void u256AddModP(u64 output[4], const u64 a[4], const u64 b[4], const u64 p[4]) {
    u64 extended[8];
    bool carry = false;
    carry = u64Add(&extended[0], a[0], b[0], carry);
    carry = u64Add(&extended[1], a[1], b[1], carry);
    carry = u64Add(&extended[2], a[2], b[2], carry);
    carry = u64Add(&extended[3], a[3], b[3], carry);
    extended[4] = carry;
    extended[5] = extended[6] = extended[7] = 0;

    u512ModU256P(output, extended, p);
}

// Subtract two 256-bit numbers modulo p
__forceinline__ __device__ void u256SubModP(u64 output[4], const u64 a[4], const u64 b[4], const u64 p[4]) {
    u64 temp[4];
    bool borrow = u256Sub(temp, a, b);
    
    if (borrow) {
        u256Add(temp, temp, p);
    }

    u256ModP(output, temp, p);
}

// Multiply two 256-bit numbers modulo p
__forceinline__ __device__ void u256MulModP(u64 output[4], const u64 a[4], const u64 b[4], const u64 p[4]) {
    u64 multiplied[8];
    u256Mul(multiplied, a, b);
    u64 result[4];
    u512ModU256P(result, multiplied, p);
    u256Copy(output, result);
}

// Exponentiate a 256-bit number modulo p
__forceinline__ __device__ void u256PowModP(u64 output[4], const u64 a[4], const u64 b[4], const u64 p[4]) {
    u64 result[4] = {1, 0, 0, 0};
    u64 baseModP[4];
    u256ModP(baseModP, a, p);

    // If base is zero, result is zero
    if (u256IsZero(baseModP)) {
        u256SetZero(result);
    } else {
        u64 bCopy[4];
        u256Copy(bCopy, b);

        for (int i = 0; i < 256; ++i) {
            if (u256GetBit(bCopy, i)) {
                u256MulModP(result, result, baseModP, p);
            }
            u256MulModP(baseModP, baseModP, baseModP, p);
        }
    }

    u256Copy(output, result);
}

// Compute the modular inverse of a 256-bit number modulo p
__forceinline__ __device__ void u256InvModP(u64 output[4], const u64 a[4], const u64 p[4]) {
    u64 pMinus2[4];
    bool borrow = u256Sub(pMinus2, p, (u64[4]){2, 0, 0, 0});
    
    u64 result[4];
    u256PowModP(result, a, pMinus2, p);
    
    u256Copy(output, result);
}

#endif
