#ifndef FP_CUH
#define FP_CUH

#include "u256.cuh"

/**
 * @brief Adds two finite field elements: output = (a + b) mod p
 *
 * @param output The resulting finite field element after addition.
 * @param a      The first operand.
 * @param b      The second operand.
 * @param p      The prime modulus.
 */
inline __device__ void fpAdd(u64 output[4], const u64 a[4],
                             const u64 b[4], const u64 p[4]) {
    u256AddModP(output, a, b, p);
}

/**
 * @brief Subtracts two finite field elements: output = (a - b) mod p
 *
 * @param output The resulting finite field element after subtraction.
 * @param a      The minuend.
 * @param b      The subtrahend.
 * @param p      The prime modulus.
 */
inline __device__ void fpSub(u64 output[4], const u64 a[4],
                             const u64 b[4], const u64 p[4]) {
    u256SubModP(output, a, b, p);
}

/**
 * @brief Multiplies two finite field elements: output = (a * b) mod p
 *
 * @param output The resulting finite field element after multiplication.
 * @param a      The first operand.
 * @param b      The second operand.
 * @param p      The prime modulus.
 */
inline __device__ void fpMul(u64 output[4], const u64 a[4],
                             const u64 b[4], const u64 p[4]) {
    u256MulModP(output, a, b, p);
}

/**
 * @brief Raises a finite field element to a power: output = (a ^ b) mod p
 *
 * @param output The resulting finite field element after exponentiation.
 * @param a      The base.
 * @param b      The exponent.
 * @param p      The prime modulus.
 */
inline __device__ void fpPow(u64 output[4], const u64 a[4],
                             const u64 b[4], const u64 p[4]) {
    u256PowModP(output, a, b, p);
}

/**
 * @brief Computes the multiplicative inverse in a finite field: output = a^{-1} mod p
 *
 * @param output The resulting finite field element after inversion.
 * @param a      The element to invert.
 * @param p      The prime modulus.
 */
inline __device__ void fpInv(u64 output[4], const u64 a[4],
                             const u64 p[4]) {
    u256InvModP(output, a, p);
}

/**
 * @brief Divides two finite field elements: output = (a / b) mod p
 *
 * @param output The resulting finite field element after division.
 * @param a      The dividend.
 * @param b      The divisor.
 * @param p      The prime modulus.
 */
inline __device__ void fpDiv(u64 output[4], const u64 a[4],
                             const u64 b[4], const u64 p[4]) {
    // Compute the multiplicative inverse of b: inversed = b^{-1} mod p
    u64 inversed[4];
    fpInv(inversed, b, p);

    // Multiply a by the inverse of b: output = (a * inversed) mod p
    u256MulModP(output, a, inversed, p);
}

/**
 * @brief Negates a finite field element: output = (-a) mod p
 *
 * @param output The resulting finite field element after negation.
 * @param a      The element to negate.
 * @param p      The prime modulus.
 */
inline __device__ void fpNeg(u64 output[4], const u64 a[4],
                             const u64 p[4]) {
    fpSub(output, p, a, p);
}

/**
 * @brief Checks if a finite field element is zero.
 *
 * @param a The finite field element to check.
 * @return true  If the element is zero.
 * @return false Otherwise.
 */
inline __device__ bool fpIsZero(const u64 a[4]) {
    return u256IsZero(a);
}

#endif // FP_CUH
