#ifndef _U64_CUH
#define _U64_CUH

typedef unsigned long long u64;

/**
 * @brief Adds two 64-bit unsigned integers along with a carry.
 *
 * @param output Pointer to store the sum of a, b, and carry.
 * @param a First operand.
 * @param b Second operand.
 * @param carry Carry-in (0 or 1).
 * @return bool Returns true if there is a carry-out, false otherwise.
 */
__forceinline__ __device__ bool u64Add(u64 *output, const u64 a, const u64 b, const bool carry) {
    u64 temp = a + b;
    *output = temp + carry;
    // Carry-out occurs if temp < a or (carry is true and temp == ULLONG_MAX)
    return (temp < a) || (carry && (temp == ULLONG_MAX));
}

/**
 * @brief Subtracts two 64-bit unsigned integers along with a borrow.
 *
 * @param output Pointer to store the result of a - b - borrow.
 * @param a Minuend.
 * @param b Subtrahend.
 * @param borrow Borrow-in (0 or 1).
 * @return bool Returns true if a borrow-out occurs, false otherwise.
 */
__forceinline__ __device__ bool u64Sub(u64 *output, const u64 a, const u64 b, const bool borrow) {
    // Perform subtraction with borrow
    *output = a - b - borrow;
    // Borrow-out occurs if a < (b + borrow)
    return a < (b + borrow);
}

/**
 * @brief Multiplies two 64-bit unsigned integers and adds a carry.
 *
 * @param output Pointer to store the lower 64 bits of the product plus carry.
 * @param a First operand.
 * @param b Second operand.
 * @param carry Carry-in (0 or 1).
 * @return u64 Returns the upper 64 bits of the product after adding carry.
 */
__forceinline__ __device__ u64 u64Mul(u64 *output, const u64 a, const u64 b, const u64 carry) {
    u64 low = a * b;
    u64 high = __umul64hi(a, b);
    
    // Add carry to the lower part
    u64 new_low = low + carry;
    
    // If overflow occurs when adding carry, increment high
    if (new_low < low) {
        high += 1;
    }
    
    *output = new_low;
    
    return high;
}

#endif // _U64_CUH
