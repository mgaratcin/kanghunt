#ifndef _SECP256K1_CUH
#define _SECP256K1_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "ptx.h"


/**
 Prime modulus 2^256 - 2^32 - 977
 */
__constant__ static unsigned int _P[8] = {
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

/**
 Base point X
 */
__constant__ static unsigned int _GX[8] = {
	0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};


/**
 Base point Y
 */
__constant__ static unsigned int _GY[8] = {
	0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};


/**
 * Group order
 */
__constant__ static unsigned int _N[8] = {
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

__constant__ static unsigned int _BETA[8] = {
	0x7AE96A2B, 0x657C0710, 0x6E64479E, 0xAC3434E9, 0x9CF04975, 0x12F58995, 0xC1396C28, 0x719501EE
};


__constant__ static unsigned int _LAMBDA[8] = {
	0x5363AD4C, 0xC05C30E0, 0xA5261C02, 0x8812645A, 0x122E22EA, 0x20816678, 0xDF02967C, 0x1B23BD72
};


__device__ __forceinline__ bool isInfinity(const unsigned int x[8])
{
	bool isf = true;

	for(int i = 0; i < 8; i++) {
		if(x[i] != 0xffffffff) {
			isf = false;
		}
	}

	return isf;
}

__device__ __forceinline__ static void copyBigInt(const unsigned int src[8], unsigned int dest[8])
{
	for(int i = 0; i < 8; i++) {
		dest[i] = src[i];
	}
}

__device__ static bool equal(const unsigned int *a, const unsigned int *b)
{
	bool eq = true;

	for(int i = 0; i < 8; i++) {
		eq &= (a[i] == b[i]);
	}

	return eq;
}

/**
 * Reads an 8-word big integer from device memory
 */
__device__ static void readInt(const unsigned int *ara, int idx, unsigned int x[8])
{
	int totalThreads = gridDim.x * blockDim.x;

	int base = idx * totalThreads * 8;

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	int index = base + threadId;

	for (int i = 0; i < 8; i++) {
		x[i] = ara[index];
		index += totalThreads;
	}
}

__device__ static unsigned int readIntLSW(const unsigned int *ara, int idx)
{
	int totalThreads = gridDim.x * blockDim.x;

	int base = idx * totalThreads * 8;

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	int index = base + threadId;

	return ara[index + totalThreads * 7];
}

/**
 * Writes an 8-word big integer to device memory
 */
__device__ static void writeInt(unsigned int *ara, int idx, const unsigned int x[8])
{
	int totalThreads = gridDim.x * blockDim.x;

	int base = idx * totalThreads * 8;

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	int index = base + threadId;

	for (int i = 0; i < 8; i++) {
		ara[index] = x[i];
		index += totalThreads;
	}
}

/**
 * Subtraction mod p
 */
__device__ static void subModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	sub_cc(c[7], a[7], b[7]);
	subc_cc(c[6], a[6], b[6]);
	subc_cc(c[5], a[5], b[5]);
	subc_cc(c[4], a[4], b[4]);
	subc_cc(c[3], a[3], b[3]);
	subc_cc(c[2], a[2], b[2]);
	subc_cc(c[1], a[1], b[1]);
	subc_cc(c[0], a[0], b[0]);

	unsigned int borrow = 0;
	subc(borrow, 0, 0);

	if (borrow) {
		add_cc(c[7], c[7], _P[7]);
		addc_cc(c[6], c[6], _P[6]);
		addc_cc(c[5], c[5], _P[5]);
		addc_cc(c[4], c[4], _P[4]);
		addc_cc(c[3], c[3], _P[3]);
		addc_cc(c[2], c[2], _P[2]);
		addc_cc(c[1], c[1], _P[1]);
		addc(c[0], c[0], _P[0]);
	}
}

__device__ static unsigned int add(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	add_cc(c[7], a[7], b[7]);
	addc_cc(c[6], a[6], b[6]);
	addc_cc(c[5], a[5], b[5]);
	addc_cc(c[4], a[4], b[4]);
	addc_cc(c[3], a[3], b[3]);
	addc_cc(c[2], a[2], b[2]);
	addc_cc(c[1], a[1], b[1]);
	addc_cc(c[0], a[0], b[0]);

	unsigned int carry = 0;
	addc(carry, 0, 0);

	return carry;
}

__device__ static unsigned int sub(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	sub_cc(c[7], a[7], b[7]);
	subc_cc(c[6], a[6], b[6]);
	subc_cc(c[5], a[5], b[5]);
	subc_cc(c[4], a[4], b[4]);
	subc_cc(c[3], a[3], b[3]);
	subc_cc(c[2], a[2], b[2]);
	subc_cc(c[1], a[1], b[1]);
	subc_cc(c[0], a[0], b[0]);

	unsigned int borrow = 0;
	subc(borrow, 0, 0);

	return (borrow & 0x01);
}


__device__ static void addModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	add_cc(c[7], a[7], b[7]);
	addc_cc(c[6], a[6], b[6]);
	addc_cc(c[5], a[5], b[5]);
	addc_cc(c[4], a[4], b[4]);
	addc_cc(c[3], a[3], b[3]);
	addc_cc(c[2], a[2], b[2]);
	addc_cc(c[1], a[1], b[1]);
	addc_cc(c[0], a[0], b[0]);

	unsigned int carry = 0;
	addc(carry, 0, 0);

	bool gt = false;
	for(int i = 0; i < 8; i++) {
		if(c[i] > _P[i]) {
			gt = true;
			break;
		} else if(c[i] < _P[i]) {
			break;
		}
	}

	if(carry || gt) {
		sub_cc(c[7], c[7], _P[7]);
		subc_cc(c[6], c[6], _P[6]);
		subc_cc(c[5], c[5], _P[5]);
		subc_cc(c[4], c[4], _P[4]);
		subc_cc(c[3], c[3], _P[3]);
		subc_cc(c[2], c[2], _P[2]);
		subc_cc(c[1], c[1], _P[1]);
		subc(c[0], c[0], _P[0]);
	}
}



__device__ static void mulModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	unsigned int high[8] = { 0 };

	unsigned int t = a[7];

	// a[7] * b (low)
	for(int i = 7; i >= 0; i--) {
		c[i] = t * b[i];
	}

	// a[7] * b (high)
	mad_hi_cc(c[6], t, b[7], c[6]);
	madc_hi_cc(c[5], t, b[6], c[5]);
	madc_hi_cc(c[4], t, b[5], c[4]);
	madc_hi_cc(c[3], t, b[4], c[3]);
	madc_hi_cc(c[2], t, b[3], c[2]);
	madc_hi_cc(c[1], t, b[2], c[1]);
	madc_hi_cc(c[0], t, b[1], c[0]);
	madc_hi(high[7], t, b[0], high[7]);



	// a[6] * b (low)
	t = a[6];
	mad_lo_cc(c[6], t, b[7], c[6]);
	madc_lo_cc(c[5], t, b[6], c[5]);
	madc_lo_cc(c[4], t, b[5], c[4]);
	madc_lo_cc(c[3], t, b[4], c[3]);
	madc_lo_cc(c[2], t, b[3], c[2]);
	madc_lo_cc(c[1], t, b[2], c[1]);
	madc_lo_cc(c[0], t, b[1], c[0]);
	madc_lo_cc(high[7], t, b[0], high[7]);
	addc(high[6], high[6], 0);

	// a[6] * b (high)
	mad_hi_cc(c[5], t, b[7], c[5]);
	madc_hi_cc(c[4], t, b[6], c[4]);
	madc_hi_cc(c[3], t, b[5], c[3]);
	madc_hi_cc(c[2], t, b[4], c[2]);
	madc_hi_cc(c[1], t, b[3], c[1]);
	madc_hi_cc(c[0], t, b[2], c[0]);
	madc_hi_cc(high[7], t, b[1], high[7]);
	madc_hi(high[6], t, b[0], high[6]);

	// a[5] * b (low)
	t = a[5];
	mad_lo_cc(c[5], t, b[7], c[5]);
	madc_lo_cc(c[4], t, b[6], c[4]);
	madc_lo_cc(c[3], t, b[5], c[3]);
	madc_lo_cc(c[2], t, b[4], c[2]);
	madc_lo_cc(c[1], t, b[3], c[1]);
	madc_lo_cc(c[0], t, b[2], c[0]);
	madc_lo_cc(high[7], t, b[1], high[7]);
	madc_lo_cc(high[6], t, b[0], high[6]);
	addc(high[5], high[5], 0);

	// a[5] * b (high)
	mad_hi_cc(c[4], t, b[7], c[4]);
	madc_hi_cc(c[3], t, b[6], c[3]);
	madc_hi_cc(c[2], t, b[5], c[2]);
	madc_hi_cc(c[1], t, b[4], c[1]);
	madc_hi_cc(c[0], t, b[3], c[0]);
	madc_hi_cc(high[7], t, b[2], high[7]);
	madc_hi_cc(high[6], t, b[1], high[6]);
	madc_hi(high[5], t, b[0], high[5]);



	// a[4] * b (low)
	t = a[4];
	mad_lo_cc(c[4], t, b[7], c[4]);
	madc_lo_cc(c[3], t, b[6], c[3]);
	madc_lo_cc(c[2], t, b[5], c[2]);
	madc_lo_cc(c[1], t, b[4], c[1]);
	madc_lo_cc(c[0], t, b[3], c[0]);
	madc_lo_cc(high[7], t, b[2], high[7]);
	madc_lo_cc(high[6], t, b[1], high[6]);
	madc_lo_cc(high[5], t, b[0], high[5]);
	addc(high[4], high[4], 0);

	// a[4] * b (high)
	mad_hi_cc(c[3], t, b[7], c[3]);
	madc_hi_cc(c[2], t, b[6], c[2]);
	madc_hi_cc(c[1], t, b[5], c[1]);
	madc_hi_cc(c[0], t, b[4], c[0]);
	madc_hi_cc(high[7], t, b[3], high[7]);
	madc_hi_cc(high[6], t, b[2], high[6]);
	madc_hi_cc(high[5], t, b[1], high[5]);
	madc_hi(high[4], t, b[0], high[4]);



	// a[3] * b (low)
	t = a[3];
	mad_lo_cc(c[3], t, b[7], c[3]);
	madc_lo_cc(c[2], t, b[6], c[2]);
	madc_lo_cc(c[1], t, b[5], c[1]);
	madc_lo_cc(c[0], t, b[4], c[0]);
	madc_lo_cc(high[7], t, b[3], high[7]);
	madc_lo_cc(high[6], t, b[2], high[6]);
	madc_lo_cc(high[5], t, b[1], high[5]);
	madc_lo_cc(high[4], t, b[0], high[4]);
	addc(high[3], high[3], 0);

	// a[3] * b (high)
	mad_hi_cc(c[2], t, b[7], c[2]);
	madc_hi_cc(c[1], t, b[6], c[1]);
	madc_hi_cc(c[0], t, b[5], c[0]);
	madc_hi_cc(high[7], t, b[4], high[7]);
	madc_hi_cc(high[6], t, b[3], high[6]);
	madc_hi_cc(high[5], t, b[2], high[5]);
	madc_hi_cc(high[4], t, b[1], high[4]);
	madc_hi(high[3], t, b[0], high[3]);



	// a[2] * b (low)
	t = a[2];
	mad_lo_cc(c[2], t, b[7], c[2]);
	madc_lo_cc(c[1], t, b[6], c[1]);
	madc_lo_cc(c[0], t, b[5], c[0]);
	madc_lo_cc(high[7], t, b[4], high[7]);
	madc_lo_cc(high[6], t, b[3], high[6]);
	madc_lo_cc(high[5], t, b[2], high[5]);
	madc_lo_cc(high[4], t, b[1], high[4]);
	madc_lo_cc(high[3], t, b[0], high[3]);
	addc(high[2], high[2], 0);

	// a[2] * b (high)
	mad_hi_cc(c[1], t, b[7], c[1]);
	madc_hi_cc(c[0], t, b[6], c[0]);
	madc_hi_cc(high[7], t, b[5], high[7]);
	madc_hi_cc(high[6], t, b[4], high[6]);
	madc_hi_cc(high[5], t, b[3], high[5]);
	madc_hi_cc(high[4], t, b[2], high[4]);
	madc_hi_cc(high[3], t, b[1], high[3]);
	madc_hi(high[2], t, b[0], high[2]);



	// a[1] * b (low)
	t = a[1];
	mad_lo_cc(c[1], t, b[7], c[1]);
	madc_lo_cc(c[0], t, b[6], c[0]);
	madc_lo_cc(high[7], t, b[5], high[7]);
	madc_lo_cc(high[6], t, b[4], high[6]);
	madc_lo_cc(high[5], t, b[3], high[5]);
	madc_lo_cc(high[4], t, b[2], high[4]);
	madc_lo_cc(high[3], t, b[1], high[3]);
	madc_lo_cc(high[2], t, b[0], high[2]);
	addc(high[1], high[1], 0);

	// a[1] * b (high)
	mad_hi_cc(c[0], t, b[7], c[0]);
	madc_hi_cc(high[7], t, b[6], high[7]);
	madc_hi_cc(high[6], t, b[5], high[6]);
	madc_hi_cc(high[5], t, b[4], high[5]);
	madc_hi_cc(high[4], t, b[3], high[4]);
	madc_hi_cc(high[3], t, b[2], high[3]);
	madc_hi_cc(high[2], t, b[1], high[2]);
	madc_hi(high[1], t, b[0], high[1]);



	// a[0] * b (low)
	t = a[0];
	mad_lo_cc(c[0], t, b[7], c[0]);
	madc_lo_cc(high[7], t, b[6], high[7]);
	madc_lo_cc(high[6], t, b[5], high[6]);
	madc_lo_cc(high[5], t, b[4], high[5]);
	madc_lo_cc(high[4], t, b[3], high[4]);
	madc_lo_cc(high[3], t, b[2], high[3]);
	madc_lo_cc(high[2], t, b[1], high[2]);
	madc_lo_cc(high[1], t, b[0], high[1]);
	addc(high[0], high[0], 0);

	// a[0] * b (high)
	mad_hi_cc(high[7], t, b[7], high[7]);
	madc_hi_cc(high[6], t, b[6], high[6]);
	madc_hi_cc(high[5], t, b[5], high[5]);
	madc_hi_cc(high[4], t, b[4], high[4]);
	madc_hi_cc(high[3], t, b[3], high[3]);
	madc_hi_cc(high[2], t, b[2], high[2]);
	madc_hi_cc(high[1], t, b[1], high[1]);
	madc_hi(high[0], t, b[0], high[0]);



	// At this point we have 16 32-bit words representing a 512-bit value
	// high[0 ... 7] and c[0 ... 7]
	const unsigned int s = 977;

	// Store high[6] and high[7] since they will be overwritten
	unsigned int high7 = high[7];
	unsigned int high6 = high[6];


	// Take high 256 bits, multiply by 2^32, add to low 256 bits
	// That is, take high[0 ... 7], shift it left 1 word and add it to c[0 ... 7]
	add_cc(c[6], high[7], c[6]);
	addc_cc(c[5], high[6], c[5]);
	addc_cc(c[4], high[5], c[4]);
	addc_cc(c[3], high[4], c[3]);
	addc_cc(c[2], high[3], c[2]);
	addc_cc(c[1], high[2], c[1]);
	addc_cc(c[0], high[1], c[0]);
	addc_cc(high[7], high[0], 0);
	addc(high[6], 0, 0);


	// Take high 256 bits, multiply by 977, add to low 256 bits
	// That is, take high[0 ... 5], high6, high7, multiply by 977 and add to c[0 ... 7]
	mad_lo_cc(c[7], high7, s, c[7]);
	madc_lo_cc(c[6], high6, s, c[6]);
	madc_lo_cc(c[5], high[5], s, c[5]);
	madc_lo_cc(c[4], high[4], s, c[4]);
	madc_lo_cc(c[3], high[3], s, c[3]);
	madc_lo_cc(c[2], high[2], s, c[2]);
	madc_lo_cc(c[1], high[1], s, c[1]);
	madc_lo_cc(c[0], high[0], s, c[0]);
	addc_cc(high[7], high[7], 0);
	addc(high[6], high[6], 0);


	mad_hi_cc(c[6], high7, s, c[6]);
	madc_hi_cc(c[5], high6, s, c[5]);
	madc_hi_cc(c[4], high[5], s, c[4]);
	madc_hi_cc(c[3], high[4], s, c[3]);
	madc_hi_cc(c[2], high[3], s, c[2]);
	madc_hi_cc(c[1], high[2], s, c[1]);
	madc_hi_cc(c[0], high[1], s, c[0]);
	madc_hi_cc(high[7], high[0], s, high[7]);
	addc(high[6], high[6], 0);


	// Repeat the same steps, but this time we only need to handle high[6] and high[7]
	high7 = high[7];
	high6 = high[6];

	// Take the high 64 bits, multiply by 2^32 and add to the low 256 bits
	add_cc(c[6], high[7], c[6]);
	addc_cc(c[5], high[6], c[5]);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], 0, 0);


	// Take the high 64 bits, multiply by 977 and add to the low 256 bits
	mad_lo_cc(c[7], high7, s, c[7]);
	madc_lo_cc(c[6], high6, s, c[6]);
	addc_cc(c[5], c[5], 0);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], high[7], 0);

	mad_hi_cc(c[6], high7, s, c[6]);
	madc_hi_cc(c[5], high6, s, c[5]);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], high[7], 0);


	bool overflow = high[7] != 0;

	unsigned int borrow = sub(c, _P, c);

	if(overflow) {
		if(!borrow) {
			sub(c, _P, c);
		}
	} else {
		if(borrow) {
			add(c, _P, c);
		}
	}
}


/**
 * Square mod P
 * b = a * a
 */
__device__ static void squareModP(const unsigned int a[8], unsigned int b[8])
{
	mulModP(a, a, b);
}

/**
 * Square mod P
 * x = x * x
 */
__device__ static void squareModP(unsigned int x[8])
{
	unsigned int tmp[8];
	squareModP(x, tmp);
	copyBigInt(tmp, x);
}

/**
 * Multiply mod P
 * c = a * c
 */
__device__ static void mulModP(const unsigned int a[8], unsigned int c[8])
{
	unsigned int tmp[8];
	mulModP(a, c, tmp);

	copyBigInt(tmp, c);
}

/**
 * Multiplicative inverse mod P using Fermat's method of x^(p-2) mod p and addition chains
 */
__device__ static void invModP(unsigned int value[8])
{
	unsigned int x[8];

	copyBigInt(value, x);

	unsigned int y[8] = { 0, 0, 0, 0, 0, 0, 0, 1 };

	// 0xd - 1101
	mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);


	// 0x2 - 0010
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);

	// 0xc = 0x1100
	//mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0xfffff
	for(int i = 0; i < 20; i++) {
		mulModP(x, y);
		squareModP(x);
	}

	// 0xe - 1110
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffff
	for(int i = 0; i < 219; i++) {
		mulModP(x, y);
		squareModP(x);
	}
	mulModP(x, y);

	copyBigInt(y, value);
}

__device__ static void invModP(const unsigned int *value, unsigned int *inverse)
{
	copyBigInt(value, inverse);

	invModP(inverse);
}

__device__ static void negModP(const unsigned int *value, unsigned int *negative)
{
	sub_cc(negative[0], _P[0], value[0]);
	subc_cc(negative[1], _P[1], value[1]);
	subc_cc(negative[2], _P[2], value[2]);
	subc_cc(negative[3], _P[3], value[3]);
	subc_cc(negative[4], _P[4], value[4]);
	subc_cc(negative[5], _P[5], value[5]);
	subc_cc(negative[6], _P[6], value[6]);
	subc(negative[7], _P[7], value[7]);
}


__device__ __forceinline__ static void beginBatchAdd(const unsigned int *px, const unsigned int *x, unsigned int *chain, int i, int batchIdx, unsigned int inverse[8])
{
	// x = Gx - x
	unsigned int t[8];
	subModP(px, x, t);

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(t, inverse);

	writeInt(chain, batchIdx, inverse);
}


__device__ __forceinline__ static void beginBatchAddWithDouble(const unsigned int *px, const unsigned int *py, unsigned int *xPtr, unsigned int *chain, int i, int batchIdx, unsigned int inverse[8])
{
	unsigned int x[8];
	readInt(xPtr, i, x);

	if(equal(px, x)) {
		addModP(py, py, x);
	} else {
		// x = Gx - x
		subModP(px, x, x);
	}

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(x, inverse);

	writeInt(chain, batchIdx, inverse);
}

__device__ static void completeBatchAddWithDouble(const unsigned int *px, const unsigned int *py, const unsigned int *xPtr, const unsigned int *yPtr, int i, int batchIdx, unsigned int *chain, unsigned int *inverse, unsigned int newX[8], unsigned int newY[8])
{
	unsigned int s[8];
	unsigned int x[8];
	unsigned int y[8];

	readInt(xPtr, i, x);
	readInt(yPtr, i, y);

	if(batchIdx >= 1) {
		unsigned int c[8];

		readInt(chain, batchIdx - 1, c);

		mulModP(inverse, c, s);

		unsigned int diff[8];
		if(equal(px, x)) {
			addModP(py, py, diff);
		} else {
			subModP(px, x, diff);
		}

		mulModP(diff, inverse);
	} else {
		copyBigInt(inverse, s);
	}


	if(equal(px, x)) {
		// currently s = 1 / 2y

		unsigned int x2[8];
		unsigned int tx2[8];

		// 3x^2
		mulModP(x, x, x2);
		addModP(x2, x2, tx2);
		addModP(x2, tx2, tx2);


		// s = 3x^2 * 1/2y
		mulModP(tx2, s);

		// s^2
		unsigned int s2[8];
		mulModP(s, s, s2);

		// Rx = s^2 - 2px
		subModP(s2, x, newX);
		subModP(newX, x, newX);

		// Ry = s(px - rx) - py
		unsigned int k[8];
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

	} else {

		unsigned int rise[8];
		subModP(py, y, rise);

		mulModP(rise, s);

		// Rx = s^2 - Gx - Qx
		unsigned int s2[8];
		mulModP(s, s, s2);

		subModP(s2, px, newX);
		subModP(newX, x, newX);

		// Ry = s(px - rx) - py
		unsigned int k[8];
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);
	}
}

__device__ static void completeBatchAdd(const unsigned int *px, const unsigned int *py, unsigned int *xPtr, unsigned int *yPtr, int i, int batchIdx, unsigned int *chain, unsigned int *inverse, unsigned int newX[8], unsigned int newY[8])
{
	unsigned int s[8];
	unsigned int x[8];

	readInt(xPtr, i, x);

	if(batchIdx >= 1) {
		unsigned int c[8];

		readInt(chain, batchIdx - 1, c);
		mulModP(inverse, c, s);

		unsigned int diff[8];
		subModP(px, x, diff);
		mulModP(diff, inverse);
	} else {
		copyBigInt(inverse, s);
	}

	unsigned int y[8];
	readInt(yPtr, i, y);

	unsigned int rise[8];
	subModP(py, y, rise);

	mulModP(rise, s);

	// Rx = s^2 - Gx - Qx
	unsigned int s2[8];
	mulModP(s, s, s2);
	subModP(s2, px, newX);
	subModP(newX, x, newX);

	// Ry = s(px - rx) - py
	unsigned int k[8];
	subModP(px, newX, k);
	mulModP(s, k, newY);
	subModP(newY, py, newY);
}

__device__ static void doubleModP(const unsigned int a[8], unsigned int result[8]) {
    addModP(a, a, result);
}

// Implement scalar multiplication
__device__ void pointMultiply(const unsigned int scalar[8], unsigned int x[8], unsigned int y[8]) {
    // Implement scalar multiplication using the double-and-add algorithm
    // Using Jacobian coordinates for efficiency

    unsigned int Rx[8], Ry[8], Rz[8]; // Result point in Jacobian coordinates
    unsigned int Px[8], Py[8], Pz[8]; // Point G in Jacobian coordinates

    // Initialize R to the point at infinity (represented by Z = 0)
    for (int i = 0; i < 8; ++i) {
        Rx[i] = 0;
        Ry[i] = 0;
        Rz[i] = 0;
    }

    // Initialize P to G (the generator point)
    copyBigInt(_GX, Px);
    copyBigInt(_GY, Py);
    for (int i = 0; i < 8; ++i) {
        Pz[i] = 0;
    }
    Pz[7] = 1; // Z = 1 for affine point converted to Jacobian coordinates

    // Loop over bits of scalar from most significant bit to least significant bit
    for (int bit = 255; bit >= 0; --bit) {
        // R = 2R
        pointDouble(Rx, Ry, Rz, Rx, Ry, Rz);

        // Check if the bit is set
        int word = bit / 32;
        int bitPos = bit % 32;
        if ((scalar[word] >> bitPos) & 1) {
            // R = R + P
            pointAdd(Rx, Ry, Rz, Px, Py, Pz, Rx, Ry, Rz);
        }
    }

    // Convert R back to affine coordinates
    jacobianToAffine(Rx, Ry, Rz, x, y);
}

// Implement point addition in Jacobian coordinates
__device__ void pointAdd(const unsigned int X1[8], const unsigned int Y1[8], const unsigned int Z1[8],
                         const unsigned int X2[8], const unsigned int Y2[8], const unsigned int Z2[8],
                         unsigned int X3[8], unsigned int Y3[8], unsigned int Z3[8]) {
    // Temporary variables
    unsigned int Z1Z1[8], Z2Z2[8];
    unsigned int U1[8], U2[8];
    unsigned int S1[8], S2[8];
    unsigned int H[8], R[8];
    unsigned int H2[8], H3[8];
    unsigned int U1H2[8];
    unsigned int temp1[8], temp2[8], temp3[8];

    // Compute Z1Z1 = Z1^2 mod p
    squareModP(Z1, Z1Z1);

    // Compute Z2Z2 = Z2^2 mod p
    squareModP(Z2, Z2Z2);

    // Compute U1 = X1 * Z2Z2 mod p
    mulModP(X1, Z2Z2, U1);

    // Compute U2 = X2 * Z1Z1 mod p
    mulModP(X2, Z1Z1, U2);

    // Compute Z1^3 and Z2^3
    mulModP(Z1, Z1Z1, temp1); // temp1 = Z1^3
    mulModP(Z2, Z2Z2, temp2); // temp2 = Z2^3

    // Compute S1 = Y1 * Z2^3 mod p
    mulModP(Y1, temp2, S1);

    // Compute S2 = Y2 * Z1^3 mod p
    mulModP(Y2, temp1, S2);

    // Compute H = U2 - U1 mod p
    subModP(U2, U1, H);

    // Compute R = S2 - S1 mod p
    subModP(S2, S1, R);

    // Check if H is zero
    bool H_is_zero = true;
    for (int i = 0; i < 8; i++) {
        if (H[i] != 0) {
            H_is_zero = false;
            break;
        }
    }

    if (H_is_zero) {
        // Check if R is zero
        bool R_is_zero = true;
        for (int i = 0; i < 8; i++) {
            if (R[i] != 0) {
                R_is_zero = false;
                break;
            }
        }

        if (R_is_zero) {
            // P1 == P2, perform point doubling
            pointDouble(X1, Y1, Z1, X3, Y3, Z3);
            return;
        } else {
            // Result is point at infinity
            for (int i = 0; i < 8; i++) {
                X3[i] = 0;
                Y3[i] = 0;
                Z3[i] = 0;
            }
            return;
        }
    }

    // Compute H2 = H^2 mod p
    squareModP(H, H2);

    // Compute H3 = H * H2 mod p
    mulModP(H, H2, H3);

    // Compute U1H2 = U1 * H2 mod p
    mulModP(U1, H2, U1H2);

    // Compute R^2
    squareModP(R, temp1); // temp1 = R^2

    // Compute 2 * U1H2
    doubleModP(U1H2, temp2); // temp2 = 2 * U1H2

    // Compute X3 = R^2 - H3 - 2 * U1H2 mod p
    subModP(temp1, H3, temp3); // temp3 = R^2 - H3
    subModP(temp3, temp2, X3); // X3 = temp3 - temp2

    // Compute Y3 = R * (U1H2 - X3) - S1 * H3 mod p
    subModP(U1H2, X3, temp1); // temp1 = U1H2 - X3
    mulModP(R, temp1, temp2); // temp2 = R * (U1H2 - X3)
    mulModP(S1, H3, temp3);   // temp3 = S1 * H3
    subModP(temp2, temp3, Y3); // Y3 = temp2 - temp3

    // Compute Z3 = H * Z1 * Z2 mod p
    mulModP(Z1, Z2, temp1);   // temp1 = Z1 * Z2
    mulModP(temp1, H, Z3);    // Z3 = temp1 * H
}

// Implement point doubling in Jacobian coordinates
__device__ void pointDouble(const unsigned int X1[8], const unsigned int Y1[8], const unsigned int Z1[8],
                            unsigned int X3[8], unsigned int Y3[8], unsigned int Z3[8]) {
    // Temporary variables
    unsigned int S1[8]; // S1 = Y1^2 mod p
    unsigned int S2[8]; // S2 = X1 * S1 mod p
    unsigned int S[8];  // S = 4 * S2 mod p
    unsigned int A[8];  // A = X1^2 mod p
    unsigned int M[8];  // M = 3 * A mod p
    unsigned int X3_temp[8]; // X3_temp = M^2 mod p
    unsigned int two_S[8];  // two_S = 2 * S mod p
    unsigned int T[8]; // T = S - X3 mod p
    unsigned int Y3_temp[8]; // Y3_temp = M * T mod p
    unsigned int S1_squared[8];  // S1_squared = S1^2 mod p
    unsigned int eight_S1_squared[8]; // eight_S1_squared = 8 * S1_squared mod p
    unsigned int temp1[8], temp2[8]; // temporary variables

    // Compute S1 = Y1^2 mod p
    squareModP(Y1, S1);

    // Compute S2 = X1 * S1 mod p
    mulModP(X1, S1, S2);

    // Compute S = 4 * S2 mod p
    doubleModP(S2, temp1); // temp1 = 2 * S2 mod p
    doubleModP(temp1, S);  // S = 4 * S2 mod p

    // Compute A = X1^2 mod p
    squareModP(X1, A);

    // Compute M = 3 * A mod p
    addModP(A, A, temp1); // temp1 = 2 * A mod p
    addModP(temp1, A, M); // M = 3 * A mod p

    // Compute X3_temp = M^2 mod p
    squareModP(M, X3_temp);

    // Compute two_S = 2 * S mod p
    doubleModP(S, two_S);

    // Compute X3 = X3_temp - two_S mod p
    subModP(X3_temp, two_S, X3);

    // Compute T = S - X3 mod p
    subModP(S, X3, T);

    // Compute Y3_temp = M * T mod p
    mulModP(M, T, Y3_temp);

    // Compute S1_squared = S1^2 mod p
    squareModP(S1, S1_squared);

    // Compute eight_S1_squared = 8 * S1_squared mod p
    doubleModP(S1_squared, temp1); // temp1 = 2 * S1_squared
    doubleModP(temp1, temp2); // temp2 = 4 * S1_squared
    doubleModP(temp2, eight_S1_squared); // eight_S1_squared = 8 * S1_squared

    // Compute Y3 = Y3_temp - eight_S1_squared mod p
    subModP(Y3_temp, eight_S1_squared, Y3);

    // Compute Z3 = 2 * Y1 * Z1 mod p
    mulModP(Y1, Z1, temp1); // temp1 = Y1 * Z1 mod p
    doubleModP(temp1, Z3);  // Z3 = 2 * Y1 * Z1 mod p
}


// Convert point from Jacobian to affine coordinates
__device__ void jacobianToAffine(const unsigned int X[8], const unsigned int Y[8], const unsigned int Z[8],
                                 unsigned int x[8], unsigned int y[8]) {
    // Compute Zinv = inverse of Z modulo p
    unsigned int Zinv[8];
    copyBigInt(Z, Zinv);
    invModP(Zinv);

    // Compute Zinv2 = Zinv^2
    unsigned int Zinv2[8];
    mulModP(Zinv, Zinv, Zinv2);

    // Compute Zinv3 = Zinv^2 * Zinv = Zinv^3
    unsigned int Zinv3[8];
    mulModP(Zinv2, Zinv, Zinv3);

    // x = X * Zinv2 mod p
    mulModP(X, Zinv2, x);

    // y = Y * Zinv3 mod p
    mulModP(Y, Zinv3, y);
}

__device__ __forceinline__ static void doBatchInverse(unsigned int inverse[8])
{
	invModP(inverse);
}

#endif
