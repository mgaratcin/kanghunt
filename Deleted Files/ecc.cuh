#ifndef __LIBSECP256K1_CUH
#define __LIBSECP256K1_CUH

#include "point.cuh"

// Kernel declaration (no extern "C" needed)
__global__ void getPublicKeyByPrivateKeyKernel(Point *output, const u64 *privateKeys, int n);

#ifdef __cplusplus
extern "C" {
#endif

// Host function declaration with C linkage
void getPublicKeyByPrivateKey(Point output[], const u64 flattenedPrivateKeys[][4], int n);

#ifdef __cplusplus
}
#endif

#endif // __LIBSECP256K1_CUH
