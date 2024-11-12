#ifndef POINT_CUH
#define POINT_CUH

#include "u256.cuh"
#include "fp.cuh"

// Structure representing a point on the elliptic curve
typedef struct {
    u64 x[4];
    u64 y[4];
} Point;

// Function to set a Point to zero (the point at infinity)
inline __device__ void pointSetZero(Point *output) {
    u256SetZero(output->x);
    u256SetZero(output->y);
}

// Function to check if a Point is zero (the point at infinity)
inline __device__ bool pointIsZero(const Point *p) {
    return fpIsZero(p->x) && fpIsZero(p->y);
}

// Function to copy one Point to another
inline __device__ void pointCopy(Point *output, const Point *p) {
    u256Copy(output->x, p->x);
    u256Copy(output->y, p->y);
}

// Function to negate a Point (reflect over the x-axis)
inline __device__ void pointNeg(Point *output, const Point *p, const u64 prime[4]) {
    u256Copy(output->x, p->x);
    fpNeg(output->y, p->y, prime);
}

// Function to add two Points on the elliptic curve
inline __device__ void pointAdd(Point *output, const Point *p, const Point *q, 
                                const u64 prime[4], const u64 a[4], const u64 b[4]) {
    // Handle special cases
    if (pointIsZero(p)) {
        pointCopy(output, q);
        return;
    }
    if (pointIsZero(q)) {
        pointCopy(output, p);
        return;
    }

    // Check if points are inverses of each other (P.x == Q.x and P.y == -Q.y)
    if (u256Compare(p->x, q->x) == 0) {
        u64 negQY[4];
        fpNeg(negQY, q->y, prime);
        if (u256Compare(p->y, negQY) == 0) {
            pointSetZero(output);
            return;
        }
    }

    Point temp;
    u64 s[4]; // Slope

    if (u256Compare(p->y, q->y) == 0) { // Point doubling
        u64 squaredPX[4];
        fpMul(squaredPX, p->x, p->x, prime); // s = (3 * P.x^2 + a) / (2 * P.y)

        u64 threePX2[4];
        fpAdd(threePX2, squaredPX, squaredPX, prime); // 2 * P.x^2
        fpAdd(threePX2, threePX2, squaredPX, prime); // 3 * P.x^2

        u64 doubledPY[4];
        fpAdd(doubledPY, p->y, p->y, prime); // 2 * P.y

        u64 numerator[4];
        fpAdd(numerator, threePX2, a, prime); // 3 * P.x^2 + a

        fpDiv(s, numerator, doubledPY, prime); // s = (3 * P.x^2 + a) / (2 * P.y)
    } else { // Point addition
        u64 diffX[4];
        u64 diffY[4];
        fpSub(diffX, q->x, p->x, prime); // Q.x - P.x
        fpSub(diffY, q->y, p->y, prime); // Q.y - P.y

        fpDiv(s, diffY, diffX, prime); // s = (Q.y - P.y) / (Q.x - P.x)
    }

    // Compute x3 = s^2 - P.x - Q.x
    u64 sSquared[4];
    fpMul(sSquared, s, s, prime); // s^2

    u64 x3Temp[4];
    fpSub(x3Temp, sSquared, p->x, prime); // s^2 - P.x

    u64 x3[4];
    fpSub(x3, x3Temp, q->x, prime); // s^2 - P.x - Q.x

    // Compute y3 = s * (P.x - x3) - P.y
    u64 pxMinusx3[4];
    fpSub(pxMinusx3, p->x, x3, prime); // P.x - x3

    u64 sTimesDiff[4];
    fpMul(sTimesDiff, s, pxMinusx3, prime); // s * (P.x - x3)

    u64 y3[4];
    fpSub(y3, sTimesDiff, p->y, prime); // s * (P.x - x3) - P.y

    // Set the resulting Point
    u256Copy(output->x, x3);
    u256Copy(output->y, y3);
}

// Function to subtract two Points on the elliptic curve
inline __device__ void pointSub(Point *output, const Point *p, const Point *q, 
                                const u64 prime[4], const u64 a[4], const u64 b[4]) {
    Point qNeg;
    pointNeg(&qNeg, q, prime);
    pointAdd(output, p, &qNeg, prime, a, b);
}

// Function to multiply a Point by a scalar k (double-and-add algorithm)
inline __device__ void pointMul(Point *output, const Point *p, const u64 k[4], 
                                const u64 prime[4], const u64 a[4], const u64 b[4]) {
    pointSetZero(output); // Initialize output to zero (point at infinity)

    if (pointIsZero(p) || fpIsZero(k)) {
        return; // Nothing to do
    }

    Point q;
    pointCopy(&q, p); // Initialize Q = P

    for (int i = 0; i < 256; ++i) {
        if (u256GetBit(k, i)) {
            Point temp;
            pointCopy(&temp, output);
            pointAdd(output, &temp, &q, prime, a, b);
        }
        pointAdd(&q, &q, &q, prime, a, b); // Q = Q + Q (Point doubling)
    }
}

// Function to check if two Points are equal
inline __device__ bool pointEqual(const Point *p, const Point *q) {
    return (u256Compare(p->x, q->x) == 0) && (u256Compare(p->y, q->y) == 0);
}

#endif // POINT_CUH
