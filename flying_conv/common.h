#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

struct FastDivmod {
    unsigned int d_;
    unsigned int magic_;
    unsigned int shift_;

    __host__ __device__ FastDivmod(unsigned int d = 1) : d_(d) {
        if (d == 0) d = 1;
        shift_ = 0;
        while ((1U << shift_) < d) shift_++;
        uint64_t magic = ((1ULL << 32) * ((1ULL << shift_) - d)) / d + 1;
        magic_ = (unsigned int)magic;
    }

    __device__ __forceinline__ void divmod(unsigned int n, unsigned int& q, unsigned int& r) const {
#ifdef __CUDA_ARCH__
        unsigned int t = __umulhi(n, magic_);
        // multiply two numbers -> 128bit result -> remove behind 64bit results
#else
        unsigned int t = (unsigned int)(((uint64_t)n * magic_) >> 32);
#endif
        q = (t + n) >> shift_;
        r = n - q * d_;
    }
};