#pragma once
#include <thrust/random.h>

__host__ __device__ inline unsigned int util_hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ inline thrust::default_random_engine make_seeded_random_engine(int iter, int index, int depth) {
  int h = util_hash((1 << 31) | (depth << 22) | iter) ^ util_hash(index);
  return thrust::default_random_engine(h);
}

__host__ __device__ inline unsigned int make_seed(int iter, int index, int depth) {
  return util_hash((1 << 31) | (depth << 22) | iter) ^ util_hash(index);
}