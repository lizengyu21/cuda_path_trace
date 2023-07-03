#pragma once
#include "ray.cuh"
#include "record.cuh"
#include <thrust/swap.h>

class Aabb {
public:
    float3 lower, upper;
    __host__ __device__ Aabb() {}
    __host__ __device__ Aabb(float3 _l, float3 _u) : lower(_l), upper(_u) {}
    __host__ __device__ bool intersect(const PathState &path_state);
};

__device__ Aabb merge(const Aabb &l, const Aabb &r);

__host__ __device__ inline float3 centerlization(const Aabb &box) {
    float3 temp;
    temp.x = (box.upper.x + box.lower.x) * 0.5f;
    temp.y = (box.upper.y + box.lower.y) * 0.5f;
    temp.z = (box.upper.z + box.lower.z) * 0.5f;
    return temp;
}

struct aabb_merger {
    __device__ Aabb operator() (const Aabb &l, const Aabb &r) const noexcept {
        return merge(l, r);
    }
};