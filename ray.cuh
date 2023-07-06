#pragma once
#include "math_utils.cuh"



class Ray {
public:
    float3 position;
    float3 direction, direction_inverse;
};

// every path matches to the pixel
class PathState {
public:
    Ray ray;
    float3 attenuation;
    float3 result;
    int pixel_index;
    int remaining_iteration;
    unsigned int seed;
    __host__ __device__ float3 intersaction_point(float t) const noexcept { return ray.position + (t * ray.direction); }
};