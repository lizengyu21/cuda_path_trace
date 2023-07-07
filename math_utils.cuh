#pragma once
#include <math.h>
#include <thrust/random.h>
#include <assert.h>
const float M_PI = 3.14159265358979323846;

__device__ __host__ inline float3 operator+(const float3 &a, const float3 &b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __host__ inline float3 operator-(const float3 &a, const float3 &b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __host__ inline float3 operator-(const float3 &a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __host__ inline float3 operator*(const float &a, const float3 &b) { return make_float3(a * b.x, a * b.y, a * b.z); }
__device__ __host__ inline float3 operator*(const float3 &a, const float3 &b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __host__ inline float3 operator/(const float3 &a, const float &b) { return make_float3(a.x / b, a.y / b, a.z / b); }
__device__ __host__ inline float3 operator/(const float &a, const float3 &b) { return make_float3(a / b.x, a / b.y, a / b.z); }
__device__ __host__ inline float3 operator/(const float3 &a, const float3 &b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
__device__ __host__ inline float length_squared(const float3 &a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
__device__ __host__ inline float3 unit(const float3 &a) { float len = sqrt(length_squared(a)); return make_float3(a.x / len, a.y / len, a.z / len); }
__device__ __host__ inline float dot(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __host__ inline float3 cross(const float3 &a, const float3 &b) { return make_float3(a.y * b.z - b.y * a.z, b.x * a.z - b.z * a.x, a.x * b.y - b.x * a.y); }
inline float degree2radian(float degree) { return degree * M_PI / 180.0f; }
__device__ __host__ inline float clamp(float min, float max, float t) { return t < min ? min : (t > max ? max : t); }
__device__ __host__ inline float3 reflect(const float3 &v, const float3 &n) { return unit(v - 2.0f * dot(v, n) * n); }

__device__ __host__ inline float3 random_on_unit_sphere(thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u_01(0, 1);
    while (true) {
        float3 v = make_float3(u_01(rng) * 2.0f - 1.0f, u_01(rng) * 2.0f - 1.0f, u_01(rng) * 2.0f - 1.0f);
        if (length_squared(v) >= 1.0f) continue;
        return unit(v);
    }
}

__device__ __host__ inline float3 random_on_hemi_sphere(thrust::default_random_engine &rng, const float3 &n) {
    float3 v = random_on_unit_sphere(rng);
    return dot(v, n) > 0.0f ? v : -v;
}