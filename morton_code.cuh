#include <cstdint>
#include "aabb.cuh"


__device__ __host__ inline unsigned int expand_bits(unsigned int d) noexcept;

__device__ __host__ unsigned int morton_code(float3 xyz) noexcept;

__device__ inline int common_prefix_bits_count(const unsigned int l, const unsigned int r) {
    return __clz(l ^ r);
}

__device__ inline int common_prefix_bits_count(const unsigned long long int l, const unsigned long long int r) {
    return __clzll(l ^ r);
}

struct morton_code_calculator {
    Aabb whole;
    morton_code_calculator(Aabb box) : whole(box) {}
    __device__ unsigned int operator() (const Aabb &box) const noexcept {
        float3 p = centerlization(box);
        p.x -= whole.lower.x;
        p.y -= whole.lower.y;
        p.z -= whole.lower.z;
        p.x /= (whole.upper.x - whole.lower.x);
        p.y /= (whole.upper.y - whole.lower.y);
        p.z /= (whole.upper.z - whole.lower.z);
        return morton_code(p);
    }
};

struct expand_morton_64 {
    __device__ unsigned long long int operator() (const unsigned int m, const unsigned int index) const noexcept {
        unsigned long long m64 = m;
        return (unsigned long long)((m64 << 32) | (unsigned long long)index);
    }
};