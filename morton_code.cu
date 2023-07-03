#include "morton_code.cuh"

__device__ __host__ inline unsigned int expand_bits(unsigned int d) noexcept {
    // only minimum 10 digits reserved

    // copy lower 16 bits to higher 16 bits
    // and apply mask to get reserve higher 8 bits and lower 8 bits
    d = ((d << 16) + d) & 0xff0000ffu;
    // copy lower 8 bits to 8-15 bits
    d = ((d << 8) + d) & 0x0f00f00fu;
    d = ((d << 4) + d) & 0xc30c30c3u;
    d = ((d << 2) + d) & 0x49249249u;
    return d;
    // d = * * * * * B C D  B = b3 b2 b1 b0
    // get bits like:
    // | * * 0 0 | b1 0 0 b0 | 0 0 c3 0 | 0 c2 0 0 | c1 0 0 c0 | 0 0 d3 0 | 0 d2 0 0 | d1 0 0 d0 |   
}

__device__ __host__ unsigned int morton_code(float3 xyz) noexcept { // choose 10 bits resolution
    const float resolution = 1024.0f;
    xyz.x = min(max(xyz.x * resolution, 0.0f), resolution - 1.0f);
    xyz.y = min(max(xyz.y * resolution, 0.0f), resolution - 1.0f);
    xyz.z = min(max(xyz.z * resolution, 0.0f), resolution - 1.0f);
    // tranverse to [0, 1023]
    const unsigned int x = expand_bits(static_cast<unsigned int>(xyz.x));
    const unsigned int y = expand_bits(static_cast<unsigned int>(xyz.y));
    const unsigned int z = expand_bits(static_cast<unsigned int>(xyz.z));
    return (x << 2) + (y << 1) + z;
}
