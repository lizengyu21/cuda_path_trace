#pragma once
#include "ray.cuh"
#include "record.cuh"
#include "util.cuh"

enum MaterialType {
    DIFFUSE,
    METAL,
};

class Material;

typedef void (*MaterialShader)(PathState &, const HitRecord &, thrust::default_random_engine &, Material);

__device__ void direct_callable_diffuse(PathState &path_state, const HitRecord &record, thrust::default_random_engine &rng, Material material);
static __device__ MaterialShader diffuse_shader = direct_callable_diffuse;

__device__ void direct_callable_metal(PathState &path_state, const HitRecord &record, thrust::default_random_engine &rng, Material material);
static __device__ MaterialShader metal_shader = direct_callable_metal;


struct Material
{
    // diffuse
    float3 albedo = make_float3(0.1, 0.2, 0.3);

    // metal
    float roughness = 0.0f;

    // lightness
    float emittance = 0.0f;

    MaterialShader shader;

    Material(MaterialType type = MaterialType::DIFFUSE) {
        switch (type) {
        case MaterialType::DIFFUSE:
            CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(&shader, diffuse_shader, sizeof(MaterialShader)));
            break;
        case MaterialType::METAL:
            CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(&shader, metal_shader, sizeof(MaterialShader)));
            break;
        default:
            break;
        }
    }
    __device__ void shade(PathState &state, const HitRecord &record, thrust::default_random_engine &rng) {
        shader(state, record, rng, *this);
    }
};

