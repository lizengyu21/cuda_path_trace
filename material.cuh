#pragma once
#include "ray.cuh"
#include "record.cuh"
#include "util.cuh"


class Material;

typedef void (*MaterialShader)(PathState &, const HitRecord &, thrust::default_random_engine &, Material);

__device__ void direct_callable_diffuse(PathState &path_state, const HitRecord &record, thrust::default_random_engine &rng, Material material);
static __device__ MaterialShader diffuse_shader = direct_callable_diffuse;


struct Material
{
    float3 albedo = make_float3(0.1, 0.2, 0.3);

    float emittance = 0.0f;

    MaterialShader shader;

    Material() {
        CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(&shader, diffuse_shader, sizeof(MaterialShader)));
    }
    __device__ void shade(PathState &state, const HitRecord &record, thrust::default_random_engine &rng) {
        shader(state, record, rng, *this);
    }
};

