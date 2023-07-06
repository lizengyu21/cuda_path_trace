#include "material.cuh"
#include "math_utils.cuh"
#include <assert.h>

static __constant__ float MPI = 3.14159265358979323846;

__device__ void direct_callable_diffuse(PathState &path_state, const HitRecord &record, thrust::default_random_engine &rng, Material material) {
    if (material.emittance > 0.00001f) {
        path_state.result = path_state.result + path_state.attenuation * (material.emittance * material.albedo);
        path_state.remaining_iteration = 0;
        return;
    }
    path_state.ray.direction = random_on_hemi_sphere(rng, record.normal);
    path_state.ray.direction_inverse = 1.0f / path_state.ray.direction;

    float cosine = dot(path_state.ray.direction, record.normal);
    assert(cosine >= 0.0f);
    path_state.ray.position = record.position + 0.00001f * record.normal;
    path_state.attenuation = 2.0f * cosine * path_state.attenuation * material.albedo;
    --(path_state.remaining_iteration);
}

__device__ void direct_callable_metal(PathState &path_state, const HitRecord &record, thrust::default_random_engine &rng, Material material) {
    path_state.ray.position = record.position + 0.00001f * record.normal;
    path_state.ray.direction = unit(reflect(path_state.ray.direction, record.normal) + material.roughness * random_on_unit_sphere(rng));
    path_state.ray.direction_inverse = 1.0f / path_state.ray.direction;

}