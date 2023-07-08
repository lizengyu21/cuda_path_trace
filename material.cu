#include "material.cuh"
#include "math_utils.cuh"
#include <assert.h>

static __constant__ float MPI = 3.14159265358979323846; 

__device__ void sample_on_aabb(const Aabb &aabb, float &pdf_inv, float3 &destination, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<unsigned int> u_03(0, 3);
    thrust::uniform_real_distribution<float> u_01(0, 1);
    unsigned int dim = u_03(rng);
    dim = 1;
    assert(dim < 3);
    float u = u_01(rng);
    float v = u_01(rng);
    float3 bound_length = aabb.upper - aabb.lower;
    switch (dim) {
    case 0: // left
        destination = make_float3(u, v, 0) * bound_length + aabb.lower;
        pdf_inv = bound_length.x * bound_length.y;
        if (pdf_inv > 0.0001f) break;
    case 1: // bottom
        destination = make_float3(u, 0, v) * bound_length + aabb.lower;
        pdf_inv = bound_length.x * bound_length.z;
        if (pdf_inv > 0.0001f) break;
    case 2: // right
        destination = make_float3(0, u, v) * bound_length + aabb.lower;
        pdf_inv = bound_length.y * bound_length.z;
        if (pdf_inv > 0.000001f) break;
    default:
        pdf_inv = -1.0f;
        break;
    }
}

__device__ void sample_on_light(PathState &path_state, const HitRecord &record, Aabb *lights, unsigned int light_count, thrust::default_random_engine &rng, DeviceBVH self, Material *materials, float3 brdf) {
    if (light_count == 0) return;
    path_state.has_collect_direct_light = true;
    
    thrust::uniform_real_distribution<unsigned int> u_0n(0, light_count);
    unsigned int index = u_0n(rng);
    assert(index < light_count);
    Aabb light_sample = lights[index];

    float pdf_inv;
    PathState path_to_direct_light;
    HitRecord record_to_direct_light;
    float3 destination;
    
    sample_on_aabb(light_sample, pdf_inv, destination, rng);
    if (pdf_inv <= 0) return;

    path_to_direct_light.ray.position = record.position + 0.00001f * record.normal;
    path_to_direct_light.ray.direction = unit(destination - path_to_direct_light.ray.position);

    float cosine = dot(path_to_direct_light.ray.direction, record.normal);

    // direct light is behind the object
    if (cosine > 1 || cosine <= 0) return;

    path_to_direct_light.ray.direction_inverse = 1.0f / path_to_direct_light.ray.direction;
    cast_ray(path_to_direct_light, record_to_direct_light, self);

    if (!record_to_direct_light.missed && materials[record_to_direct_light.material_index].emittance > 0.00001f) {
        // hit the direct light
        float distance_squared = length_squared(record_to_direct_light.position - path_to_direct_light.ray.position);

        float cosine_prime = dot(record_to_direct_light.normal, -path_to_direct_light.ray.direction);
        
        if (cosine_prime > 0 && distance_squared > 0.1f)
            path_state.result = path_state.result + materials[record_to_direct_light.material_index].emittance * cosine * cosine_prime * pdf_inv * path_state.attenuation * brdf / distance_squared;
    }
}

__device__ void direct_callable_diffuse(PathState &path_state, const HitRecord &record, thrust::default_random_engine &rng, Material material, Aabb *light_bounds, unsigned int light_count, DeviceBVH device_bvh, Material *materials) {
    if (material.emittance > 0.00001f) { // hit direct light
        path_state.remaining_iteration = 0;
        // has collected direct light
        if (!path_state.has_collect_direct_light) 
            path_state.result = path_state.result + material.emittance * material.albedo * path_state.attenuation;
        return;
    } else {
        float3 brdf = material.albedo / MPI;
        sample_on_light(path_state, record, light_bounds, light_count, rng, device_bvh, materials, brdf);
    }
    path_state.ray.direction = random_on_hemi_sphere(rng, record.normal);
    path_state.ray.direction_inverse = 1.0f / path_state.ray.direction;

    float cosine = dot(path_state.ray.direction, record.normal);
    assert(cosine >= 0.0f);
    path_state.ray.position = record.position + 0.00001f * record.normal;
    path_state.attenuation = 2.0f * cosine * path_state.attenuation * material.albedo;
    --(path_state.remaining_iteration);
}

__device__ void direct_callable_metal(PathState &path_state, const HitRecord &record, thrust::default_random_engine &rng, Material material, Aabb *light_bounds, unsigned int light_count, DeviceBVH device_bvh, Material *materials) {

    path_state.ray.position = record.position + 0.00001f * record.normal;
    path_state.ray.direction = unit(reflect(path_state.ray.direction, record.normal) + material.roughness * random_on_unit_sphere(rng));
    path_state.ray.direction_inverse = 1.0f / path_state.ray.direction;
    --(path_state.remaining_iteration);
}

static __forceinline__ __device__ float schlick_fresnel (float u) {
    float m = clamp(0.0f, 1.0f, 1 - u);
    float m_2 = m * m;
    return m_2 * m_2 * m;
}

static __forceinline__ __device__ float schlick_fresnel_flectance (float cosine, float refractivity) {
    float r0 = (1.0f - refractivity) / (1.0f + refractivity);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * schlick_fresnel(cosine);
}


__device__ void direct_callable_dielectric(PathState &path_state, const HitRecord &record, thrust::default_random_engine &rng, Material material, Aabb *light_bounds, unsigned int light_count, DeviceBVH device_bvh, Material *materials) {
    
    float refractivity = !record.outer ? material.refractivity : 1.0f / material.refractivity;
    float cos_theta = fmin(dot(-path_state.ray.direction, record.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    
    thrust::uniform_real_distribution<float> u_01(0, 1);
    if (0 && refractivity * sin_theta > 1.0f || schlick_fresnel_flectance(cos_theta, refractivity) > u_01(rng)) {
        path_state.ray.position = record.position + 0.00001f * record.normal;
        path_state.ray.direction = reflect(path_state.ray.direction, record.normal);
    } else {
        path_state.ray.position = record.position - 0.00001f * record.normal;
        path_state.ray.direction = refract(path_state.ray.direction, record.normal, refractivity);
    }
    // path_state.ray.position = record.position - 0.00001f * record.normal;
    // path_state.ray.direction = refract(path_state.ray.direction, record.normal, refractivity);
    path_state.ray.direction_inverse = 1.0f / path_state.ray.direction;
    --(path_state.remaining_iteration);
}