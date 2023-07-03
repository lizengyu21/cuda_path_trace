#include "aabb.cuh"

__host__ __device__ bool Aabb::intersect(const PathState &path_state) {
// if (ray.direction.x == 0.0f)
    float3 t_min = (lower - path_state.ray.position) * path_state.ray.direction_inverse;
    float3 t_max = (upper - path_state.ray.position) * path_state.ray.direction_inverse;
    if (path_state.ray.direction.x <= 0) thrust::swap(t_min.x, t_max.x);
    if (path_state.ray.direction.y <= 0) thrust::swap(t_min.y, t_max.y);
    if (path_state.ray.direction.z <= 0) thrust::swap(t_min.z, t_max.z);
    // assert(t_min.x <= t_max.x && t_min.y <= t_max.y && t_min.z <= t_max.z);
    float t_exit = min(t_max.x, min(t_max.y, t_max.z));
    if (t_exit < 0) return false;
    float t_enter = max(t_min.x, max(t_min.y, t_min.z));
    return t_exit >= t_enter;
}

__device__ Aabb merge(const Aabb &l, const Aabb &r) {
    float3 lower;
    lower.x = min(l.lower.x, r.lower.x);
    lower.y = min(l.lower.y, r.lower.y);
    lower.z = min(l.lower.z, r.lower.z);
    float3 upper;
    upper.x = max(l.upper.x, r.upper.x);
    upper.y = max(l.upper.y, r.upper.y);
    upper.z = max(l.upper.z, r.upper.z);
    return Aabb(lower, upper);
}

