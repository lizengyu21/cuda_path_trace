#include "object.cuh"

__device__ void Sphere::intersect(const PathState &path_state, HitRecord &record) {
    float3 oc = path_state.ray.position - origin;
    auto a = length_squared(path_state.ray.direction);
    auto half_b = dot(oc, path_state.ray.direction);
    auto c = length_squared(oc) - radius * radius;
    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) 
        return;
    float t = (-half_b - sqrt(discriminant)) / a;
    if (t < 0) t = (-half_b + sqrt(discriminant)) / a;
    if (t < record.t && t > 0) {
        record.t = t;
        float3 hit_point = path_state.intersaction_point(t);
        record.missed = false;
        record.position = hit_point;
        record.material_index = material_index;
        record.normal = unit(hit_point - origin);
        record.outer = dot(path_state.ray.direction, record.normal) < 0;
        record.normal = record.outer ? record.normal : -record.normal;
    }
}

__device__ void Triangle::intersect(const PathState &path_state, HitRecord &record) {
    float u, v, t_tmp = 0;
    float3 p_vec = cross(path_state.ray.direction, e2);
    float det = dot(e1, p_vec);
    if (det < 0.00001f && det > -0.00001f) return;
    float det_inv = 1.0f / det;
    float3 t_vec = path_state.ray.position - v0;
    u = dot(t_vec, p_vec) * det_inv;
    if (u < 0 || u > 1) return;
    float3 q_vec = cross(t_vec, e1);
    v = dot(path_state.ray.direction, q_vec) * det_inv;
    if (v < 0 || u + v > 1) return;
    t_tmp = dot(e2, q_vec) * det_inv;
    if (t_tmp > 0 && t_tmp < record.t) {
        record.t = t_tmp;
        record.missed = false;
        record.outer = dot(path_state.ray.direction, this->normal) < 0;
        record.normal = record.outer ? this->normal : -this->normal;
        record.position = path_state.intersaction_point(t_tmp);
        record.material_index = this->material_index;
    }
}