#include "object.cuh"

// bool Sphere::intersect(const Ray &r, HitRecord &record) {
    
// }

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
    }
}