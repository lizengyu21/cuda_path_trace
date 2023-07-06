#pragma once
#include "ray.cuh"
#include "record.cuh"
#include "aabb.cuh"
#include <thrust/host_vector.h>
#include <string>

class Sphere {
public:
    float3 origin;
    float radius;
    unsigned int material_index;
    Sphere(const float3 &o, float r, unsigned int m_id) : origin(o), radius(r), material_index(m_id) {}
    __device__ void intersect(const PathState &path_state, HitRecord &record);
};

struct sphere_aabb_getter {
    __device__ Aabb operator() (const Sphere &s) const noexcept {
        float3 lower = make_float3(s.origin.x - s.radius, s.origin.y - s.radius, s.origin.z - s.radius);
        float3 upper = make_float3(s.origin.x + s.radius, s.origin.y + s.radius, s.origin.z + s.radius);
        return Aabb(lower, upper);
    }
};

class Triangle {
public:
    float3 v0, v1, v2;
    float3 e1, e2;
    float3 normal;
    unsigned int material_index;
    Triangle(float3 a, float3 b, float3 c, unsigned int m_id) : v0(a), v1(b), v2(c), material_index(m_id) {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = unit(cross(e1, e2));
    }
    __device__ void intersect(const PathState &path_state, HitRecord &record);
};

struct triangle_aabb_getter {
    __device__ Aabb operator() (const Triangle &t) const noexcept {
        float x_min = min(t.v0.x, min(t.v1.x, t.v2.x));
        float y_min = min(t.v0.y, min(t.v1.y, t.v2.y));
        float z_min = min(t.v0.z, min(t.v1.z, t.v2.z));

        float x_max = max(t.v0.x, max(t.v1.x, t.v2.x));
        float y_max = max(t.v0.y, max(t.v1.y, t.v2.y));
        float z_max = max(t.v0.z, max(t.v1.z, t.v2.z));
        float3 lower = make_float3(x_min, y_min, z_min);
        float3 upper = make_float3(x_max, y_max, z_max);
        
        return Aabb(lower, upper);
    }
};

