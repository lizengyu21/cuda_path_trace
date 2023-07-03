#pragma once
#include "ray.cuh"
#include "record.cuh"
#include "aabb.cuh"

class Object {
public:
    unsigned int material_index = 0xFFFFFFFF;
};

class Sphere : Object {
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


