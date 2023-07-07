#pragma once

#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include "morton_code.cuh"
#include <vector>
#include "object.cuh"
#include "aabb.cuh"
#include "ray.cuh"
#include "record.cuh"
#include "mesh.cuh"

std::ostream &operator<<(std::ostream &out, float3 c);

struct Node {
    unsigned int parent = 0xFFFFFFFF;
    unsigned int left_child = 0xFFFFFFFF;
    unsigned int right_child = 0xFFFFFFFF;
    unsigned int object_index = 0xFFFFFFFF;
};

struct DeviceBVH {
    unsigned int node_count;
    unsigned int sphere_count;
    unsigned int triangle_count;
    unsigned int object_count;
    Node *nodes;
    Aabb *aabbs;
    Sphere *spheres;
    Triangle *triangles;
};


class BVH {
public:
    thrust::host_vector<Sphere> host_spheres;
    thrust::device_vector<Sphere> dev_spheres;

    thrust::host_vector<Triangle> host_triangles;
    thrust::device_vector<Triangle> dev_triangles;

    thrust::device_vector<Aabb> dev_aabbs;
    thrust::device_vector<Node> dev_nodes;

    template <class SphereInputIterator, class TriangleInputIterator>
    BVH(SphereInputIterator sphere_first = 0, SphereInputIterator sphere_end = 0, TriangleInputIterator triangle_first = 0, TriangleInputIterator triangle_end = 0)
        : host_spheres(sphere_first, sphere_end), dev_spheres(host_spheres)
        , host_triangles(triangle_first, triangle_end), dev_triangles(host_triangles) {}

        template <class TriangleInputIterator>
    BVH(TriangleInputIterator triangle_first = 0, TriangleInputIterator triangle_end = 0)
        : host_triangles(triangle_first, triangle_end), dev_triangles(host_triangles) {}

    BVH() = default;
    __host__ DeviceBVH get_dev_bvh() noexcept;

    void build();
    __host__ bool is_empty() { return dev_nodes.size() == 0; }
    template <class MortonType>
    void construct_internal_nodes(const DeviceBVH &self, const MortonType *morton, const unsigned int objects_count);
};

template <class MortonType>
__device__ uint2 calc_range(const unsigned int index, const MortonType *morton, const unsigned int leaves_count);

template <class MortonType>
__device__ unsigned int find_split_index(const unsigned int leaves_count, const uint2 range, const MortonType *morton);

__device__ void cast_ray(const PathState &ray, HitRecord &record, const DeviceBVH &self);
