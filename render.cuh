#pragma once
#include "camera.cuh"
#include "ray.cuh"
#include "scene.cuh"
#include "bvh.cuh"
#include "object.cuh"
#include "material.cuh"

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
#include <thrust/random.h>

class Render {
public:
    int trace_depth = 10; // number of bounce times
    int SPP = 4096;
    Camera camera;
    BVH bvh;
    thrust::host_vector<PathState> host_path_state_buffer;
    thrust::host_vector<HitRecord> host_hit_record_buffer;
    thrust::device_vector<PathState> dev_path_state_buffer;
    thrust::device_vector<HitRecord> dev_hit_record_buffer;
    thrust::device_vector<Material> dev_material_buffer;
    thrust::host_vector<Material> host_material_buffer;
    Render() {}
    void print_image(const float3 *);
    void init();
    void path_trace();
};

