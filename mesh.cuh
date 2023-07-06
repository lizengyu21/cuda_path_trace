#pragma once

#include "aabb.cuh"
#include "object.cuh"
#include <thrust/host_vector.h>
#include "obj_loader.cuh"
#include <thrust/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <string>



class MeshTriangle {
public:
    thrust::host_vector<Triangle> host_triangles;
    Aabb aabb;
    MeshTriangle(const std::string &filename, unsigned int m_id) {
        objl::Loader loader;
        loader.LoadFile(filename);
        assert(loader.LoadedMeshes.size() == 1);
        auto mesh = loader.LoadedMeshes[0];
        float3 lower = make_float3(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        float3 upper = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            float3 face_vertices[3];
            for (int j = 0; j < 3; ++j) {
                float3 vert = 60.0f * make_float3(mesh.Vertices[i + j].Position.X, mesh.Vertices[i + j].Position.Y, mesh.Vertices[i + j].Position.Z);
                face_vertices[j] = vert;
                lower = make_float3(min(lower.x, vert.x), min(lower.y, vert.y), min(lower.z, vert.z));
                upper = make_float3(max(upper.x, vert.x), max(upper.y, vert.y), max(upper.z, vert.z));
            }
            host_triangles.push_back(Triangle(face_vertices[0], face_vertices[1], face_vertices[2], m_id));
        }
        aabb = Aabb(lower, upper);
    }

    void contact_to_whole(thrust::host_vector<Triangle> &whole) {
        // push this->host_triangles back to whole
        whole.insert(whole.end(), this->host_triangles.begin(), this->host_triangles.end());
    }
};