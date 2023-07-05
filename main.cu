#include "camera.cuh"
#include <iostream>
#include <vector>
#include "render.cuh"
#include "material.cuh"

std::ostream &operator<<(std::ostream &out, float3 c) {
    out << c.x << ' ' << c.y << ' ' << c.z;
    return out;
}



int main() {
    int SPP;
    float roughness;
    float emittance;
    std::clog << "SPP: ";
    std::cin >> SPP;
    std::clog << "roughness: ";
    std::cin >> roughness;
    std::clog << "emittance: ";
    std::cin >>emittance;

    Render render;
    render.SPP = SPP;
    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[0].albedo = make_float3(1.0, 0.1, 0.1);
    // render.host_material_buffer[0].has_reflective = 1.0f;
    // render.host_material_buffer[0].specular.color = make_float3(1, 0, 0);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[1].albedo = make_float3(0.1, 1.0, 0.1);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[2].albedo = make_float3(0.1, 0.1f, 1);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[3].emittance = emittance;
    render.host_material_buffer[3].albedo = make_float3(1, 1, 1);

    render.host_material_buffer.push_back(Material(MaterialType::METAL));
    render.host_material_buffer[4].roughness = roughness;

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[5].albedo = make_float3(0.5, 0.5, 0.5);
    
    std::vector<Sphere> spheres;
    // spheres.push_back(Sphere(make_float3(0, 0, -15), 3, 0));
    spheres.push_back(Sphere(make_float3(1000, 0, -15), 995, 1));
    spheres.push_back(Sphere(make_float3(-1000, 0, -15), 995, 2));
    spheres.push_back(Sphere(make_float3(0, 10, -14), 3, 3));
    spheres.push_back(Sphere(make_float3(0, 0, -1000), 980, 5));
    // spheres.push_back(Sphere(make_float3(6, -7, -14), 3, 3));
    // spheres.push_back(Sphere(make_float3(0, -1000, 0), 993, 5));
    // spheres.push_back(Sphere(make_float3(0, 0, 0), 1000, 5));
    std::vector<Triangle> triangles;
    triangles.push_back(Triangle(make_float3(-5, 0, -13), make_float3(5, 0, -13), make_float3(0, 5, -16), 4));
    // triangles.push_back(Triangle(make_float3(5, 0, -10), make_float3(-5, 0, -10), make_float3(0, 5, -13), 0));
    render.bvh = BVH(spheres.begin(), spheres.end(), triangles.begin(), triangles.end());
    CameraSetting camera_setting;
    // float a;
    // std::cin >> a;
    // camera_setting.aperture = a;
    camera_setting.set_aspect_ratio(16.0f / 9.0f);
    camera_setting.set_focal_length(10);
    render.camera = camera_setting;
    std::clog << "init.\n";
    render.init();
    std::clog << "start.\n";
    render.path_trace();
    cudaDeviceSynchronize();
    std::clog << "finish.\n";
    // render.host_path_state_buffer = render.dev_path_state_buffer;
    // for (auto p : render.host_path_state_buffer) {
    //     using std::cout;
    //     cout << p.ray.position << ' ';
    //     cout << p.ray.direction << ' ';
    //     cout << p.ray.position + p.ray.direction << '\n';
    // }
}