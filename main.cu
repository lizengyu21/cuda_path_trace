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

    Render render;

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[0].color = make_float3(1.0, 0.1, 0.1);
    // render.host_material_buffer[0].has_reflective = 1.0f;
    // render.host_material_buffer[0].specular.color = make_float3(1, 0, 0);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[1].color = make_float3(0.1, 1.0, 0.1);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[2].color = make_float3(0.1, 0.1f, 1);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[3].emittance = 16.0f;
    render.host_material_buffer[3].color = make_float3(1, 1, 1);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[4].color = make_float3(0.7, 0.7f, 0.5);
    render.host_material_buffer[4].specular.color = make_float3(1, 1, 1);
    render.host_material_buffer[4].has_reflective = 0.01;

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[5].emittance = 6.0f;
    render.host_material_buffer[5].color = make_float3(1, 1, 1);
    
    std::vector<Sphere> spheres;
    // spheres.push_back(Sphere(make_float3(0, 0, -15), 3, 0));
    // spheres.push_back(Sphere(make_float3(6, 0, -15), 3, 1));
    // spheres.push_back(Sphere(make_float3(12, 0, -15), 3, 2));
    // spheres.push_back(Sphere(make_float3(0, 7, -14), 3, 3));
    // spheres.push_back(Sphere(make_float3(6, -7, -14), 3, 3));
    // spheres.push_back(Sphere(make_float3(0, -1000, 0), 993, 4));
    // spheres.push_back(Sphere(make_float3(0, 0, 0), 1000, 5));

    
    render.bvh = BVH(spheres.begin(), spheres.end());
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