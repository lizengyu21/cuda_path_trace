#include "camera.cuh"
#include <iostream>
#include <vector>
#include "render.cuh"
#include "material.cuh"

std::ostream &operator<<(std::ostream &out, float3 c) {
    out << c.x << ' ' << c.y << ' ' << c.z;
    return out;
}

void SCENE1() {
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
    render.host_material_buffer[0].albedo = make_float3(0.83f, 0.065f, 0.05f);
    // render.host_material_buffer[0].has_reflective = 1.0f;
    // render.host_material_buffer[0].specular.color = make_float3(1, 0, 0);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[1].albedo = make_float3(0.14f, 0.85f, 0.091f);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[2].albedo = make_float3(0.1, 0.1f, 1);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[3].emittance = emittance;
    render.host_material_buffer[3].albedo = make_float3(1, 1, 1);

    render.host_material_buffer.push_back(Material(MaterialType::METAL));
    render.host_material_buffer[4].roughness = roughness;

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[5].albedo = make_float3(0.6, 0.6, 0.6);
    
    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(make_float3(-2.5, -3, -11), 2, 4));
    // spheres.push_back(Sphere(make_float3(1000, 0, -15), 995, 1));
    // spheres.push_back(Sphere(make_float3(-1000, 0, -15), 995, 2));
    // spheres.push_back(Sphere(make_float3(0, 10, -14), 3, 3));
    // spheres.push_back(Sphere(make_float3(0, 0, -1000), 980, 5));
    // spheres.push_back(Sphere(make_float3(6, -7, -14), 3, 3));
    // spheres.push_back(Sphere(make_float3(0, -1000, 0), 993, 5));
    // spheres.push_back(Sphere(make_float3(0, 0, 0), 1000, 5));
    std::vector<Triangle> triangles;
    triangles.push_back(Triangle(make_float3(-5, -5, -15), make_float3(-5, 5, -15), make_float3(5, -5, -15), 5));
    triangles.push_back(Triangle(make_float3(5, 5, -15), make_float3(-5, 5, -15), make_float3(5, -5, -15), 5));
    triangles.push_back(Triangle(make_float3(-5, -5, -15), make_float3(-5, -5, -5), make_float3(-5, 5, -5), 0));
    triangles.push_back(Triangle(make_float3(-5, -5, -15), make_float3(-5, 5, -15), make_float3(-5, 5, -5), 0));
    triangles.push_back(Triangle(make_float3(5, -5, -15), make_float3(5, -5, -5), make_float3(5, 5, -5), 1));
    triangles.push_back(Triangle(make_float3(5, -5, -15), make_float3(5, 5, -15), make_float3(5, 5, -5), 1));

    triangles.push_back(Triangle(make_float3(-5, 5, -15), make_float3(-5, 5, -5), make_float3(5, 5, -15), 5));
    triangles.push_back(Triangle(make_float3(-5, 5, -5), make_float3(5, 5, -5), make_float3(5, 5, -15), 5));

    triangles.push_back(Triangle(make_float3(-5, -5, -15), make_float3(-5, -5, -5), make_float3(5, -5, -15), 5));
    triangles.push_back(Triangle(make_float3(-5, -5, -5), make_float3(5, -5, -5), make_float3(5, -5, -15), 5));
    
    triangles.push_back(Triangle(make_float3(-1, 4.995, -11), make_float3(-1, 4.995, -9), make_float3(1, 4.995, -11), 3));
    triangles.push_back(Triangle(make_float3(-1, 4.995, -9), make_float3(1, 4.995, -9), make_float3(1, 4.995, -11), 3));
    
    // triangles.push_back(Triangle(make_float3(5, 10, -13), make_float3(-5, 10, -13), make_float3(0, 5, -16), 3));
    // triangles.push_back(Triangle(make_float3(5, 0, -13), make_float3(-5, 0, -13), make_float3(0, 5, -16), 0));
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

void SCENE2() {
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
    render.host_material_buffer[0].albedo = make_float3(0.83f, 0.065f, 0.05f);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[1].albedo = make_float3(0.14f, 0.85f, 0.091f);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[2].albedo = make_float3(0.1, 0.1f, 1);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[3].emittance = emittance;
    render.host_material_buffer[3].albedo = make_float3(1, 1, 1);

    render.host_material_buffer.push_back(Material(MaterialType::METAL));
    render.host_material_buffer[4].roughness = roughness;

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[5].albedo = make_float3(0.6, 0.6, 0.6);
    
    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(make_float3(1000, 0, -15), 995, 1));
    spheres.push_back(Sphere(make_float3(-1000, 0, -15), 995, 2));
    spheres.push_back(Sphere(make_float3(0, 5, -15), 6, 3));
    spheres.push_back(Sphere(make_float3(0, 0, -1000), 980, 5));
    // spheres.push_back(Sphere(make_float3(6, -7, -14), 3, 3));
    spheres.push_back(Sphere(make_float3(0, -1000, 0), 993, 5));
    spheres.push_back(Sphere(make_float3(0, 0, 0), 1000, 5));
    std::vector<Triangle> triangles;
    // triangles.push_back(Triangle(make_float3(5, 10, -13), make_float3(-5, 10, -13), make_float3(0, 5, -16), 3));
    // triangles.push_back(Triangle(make_float3(5, 0, -13), make_float3(-5, 0, -13), make_float3(0, 5, -16), 0));
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

void SCENE3() {
    int SPP;
    float roughness;
    float emittance;
    float x = -5;
    float y = -5;
    float z = -15;
    float length = 10;
    std::clog << "SPP: ";
    std::cin >> SPP;
    std::clog << "roughness: ";
    std::cin >> roughness;
    std::clog << "emittance: ";
    std::cin >>emittance;

    std::clog << "x: ";
    std::cin >> x;
    std::clog << "y: ";
    std::cin >> y;
    std::clog << "z: ";
    std::cin >> z;
    std::clog << "length: ";
    std::cin >> length;

    Render render;
    render.SPP = SPP;
    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[0].albedo = make_float3(0.83f, 0.065f, 0.05f);
    // render.host_material_buffer[0].has_reflective = 1.0f;
    // render.host_material_buffer[0].specular.color = make_float3(1, 0, 0);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[1].albedo = make_float3(0.14f, 0.85f, 0.091f);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[2].albedo = make_float3(0.1, 0.1f, 1);

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[3].emittance = emittance;
    render.host_material_buffer[3].albedo = make_float3(1, 1, 1);

    render.host_material_buffer.push_back(Material(MaterialType::METAL));
    render.host_material_buffer[4].roughness = roughness;

    render.host_material_buffer.push_back(Material());
    render.host_material_buffer[5].albedo = make_float3(0.6, 0.6, 0.6);
    
    thrust::host_vector<Sphere> spheres;
    // spheres.push_back(Sphere(make_float3(-2.5, -3, -11), 2, 4));
    // spheres.push_back(Sphere(make_float3(1000, 0, -15), 995, 1));
    // spheres.push_back(Sphere(make_float3(-1000, 0, -15), 995, 2));
    // spheres.push_back(Sphere(make_float3(0, 10, -14), 3, 3));
    // spheres.push_back(Sphere(make_float3(0, 0, -1000), 980, 5));
    // spheres.push_back(Sphere(make_float3(6, -7, -14), 3, 3));
    // spheres.push_back(Sphere(make_float3(0, -1000, 0), 993, 5));
    // spheres.push_back(Sphere(make_float3(0, 0, 0), 1000, 5));
    thrust::host_vector<Triangle> triangles;
    MeshTriangle mesh("./modules/bunny.obj", 4);
    

    // back plane
    triangles.push_back(Triangle(make_float3(x, y, z), make_float3(x, y + length, z), make_float3(x + length, y, z), 5));
    triangles.push_back(Triangle(make_float3(x + length, y + length, z), make_float3(x, y + length, z), make_float3(x + length, y, z), 5));
    // left plane
    triangles.push_back(Triangle(make_float3(x, y, z), make_float3(x, y + length, z), make_float3(x, y, z + length), 0));
    triangles.push_back(Triangle(make_float3(x, y + length, z + length), make_float3(x, y + length, z), make_float3(x, y, z + length), 0));
    // right plane
    triangles.push_back(Triangle(make_float3(x + length, y, z), make_float3(x + length, y + length, z), make_float3(x + length, y, z + length), 1));
    triangles.push_back(Triangle(make_float3(x + length, y + length, z + length), make_float3(x + length, y + length, z), make_float3(x + length, y, z + length), 1));
    // bottom plane
    triangles.push_back(Triangle(make_float3(x, y, z), make_float3(x + length, y, z), make_float3(x, y, z + length), 5));
    triangles.push_back(Triangle(make_float3(x + length, y, z + length), make_float3(x + length, y, z), make_float3(x, y, z + length), 5));
    // top plane
    triangles.push_back(Triangle(make_float3(x, y + length, z), make_float3(x + length, y + length, z), make_float3(x, y + length, z + length), 5));
    triangles.push_back(Triangle(make_float3(x + length, y + length, z + length), make_float3(x + length, y + length, z), make_float3(x, y + length, z + length), 5));

    // light
    float offset = 0.4f * length;
    triangles.push_back(Triangle(make_float3(x + offset, y + 0.999f * length, z + offset), make_float3(x + length - offset, y + 0.999f * length, z + offset), make_float3(x + offset, y + 0.999f * length, z + length - offset), 3));
    triangles.push_back(Triangle(make_float3(x + length - offset, y + 0.999f * length, z + length - offset), make_float3(x + length - offset, y + 0.999f * length, z + offset), make_float3(x + offset, y + 0.999f * length, z + length - offset), 3));

    
    mesh.contact_to_whole(triangles);
    render.bvh = BVH(spheres.begin(), spheres.end(), triangles.begin(), triangles.end());
    CameraSetting camera_setting;

    camera_setting.look_from = make_float3(-1, 6.6, 15);
    camera_setting.look_at = make_float3(-1, 6.6, -10);

    camera_setting.set_aspect_ratio(16.0f / 9.0f);
    camera_setting.set_focal_length(10);
    render.camera = camera_setting;
    std::clog << "init.\n";
    render.init();
    std::clog << "start.\n";
    render.path_trace();
    cudaDeviceSynchronize();
    std::clog << "\nfinish.\n";
    // render.host_path_state_buffer = render.dev_path_state_buffer;
    // for (auto p : render.host_path_state_buffer) {
    //     using std::cout;
    //     cout << p.ray.position << ' ';
    //     cout << p.ray.direction << ' ';
    //     cout << p.ray.position + p.ray.direction << '\n';
    // }
}

int main() {


    int wich;
    std::cin >> wich;
    switch (wich)
    {
    case 1:
        SCENE1();
        break;
    case 2:
        SCENE2();
        break;
    case 3:
        SCENE3();
        break;
    default:
        break;
    }

}