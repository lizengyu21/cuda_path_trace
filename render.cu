#include "render.cuh"
#include <assert.h>
#include <iomanip>
#include "rand.cuh"

struct split_by_completed {
    __host__ __device__ bool operator() (const PathState &segment) {
        return segment.remaining_iteration > 0;
    }
};


class BVH;

__global__ void compute_intersections(
    int total_path_count,
    bool is_empty,
    PathState *dev_path_state_buffer,
    HitRecord *dev_hit_record_buffer,
    DeviceBVH device_bvh) {
    int path_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_index < total_path_count) {
        if (is_empty) return;
        PathState state = dev_path_state_buffer[path_index];
        cast_ray(state, dev_hit_record_buffer[path_index], device_bvh);
    }
    __syncthreads();
}

__global__ void gather(const unsigned int path_count, const PathState *path_state, float3 *image) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < path_count) {
        image[path_state[index].pixel_index] = image[path_state[index].pixel_index] + path_state[index].result;
    }
    __syncthreads();
}




__global__ void shade_material(const int path_count, PathState *path_state, HitRecord *records, Material *materials, const int iter) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < path_count && path_state[index].remaining_iteration > 0) {
        HitRecord hit_record = records[index];
        if (hit_record.missed) {
            path_state[index].attenuation = make_float3(0.0, 0.0, 0.0);
            path_state[index].remaining_iteration = 0;
        } else {
            Material material = materials[hit_record.material_index];
            thrust::default_random_engine rng;
            rng = make_seeded_random_engine(iter, index, path_state[index].remaining_iteration);
            material.shade(path_state[index], hit_record, rng);
        }
    }
    
    __syncthreads();
}

__global__ void generate_ray_from_camera(PathState *dev_path_state_buffer, Camera camera, int trace_depth, int iter) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < camera.pixel_horizontal_length && y < camera.pixel_vertical_length) {
        int index = x + y * camera.pixel_horizontal_length;
        PathState &state = dev_path_state_buffer[index];
        
        // calculat the ray params
        float x_offseted = x;
        float y_offseted = y;

        // anti_alias start
        thrust::default_random_engine rng = make_seeded_random_engine(iter, index, state.remaining_iteration);
        thrust::uniform_real_distribution<float> u_01(0.0f, 1.0f);
        x_offseted += u_01(rng);
        y_offseted += u_01(rng);
        // end
        const float u = (1.0f * x_offseted) / (float)camera.pixel_horizontal_length;
        const float v = (1.0f * y_offseted) / (float)camera.pixel_vertical_length;
        float3 o_offset = make_float3(0, 0, 0);
        if (camera.radius > 0.00001f) {
            float3 o_offset = random_on_unit_sphere(rng);
            o_offset = camera.radius * (o_offset.x * unit(camera.horizontal) + o_offset.y * unit(camera.vertical));
        }
        state.ray.position = camera.origin + o_offset;
        state.ray.direction = unit(camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin - o_offset);
        state.ray.direction_inverse = 1.0f / state.ray.direction;
        // set other params
        state.attenuation = make_float3(1, 1, 1);
        state.result = make_float3(0, 0, 0);
        state.pixel_index = index;
        state.remaining_iteration = trace_depth;
    }
}

static void write_color(std::ostream &out, float3 pixel_color) {
    // assert(pixel_color.x <= 1.0f || pixel_color.y <= 1.0f || pixel_color.z <= 1.0f);
    // assert(pixel_color.x >= 0.0f || pixel_color.y >= 0.0f || pixel_color.z >= 0.0f);
    out << static_cast<int>(255.999 * clamp(0, 1, pixel_color.x)) << ' '
        << static_cast<int>(255.999 * clamp(0, 1, pixel_color.y)) << ' '
        << static_cast<int>(255.999 * clamp(0, 1, pixel_color.z)) << '\n';
}

static void show_progress_bar(int now, int total) {
    std::clog << '\r';
    std::clog << std::setw(6) << std::fixed << std::setprecision(2) << (float) now / total * 100.0f << "%";
}

void Render::path_trace() {
    using std::clog;
    dim3 threads_per_block2d(8, 8);
    const unsigned int pixel_count = camera.pixel_horizontal_length * camera.pixel_vertical_length;
    dim3 blocks_per_grid2d((camera.pixel_horizontal_length + threads_per_block2d.x - 1) / threads_per_block2d.x, (camera.pixel_vertical_length + threads_per_block2d.y - 1) / threads_per_block2d.y);
        
    int cur_depth = 0;
    int path_count = pixel_count;
    const unsigned int threads_per_block1d = 128;
    thrust::device_vector<float3> dev_image;
    thrust::host_vector<float3> host_image;
    dev_image.resize(pixel_count, make_float3(0, 0, 0));

    for (int i = 0; i < SPP; i++) {
        
        cur_depth = 0;
        generate_ray_from_camera<<<blocks_per_grid2d, threads_per_block2d>>>(thrust::raw_pointer_cast(dev_path_state_buffer.data()), camera, trace_depth, i);

        while (cur_depth < trace_depth) {
            dim3 blocks_per_grid1d((path_count + threads_per_block1d - 1) / threads_per_block1d);
            dev_hit_record_buffer.clear();
            dev_hit_record_buffer.resize(pixel_count, HitRecord());

            compute_intersections<<< blocks_per_grid1d, threads_per_block1d >>>(path_count, bvh.is_empty(), thrust::raw_pointer_cast(dev_path_state_buffer.data()), thrust::raw_pointer_cast(dev_hit_record_buffer.data()), bvh.get_dev_bvh());
            cudaDeviceSynchronize();

            shade_material<<< blocks_per_grid1d, threads_per_block1d >>>(path_count, thrust::raw_pointer_cast(dev_path_state_buffer.data()), thrust::raw_pointer_cast(dev_hit_record_buffer.data()), thrust::raw_pointer_cast(dev_material_buffer.data()), i);
            auto pivot = thrust::partition(thrust::device, dev_path_state_buffer.begin(), dev_path_state_buffer.begin() + path_count, split_by_completed());
            path_count = pivot - dev_path_state_buffer.begin();
            if (path_count < 1) cur_depth = trace_depth;
            ++cur_depth;
        }
        path_count = pixel_count;
        dim3 blocks_per_grid1d((path_count + threads_per_block1d - 1) / threads_per_block1d);
        gather<<< blocks_per_grid1d, threads_per_block1d >>>(path_count, dev_path_state_buffer.data().get(), thrust::raw_pointer_cast(dev_image.data()));
        
        show_progress_bar(i, SPP);
    }
    host_image = dev_image;
    std::cout << "P3\n" << camera.pixel_horizontal_length << ' ' << camera.pixel_vertical_length << "\n255\n";
    for (int j = camera.pixel_vertical_length - 1; j >= 0 ; --j)
        for (int i = 0; i < camera.pixel_horizontal_length; ++i) {
            int index = i + j * camera.pixel_horizontal_length;
            write_color(std::cout, host_image[index] / SPP);
        }
    // print_image(thrust::raw_pointer_cast(host_image));
}

void Render::print_image(const float3 *image) {
    std::cout << "P3\n" << camera.pixel_horizontal_length << ' ' << camera.pixel_vertical_length << "\n255\n";
    for (int j = 0; j < camera.pixel_vertical_length; ++j)
        for (int i = 0; i < camera.pixel_horizontal_length; ++i) {
            int index = i + j * camera.pixel_horizontal_length;
            write_color(std::cout, host_path_state_buffer[index].attenuation / SPP);
        }
}

void Render::init() {
    bvh.build();
    bvh.host_nodes = bvh.dev_nodes;
    bvh.host_aabbs = bvh.dev_aabbs;
    // for (int i = 0; i < bvh.host_nodes.size(); ++i) {
    //     std::clog << "node " << i << '\n';
    //     std::clog << bvh.host_nodes[i].parent << ' ' << bvh.host_nodes[i].left_child << ' ' << bvh.host_nodes[i].right_child << ' ' << bvh.host_nodes[i].object_index << '\n';
    //     std::clog << bvh.host_aabbs[i].lower << ' ' << bvh.host_aabbs[i].upper << '\n';
    // }
    dev_material_buffer = host_material_buffer;
    const int pixel_count = camera.pixel_vertical_length * camera.pixel_horizontal_length;
    host_path_state_buffer.resize(pixel_count);
    host_hit_record_buffer.resize(pixel_count);
    dev_path_state_buffer.resize(pixel_count);
    dev_hit_record_buffer.resize(pixel_count);
    // dev_path_state_buffer.alloc(pixel_count * sizeof(PathState));
    // dev_hit_record_buffer.alloc(pixel_count * sizeof(HitRecord));
}




