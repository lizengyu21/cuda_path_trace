#include "bvh.cuh"

__host__ DeviceBVH BVH::get_dev_bvh() noexcept {
    return DeviceBVH{
        static_cast<unsigned int>(dev_nodes.size()),
        static_cast<unsigned int>(dev_spheres.size()),
        static_cast<unsigned int>(dev_triangles.size()),
        static_cast<unsigned int>(dev_spheres.size() + 0),
        thrust::raw_pointer_cast(dev_nodes.data()), 
        thrust::raw_pointer_cast(dev_aabbs.data()), 
        thrust::raw_pointer_cast(dev_spheres.data()),
        thrust::raw_pointer_cast(dev_triangles.data()),
    };
}

template <class MortonType>
void BVH::construct_internal_nodes(const DeviceBVH &self, const MortonType *morton, const unsigned int objects_count) {
    // build internal nodes parallel
    thrust::for_each(thrust::device, thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(objects_count - 1),
        [self, morton, objects_count] __device__ (const unsigned int index) {
            self.nodes[index].object_index = 0xFFFFFFFF;
            const uint2 range = calc_range(index, morton, objects_count);
            const int gamma = find_split_index(objects_count, range, morton);
            self.nodes[index].left_child = gamma;
            self.nodes[index].right_child = gamma + 1;
            if (thrust::min(range.x, range.y) == gamma)
                self.nodes[index].left_child += objects_count - 1; // bound to object
            if (thrust::max(range.x, range.y) == gamma + 1)
                self.nodes[index].right_child += objects_count - 1;
                self.nodes[self.nodes[index].left_child].parent = index;
                self.nodes[self.nodes[index].right_child].parent = index;
                return;
            }
        );
}

template <class MortonType>
__device__ uint2 calc_range(const unsigned int index, const MortonType *morton, const unsigned int leaves_count) {
    if (index == 0) {
        // root
        return make_uint2(0, leaves_count - 1);
    }
    MortonType self_morton = morton[index];
    // calculate the direction of range
    const int l_delta = common_prefix_bits_count(self_morton, morton[index - 1]);
    const int r_delta = common_prefix_bits_count(self_morton, morton[index + 1]);
    const int d = (l_delta > r_delta) ? -1 : 1;

    // determine the general range scope
    const int delta_min = common_prefix_bits_count(self_morton, morton[index - d]);
    int l_max = 2;
    int delta = -1;
    int temp_index = index + d * l_max;
    // boundary condition
    if (0 <= temp_index && temp_index < leaves_count)
        delta = common_prefix_bits_count(self_morton, morton[temp_index]);
    while (delta > delta_min) {
        l_max <<= 1;
        temp_index = index + d * l_max;
        if (temp_index < 0 || temp_index >= leaves_count) break;
        delta = common_prefix_bits_count(self_morton, morton[temp_index]);
    }
    // general scope is [index, temp_index] or [temp_index, index]
        
    int l = 0;
    int t = l_max >> 1;
    while (t > 0) {
        temp_index = index + (l + t) * d;
        if (0 <= temp_index && temp_index < leaves_count && (common_prefix_bits_count(self_morton, morton[temp_index]) > delta_min)) // range scope is not tight enough
            l = l + t;
        t >>= 1;
    }
    // now we get the tight scope [i, l * d)

    unsigned int j = index + l * d;
    return j > index ? make_uint2(index, j) : make_uint2(j, index);
}


template <class MortonType>
__device__ unsigned int find_split_index(const unsigned int leaves_count, const uint2 range, const MortonType *morton) {
    const MortonType l_morton = morton[range.x];
    const MortonType r_morton = morton[range.y];
    const int delta_node = common_prefix_bits_count(l_morton, r_morton);
    
    if (l_morton == r_morton)
        return (range.x + range.y) >> 1;
    // do binary search
    int split = range.x;
    int stride = range.y - range.x;
    do {
        stride = (stride + 1) >> 1;
        if (split + stride < range.y && common_prefix_bits_count(l_morton, morton[split + stride]) > delta_node)
            split = split + stride;
    } while (stride > 1);

    return split;
}

__device__ void cast_ray(const PathState &path_state, HitRecord &record, const DeviceBVH &self) {
    unsigned int stack[10000];
    int size = 0;
    stack[size++] = 0;
    while (size != 0) {
        int cur_index = stack[--size];
        if (self.aabbs[cur_index].intersect(path_state)) {
            // hit the aabb
            if (self.nodes[cur_index].left_child != 0xFFFFFFFF) stack[size++] = self.nodes[cur_index].left_child;
            if (self.nodes[cur_index].right_child != 0xFFFFFFFF) stack[size++] = self.nodes[cur_index].right_child;
            unsigned int object_index = self.nodes[cur_index].object_index;
            if (object_index != 0xFFFFFFFF) {
                // hit the leaves node
                // TODO:
                if (object_index < self.sphere_count)
                    self.spheres[object_index].intersect(path_state, record);
                else if (object_index < self.sphere_count + self.triangle_count)
                    self.triangles[object_index - self.sphere_count].intersect(path_state, record);
            }
        }
    }
}



void BVH::build() {
    // check correctness of objects
    assert(host_spheres.size() == dev_spheres.size());
    assert(host_triangles.size() == dev_triangles.size());

    if (host_spheres.size() == 0 && host_triangles.size() == 0) return;
    // record counts of all objects
    const unsigned int spheres_count = host_spheres.size();
    const unsigned int triangles_count = host_triangles.size();
    const unsigned int objects_count = spheres_count + triangles_count;
    const unsigned int internal_nodes_count = objects_count - 1;
    const unsigned int nodes_count = objects_count + internal_nodes_count;
    // construct defalut aabb data
    const auto inf = std::numeric_limits<float>::infinity();
    Aabb default_aabb(make_float3(inf, inf, inf), make_float3(-inf, -inf, -inf));
    dev_aabbs.resize(nodes_count, default_aabb);
    // create sphere aabbs in GPU
    thrust::transform(dev_spheres.begin(), dev_spheres.end(), dev_aabbs.begin() + internal_nodes_count, sphere_aabb_getter());
    thrust::transform(dev_triangles.begin(), dev_triangles.end(), dev_aabbs.begin() + internal_nodes_count + spheres_count, triangle_aabb_getter());

    const auto whole_aabb = thrust::reduce(dev_aabbs.begin() + internal_nodes_count, dev_aabbs.end(), default_aabb, aabb_merger());
    

    // calculate morton code
    thrust::device_vector<unsigned int> morton(objects_count);
    thrust::transform(dev_aabbs.begin() + internal_nodes_count, dev_aabbs.end(), morton.begin(), morton_code_calculator(whole_aabb));

    // sort aabbs in morton code indices
    // generate indexes from 0 to objects_num - 1
    thrust::device_vector<unsigned int> indexes(thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(objects_count));
    thrust::stable_sort_by_key(morton.begin(), morton.end(), thrust::make_zip_iterator(thrust::make_tuple(dev_aabbs.begin() + internal_nodes_count, indexes.begin())));

    bool is_unique_morton = (morton.size() == thrust::unique_count(morton.begin(), morton.end(), thrust::equal_to<unsigned int>()));
    thrust::device_vector<unsigned long long> morton64(objects_count);
    if (!is_unique_morton) {
        // has the same morton code
        // expand 32-bit morton code to 64-bit morton code
        thrust::transform(morton.begin(), morton.end(), indexes.begin(), morton64.begin(), expand_morton_64());
        morton.clear();
    } else {
        morton64.clear();
    }
    // initialize the leaf node bonding objects
    Node default_node;
    dev_nodes.resize(nodes_count, default_node);
    thrust::transform(indexes.begin(), indexes.end(), dev_nodes.begin() + internal_nodes_count, [] __device__ (const unsigned int index) {
        Node temp;
        temp.object_index = index;
        return temp;
    });

    // get raw pointers to compute internal nodes parallel
    auto self = this->get_dev_bvh();
    // construct internal nodes
    if (is_unique_morton) {
        const unsigned int *morton_codes = morton.data().get();
        this->construct_internal_nodes(self, morton_codes, objects_count);
    } else {
        const unsigned long long int *morton_codes = morton64.data().get();
        this->construct_internal_nodes(self, morton_codes, objects_count);
    }

    // create aabb
    // generate threads for every leaf node
    thrust::device_vector<unsigned int> dev_flags(internal_nodes_count, 0);
    auto flags = thrust::raw_pointer_cast(dev_flags.data());
    thrust::for_each(thrust::make_counting_iterator<unsigned int>(internal_nodes_count),
        thrust::make_counting_iterator<unsigned int>(nodes_count),
        [self, flags] __device__ (const unsigned int index) {
            unsigned int p = self.nodes[index].parent;
            while (p != 0xFFFFFFFF) {
                // the first thread visit the parent node, which enable the other child to visit the parent
                unsigned int old = atomicCAS(flags + p, 0, 1);
                if (old == 0) return;

                self.aabbs[p] = merge(self.aabbs[self.nodes[p].left_child], self.aabbs[self.nodes[p].right_child]);
                p = self.nodes[p].parent;
            }
        }
    );
}
