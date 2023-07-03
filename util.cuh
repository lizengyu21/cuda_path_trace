#pragma once
#include <iostream>


// Naive error_handler
#define CHECK_CUDA_ERRORS(val) check_cuda((val), __FILE__, __LINE__, #val)
inline void check_cuda(cudaError_t result, char const *const file, unsigned int const line, char const *const func) {
    using std::cerr;
    using std::endl;
    if (result) {
        cerr << "CUDA Error= " << static_cast<unsigned int>(result) 
            << " at " << file << " : " << line << "'" << func << endl;
        cudaDeviceReset();
        exit(99);
    }
}

// std::ostream &operator<<(std::ostream &out, float3 c) {
//     out << c.x << ' ' << c.y << ' ' << c.z;
//     return out;
// }
