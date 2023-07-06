#include "obj_loader.cuh"

objl::Vector3 objl::math::CrossV3(const Vector3 a, const Vector3 b) {
    return Vector3(a.Y * b.Z - a.Z * b.Y,
                   a.Z * b.X - a.X * b.Z,
                   a.X * b.Y - a.Y * b.X);
}