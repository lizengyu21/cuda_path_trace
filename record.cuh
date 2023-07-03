#pragma once

struct HitRecord
{
    bool missed = true;
    float t = 3.4e+38;
    unsigned int material_index = 0xFFFFFFFF;
    float3 position;
    float3 normal;
};