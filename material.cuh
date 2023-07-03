#pragma once


struct Material
{
    float3 color = make_float3(0.1, 0.2, 0.3);
    struct {
        float exponent;
        float3 color;
    } specular;
    float has_reflective = 0.0f;
    float has_refractive = 0.0f;
    float emittance = 0.0f;
};

