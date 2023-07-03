#pragma once
#include <math.h>
#include <iostream>
#include "math_utils.cuh"
#include "util.cuh"

struct CameraSetting
{
    int pixel_vertical_length = 756;
    float aperture = 0;
    float3 look_from = make_float3(0, 10, 0);
    float3 look_at = make_float3(0, 0, -15);
    float3 up = make_float3(0, 1.0f, 0);
    float fovY = 90.0f;
    float focal_length = 2.0f;
    float aspect_ratio = 16.0f / 9.0f;
    void set_fovY(float _fovY) { fovY = _fovY; }
    void set_focal_length(float _focal_length) { focal_length = _focal_length; }
    void set_aspect_ratio(float _aspect_ratio) { aspect_ratio = _aspect_ratio; }
    void set_pixels(int max_y) { pixel_vertical_length = max_y; }
};

class Camera {
public:
    int pixel_horizontal_length, pixel_vertical_length; // pixels
    float radius;
    float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
    Camera() {}
    Camera(const CameraSetting &camera_setting) {
        this->radius = camera_setting.aperture / 2.0f;
        this->pixel_horizontal_length = camera_setting.pixel_vertical_length * camera_setting.aspect_ratio;
        this->pixel_vertical_length = camera_setting.pixel_vertical_length;
        this->origin = camera_setting.look_from;
        float view_height = camera_setting.focal_length * tan(degree2radian(camera_setting.fovY / 2.0f)) * 2.0f;
        float view_width = view_height * camera_setting.aspect_ratio;
        // Camera coordinates
        float3 w = unit(camera_setting.look_from - camera_setting.look_at);
        float3 u = unit(cross(camera_setting.up, w));
        float3 v = cross(w, u);
        this->horizontal = view_width * u;
        this->vertical = view_height * v;
        this->lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - camera_setting.focal_length * w;
    }
};