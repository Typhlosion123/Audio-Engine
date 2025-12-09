#pragma once

#include <cuda_runtime.h>

struct Ray
{
    /* data */
    float3 origin;
    float3 direction;
};

struct RayState {
    float energy; // starts aat 1.0, decreases with each bounce
    float time_delay; // total distance traveled / speed of sound
    bool is_active; //false if it hits the sky or energy < threshold
};

struct AABB {
    float3 min; // Bottom-left-back corner
    float3 max; // Top-right-front corner
};

struct HitInfo {
    float t;          // Distance to hit
    float3 normal;    // Surface normal (for bouncing)
    bool hit;         // Did we hit anything?
};
