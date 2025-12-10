#pragma once

#include <cuda_runtime.h>

#define MAX_BOUNCES 50 
#define MIN_ENERGY 0.001f

#define OBJ_SPHERE 0
#define OBJ_BOX 1

struct Ray
{
    /* data */
    float3 origin;
    float3 direction;
};


struct AABB {
    float3 min; // Bottom-left-back corner
    float3 max; // Top-right-front corner
};

struct Object {
    int type;
    float3 param1; //if it is a box, will be the min corner. Otherwise its the sphere orogin
    float3 param2; //if it is a bhox, max corner, otherwise its the spehre radius
};

struct SceneHeader {
    AABB room;
    float3 source;
    float3 listener;
    float listener_r;
    int num_objects;
};


