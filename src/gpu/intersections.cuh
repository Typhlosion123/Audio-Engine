#pragma once
#include <cuda_runtime.h>
#include <types.h>

#define MAX_BOUNCES 50 
#define MIN_ENERGY 0.001f

/*
--4 inputs, The ray, the AABB box that checks for intersection,
t_out = distance to hit, n_out = normal vector 
*/
__device__  bool intersectBox(const Ray& r, const AABB& box, float& t_out, float3& normal_out) {
    float3 invDir = make_float3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);

    //x plan intersection of AABB box

    float t1 = (box.min.x - r.origin.x) * invDir.x;
    float t2 = (box.max.x - r.origin.x) * invDir.x;
    float tmin = fminf(t1, t2);
    float tmax = fmaxf(t1, t2);

    //y plane intersection
    float t3 = (box.min.y - r.origin.y) * invDir.y;
    float t4 = (box.max.y - r.origin.y) * invDir.y;
    tmin = fmaxf(tmin, fminf(t3, t4));
    tmax = fminf(tmax, fmaxf(t3, t4));

    // Calculate intersection with Z planes
    float t5 = (box.min.z - r.origin.z) * invDir.z;
    float t6 = (box.max.z - r.origin.z) * invDir.z;
    tmin = fmaxf(tmin, fminf(t5, t6));
    tmax = fminf(tmax, fmaxf(t5, t6));

    if (tmax < 0 || tmin > tmax) {
        return false;
    }

    t_out = tmax;

    float3 hitPoint;
    hitPoint.x = r.origin.x + tmax * r.direction.x;
    hitPoint.y = r.origin.y + tmax * r.direction.y;
    hitPoint.z = r.origin.z + tmax * r.direction.z;


    float eps = 1e-3f;
    if (fabsf(hitPoint.x - box.max.x) < eps)      normal_out = make_float3(-1, 0, 0); // Hit Right Wall
    else if (fabsf(hitPoint.x - box.min.x) < eps) normal_out = make_float3(1, 0, 0);  // Hit Left Wall
    else if (fabsf(hitPoint.y - box.max.y) < eps) normal_out = make_float3(0, -1, 0); // Hit Ceiling
    else if (fabsf(hitPoint.y - box.min.y) < eps) normal_out = make_float3(0, 1, 0);  // Hit Floor
    else if (fabsf(hitPoint.z - box.max.z) < eps) normal_out = make_float3(0, 0, -1); // Hit Front
    else                                          normal_out = make_float3(0, 0, 1);  // Hit Back

    return true;
} 


/*
Helper to find the new angle after bouncing off a wall / material

Equation for reflection is i - 2 * (i dot n) * n

params: i input vector, n normal vector (unit vector)
*/

__device__ float3 reflect(float3 i, float3 n) {
    float dot = i.x * n.x + i.y * n.y + i.z * n.z;

    return make_float3(
        i.x - 2.0f * dot * n.x,
        i.y - 2.0f * dot * n.y,
        i.z - 2.0f * dot * n.z
    );
}

__device__ bool intersectSphere(float3 origin, float3 direction, float max_dist, float3 center, float radius) {
    float3 L = make_float3(center.x - origin.x, center.y - origin.y, center.z - origin.z);
    float tca = L.x * direction.x + L.y * direction.y + L.z * direction.z;
    if (tca < 0) return false;

    float d2 = (L.x * L.x + L.y * L.y + L.z * L.z) - (tca * tca);
    float r2 = radius * radius;

    if (d2 > r2) return false;

    float thc = sqrtf(r2 - d2);
    float t0 = tca - thc; 
    
    if (t0 > 0 && t0 < max_dist) return true;

    return false;
}