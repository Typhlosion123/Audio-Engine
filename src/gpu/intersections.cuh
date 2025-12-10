#pragma once
#include <cuda_runtime.h>
#include <types.h>


/*
Helper to find the new angle after bouncing off a wall / material

Equation for reflection is i - 2 * (i dot n) * n

params: i input vector, n normal vector (unit vector)
*/

__device__ float3 reflect(float3 i, float3 n) {
    float dot = i.x * n.x + i.y * n.y + i.z * n.z;
    return make_float3(i.x - 2.0f * dot * n.x, i.y - 2.0f * dot * n.y, i.z - 2.0f * dot * n.z);
}

__device__ bool intersectAABB(const Ray& r, float3 min, float3 max, float& t_out, float3& normal_out, bool inside) {
    float3 dir = r.direction;
    // Avoid NaN
    if (fabsf(dir.x) < 1e-6f) dir.x = 1e-6f;
    if (fabsf(dir.y) < 1e-6f) dir.y = 1e-6f;
    if (fabsf(dir.z) < 1e-6f) dir.z = 1e-6f;

    float3 invDir = make_float3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

    float t1 = (min.x - r.origin.x) * invDir.x;
    float t2 = (max.x - r.origin.x) * invDir.x;
    float tmin = fminf(t1, t2);
    float tmax = fmaxf(t1, t2);

    float t3 = (min.y - r.origin.y) * invDir.y;
    float t4 = (max.y - r.origin.y) * invDir.y;
    tmin = fmaxf(tmin, fminf(t3, t4));
    tmax = fminf(tmax, fmaxf(t3, t4));

    float t5 = (min.z - r.origin.z) * invDir.z;
    float t6 = (max.z - r.origin.z) * invDir.z;
    tmin = fmaxf(tmin, fminf(t5, t6));
    tmax = fminf(tmax, fmaxf(t5, t6));

    if (tmax < 0 || tmin > tmax) return false;

    // Inside (Room walls) vs Outside (Obstacle walls)
    float t_hit = inside ? tmax : tmin;
    
    // If obstacle is behind us (t_hit < 0), return false
    if (t_hit < 0.001f) return false;

    t_out = t_hit;

    float3 hitPoint;
    hitPoint.x = r.origin.x + t_hit * r.direction.x;
    hitPoint.y = r.origin.y + t_hit * r.direction.y;
    hitPoint.z = r.origin.z + t_hit * r.direction.z;

    float eps = 1e-3f;
    if (fabsf(hitPoint.x - max.x) < eps)      normal_out = make_float3(inside ? -1 : 1, 0, 0);
    else if (fabsf(hitPoint.x - min.x) < eps) normal_out = make_float3(inside ? 1 : -1, 0, 0);
    else if (fabsf(hitPoint.y - max.y) < eps) normal_out = make_float3(0, inside ? -1 : 1, 0);
    else if (fabsf(hitPoint.y - min.y) < eps) normal_out = make_float3(0, inside ? 1 : -1, 0);
    else if (fabsf(hitPoint.z - max.z) < eps) normal_out = make_float3(0, 0, inside ? -1 : 1);
    else                                      normal_out = make_float3(0, 0, inside ? 1 : -1);

    return true;
}


__device__ bool intersectSphereObj(const Ray& r, float3 center, float radius, float& t_out, float3& normal_out) {
    float3 oc = make_float3(r.origin.x - center.x, r.origin.y - center.y, r.origin.z - center.z);
    float b = oc.x * r.direction.x + oc.y * r.direction.y + oc.z * r.direction.z;
    float c = (oc.x * oc.x + oc.y * oc.y + oc.z * oc.z) - radius * radius;
    float disc = b * b - c;
    
    if (disc > 0) {
        float t = -b - sqrtf(disc);
        if (t > 0.001f) {
            t_out = t;
            float3 hitPoint = make_float3(r.origin.x + t*r.direction.x, r.origin.y + t*r.direction.y, r.origin.z + t*r.direction.z);
            normal_out = make_float3((hitPoint.x - center.x)/radius, (hitPoint.y - center.y)/radius, (hitPoint.z - center.z)/radius);
            return true;
        }
    }
    return false;
}

__device__ bool intersectObject(const Ray& r, const Object& obj, float& t_out, float3& normal_out) {
    if (obj.type == OBJ_SPHERE) {
        return intersectSphereObj(r, obj.param1, obj.param2.x, t_out, normal_out);
    } else if (obj.type == OBJ_BOX) {
        return intersectAABB(r, obj.param1, obj.param2, t_out, normal_out, false);
    }
    return false;
}


__device__ bool intersectGhostSphere(float3 origin, float3 dir, float max_dist, float3 center, float radius) {
    float3 L = make_float3(center.x - origin.x, center.y - origin.y, center.z - origin.z);
    float tca = L.x * dir.x + L.y * dir.y + L.z * dir.z;

    if (tca < 0) return false;
    float d2 = (L.x * L.x + L.y * L.y + L.z * L.z) - (tca * tca);

    if (d2 > radius * radius) return false;

    float thc = sqrtf(radius * radius - d2);
    float t0 = tca - thc;

    return (t0 > 0 && t0 < max_dist);
}