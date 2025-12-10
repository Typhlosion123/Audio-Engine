#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "types.h"
#include "intersections.cuh"

curandState* d_states = nullptr;
Object* d_objects = nullptr;
int g_num_objects = 0;

__global__ void initRandKernel(curandState* states, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(1234, idx, 0, &states[idx]);
}

__global__ void genRaysKernel(Ray* rays, curandState* states, float3* paths, int* hits, int n, 
                              SceneHeader scene, Object* objects) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curandState localState = states[idx];

        float theta = curand_uniform(&localState) * 2.0f * 3.14159f;
        float phi = acosf(1.0f - 2.0f * curand_uniform(&localState));

        Ray r;
        r.origin = scene.source;
        r.direction = make_float3(sinf(phi)*cosf(theta), sinf(phi)*sinf(theta), cosf(phi));

        float energy = 1.0f;
        bool has_hit_listener = false;

        int stride = MAX_BOUNCES + 1;
        int path_base_idx = idx * stride;
        paths[path_base_idx] = r.origin;

        int b = 1;
        while (energy > MIN_ENERGY && b <= MAX_BOUNCES) {
            float t_closest = 1e9f;
            float3 n_closest;
            bool hit_something = false;

            float t_wall; float3 n_wall;
            if (intersectAABB(r, scene.room.min, scene.room.max, t_wall, n_wall, true)) {
                if (t_wall < t_closest) {
                    t_closest = t_wall;
                    n_closest = n_wall;
                    hit_something = true;
                }
            }

            for (int i = 0; i < scene.num_objects; i++) {
                float t_obj; float3 n_obj;
                if (intersectObject(r, objects[i], t_obj, n_obj)) {
                    if (t_obj < t_closest) {
                        t_closest = t_obj;
                        n_closest = n_obj;
                        hit_something = true;
                    }
                }
            }

            if (hit_something) {
                if (intersectGhostSphere(r.origin, r.direction, t_closest, scene.listener, scene.listener_r)) {
                    has_hit_listener = true;
                }

                r.origin.x += r.direction.x * t_closest;
                r.origin.y += r.direction.y * t_closest;
                r.origin.z += r.direction.z * t_closest;

                energy *= 0.8f; 
                r.direction = reflect(r.direction, n_closest);
                paths[path_base_idx + b] = r.origin;
            } else {
                paths[path_base_idx + b] = r.origin;
                energy = 0.0f;
            }
            b++;
        } 
        
        while (b <= MAX_BOUNCES) { paths[path_base_idx + b] = r.origin; b++; }

        states[idx] = localState;
        if (hits != nullptr) hits[idx] = has_hit_listener ? 1 : 0;
    }
}

extern "C" void initRandom(int n_rays) {
    if (d_states == nullptr) cudaMalloc(&d_states, n_rays * sizeof(curandState));
    int blockSize = 256;
    int numBlocks = (n_rays + blockSize - 1) / blockSize;
    initRandKernel<<<numBlocks, blockSize>>>(d_states, n_rays);
    cudaDeviceSynchronize();
}

extern "C" void generateRays(Ray* d_rays, float3* d_paths, int* d_hits, int n_rays, 
                             SceneHeader scene, Object* host_objects) {
    
    Object* d_scene_objects;
    if (scene.num_objects > 0) {
        cudaMalloc(&d_scene_objects, scene.num_objects * sizeof(Object));
        cudaMemcpy(d_scene_objects, host_objects, scene.num_objects * sizeof(Object), cudaMemcpyHostToDevice);
    } else {
        d_scene_objects = nullptr;
    }

    int blockSize = 256;
    int numBlocks = (n_rays + blockSize - 1) / blockSize;
    
    genRaysKernel<<<numBlocks, blockSize>>>(d_rays, d_states, d_paths, d_hits, n_rays, scene, d_scene_objects);
    
    cudaDeviceSynchronize();
    if (scene.num_objects > 0) cudaFree(d_scene_objects);
}