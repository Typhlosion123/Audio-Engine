#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "types.h"
#include "intersections.cuh"

curandState* d_states = nullptr;

__global__ void initRandKernel(curandState* states, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(1234, idx, 0, &states[idx]);
    }
}

__global__ void genRaysKernel(Ray* rays, curandState* states, float3* paths, int* hits, int n, float3 source_pos, float3 listener_pos, float listener_radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curandState localState = states[idx];

        float theta = curand_uniform(&localState) * 2.0f * 3.14159f;
        float phi = acosf(1.0f - 2.0f * curand_uniform(&localState));

        Ray r;
        r.origin = source_pos;
        r.direction = make_float3(sinf(phi)*cosf(theta), sinf(phi)*sinf(theta), cosf(phi));

        float energy = 1.0f;

        bool has_hit_listener = false;

        AABB room;
        room.min = make_float3(-10, -20, -10);
        room.max = make_float3(10, 20, 10);

        int path_base_idx = idx * (MAX_BOUNCES + 1);
        paths[path_base_idx] = r.origin;

        int b = 1;
        while (energy > MIN_ENERGY && b <= MAX_BOUNCES) {
            float t_wall;
            float3 normal;

            bool hit_wall = intersectBox(r, room, t_wall, normal);

            if (hit_wall) {
                if (intersectSphere(r.origin, r.direction, t_wall, listener_pos, listener_radius)) {
                    has_hit_listener = true;
                }

                r.origin.x += r.direction.x * t_wall;
                r.origin.y += r.direction.y * t_wall;
                r.origin.z += r.direction.z * t_wall;

                energy *= exp(-0.01f * t_wall); //air decay
                energy *= 0.8f; //bounce decay ~20%

                r.direction = reflect(r.direction, normal);

                paths[path_base_idx + b] = r.origin;
            } else {
                paths[path_base_idx + b] = r.origin;
                energy = 0.0f;
            }
            b++;
        } 

        while (b <= MAX_BOUNCES) { // fill in the rest of our memeory
            paths[path_base_idx + b] = r.origin;
            b++;
        }

        states[idx] = localState;
        rays[idx] = r;

        if (hits != nullptr) {
            hits[idx] = has_hit_listener ? 1 : 0;
        }

    }
}

extern "C" void initRandom(int n_rays) {
    if (d_states == nullptr) {
        cudaMalloc(&d_states, n_rays * sizeof(curandState));
    }
    
    int blockSize = 256;
    int numBlocks = (n_rays + blockSize - 1) / blockSize;
    
    initRandKernel<<<numBlocks, blockSize>>>(d_states, n_rays);
    cudaDeviceSynchronize();
}

extern "C" void generateRays(Ray* d_rays, float3* d_paths, int* d_hits, int n_rays, float3 source, float3 listener) {
    int blockSize = 256;
    int numBlocks = (n_rays + blockSize - 1) / blockSize;
    float listener_radius = 2.0f;

    genRaysKernel<<<numBlocks, blockSize>>>(d_rays, d_states, d_paths, d_hits, n_rays, source, listener, listener_radius);
    
    cudaDeviceSynchronize();
}