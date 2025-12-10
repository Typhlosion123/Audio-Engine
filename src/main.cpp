#include <iostream>
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include "types.h"

const int N_RAYS = 250;
const int MAX_BOUNCES = 50;

extern "C" void initRandom(int n_rays);
extern "C" void generateRays(Ray* d_rays, float3* d_paths, int* d_hits, int n_rays, float3 source, float3 listener);

int main() {
    std::cout << "=== Starting Audio Ray Tracer (Energy Mode) ===" << std::endl;

    Ray* d_rays;
    cudaMalloc(&d_rays, N_RAYS * sizeof(Ray));

    int path_size = N_RAYS * (MAX_BOUNCES + 1);
    float3* d_paths;
    cudaMalloc(&d_paths, path_size * sizeof(float3));

    int* d_hits;
    cudaMalloc(&d_hits, N_RAYS * sizeof(int)); 

    float3 source_pos = make_float3(0.0f, -10.0f, 0.0f);   
    float3 listener_pos = make_float3(0.0f, 10.0f, 2.5f); 

    std::cout << "Initializing..." << std::endl;
    initRandom(N_RAYS);

    std::cout << "Simulating (Source: " << source_pos.x << ", " << source_pos.z << ")..." << std::endl;
    generateRays(d_rays, d_paths, d_hits, N_RAYS, source_pos, listener_pos);

    std::vector<float3> host_paths(path_size);
    cudaMemcpy(host_paths.data(), d_paths, path_size * sizeof(float3), cudaMemcpyDeviceToHost);

    std::vector<int> host_hits(N_RAYS);
    cudaMemcpy(host_hits.data(), d_hits, N_RAYS * sizeof(int), cudaMemcpyDeviceToHost);

    FILE* f = fopen("paths.csv", "w");
    if (f) {
        fprintf(f, "ray_id,bounce_id,x,y,z,hit_listener\n");
        for (int i = 0; i < N_RAYS; i++) {
            int hit_status = host_hits[i];
            for (int b = 0; b <= MAX_BOUNCES; b++) {
                int idx = i * (MAX_BOUNCES + 1) + b;
                float3 p = host_paths[idx];
                fprintf(f, "%d,%d,%f,%f,%f,%d\n", i, b, p.x, p.y, p.z, hit_status);
            }
        }
        fclose(f);
        std::cout << "Saved to paths.csv" << std::endl;
    }

    cudaFree(d_rays);
    cudaFree(d_paths);
    cudaFree(d_hits);
    
    return 0;
}