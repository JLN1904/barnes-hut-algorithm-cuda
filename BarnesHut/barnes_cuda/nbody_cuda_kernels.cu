#include "nbody_cuda_kernels.h"
#include <iostream>

__constant__ float G_const, THETA_const_sq, SOFTENING_const_sq, DT_const;

void set_cuda_constants(float G_val, float THETA_val, float SOFTENING_val) {
    cudaMemcpyToSymbol(G_const, &G_val, sizeof(float));
    float theta_sq = THETA_val * THETA_val;
    cudaMemcpyToSymbol(THETA_const_sq, &theta_sq, sizeof(float));
    float softening_sq = SOFTENING_val * SOFTENING_val;
    cudaMemcpyToSymbol(SOFTENING_const_sq, &softening_sq, sizeof(float));
}

__global__ void calculateForceKernel(Particle* d_particles, QuadtreeNodeGPU* d_nodes, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    Particle p = d_particles[idx];
    Vector2D accel = {0.0f, 0.0f};
    int stack[64], sp = 0;
    stack[sp++] = 0; // raíz

    while (sp > 0) {
        int node_idx = stack[--sp];
        QuadtreeNodeGPU node = d_nodes[node_idx];
        float dx = node.centerX - p.position.x;
        float dy = node.centerY - p.position.y;
        float distSq = dx * dx + dy * dy;
        bool is_leaf = true;
        for (int i = 0; i < 4; ++i)
            if (node.children[i] != -1) { is_leaf = false; break; }
        if (!is_leaf && (node.maxX - node.minX) * (node.maxY - node.minY) > distSq * THETA_const_sq) {
            for (int i = 0; i < 4; ++i)
                if (node.children[i] != -1)
                    stack[sp++] = node.children[i];
        } else {
            float dist = sqrtf(distSq + SOFTENING_const_sq);
            //printf("ELSE: idx=%d, node.mass=%f, dist=%f\n", idx, node.mass, dist);
            if (distSq > 0) {
                float dist = sqrtf(distSq + SOFTENING_const_sq);
                double force = (G_const * node.mass) / (dist * dist * dist);
                accel.x += force * dx;
                accel.y += force * dy;
                //printf("idx=%d, node_idx=%d, node.mass=%f, force=%e, dist=%f\n", idx, node_idx, node.mass, force, dist);
            }
        }
    }
    d_particles[idx].acceleration = accel;
    //if(idx == 0) printf("Particle %d: accel = (%f, %f)\n", idx+1, d_particles[idx].acceleration.x, d_particles[idx].acceleration.y);
}

__global__ void updateParticlesKernel(Particle* d_particles, int num_particles, float DT_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    d_particles[idx].velocity.x += d_particles[idx].acceleration.x * DT_val;
    d_particles[idx].velocity.y += d_particles[idx].acceleration.y * DT_val;
    d_particles[idx].position.x += d_particles[idx].velocity.x * DT_val;
    d_particles[idx].position.y += d_particles[idx].velocity.y * DT_val;
    //if(idx == 0) printf("Particle %d: vel = (%f, %f)\n", idx+1, d_particles[idx].velocity.x, d_particles[idx].velocity.y);
    //if(idx == 0) printf("Particle %d: pos = (%f, %f)\n", idx+1, d_particles[idx].position.x, d_particles[idx].position.y);
    //if(idx == num_particles - 1) printf("Particle %d: pos = (%f, %f)\n", idx+1, d_particles[idx].position.x, d_particles[idx].position.y);
    
}
