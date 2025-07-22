#ifndef NBODY_CUDA_KERNELS_H
#define NBODY_CUDA_KERNELS_H

#include "common_structs.h"
#include "quadtree_gpu.h"

void set_cuda_constants(float G_val, float THETA_val, float SOFTENING_val);

__global__ void calculateForceKernel(
    Particle* d_particles, QuadtreeNodeGPU* d_nodes, int num_particles
);

__global__ void updateParticlesKernel(
    Particle* d_particles, int num_particles, float DT_val
);

#endif
