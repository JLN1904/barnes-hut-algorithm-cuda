#ifndef QUADTREE_GPU_H
#define QUADTREE_GPU_H

struct QuadtreeNodeGPU {
    float minX, minY, maxX, maxY;
    int start, end, count, id;
    int children[4]; // índices de hijos (-1 si no existe)
    float mass, centerX, centerY;
};

struct QuadtreeParams {
    int maxDepth;
    int minParticlesPerNode;
};

void build_quadtree_gpu(
    float* d_x, float* d_y, float* d_mass, int n,
    QuadtreeNodeGPU* d_nodes, int maxNodes,
    QuadtreeParams params,
    float minX, float minY, float maxX, float maxY
);

#endif
