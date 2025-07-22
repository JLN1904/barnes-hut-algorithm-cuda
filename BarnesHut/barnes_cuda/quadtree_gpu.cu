#include "quadtree_gpu.h"
#include "common_structs.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Kernel para inicializar el nodo raíz del árbol en GPU
__global__ void init_root_kernel(
    QuadtreeNodeGPU* nodes,
    float minX, float minY, float maxX, float maxY,
    int n_particles
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nodes[0].minX = minX;
        nodes[0].minY = minY;
        nodes[0].maxX = maxX;
        nodes[0].maxY = maxY;
        nodes[0].start = 0;
        nodes[0].count = n_particles;
        for (int i = 0; i < 4; ++i) nodes[0].children[i] = -1;
        nodes[0].mass = 0.0f;
        nodes[0].centerX = 0.0f;
        nodes[0].centerY = 0.0f;
    }
}

// Inspirado en cdpQuadtree.cu de NVIDIA
template<int NUM_THREADS_PER_BLOCK>
__global__ void build_quadtree_kernel(
    QuadtreeNodeGPU* nodes,
    float* x, float* y, float* mass,
    int maxNodes,
    QuadtreeParams params,
    int nodeIdx_base,
    int depth
) {
    const int warpSize = 32;
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

    extern __shared__ int smem[];
    volatile int* s_num_pts[4];
    for (int i = 0; i < 4; ++i)
        s_num_pts[i] = (volatile int*)&smem[i * NUM_WARPS_PER_BLOCK];

    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int lane_mask_lt = (1 << lane_id) - 1;

    int nodeIdx = nodeIdx_base + blockIdx.x; // Este bloque procesa el nodo nodeIdx
    //if (threadIdx.x == 0) printf("Bloque %d: Procesando nodo %d en profundidad %d\n", blockIdx.x, nodeIdx, depth);
    if (nodeIdx >= maxNodes) return;
    QuadtreeNodeGPU &node = nodes[nodeIdx];
    int num_points = node.count;
    int start_idx = node.start;

    // Parada de recursión
    if (depth >= params.maxDepth || num_points <= params.minParticlesPerNode)
        return;

    float centerX = 0.5f * (node.minX + node.maxX);
    float centerY = 0.5f * (node.minY + node.maxY);

    int num_points_per_warp = max(warpSize, (num_points + NUM_WARPS_PER_BLOCK - 1) / NUM_WARPS_PER_BLOCK);
    int range_begin = start_idx + warp_id * num_points_per_warp;
    int range_end = min(range_begin + num_points_per_warp, start_idx + num_points);

    // Inicializar contadores
    if (lane_id == 0)
        for (int q = 0; q < 4; ++q) s_num_pts[q][warp_id] = 0;
    __syncthreads();

    // Contar partículas por cuadrante localmente por warp
    for (int i = range_begin + lane_id; __any_sync(0xffffffff, i < range_end); i += warpSize) {
        bool active = i < range_end;
        float px = active ? x[i] : 0.0f;
        float py = active ? y[i] : 0.0f;

        // Calcula el cuadrante
        int q = 0;
        if (px >= centerX) q += 1;
        if (py < centerY) q += 2;

        int ballot = __ballot_sync(0xffffffff, active && (q == 0));
        if (q == 0 && lane_id == 0) s_num_pts[0][warp_id] += __popc(ballot);

        ballot = __ballot_sync(0xffffffff, active && (q == 1));
        if (q == 1 && lane_id == 0) s_num_pts[1][warp_id] += __popc(ballot);
        
        ballot = __ballot_sync(0xffffffff, active && (q == 2));
        if (q == 2 && lane_id == 0) s_num_pts[2][warp_id] += __popc(ballot);

        ballot = __ballot_sync(0xffffffff, active && (q == 3));
        if (q == 3 && lane_id == 0) s_num_pts[3][warp_id] += __popc(ballot);
    }
    __syncthreads();

    // Scan global de conteos por cuadrante (del ejemplo de NVIDIA)
    if (warp_id < 4) {
        int num_pts = (lane_id < NUM_WARPS_PER_BLOCK) ? s_num_pts[warp_id][lane_id] : 0;
        for (int offset = 1; offset < NUM_WARPS_PER_BLOCK; offset <<= 1) {
            int n = __shfl_up_sync(0xffffffff, num_pts, offset, NUM_WARPS_PER_BLOCK);
            if (lane_id >= offset) num_pts += n;
        }
        if (lane_id < NUM_WARPS_PER_BLOCK) s_num_pts[warp_id][lane_id] = num_pts;
    }
    __syncthreads();

    // Acumulado global entre cuadrantes (scan entre warps)
    if (warp_id == 0) {
        int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK - 1];
        for (int q = 1; q < 4; ++q) {
            int tmp = s_num_pts[q][NUM_WARPS_PER_BLOCK - 1];
            if (lane_id < NUM_WARPS_PER_BLOCK)
                s_num_pts[q][lane_id] += sum;
            sum += tmp;
        }
    }
    __syncthreads();

    // Reordenar partículas según cuadrante
    float* out_x = x;
    float* out_y = y;
    float* out_mass = mass;
    for (int i = range_begin + lane_id; __any_sync(0xffffffff, i < range_end); i += warpSize) {
        bool active = i < range_end;
        float px = active ? x[i] : 0.0f;
        float py = active ? y[i] : 0.0f;
        float pm = active ? mass[i] : 0.0f;
        int q = 0;
        if (px >= centerX) q += 1;
        if (py < centerY) q += 2;
        int dest = node.start + (q == 0 ? 0 :
                                 q == 1 ? s_num_pts[0][NUM_WARPS_PER_BLOCK - 1] :
                                 q == 2 ? s_num_pts[1][NUM_WARPS_PER_BLOCK - 1] :
                                          s_num_pts[2][NUM_WARPS_PER_BLOCK - 1]);
        // offset local dentro del warp
        int offset = 0;
        for (int j = 0; j < warp_id; ++j)
            offset += s_num_pts[q][j];
        dest += offset;
        out_x[dest] = px; out_y[dest] = py; out_mass[dest] = pm;
    }
    __syncthreads();
    
    // Crear hijos y lanzar kernels recursivos
    if (threadIdx.x == NUM_THREADS_PER_BLOCK - 1) {
        int firstChildIdx = 4 * nodeIdx + 1;
        int starts[4], counts[4];
        starts[0] = node.start;
        for (int q = 1; q < 4; ++q)
            starts[q] = node.start + s_num_pts[q - 1][NUM_WARPS_PER_BLOCK - 1];
        for (int q = 0; q < 4; ++q) {
            counts[q] = s_num_pts[q][NUM_WARPS_PER_BLOCK - 1] - (q == 0 ? 0 : s_num_pts[q - 1][NUM_WARPS_PER_BLOCK - 1]);
            int childIdx = firstChildIdx + q;
            nodes[childIdx].minX = (q & 1) ? centerX : node.minX;
            nodes[childIdx].maxX = (q & 1) ? node.maxX : centerX;
            nodes[childIdx].minY = (q & 2) ? centerY : node.minY;
            nodes[childIdx].maxY = (q & 2) ? node.maxY : centerY;
            nodes[childIdx].start = starts[q];
            nodes[childIdx].count = counts[q];
            for (int c = 0; c < 4; ++c)
                nodes[childIdx].children[c] = -1;
        }
        int suma = 0;
        for (int q = 0; q < 4; ++q) {
            suma += counts[q];
        }
        //printf("Node %d at depth %d: total = %d\n", nodeIdx, depth, suma);
        // Si hay hijos con partículas, relanza el kernel para cada cuadrante válido
        build_quadtree_kernel<NUM_THREADS_PER_BLOCK>
            <<<4, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK * sizeof(int)>>>(
                nodes, x, y, mass, maxNodes, params, firstChildIdx, depth + 1
                );
        
    }
}


// Kernel para calcular masa y centro de masa
__global__ void compute_mass_kernel(
    QuadtreeNodeGPU* nodes,
    float* x, float* y, float* mass,
    int maxNodes
) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= maxNodes) return;
    QuadtreeNodeGPU& node = nodes[idx];
    if (node.count == 0) return;

    float m = 0.0f, cx = 0.0f, cy = 0.0f;
    for (int i = node.start; i < node.start + node.count; ++i) {
        m += mass[i];
        cx += x[i] * mass[i];
        cy += y[i] * mass[i];
    }
    if (m > 0) {
        node.mass = m;
        node.centerX = cx / m;
        node.centerY = cy / m;
    }
    //printf("Thread %d Node %d: mass = %f, center = (%f, %f)\n", threadIdx.x, idx, node.mass, node.centerX, node.centerY);
}

// Función para construir el quadtree en GPU
void build_quadtree_gpu(
    float* d_x, float* d_y, float* d_mass, int n,
    QuadtreeNodeGPU* d_nodes, int maxNodes,
    QuadtreeParams params,
    float minX, float minY, float maxX, float maxY
) {


    // Inicializa nodo raíz
    checkCudaError(cudaGetLastError(), "Before init_root_kernel");
    init_root_kernel<<<1,1>>>(d_nodes, minX, minY, maxX, maxY, n);
    checkCudaError(cudaGetLastError(), "After init_root_kernel");
    cudaDeviceSynchronize();

    // Lanza kernel recursivo para construir el árbol
    checkCudaError(cudaGetLastError(), "Before build_quadtree_kernel");
    build_quadtree_kernel<128>
        <<<1, 128, 4 * (128 / 32) * sizeof(int)>>>(
            d_nodes, d_x, d_y, d_mass, maxNodes, params, 0, 0
        );
    checkCudaError(cudaGetLastError(), "After build_quadtree_kernel");
    cudaDeviceSynchronize();

    // Calcula masas y centros de masa
    int blockSize = 128;
    int numBlocks = (maxNodes + blockSize - 1) / blockSize;
    checkCudaError(cudaGetLastError(), "Before compute_mass_kernel");
    compute_mass_kernel<<<numBlocks, blockSize>>>(d_nodes, d_x, d_y, d_mass, maxNodes);
    checkCudaError(cudaGetLastError(), "After compute_mass_kernel");
    cudaDeviceSynchronize();
}
