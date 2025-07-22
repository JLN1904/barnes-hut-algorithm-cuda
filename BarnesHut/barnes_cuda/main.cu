#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>
#include <cuda_runtime.h>
#include "common_structs.h"
#include "plummer.h"
#include "quadtree_gpu.h"
#include "nbody_cuda_kernels.h"

// Constantes de simulación
const double G = 100;
const double THETA = 0.5;
const double DT = 0.15;
const double SOFTENING = 1e-2;
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N[1] = {40000}; // Diferentes tamaños de simulación
    for (int n : N) {
        std::cout << "Iniciando simulación N-Cuerpos con Barnes-Hut (CUDA) para " << n << " partículas..." << std::endl;
        std::vector<double> times; // Para almacenar tiempos de simulación
        for(int rep=0; rep<10; rep++){
            // Genera partículas en host
            Vector2D cluster1_pos = {-1.0e2, 0.0};
            Vector2D cluster1_vel = {2.0, 0.0};
            std::vector<Particle> particles_1 = generate_plummer_cluster(n/2, n, 5.0e2, cluster1_pos, cluster1_vel);

            Vector2D cluster2_pos = {1.0e2, 0.0};
            Vector2D cluster2_vel = {-2.0, 0.0};
            std::vector<Particle> particles_2 = generate_plummer_cluster(n/2, n, 5.0e2, cluster2_pos, cluster2_vel);
            // Combina las partículas de ambos clusters
            std::vector<Particle> particles = particles_1;
            particles.insert(particles.end(), particles_2.begin(), particles_2.end());
            //std::cout << "Particula 1: " << particles[0].position.x << ", " << particles[0].position.y << " Velocidad: " << particles[0].velocity.x << ", " << particles[0].velocity.y << std::endl;
            //std::cout << "Particula " << n << ": " << particles[n-1].position.x << ", " << particles[n-1].position.y << " Velocidad: " << particles[n-1].velocity.x << ", " << particles[n-1].velocity.y << std::endl;
            // Convierte a arreglos planos para GPU
            std::vector<float> h_x(n), h_y(n), h_mass(n);
            

            // Reserva y copia a GPU
            float *d_x, *d_y, *d_mass;
            checkCudaError(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc d_x");
            checkCudaError(cudaMalloc(&d_y, n * sizeof(float)), "cudaMalloc d_y");
            checkCudaError(cudaMalloc(&d_mass, n * sizeof(float)), "cudaMalloc d_mass");
            

            // Reserva nodos del árbol en GPU
            int max_depth = 10;
            int max_nodes = (pow(4, max_depth + 1) - 1) / 3;

            QuadtreeNodeGPU* d_nodes;
            checkCudaError(cudaMalloc(&d_nodes, max_nodes * sizeof(QuadtreeNodeGPU)), "cudaMalloc d_nodes");

            

            // Parámetros del árbol y construcción en GPU
            QuadtreeParams params = {max_depth, 8};
            
            
            
            // Reserva partículas en GPU (estructura completa)
            Particle* d_particles;
            checkCudaError(cudaMalloc(&d_particles, n * sizeof(Particle)), "cudaMalloc d_particles");
            checkCudaError(cudaMemcpy(d_particles, particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice), "cudaMemcpy H2D d_particles");

            // Establece constantes en GPU
            set_cuda_constants(G, THETA, SOFTENING);

            // Simulación principal
            std::vector<std::vector<float>> snapshots_x;
            std::vector<std::vector<float>> snapshots_y;
            int numSteps = 1000; // Número de pasos de simulación
            int blockSize = 256;
            int numBlocks = (n + blockSize - 1) / blockSize;
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int step = 0; step < numSteps; ++step) {
                
                // Copia posiciones y masas actuales de GPU a host
                cudaMemcpy(particles.data(), d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
                for (int i = 0; i < n; ++i) {
                    //printf("Partícula %d: Posición=(%.2f, %.2f), Masa=%.2f\n", i + 1, particles[i].position.x, particles[i].position.y, particles[i].mass);
                    h_x[i] = static_cast<float>(particles[i].position.x);
                    h_y[i] = static_cast<float>(particles[i].position.y);
                    h_mass[i] = static_cast<float>(particles[i].mass);
                    //printf("Partícula post casteo %d: Posición=(%.2f, %.2f), Masa=%.2f\n", i + 1, h_x[i], h_y[i], h_mass[i]);
                }
                
                // Copia posiciones y masas actualizadas a GPU (arreglos planos)
                checkCudaError(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D d_x");
                checkCudaError(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D d_y");
                checkCudaError(cudaMemcpy(d_mass, h_mass.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D d_mass");

                // Calcula bounding box global en host
                float minX = *std::min_element(h_x.begin(), h_x.end());
                float maxX = *std::max_element(h_x.begin(), h_x.end());
                float minY = *std::min_element(h_y.begin(), h_y.end());
                float maxY = *std::max_element(h_y.begin(), h_y.end());
                //printf("Bounding box: minX=%.2f, maxX=%.2f, minY=%.2f, maxY=%.2f\n", minX, maxX, minY, maxY);

                // Reconstruye el quadtree en GPU
                build_quadtree_gpu(d_x, d_y, d_mass, n, d_nodes, max_nodes, params, minX, minY, maxX, maxY);

                // Calcula fuerzas y actualiza partículas
                calculateForceKernel<<<numBlocks, blockSize>>>(d_particles, d_nodes, n);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA error after calculateForceKernel: " << cudaGetErrorString(err) << std::endl;
                    break;
                }
                updateParticlesKernel<<<numBlocks, blockSize>>>(d_particles, n, DT);
                cudaDeviceSynchronize();
                
                /*
                if(step %100 == 99 || step == 0) {
                    std::cout << "Paso " << step + 1 << " de " << numSteps << "..." << std::endl;
                    cudaMemcpy(particles.data(), d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
                    std::cout << "Posición final Partícula 1: (" << particles[0].position.x << ", " << particles[0].position.y << ") Velocidad: (" << particles[0].velocity.x << ", " << particles[0].velocity.y << ")" << std::endl;
                    std::cout << "Posición final Partícula " << n << ": (" << particles[n-1].position.x << ", " << particles[n-1].position.y << ") Velocidad: (" << particles[n-1].velocity.x << ", " << particles[n-1].velocity.y << ")" << std::endl;
                }*/
            }
            // Calcula tiempo de simulación
            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0; // Convertir a segundos
            times.push_back(elapsed_time);

            // Copia resultados finales a host
            cudaMemcpy(particles.data(), d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);

            // Libera memoria
            cudaFree(d_x); cudaFree(d_y); cudaFree(d_mass);
            cudaFree(d_nodes); cudaFree(d_particles);
        }
        double average_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double standard_deviation = std::sqrt(std::accumulate(times.begin(), times.end(), 0.0, 
            [average_time](double sum, double time) { return sum + (time - average_time) * (time - average_time); }) / times.size());
        std::cout << "Tiempo promedio de simulación para " << n << " partículas: " << average_time << " segundos, con desviacion estandar: " << standard_deviation << " segundos."  << std::endl;
    }
    

    std::cout << "Simulación finalizada." << std::endl;
    return 0;
}
