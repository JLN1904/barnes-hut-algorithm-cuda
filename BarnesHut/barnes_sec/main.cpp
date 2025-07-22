#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random> 
#include <numeric> 
#include <ctime>
#include <chrono>
#include "plummer.h"

// Constantes de la simulación
const double G = 100;
const double THETA = 0.5;
const double DT = 0.01;
const double SOFTENING = 1e-2;


class QuadtreeNode {
public:
    // Límites del nodo
    Vector2D center;
    float halfWidth;

    // Hijos del nodo 
    std::unique_ptr<QuadtreeNode> children[4] = {nullptr, nullptr, nullptr, nullptr};
    
    // Datos del nodo
    Particle* particle = nullptr; // Puntero a la partícula si es una hoja
    float totalMass = 0.0f;
    Vector2D centerOfMass;

    QuadtreeNode(Vector2D p_center, float p_halfWidth) : center(p_center), halfWidth(p_halfWidth) {}

    bool isLeaf() const {
        return children[0] == nullptr;
    }
};

class Quadtree {
private:
    std::unique_ptr<QuadtreeNode> root;

    // Inserta una partícula recursivamente.
    // Si el nodo está ocupado, lo subdivide y reinserta ambas partículas.
    void insert(QuadtreeNode* node, Particle* p) {
        if (node->particle == nullptr && node->isLeaf()) {
            node->particle = p;
            return;
        }

        if (node->isLeaf()) {
            // Subdivide el nodo
            node->children[0] = std::make_unique<QuadtreeNode>(Vector2D{node->center.x - node->halfWidth / 2, node->center.y + node->halfWidth / 2}, node->halfWidth / 2);
            node->children[1] = std::make_unique<QuadtreeNode>(Vector2D{node->center.x + node->halfWidth / 2, node->center.y + node->halfWidth / 2}, node->halfWidth / 2);
            node->children[2] = std::make_unique<QuadtreeNode>(Vector2D{node->center.x - node->halfWidth / 2, node->center.y - node->halfWidth / 2}, node->halfWidth / 2);
            node->children[3] = std::make_unique<QuadtreeNode>(Vector2D{node->center.x + node->halfWidth / 2, node->center.y - node->halfWidth / 2}, node->halfWidth / 2);
            
            // Reinserta la partícula que ya estaba
            insert(node->children[getQuadrant(node, node->particle->position)].get(), node->particle);
            node->particle = nullptr;
        }
        
        // Inserta la nueva partícula en el cuadrante correcto
        insert(node->children[getQuadrant(node, p->position)].get(), p);
    }
    
    int getQuadrant(const QuadtreeNode* node, const Vector2D& pos) const {
        if (pos.y > node->center.y) {
            return (pos.x > node->center.x) ? 1 : 0; // Noreste o Noroeste
        } else {
            return (pos.x > node->center.x) ? 3 : 2; // Sureste o Suroeste
        }
    }

    // Calcula el centro de masa de los nodos de forma ascendente (bottom-up).
    void calculateMassDistribution(QuadtreeNode* node) {
        if (node->isLeaf()) {
            if (node->particle) {
                node->totalMass = node->particle->mass;
                node->centerOfMass = node->particle->position;
            }
            return;
        }

        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                calculateMassDistribution(node->children[i].get());
                node->totalMass += node->children[i]->totalMass;
                node->centerOfMass = node->centerOfMass + node->children[i]->centerOfMass * node->children[i]->totalMass;
            }
        }

        if (node->totalMass > 0) {
            node->centerOfMass = node->centerOfMass * (1.0 / node->totalMass);
        }
        printf("Node : mass = %f, center = (%f, %f)\n", node->totalMass, node->centerOfMass.x, node->centerOfMass.y);
    }

    // Calcula la fuerza sobre una partícula recorriendo el árbol.
    void calculateForce(QuadtreeNode* node, Particle* target, double theta) {
        if (!node || node->totalMass == 0 || (node->isLeaf() && node->particle == target)) {
            return;
        }

        Vector2D direction = node->centerOfMass - target->position;
        double distance_sq = direction.magnitude_sq();
        double width = node->halfWidth * 2;
        
        // Condición de Aceptación 
        if ((width * width / distance_sq) < (theta * theta) || node->isLeaf()) {
            double distance = std::sqrt(distance_sq + SOFTENING * SOFTENING);
            double force_magnitude = (G * target->mass * node->totalMass) / (distance * distance);
            Vector2D force = direction * (force_magnitude / distance);
            target->acceleration = target->acceleration + force * (1.0 / target->mass);
        } else {
            // El nodo está demasiado cerca, se abre y se exploran sus hijos. 
            for (int i = 0; i < 4; ++i) {
                calculateForce(node->children[i].get(), target, theta);
            }
        }
    }

public:
    Quadtree(const Vector2D& center, double halfWidth) {
        root = std::make_unique<QuadtreeNode>(center, halfWidth);
    }
    
    void insert(Particle* p) { insert(root.get(), p); }
    void calculateMassDistribution() { calculateMassDistribution(root.get()); }
    void calculateForce(Particle* target, double theta) { calculateForce(root.get(), target, theta); }
};


int main() {
    std::cout << "Iniciando simulación N-Cuerpos con Barnes-Hut..." << std::endl;
    int N[1] = {10000}; // Diferentes tamaños de simulación

    for( int n : N ) {
        std::cout << "Número de partículas: " << n << std::endl;
        std::vector<double> times;
        for(int rep = 0; rep < 1; rep++) {
            // --- Parámetros de los cúmulos de Plummer ---
            int num_particles_per_cluster = n / 2; // Número de partículas por cúmulo
            double total_mass_cluster = n; // Masa total de cada cúmulo
            double plummer_radius = 5.0e2;     // Radio de Plummer de cada cúmulo

            // Posiciones y velocidades iniciales de los cúmulos
            Vector2D cluster1_pos = {-1.0e2, 0.0}; // Cúmulo 1 a la izquierda
            Vector2D cluster1_vel = {2.0, 0.0};  // Moviéndose hacia la derecha

            Vector2D cluster2_pos = {1.0e2, 0.0};
            Vector2D cluster2_vel = {-2.0, 0.0};

            // Generar los cúmulos de Plummer
            std::vector<Particle> particles_cluster1 = generate_plummer_cluster(
                num_particles_per_cluster, total_mass_cluster, plummer_radius,
                cluster1_pos, cluster1_vel
            );
            std::vector<Particle> particles_cluster2 = generate_plummer_cluster(
                num_particles_per_cluster, total_mass_cluster, plummer_radius,
                cluster2_pos, cluster2_vel
            );

            // Combinar todas las partículas en un solo vector
            std::vector<Particle> particles;
            particles.reserve(particles_cluster1.size() + particles_cluster2.size());
            particles.insert(particles.end(), particles_cluster1.begin(), particles_cluster1.end());
            particles.insert(particles.end(), particles_cluster2.begin(), particles_cluster2.end());
            //std::cout << "Partícula 1: " << particles[0].position.x << ", " << particles[0].position.y<< " Velocidad: " << particles[0].velocity.x << ", " << particles[0].velocity.y << std::endl;
            //std::cout << "Partícula " << n << ": " << particles[particles.size() - 1].position.x << ", " << particles[particles.size() - 1].position.y << " Velocidad: " << particles[particles.size() - 1].velocity.x << ", " << particles[particles.size() - 1].velocity.y << std::endl;
            int numSteps = 10;
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int step = 0; step < numSteps; ++step) {
                //printf("Paso %d de %d...\n", step + 1, numSteps);
                // Determinar el tamaño del dominio para crear el Quadtree
                Vector2D min_bounds = particles[0].position, max_bounds = particles[0].position;
                for (const auto& p : particles) {
                    min_bounds.x = std::min(min_bounds.x, p.position.x);
                    min_bounds.y = std::min(min_bounds.y, p.position.y);
                    max_bounds.x = std::max(max_bounds.x, p.position.x);
                    max_bounds.y = std::max(max_bounds.y, p.position.y);
                }
                Vector2D center = {(min_bounds.x + max_bounds.x) / 2, (min_bounds.y + max_bounds.y) / 2};
                double halfWidth = std::max(max_bounds.x - min_bounds.x, max_bounds.y - min_bounds.y) / 2;
                
                // Construir el Quadtree e insertar las partículas
                Quadtree tree(center, halfWidth);
                for (auto& p : particles) {
                    tree.insert(&p);
                }

                // Calcular los centros de masa del árbol
                tree.calculateMassDistribution();
                
                // Calcular las fuerzas para cada partícula
                for (auto& p : particles) {
                    p.acceleration = {0, 0}; // Resetea la aceleración
                    tree.calculateForce(&p, THETA);
                }

                // Actualizar posiciones y velocidades (Integración de Euler simple)
                for (auto& p : particles) {
                    p.velocity = p.velocity + p.acceleration * DT;
                    
                    p.position = p.position + p.velocity * DT;
                }
                /*
                if (step % 100 == 0) {
                        std::cout << "Paso " << step << ": Posición Particula 1 (" << particles[1].position.x << ", " << particles[1].position.y << ")" << std::endl;
                        std::cout << "Paso " << step << ": Posición Particula 1000 (" << particles[particles.size()-1].position.x << ", " << particles[particles.size() - 1].position.y << ")" << std::endl;
                }*/
            }
            std::cout << "Posición Particula 1 (" << particles[1].position.x << ", " << particles[1].position.y << ")" << std::endl;
            std::cout << "Posición Particula 100 (" << particles[particles.size()-1].position.x << ", " << particles[particles.size() - 1].position.y << ")" << std::endl;
            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0; // Convertir a segundos
            times.push_back(elapsed_time);
        }
        double average_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double standard_deviation = std::sqrt(std::accumulate(times.begin(), times.end(), 0.0, 
            [average_time](double sum, double time) { return sum + (time - average_time) * (time - average_time); }) / times.size());
        std::cout << "Tiempo promedio de simulación para " << n << " partículas: " << average_time << " segundos, con desviacion estandar: " << standard_deviation << " segundos."  << std::endl;
        
    }
    
    std::cout << "Simulación finalizada." << std::endl;
    return 0;
}