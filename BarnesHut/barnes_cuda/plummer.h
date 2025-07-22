#ifndef PLUMMER_H
#define PLUMMER_H

#include <vector>
#include "common_structs.h"

// Genera un cúmulo de Plummer de n partículas con masa total, radio y centro dados
std::vector<Particle> generate_plummer_cluster(
    int n_particles,
    float total_mass,
    float plummer_radius,
    Vector2D center_pos,
    Vector2D initial_velocity
);

#endif // PLUMMER_H
