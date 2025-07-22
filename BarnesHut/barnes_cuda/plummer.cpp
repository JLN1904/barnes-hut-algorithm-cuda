#include "plummer.h"
#include <random>
#include <cmath>

// Constante gravitacional para la generación del modelo Plummer
const float G_PLUMMER = 6.67430e-11f;

// Generador de números aleatorios
static std::mt19937 rng(3);
static std::uniform_real_distribution<float> dist_uniform(0.0f, 1.0f);

// Devuelve un float aleatorio entre min y max
float random_float(float min, float max) {
    return min + (max - min) * dist_uniform(rng);
}

// Genera un cúmulo de Plummer bidimensional
std::vector<Particle> generate_plummer_cluster(
    int n_particles,
    float total_mass,
    float plummer_radius,
    Vector2D center_pos,
    Vector2D initial_velocity
) {
    std::vector<Particle> cluster_particles;
    cluster_particles.reserve(n_particles);
    float mass_per_particle = total_mass / n_particles;

    for (int i = 0; i < n_particles; ++i) {
        Particle p;
        p.mass = mass_per_particle;
        p.acceleration = {0.0f, 0.0f};

        // Posición radial usando la función de masa acumulada inversa
        float x = random_float(0.0f, 0.85f); 
        float r_norm_sq_inv = std::pow(x, -2.0f/3.0f) - 1.0f;
        float r = plummer_radius / std::sqrt(r_norm_sq_inv);

        // Ángulo aleatorio para posición en 2D
        float theta = random_float(0.0f, 2.0f * M_PI);
        p.position.x = center_pos.x + r * std::cos(theta);
        p.position.y = center_pos.y + r * std::sin(theta);

        // Velocidad usando muestreo por rechazo del modelo Plummer
        float v_esc_r, v_mag, q, g_q, Y;
        do {
            q = random_float(0.0f, 1.0f);
            Y = random_float(0.0f, 0.1f); // Máximo de g(q) ~0.09
            g_q = q * q * std::pow((1.0f - q * q), 3.5f);
            v_esc_r = std::sqrt(2.0f * G_PLUMMER * total_mass / std::sqrt(r*r + plummer_radius*plummer_radius));
            v_mag = q * v_esc_r;
        } while (Y > g_q);

        // Ángulo aleatorio para dirección de velocidad
        float phi = random_float(0.0f, 2.0f * M_PI);
        p.velocity.x = initial_velocity.x + v_mag * std::cos(phi);
        p.velocity.y = initial_velocity.y + v_mag * std::sin(phi);

        cluster_particles.push_back(p);
    }

    return cluster_particles;
}
