#ifndef COMMON_STRUCTS_H
#define COMMON_STRUCTS_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

struct Vector2D {
    float x, y;
    __host__ __device__ Vector2D operator+(const Vector2D& o) const { return {x + o.x, y + o.y}; }
    __host__ __device__ Vector2D operator-(const Vector2D& o) const { return {x - o.x, y - o.y}; }
    __host__ __device__ Vector2D operator*(float s) const { return {x * s, y * s}; }
};

struct Particle {
    Vector2D position;
    Vector2D velocity;
    Vector2D acceleration;
    float mass;
};

#endif
