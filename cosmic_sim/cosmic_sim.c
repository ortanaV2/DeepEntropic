#include <SDL2/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define WIDTH 800
#define HEIGHT 600
#define NUM_PARTICLES 2000
#define RADIUS 2
#define GRAVITY_CONST 5.0f
#define DAMPING 0.999f
#define FRAME_TIME 16
#define RECORD_SECONDS 200
#define MAX_FRAMES (RECORD_SECONDS * 1000 / FRAME_TIME)

typedef struct {
    float x, y;
    float vx, vy;
    float mass;
} Particle;

Particle particles[NUM_PARTICLES];
Particle particles_copy[NUM_PARTICLES];

float randf(float range) {
    return ((float)rand() / RAND_MAX) * 2 * range - range;
}

void init_particles() {
    for (int i = 0; i < NUM_PARTICLES / 2; i++) {
        particles[i].x = WIDTH / 2 - 200 + randf(100);
        particles[i].y = HEIGHT / 2 + randf(100);
        particles[i].vx = 0;
        particles[i].vy = 0;
        particles[i].mass = 1.0f;
    }

    for (int i = NUM_PARTICLES / 2; i < NUM_PARTICLES; i++) {
        particles[i].x = WIDTH / 2 + 200 + randf(100);
        particles[i].y = HEIGHT / 2 + randf(100);
        particles[i].vx = 0;
        particles[i].vy = 0;
        particles[i].mass = 1.0f;
    }
}

// void init_particles() {
//     for (int i = 0; i < NUM_PARTICLES; i++) {
//         particles[i].x = ((float)rand() / RAND_MAX) * WIDTH;
//         particles[i].y = ((float)rand() / RAND_MAX) * HEIGHT;
//         particles[i].vx = 0;
//         particles[i].vy = 0;
//         particles[i].mass = 1.0f;
//     }
// }

void resolve_collision(Particle *a, Particle *b) {
    float dx = b->x - a->x;
    float dy = b->y - a->y;
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist == 0) return;

    float minDist = 2 * RADIUS;
    if (dist < minDist) {
        float nx = dx / dist;
        float ny = dy / dist;
        float dvx = b->vx - a->vx;
        float dvy = b->vy - a->vy;
        float relVel = dvx * nx + dvy * ny;
        if (relVel > 0) return;

        float e = 1.0f;
        float j = -(1 + e) * relVel / (1 / a->mass + 1 / b->mass);
        float impulseX = j * nx;
        float impulseY = j * ny;

        a->vx -= impulseX / a->mass;
        a->vy -= impulseY / a->mass;
        b->vx += impulseX / b->mass;
        b->vy += impulseY / b->mass;

        float overlap = 0.5f * (minDist - dist);
        a->x -= overlap * nx;
        a->y -= overlap * ny;
        b->x += overlap * nx;
        b->y += overlap * ny;
    }
}

void update_particles(float dt) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float fx = 0, fy = 0;

        for (int j = 0; j < NUM_PARTICLES; j++) {
            if (i == j) continue;

            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float distSqr = dx * dx + dy * dy + 1e-4f;
            float dist = sqrtf(distSqr);
            float force = (GRAVITY_CONST * particles[i].mass * particles[j].mass) / distSqr;

            fx += force * dx / dist;
            fy += force * dy / dist;
        }

        particles[i].vx += fx / particles[i].mass * dt;
        particles[i].vy += fy / particles[i].mass * dt;
    }

    #pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].vx *= DAMPING;
        particles[i].vy *= DAMPING;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_PARTICLES; i++) {
        for (int j = i + 1; j < NUM_PARTICLES; j++) {
            resolve_collision(&particles[i], &particles[j]);
        }
    }
}

void draw_filled_circle(SDL_Renderer* renderer, int cx, int cy, int radius) {
    for (int dy = -radius; dy <= radius; dy++) {
        int dx_limit = (int)sqrt(radius * radius - dy * dy);
        for (int dx = -dx_limit; dx <= dx_limit; dx++) {
            SDL_RenderDrawPoint(renderer, cx + dx, cy + dy);
        }
    }
}

void render_particles(SDL_Renderer *renderer) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int cx = (int)particles[i].x;
        int cy = (int)particles[i].y;
        draw_filled_circle(renderer, cx, cy, RADIUS);
    }
}

void save_frame(FILE *file, int frame, int total_frames) {
    // copy positions before update
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles_copy[i] = particles[i];
    }

    fprintf(file, "{\"inputs\": [");
    for (int i = 0; i < NUM_PARTICLES; i++) {
        fprintf(file, "%s%.2f,%.2f", i == 0 ? "" : ",", particles_copy[i].x, particles_copy[i].y);
    }

    update_particles(FRAME_TIME / 1000.0f);

    fprintf(file, "], \"targets\": [");
    for (int i = 0; i < NUM_PARTICLES; i++) {
        fprintf(file, "%s%.2f,%.2f", i == 0 ? "" : ",", particles[i].x, particles[i].y);
    }

    fprintf(file, "]}%s\n", frame == total_frames - 1 ? "" : ",");
}

int main(int argc, char *argv[]) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("Cosmic Gravity Simulation",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          WIDTH, HEIGHT, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    init_particles();

    FILE *json_file = fopen("simulation_data.json", "w");
    if (!json_file) {
        fprintf(stderr, "Could not open output file!\n");
        return 1;
    }

    fprintf(json_file, "[\n");

    int running = 1;
    Uint32 last_time = SDL_GetTicks();
    int frame_count = 0;

    while (running && frame_count < MAX_FRAMES) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = 0;
        }

        Uint32 now = SDL_GetTicks();
        float dt = (now - last_time) / 1000.0f;
        if (dt > 0.05f) dt = 0.05f;
        last_time = now;

        save_frame(json_file, frame_count, MAX_FRAMES);
        frame_count++;

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        render_particles(renderer);

        SDL_RenderPresent(renderer);
        SDL_Delay(FRAME_TIME);
    }

    fprintf(json_file, "]\n");
    fclose(json_file);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
