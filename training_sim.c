#include <SDL2/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define WIDTH 800
#define HEIGHT 600
#define NUM_PARTICLES 2000
#define RADIUS 6
#define DIAMETER (RADIUS * 2)
#define GRAVITY 0.2f
#define PRESSURE 0.04f
#define VISCOSITY 0.03f
#define DAMPING 0.2f

typedef struct {
    float x, y;
    float vx, vy;
    float fx, fy;
    Uint8 r, g, b;
} Particle;

Particle particles[NUM_PARTICLES];

void init_particles() {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].x = 300 + rand() % 200;
        particles[i].y = 50 + rand() % 100;
        particles[i].vx = 0;
        particles[i].vy = 0;
        particles[i].fx = 0;
        particles[i].fy = 0;

        // Pastellfarben zufällig (hell)
        particles[i].r = 128 + rand() % 128;
        particles[i].g = 128 + rand() % 128;
        particles[i].b = 128 + rand() % 128;
    }
}

void draw_filled_circle(SDL_Renderer *renderer, int cx, int cy, int radius) {
    for (int dy = -radius; dy <= radius; dy++) {
        int dx_limit = (int)sqrt(radius * radius - dy * dy);
        for (int dx = -dx_limit; dx <= dx_limit; dx++) {
            SDL_RenderDrawPoint(renderer, cx + dx, cy + dy);
        }
    }
}

void compute_forces() {
    #pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].fx = 0;
        particles[i].fy = GRAVITY;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_PARTICLES; i++) {
        for (int j = i + 1; j < NUM_PARTICLES; j++) {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < DIAMETER && dist > 0.01f) {
                float overlap = DIAMETER - dist;
                float nx = dx / dist;
                float ny = dy / dist;

                float fx = nx * overlap * PRESSURE;
                float fy = ny * overlap * PRESSURE;

                #pragma omp atomic
                particles[i].fx -= fx;
                #pragma omp atomic
                particles[i].fy -= fy;

                #pragma omp atomic
                particles[j].fx += fx;
                #pragma omp atomic
                particles[j].fy += fy;

                float dvx = particles[j].vx - particles[i].vx;
                float dvy = particles[j].vy - particles[i].vy;
                float viscx = dvx * VISCOSITY;
                float viscy = dvy * VISCOSITY;

                #pragma omp atomic
                particles[i].vx += viscx;
                #pragma omp atomic
                particles[i].vy += viscy;

                #pragma omp atomic
                particles[j].vx -= viscx;
                #pragma omp atomic
                particles[j].vy -= viscy;
            }
        }
    }
}

void update_particles() {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        Particle *p = &particles[i];
        p->vx += p->fx;
        p->vy += p->fy;
        p->x += p->vx;
        p->y += p->vy;

        if (p->x < RADIUS) {
            p->x = RADIUS;
            p->vx *= -DAMPING;
        }
        if (p->x > WIDTH - RADIUS) {
            p->x = WIDTH - RADIUS;
            p->vx *= -DAMPING;
        }
        if (p->y > HEIGHT - RADIUS) {
            p->y = HEIGHT - RADIUS;
            p->vy *= -DAMPING;
        }
        if (p->y < RADIUS) {
            p->y = RADIUS;
            p->vy *= -DAMPING;
        }
    }
}

void draw_particles(SDL_Renderer *renderer) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        SDL_SetRenderDrawColor(renderer, particles[i].r, particles[i].g, particles[i].b, 255);
        draw_filled_circle(renderer, (int)particles[i].x, (int)particles[i].y, RADIUS);
    }
}

int main(int argc, char *argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("OpenMP Fluid Simulation - 4000 Particles",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        printf("SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    init_particles();

    int running = 1;
    SDL_Event event;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = 0;
        }

        compute_forces();
        update_particles();

        SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
        SDL_RenderClear(renderer);

        draw_particles(renderer);

        SDL_RenderPresent(renderer);

        SDL_Delay(16);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
