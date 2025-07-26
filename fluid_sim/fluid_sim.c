#include <SDL2/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>

#define WIDTH 800
#define HEIGHT 600
#define RADIUS 8
#define NUM_PARTICLES 10
#define PARTICLE_RADIUS 1.5f*RADIUS
#define DIAMETER (RADIUS * 2)
#define GRAVITY 0.2f
#define PRESSURE 0.25f
#define VISCOSITY 0.03f
#define DAMPING 0.2f

#define FRAME_TIME 8
#define RECORD_SECONDS 6

bool use_gravity = true;
bool use_boundaries = true;

typedef struct {
    float x, y;
    float vx, vy;
    float fx, fy;
    Uint8 r, g, b;
} Particle;

Particle particles[NUM_PARTICLES];
float prev_positions[NUM_PARTICLES][2];

void init_particles() {
    srand((unsigned int)time(NULL));
    const int max_attempts = 1000;
    const float min_dist = 2.5f * PARTICLE_RADIUS;

    const float spawn_width = 240;
    const float spawn_height = 120;
    const float spawn_x_min = (WIDTH / 2.0f) - (spawn_width / 2.0f);
    const float spawn_y_min = (HEIGHT / 2.0f) - (spawn_height / 2.0f);

    for (int i = 0; i < NUM_PARTICLES; i++) {
        int attempts = 0;
        bool valid;

        do {
            valid = true;
            float x = spawn_x_min + (rand() % (int)spawn_width);
            float y = spawn_y_min + (rand() % (int)spawn_height);

            for (int j = 0; j < i; j++) {
                float dx = particles[j].x - x;
                float dy = particles[j].y - y;
                float dist_sq = dx * dx + dy * dy;
                if (dist_sq < min_dist * min_dist) {
                    valid = false;
                    break;
                }
            }

            if (valid || ++attempts > max_attempts) {
                particles[i].x = x;
                particles[i].y = y;
                break;
            }
        } while (true);

        particles[i].vx = particles[i].vy = 0;
        particles[i].fx = particles[i].fy = 0;
        particles[i].r = 128 + rand() % 128;
        particles[i].g = 128 + rand() % 128;
        particles[i].b = 128 + rand() % 128;

        prev_positions[i][0] = particles[i].x / (float)WIDTH;
        prev_positions[i][1] = particles[i].y / (float)HEIGHT;
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
        particles[i].fy = use_gravity ? GRAVITY : 0;
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

        if (use_boundaries) {
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
}

void draw_particles(SDL_Renderer *renderer) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        SDL_SetRenderDrawColor(renderer, particles[i].r, particles[i].g, particles[i].b, 255);
        draw_filled_circle(renderer, (int)particles[i].x, (int)particles[i].y, RADIUS);
    }
}

void save_frame(FILE *file, int frame, int total_frames) {
    fprintf(file, "{\"inputs\": [");
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float x_norm = particles[i].x / (float)WIDTH;
        float y_norm = particles[i].y / (float)HEIGHT;

        fprintf(file, "%s%.5f,%.5f", i == 0 ? "" : ",", x_norm, y_norm);
    }

    compute_forces();
    update_particles();

    fprintf(file, "], \"targets\": [");
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float x_norm = particles[i].x / (float)WIDTH;
        float y_norm = particles[i].y / (float)HEIGHT;

        float dx = x_norm - prev_positions[i][0];
        float dy = y_norm - prev_positions[i][1];

        fprintf(file, "%s%.5f,%.5f", i == 0 ? "" : ",", dx, dy);

        prev_positions[i][0] = x_norm;
        prev_positions[i][1] = y_norm;
    }
    fprintf(file, "]}%s\n", frame == total_frames - 1 ? "" : ",");
}

int main(int argc, char *argv[]) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("Simulation", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    init_particles();

    FILE *outfile = fopen("simulation_data.json", "w");
    if (!outfile) {
        fprintf(stderr, "Failed to open file for writing.\n");
        return 1;
    }
    fprintf(outfile, "[\n");

    int total_frames = (RECORD_SECONDS * 1000) / FRAME_TIME;
    for (int frame = 0; frame < total_frames; frame++) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) goto done;
        }

        draw_particles(renderer);
        SDL_RenderPresent(renderer);
        SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
        SDL_RenderClear(renderer);

        save_frame(outfile, frame, total_frames);

        SDL_Delay(FRAME_TIME);
    }

done:
    fprintf(outfile, "\n]\n");
    fclose(outfile);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
