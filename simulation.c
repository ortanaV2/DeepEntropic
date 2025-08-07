#include <SDL2/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>
#include <sqlite3.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

#define WIDTH 800
#define HEIGHT 600
#define RADIUS 3
#define NUM_PARTICLES 1000
#define HALF_PARTICLES (NUM_PARTICLES / 2)
#define PARTICLE_RADIUS (2.0f * RADIUS)
#define DIAMETER (RADIUS * 2)
#define G_CONST 0.1f  // gravitational constant (scaled for visible simulation effects)

#define PRESSURE 0.25f   // repelling force in collisions
#define VISCOSITY 0.03f  // velocity smoothing between particles
#define DAMPING 0.2f     // wall bounce damping

#define FRAME_TIME 8
#define RECORD_SECONDS 10

bool use_gravity = true;
bool use_boundaries = true;
bool enable_visualization = true;

// Core particle structure including forces and color attributes
typedef struct {
    float x, y;
    float vx, vy;
    float fx, fy;
    Uint8 r, g, b;
    bool collision_flag;  // neu: Kollisionsflag pro Partikel
} Particle;

Particle particles[NUM_PARTICLES];
float prev_positions[NUM_PARTICLES][2];  // previous normalized positions for delta target generation

sqlite3 *db = NULL;

#define MAX_FRAMES ((RECORD_SECONDS * 1000) / FRAME_TIME)

// Buffers to store simulation inputs/targets for ML dataset creation
float **all_inputs = NULL;
float **all_targets = NULL;

// Allocate 2D buffers for inputs and targets per frame
void allocate_frame_buffers(int total_frames) {
    all_inputs = malloc(sizeof(float*) * total_frames);
    all_targets = malloc(sizeof(float*) * total_frames);
    for (int i = 0; i < total_frames; i++) {
        all_inputs[i] = malloc(NUM_PARTICLES * 5 * sizeof(float));   // 5 Inputs: x,y,vx,vy,collision_flag
        all_targets[i] = malloc(NUM_PARTICLES * 4 * sizeof(float));  // 4 Targets: dx,dy,vx,vy
        if (!all_inputs[i] || !all_targets[i]) {
            fprintf(stderr, "Memory allocation failed for frame %d\n", i);
            exit(1);
        }
    }
}

// Cleanup for dataset buffers
void free_frame_buffers(int total_frames) {
    for (int i = 0; i < total_frames; i++) {
        free(all_inputs[i]);
        free(all_targets[i]);
    }
    free(all_inputs);
    free(all_targets);
}

// Initialize a random seed that avoids collisions across multiple processes
void init_random_seed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned int seed = (unsigned int)(tv.tv_sec ^ tv.tv_usec ^ getpid());
    srand(seed);
}

// Place particles into two well-separated circular clusters ("planets").
// Randomized position generation with minimal inter-particle distance enforcement.
void init_particles() {
    init_random_seed();

    const float cluster_radius = 150.0f;
    const float separation = 400.0f;
    const float min_dist = 2.0f * PARTICLE_RADIUS;

    const float cx1 = WIDTH / 2.0f - separation / 2.0f;
    const float cx2 = WIDTH / 2.0f + separation / 2.0f;
    const float cy1 = HEIGHT / 2.0f;
    const float cy2 = HEIGHT / 2.0f;

    const int max_attempts = 3000;

    // Ensures particles are not too close within a cluster
    bool is_position_valid(int cluster_start, int cluster_end, float x, float y) {
        for (int j = cluster_start; j < cluster_end; j++) {
            float dx = particles[j].x - x;
            float dy = particles[j].y - y;
            if (dx*dx + dy*dy < min_dist * min_dist) return false;
        }
        return true;
    }

    // First (left) cluster
    for (int i = 0; i < HALF_PARTICLES; i++) {
        int attempts = 0;
        float x, y;
        do {
            float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
            float radius = sqrtf((float)rand() / RAND_MAX) * cluster_radius;
            x = cx1 + cosf(angle) * radius;
            y = cy1 + sinf(angle) * radius;
        } while (!is_position_valid(0, i, x, y) && ++attempts < max_attempts);

        particles[i] = (Particle){ x, y, 0, 0, 0, 0, 50, 100 + rand() % 156, 200 + rand() % 55, false };
        prev_positions[i][0] = x / (float)WIDTH;
        prev_positions[i][1] = y / (float)HEIGHT;
    }

    // Second (right) cluster
    for (int i = HALF_PARTICLES; i < NUM_PARTICLES; i++) {
        int attempts = 0;
        float x, y;
        do {
            float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
            float radius = sqrtf((float)rand() / RAND_MAX) * cluster_radius;
            x = cx2 + cosf(angle) * radius;
            y = cy2 + sinf(angle) * radius;
        } while (!is_position_valid(HALF_PARTICLES, i, x, y) && ++attempts < max_attempts);

        particles[i] = (Particle){ x, y, 0, 0, 0, 0, 200 + rand() % 55, 50, 50 + rand() % 100, false };
        prev_positions[i][0] = x / (float)WIDTH;
        prev_positions[i][1] = y / (float)HEIGHT;
    }
}

// Simple filled-circle rasterization using scanlines
void draw_filled_circle(SDL_Renderer *renderer, int cx, int cy, int radius) {
    for (int dy = -radius; dy <= radius; dy++) {
        int dx_limit = (int)sqrt(radius * radius - dy * dy);
        for (int dx = -dx_limit; dx <= dx_limit; dx++) {
            SDL_RenderDrawPoint(renderer, cx + dx, cy + dy);
        }
    }
}

// Calculates pairwise interactions between particles:
// gravitational attraction + pressure and viscosity (short-range collision handling)
// -> setzt auch collision_flag für beteiligte Partikel
void compute_forces() {
    #pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].fx = 0;
        particles[i].fy = 0;
        particles[i].collision_flag = false; // Reset collision flag vor neuer Berechnung
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_PARTICLES; i++) {
        for (int j = i + 1; j < NUM_PARTICLES; j++) {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dist_sq = dx * dx + dy * dy;
            float dist = sqrtf(dist_sq);

            if (dist > 0.01f) {
                float force_mag = G_CONST / dist_sq;
                float fx = force_mag * dx / dist;
                float fy = force_mag * dy / dist;

                #pragma omp atomic
                particles[i].fx += fx;
                #pragma omp atomic
                particles[i].fy += fy;

                #pragma omp atomic
                particles[j].fx -= fx;
                #pragma omp atomic
                particles[j].fy -= fy;

                // Short-range collision response
                if (dist < DIAMETER) {
                    float overlap = DIAMETER - dist;
                    float nx = dx / dist, ny = dy / dist;

                    float pressure_fx = nx * overlap * PRESSURE;
                    float pressure_fy = ny * overlap * PRESSURE;

                    #pragma omp atomic
                    particles[i].fx -= pressure_fx;
                    #pragma omp atomic
                    particles[i].fy -= pressure_fy;
                    #pragma omp atomic
                    particles[j].fx += pressure_fx;
                    #pragma omp atomic
                    particles[j].fy += pressure_fy;

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

                    // Setze collision_flag bei Kollision für beide Partikel
                    particles[i].collision_flag = true;
                    particles[j].collision_flag = true;
                }
            }
        }
    }
}

// Apply accumulated forces and update position; include boundary handling
void update_particles() {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        Particle *p = &particles[i];
        p->vx += p->fx;
        p->vy += p->fy;
        p->x += p->vx;
        p->y += p->vy;

        if (use_boundaries) {
            if (p->x < RADIUS) { p->x = RADIUS; p->vx *= -DAMPING; }
            if (p->x > WIDTH - RADIUS) { p->x = WIDTH - RADIUS; p->vx *= -DAMPING; }
            if (p->y < RADIUS) { p->y = RADIUS; p->vy *= -DAMPING; }
            if (p->y > HEIGHT - RADIUS) { p->y = HEIGHT - RADIUS; p->vy *= -DAMPING; }
        }
    }
}

// Render all particles to screen
void draw_particles(SDL_Renderer *renderer) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        SDL_SetRenderDrawColor(renderer, particles[i].r, particles[i].g, particles[i].b, 255);
        draw_filled_circle(renderer, (int)particles[i].x, (int)particles[i].y, RADIUS);
    }
}

// Save per-frame input/output data for ML model training
void save_frame_to_buffer(int frame) {
    float *inputs = all_inputs[frame];
    float *targets = all_targets[frame];

    // Inputs: x, y, vx, vy, nearest_distance (normalized pos & raw velocity & dist)
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float min_dist_sq = 1e9f;
        for (int j = 0; j < NUM_PARTICLES; j++) {
            if (i == j) continue;
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
        float nearest_dist = sqrtf(min_dist_sq);

        inputs[i*5 + 0] = particles[i].x / (float)WIDTH;
        inputs[i*5 + 1] = particles[i].y / (float)HEIGHT;
        inputs[i*5 + 2] = particles[i].vx;
        inputs[i*5 + 3] = particles[i].vy;
        inputs[i*5 + 4] = nearest_dist / (float)WIDTH;
    }

    compute_forces();
    update_particles();

    // Targets: delta-x, delta-y, vx, vy (positions delta + velocity)
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float x_norm = particles[i].x / (float)WIDTH;
        float y_norm = particles[i].y / (float)HEIGHT;
        targets[i*4 + 0] = x_norm - inputs[i*5 + 0];
        targets[i*4 + 1] = y_norm - inputs[i*5 + 1];
        targets[i*4 + 2] = particles[i].vx;
        targets[i*4 + 3] = particles[i].vy;
    }
}

// Write all frame data to SQLite database (as blobs)
int write_all_frames_to_db(const char *table_name, int total_frames) {
    char sql[256];
    snprintf(sql, sizeof(sql), "INSERT INTO %s (inputs, targets) VALUES (?, ?);", table_name);

    sqlite3_stmt *stmt;
    int rc = sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
        return rc;
    }

    for (int i = 0; i < total_frames; i++) {
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);

        sqlite3_bind_blob(stmt, 1, all_inputs[i], NUM_PARTICLES * 5 * sizeof(float), SQLITE_STATIC);
        sqlite3_bind_blob(stmt, 2, all_targets[i], NUM_PARTICLES * 4 * sizeof(float), SQLITE_STATIC);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
            return rc;
        }
    }

    sqlite3_finalize(stmt);
    return sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
}

// Main simulation loop: handles rendering, recording, and database writing
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <table_name>\n", argv[0]);
        return 1;
    }
    const char *table_name = argv[1];

    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;

    if (enable_visualization && SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    if (sqlite3_open("dataset.db", &db) != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);

    char sql_create[512];
    snprintf(sql_create, sizeof(sql_create),
        "CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, inputs BLOB NOT NULL, targets BLOB NOT NULL);", table_name);

    char *err_msg = NULL;
    if (sqlite3_exec(db, sql_create, 0, 0, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "Failed to create table '%s': %s\n", table_name, err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return 1;
    }

    if (enable_visualization) {
        window = SDL_CreateWindow("Simulation", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!window || !renderer) {
            fprintf(stderr, "SDL init failed\n");
            sqlite3_close(db);
            return 1;
        }
    }

    init_particles();
    int total_frames = MAX_FRAMES;
    allocate_frame_buffers(total_frames);

    for (int frame = 0; frame < total_frames; frame++) {
        if (enable_visualization) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) if (event.type == SDL_QUIT) goto done;
            SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
            SDL_RenderClear(renderer);
            draw_particles(renderer);
            SDL_RenderPresent(renderer);
        }
        save_frame_to_buffer(frame);
        SDL_Delay(FRAME_TIME);
    }

done:
    if (write_all_frames_to_db(table_name, total_frames) != SQLITE_OK) {
        fprintf(stderr, "Error writing frames to database\n");
    }

    free_frame_buffers(total_frames);
    sqlite3_close(db);
    if (enable_visualization) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    return 0;
}
