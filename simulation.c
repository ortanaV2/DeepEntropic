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
#define FRAME_SAMPLING 10  // Nur jeden 10. Frame speichern

#define NUM_NEIGHBORS 500
#define INPUT_DIM (4 + NUM_NEIGHBORS * 4 + 2)

// Movement detection parameters for early stopping
#define VELOCITY_THRESHOLD 0.1f  // speed threshold for considering particles as stopped
#define STOP_RATIO 0.85f         // fraction of particles that must be stopped to end recording

static const float PI_F = 3.14159265358979323846f;

bool use_gravity = true;
bool use_boundaries = true;
bool enable_visualization = true;

typedef struct {
    float x, y;
    float vx, vy;
    float fx, fy;
    Uint8 r, g, b;
} Particle;

Particle particles[NUM_PARTICLES];
float prev_positions[NUM_PARTICLES][2];

sqlite3 *db = NULL;

#define MAX_FRAMES ((RECORD_SECONDS * 1000) / FRAME_TIME)
#define MAX_SAVED_FRAMES (MAX_FRAMES / FRAME_SAMPLING + 1)

float **all_inputs = NULL;
float **all_targets = NULL;

void compute_forces();
void update_particles();

void allocate_frame_buffers(int max_saved_frames) {
    all_inputs = malloc(sizeof(float*) * max_saved_frames);
    all_targets = malloc(sizeof(float*) * max_saved_frames);
    if (!all_inputs || !all_targets) {
        fprintf(stderr, "Memory allocation failed for frame pointers\n");
        exit(1);
    }
    for (int i = 0; i < max_saved_frames; i++) {
        all_inputs[i] = malloc(NUM_PARTICLES * INPUT_DIM * sizeof(float));
        all_targets[i] = malloc(NUM_PARTICLES * 6 * sizeof(float));  // 6 targets: dx, dy, dvx, dvy, vx, vy
        if (!all_inputs[i] || !all_targets[i]) {
            fprintf(stderr, "Memory allocation failed for frame %d\n", i);
            exit(1);
        }
    }
    printf("Allocated buffers: INPUT_DIM = %d, Total memory per saved frame: %.2f MB\n", 
           INPUT_DIM, (NUM_PARTICLES * INPUT_DIM * sizeof(float)) / (1024.0f * 1024.0f));
    printf("Frame sampling: saving every %d frames, max saved frames: %d\n", FRAME_SAMPLING, max_saved_frames);
}

void free_frame_buffers(int total_saved_frames) {
    if (!all_inputs || !all_targets) return;
    for (int i = 0; i < total_saved_frames; i++) {
        free(all_inputs[i]);
        free(all_targets[i]);
    }
    free(all_inputs);
    free(all_targets);
    all_inputs = NULL;
    all_targets = NULL;
}

void init_random_seed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned int seed = (unsigned int)(tv.tv_sec ^ tv.tv_usec ^ getpid());
    srand(seed);
}

static bool is_position_valid_range(int cluster_start, int cluster_end, float x, float y, float min_dist_sq) {
    for (int j = cluster_start; j < cluster_end; j++) {
        float dx = particles[j].x - x;
        float dy = particles[j].y - y;
        if (dx*dx + dy*dy < min_dist_sq) return false;
    }
    return true;
}

void init_particles() {
    init_random_seed();

    const float min_dist = 2.0f * PARTICLE_RADIUS;
    const int max_attempts = 3000;

    int num_planets = 2 + rand() % 4; 

    int base_particles = NUM_PARTICLES / num_planets;
    int remainder = NUM_PARTICLES % num_planets;

    bool is_position_valid(int cluster_start, int cluster_end, float x, float y) {
        for (int j = cluster_start; j < cluster_end; j++) {
            float dx = particles[j].x - x;
            float dy = particles[j].y - y;
            if (dx*dx + dy*dy < min_dist * min_dist) return false;
        }
        return true;
    }

    int particle_index = 0;

    // Generate random clusters
    for (int p = 0; p < num_planets; p++) {
        float cluster_radius = 80.0f + (rand() % 100);
        float cx = cluster_radius + (rand() % (int)(WIDTH - 2 * cluster_radius));
        float cy = cluster_radius + (rand() % (int)(HEIGHT - 2 * cluster_radius));

        int count = base_particles + (p == num_planets - 1 ? remainder : 0);

        unsigned char r = rand() % 256;
        unsigned char g = rand() % 256;
        unsigned char b = rand() % 256;

        for (int i = 0; i < count; i++) {
            int attempts = 0;
            float x, y;
            do {
                float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
                float radius = sqrtf((float)rand() / RAND_MAX) * cluster_radius;
                x = cx + cosf(angle) * radius;
                y = cy + sinf(angle) * radius;
            } while (!is_position_valid(particle_index, particle_index + i, x, y) && ++attempts < max_attempts);

            particles[particle_index + i] = (Particle){
                x, y,
                0, 0,
                0, 0,
                r, g, b
            };
            prev_positions[particle_index + i][0] = x / (float)WIDTH;
            prev_positions[particle_index + i][1] = y / (float)HEIGHT;
        }

        particle_index += count;
    }
}

void draw_filled_circle(SDL_Renderer *renderer, int cx, int cy, int radius) {
    for (int dy = -radius; dy <= radius; dy++) {
        int dx_limit = (int)sqrtf((float)(radius * radius - dy * dy));
        for (int dx = -dx_limit; dx <= dx_limit; dx++) {
            SDL_RenderDrawPoint(renderer, cx + dx, cy + dy);
        }
    }
}

void compute_forces() {
    #pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].fx = 0.0f;
        particles[i].fy = 0.0f;
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

                // Collision handling
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
                }
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
            if (p->x < RADIUS) { p->x = RADIUS; p->vx *= -DAMPING; }
            if (p->x > WIDTH - RADIUS) { p->x = WIDTH - RADIUS; p->vx *= -DAMPING; }
            if (p->y < RADIUS) { p->y = RADIUS; p->vy *= -DAMPING; }
            if (p->y > HEIGHT - RADIUS) { p->y = HEIGHT - RADIUS; p->vy *= -DAMPING; }
        }
    }
}

void draw_particles(SDL_Renderer *renderer) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        SDL_SetRenderDrawColor(renderer, particles[i].r, particles[i].g, particles[i].b, 255);
        draw_filled_circle(renderer, (int)particles[i].x, (int)particles[i].y, RADIUS);
    }
}

void save_frame_to_buffer(int saved_frame_index) {
    float *inputs = all_inputs[saved_frame_index];
    float *targets = all_targets[saved_frame_index];

    // Global gravity force (constant for all particles)
    float gx = 0.0f;
    float gy = G_CONST;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        // Find NUM_NEIGHBORS nearest neighbors
        float min_dists_sq[NUM_NEIGHBORS];
        int min_idx[NUM_NEIGHBORS];
        
        // Initialize with large distances
        for (int k = 0; k < NUM_NEIGHBORS; k++) {
            min_dists_sq[k] = 1e30f;
            min_idx[k] = -1;
        }

        // Find the NUM_NEIGHBORS closest particles
        for (int j = 0; j < NUM_PARTICLES; j++) {
            if (i == j) continue;
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dsq = dx*dx + dy*dy;

            // Insert this distance in sorted order
            for (int k = 0; k < NUM_NEIGHBORS; k++) {
                if (dsq < min_dists_sq[k]) {
                    // Shift everything down
                    for (int l = NUM_NEIGHBORS - 1; l > k; l--) {
                        min_dists_sq[l] = min_dists_sq[l-1];
                        min_idx[l] = min_idx[l-1];
                    }
                    // Insert at position k
                    min_dists_sq[k] = dsq;
                    min_idx[k] = j;
                    break;
                }
            }
        }

        int base_idx = i * INPUT_DIM;
        
        // Particle state (normalized to [0,1])
        inputs[base_idx + 0] = particles[i].x / (float)WIDTH;
        inputs[base_idx + 1] = particles[i].y / (float)HEIGHT;
        inputs[base_idx + 2] = particles[i].vx / (float)WIDTH;
        inputs[base_idx + 3] = particles[i].vy / (float)HEIGHT;

        // Relative positions and velocities of NUM_NEIGHBORS nearest neighbors
        for (int k = 0; k < NUM_NEIGHBORS; k++) {
            int ni = min_idx[k];
            int neighbor_base = base_idx + 4 + k * 4;
            
            if (ni >= 0) {
                float dx = (particles[ni].x - particles[i].x) / (float)WIDTH;
                float dy = (particles[ni].y - particles[i].y) / (float)HEIGHT;
                float dvx = (particles[ni].vx - particles[i].vx) / (float)WIDTH;
                float dvy = (particles[ni].vy - particles[i].vy) / (float)HEIGHT;
                inputs[neighbor_base + 0] = dx;
                inputs[neighbor_base + 1] = dy;
                inputs[neighbor_base + 2] = dvx;
                inputs[neighbor_base + 3] = dvy;
            } else {
                inputs[neighbor_base + 0] = 0.0f;
                inputs[neighbor_base + 1] = 0.0f;
                inputs[neighbor_base + 2] = 0.0f;
                inputs[neighbor_base + 3] = 0.0f;
            }
        }

        // Global gravity as additional features (korrekte Indizes)
        int gravity_base = base_idx + 4 + NUM_NEIGHBORS * 4;
        inputs[gravity_base + 0] = gx;
        inputs[gravity_base + 1] = gy;
    }

    compute_forces();
    update_particles();

    for (int i = 0; i < NUM_PARTICLES; i++) {
        int base_idx = i * INPUT_DIM;
        float x_new = particles[i].x / (float)WIDTH;
        float y_new = particles[i].y / (float)HEIGHT;

        // Position changes
        targets[i*6 + 0] = x_new - inputs[base_idx + 0];
        targets[i*6 + 1] = y_new - inputs[base_idx + 1];
        // Velocity changes (acceleration)
        targets[i*6 + 2] = particles[i].vx / (float)WIDTH - inputs[base_idx + 2];
        targets[i*6 + 3] = particles[i].vy / (float)HEIGHT - inputs[base_idx + 3];
        // New velocities
        targets[i*6 + 4] = particles[i].vx / (float)WIDTH;
        targets[i*6 + 5] = particles[i].vy / (float)HEIGHT;
    }
}

int write_all_frames_to_db(const char *table_name, int total_saved_frames) {
    char sql[512];
    snprintf(sql, sizeof(sql), "INSERT INTO %s (inputs, targets) VALUES (?, ?);", table_name);

    sqlite3_stmt *stmt;
    int rc = sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
        return rc;
    }

    for (int i = 0; i < total_saved_frames; i++) {
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);

        rc = sqlite3_bind_blob(stmt, 1, all_inputs[i], (int)(NUM_PARTICLES * INPUT_DIM * sizeof(float)), SQLITE_TRANSIENT);
        if (rc != SQLITE_OK) { sqlite3_finalize(stmt); sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL); return rc; }
        rc = sqlite3_bind_blob(stmt, 2, all_targets[i], (int)(NUM_PARTICLES * 6 * sizeof(float)), SQLITE_TRANSIENT);
        if (rc != SQLITE_OK) { sqlite3_finalize(stmt); sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL); return rc; }

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

// Check if majority of particles have stopped moving
bool check_particles_stopped() {
    int count_slow = 0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float speed = sqrtf(particles[i].vx * particles[i].vx + particles[i].vy * particles[i].vy);
        if (speed < VELOCITY_THRESHOLD) {
            count_slow++;
        }
    }
    float ratio = (float)count_slow / NUM_PARTICLES;
    return ratio >= STOP_RATIO;
}

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

    char sql_create[1024];
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
    int max_saved_frames = MAX_SAVED_FRAMES;
    allocate_frame_buffers(max_saved_frames);

    int frame = 0;
    int saved_frame_count = 0;
    
    while (frame < MAX_FRAMES && saved_frame_count < max_saved_frames) {
        if (enable_visualization) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) if (event.type == SDL_QUIT) goto done;
            SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
            SDL_RenderClear(renderer);
            draw_particles(renderer);
            SDL_RenderPresent(renderer);
        }

        if (frame % FRAME_SAMPLING == 0) {
            save_frame_to_buffer(saved_frame_count);
            printf("Saved frame %d (simulation frame %d)\n", saved_frame_count, frame);
            saved_frame_count++;
            
            // Early stopping when particles settle
            if (check_particles_stopped()) {
                printf("Recording stopped at saved frame %d (simulation frame %d): >85%% particles slow\n", 
                       saved_frame_count - 1, frame);
                break;
            }
        } else {
            compute_forces();
            update_particles();
            if (frame % 50 == 0) {
                printf("Simulation frame %d (not saved)\n", frame);
            }
        }

        SDL_Delay(FRAME_TIME);
        frame++;
    }

done:
    printf("Total simulation frames: %d, Saved frames: %d\n", frame, saved_frame_count);
    
    if (write_all_frames_to_db(table_name, saved_frame_count) != SQLITE_OK) {
        fprintf(stderr, "Error writing frames to database\n");
    }

    free_frame_buffers(max_saved_frames);
    sqlite3_close(db);

    if (enable_visualization) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    return 0;
}