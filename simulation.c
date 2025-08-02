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
#define RADIUS 14
#define NUM_PARTICLES 10
#define PARTICLE_RADIUS (2.0f * RADIUS)
#define DIAMETER (RADIUS * 2)
#define GRAVITY 0.2f
#define PRESSURE 0.25f
#define VISCOSITY 0.03f
#define DAMPING 0.2f

#define FRAME_TIME 8
#define RECORD_SECONDS 7

bool use_gravity = true;
bool use_boundaries = true;
bool enable_visualization = false;

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

float **all_inputs = NULL;
float **all_targets = NULL;

void allocate_frame_buffers(int total_frames) {
    all_inputs = malloc(sizeof(float*) * total_frames);
    all_targets = malloc(sizeof(float*) * total_frames);
    for (int i = 0; i < total_frames; i++) {
        all_inputs[i] = malloc(NUM_PARTICLES * 4 * sizeof(float));
        all_targets[i] = malloc(NUM_PARTICLES * 2 * sizeof(float));
        if (!all_inputs[i] || !all_targets[i]) {
            fprintf(stderr, "Memory allocation failed for frame %d\n", i);
            exit(1);
        }
    }
}

void free_frame_buffers(int total_frames) {
    for (int i = 0; i < total_frames; i++) {
        free(all_inputs[i]);
        free(all_targets[i]);
    }
    free(all_inputs);
    free(all_targets);
}

void init_random_seed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned int seed = (unsigned int)(tv.tv_sec ^ tv.tv_usec ^ getpid());
    srand(seed);
}

void init_particles() {
    init_random_seed();

    const int max_attempts = 1000;
    const float min_dist = 2.5f * PARTICLE_RADIUS;

    const float margin = PARTICLE_RADIUS;
    const float floor_clearance = 50.0f;

    const float spawn_x_min = margin;
    const float spawn_x_max = WIDTH - margin;
    const float spawn_y_min = margin;
    const float spawn_y_max = HEIGHT - margin - floor_clearance;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        int attempts = 0;
        bool valid;

        do {
            valid = true;
            float x = spawn_x_min + ((float)rand() / RAND_MAX) * (spawn_x_max - spawn_x_min);
            float y = spawn_y_min + ((float)rand() / RAND_MAX) * (spawn_y_max - spawn_y_min);

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

        particles[i].vx = 0;
        particles[i].vy = 0;
        particles[i].fx = 0;
        particles[i].fy = 0;
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

void save_frame_to_buffer(int frame) {
    float *inputs = all_inputs[frame];
    float *targets = all_targets[frame];

    for (int i = 0; i < NUM_PARTICLES; i++) {
        inputs[i*4 + 0] = prev_positions[i][0];
        inputs[i*4 + 1] = prev_positions[i][1];
        inputs[i*4 + 2] = particles[i].vx / WIDTH;
        inputs[i*4 + 3] = particles[i].vy / HEIGHT;
    }

    compute_forces();
    update_particles();

    for (int i = 0; i < NUM_PARTICLES; i++) {
        float x_norm = particles[i].x / (float)WIDTH;
        float y_norm = particles[i].y / (float)HEIGHT;

        targets[i*2 + 0] = x_norm - prev_positions[i][0];
        targets[i*2 + 1] = y_norm - prev_positions[i][1];

        prev_positions[i][0] = x_norm;
        prev_positions[i][1] = y_norm;
    }
}

int write_all_frames_to_db(const char *table_name, int total_frames) {
    char sql[256];
    snprintf(sql, sizeof(sql), "INSERT INTO %s (inputs, targets) VALUES (?, ?);", table_name);

    sqlite3_stmt *stmt;
    int rc;

    rc = sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to begin transaction: %s\n", sqlite3_errmsg(db));
        return rc;
    }

    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
        return rc;
    }

    for (int i = 0; i < total_frames; i++) {
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);

        sqlite3_bind_blob(stmt, 1, all_inputs[i], NUM_PARTICLES * 4 * sizeof(float), SQLITE_STATIC);
        sqlite3_bind_blob(stmt, 2, all_targets[i], NUM_PARTICLES * 2 * sizeof(float), SQLITE_STATIC);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Failed to execute statement at frame %d: %s\n", i, sqlite3_errmsg(db));
            sqlite3_finalize(stmt);
            sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
            return rc;
        }
    }

    sqlite3_finalize(stmt);

    rc = sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to commit transaction: %s\n", sqlite3_errmsg(db));
        return rc;
    }

    return SQLITE_OK;
}

int main(int argc, char *argv[]) {
    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <table_name>\n", argv[0]);
        return 1;
    }
    const char *table_name = argv[1];

    if (enable_visualization) {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
            return 1;
        }
    }

    if (sqlite3_open("dataset.db", &db) != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);  // enable parallel db-writing

    char sql_create[512];
    snprintf(sql_create, sizeof(sql_create),
        "CREATE TABLE IF NOT EXISTS %s ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "inputs BLOB NOT NULL,"
        "targets BLOB NOT NULL);", table_name);

    char *err_msg = NULL;
    if (sqlite3_exec(db, sql_create, 0, 0, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "Failed to create table '%s': %s\n", table_name, err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return 1;
    }

    if (enable_visualization) {
        window = SDL_CreateWindow("Simulation", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
        if (!window) {
            fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
            sqlite3_close(db);
            return 1;
        }

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) {
            fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());
            SDL_DestroyWindow(window);
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
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) goto done;
            }

            draw_particles(renderer);
            SDL_RenderPresent(renderer);
            SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
            SDL_RenderClear(renderer);
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