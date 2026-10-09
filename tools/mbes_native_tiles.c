#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define TILE_SIZE 256
#define CACHE_SLOTS 64

typedef struct {
    int valid;
    int dirty;
    int tx;
    int ty;
    uint64_t stamp;
    float *pixels;
} TileSlot;

static const char *output_root;
static TileSlot slots[CACHE_SLOTS];
static uint64_t clock_tick = 1;
static unsigned long long tile_writes = 0;

static char *parse_number(char *p, double *value) {
    while (*p == ' ' || *p == '\t' || *p == ',') p++;
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    else if (*p == '+') p++;
    if ((*p < '0' || *p > '9') && *p != '.') return NULL;
    double result = 0.0;
    while (*p >= '0' && *p <= '9') { result = result * 10.0 + (*p - '0'); p++; }
    if (*p == '.') {
        p++;
        double place = 0.1;
        while (*p >= '0' && *p <= '9') { result += (*p - '0') * place; place *= 0.1; p++; }
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        int exp_sign = 1;
        if (*p == '-') { exp_sign = -1; p++; }
        else if (*p == '+') p++;
        int exponent = 0;
        while (*p >= '0' && *p <= '9') { exponent = exponent * 10 + (*p - '0'); p++; }
        result *= pow(10.0, exp_sign * exponent);
    }
    *value = sign * result;
    return p;
}

static void tile_path(char *path, size_t path_size, char *dir, size_t dir_size, int tx, int ty) {
    snprintf(dir, dir_size, "%s/%d", output_root, tx);
    snprintf(path, path_size, "%s/%d.f32", dir, ty);
}

static int flush_slot(TileSlot *slot) {
    if (!slot->valid || !slot->dirty) return 0;
    char path[4096], dir[4096];
    tile_path(path, sizeof(path), dir, sizeof(dir), slot->tx, slot->ty);
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "mkdir %s failed: %s\n", dir, strerror(errno));
        return -1;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
        return -1;
    }
    size_t count = TILE_SIZE * TILE_SIZE;
    int ok = fwrite(slot->pixels, sizeof(float), count, file) == count;
    fclose(file);
    if (!ok) {
        fprintf(stderr, "write %s failed\n", path);
        return -1;
    }
    slot->dirty = 0;
    tile_writes++;
    return 0;
}

static TileSlot *get_slot(int tx, int ty) {
    TileSlot *victim = NULL;
    for (int i = 0; i < CACHE_SLOTS; i++) {
        if (slots[i].valid && slots[i].tx == tx && slots[i].ty == ty) {
            slots[i].stamp = clock_tick++;
            return &slots[i];
        }
        if (!slots[i].valid) victim = &slots[i];
    }
    if (!victim) {
        victim = &slots[0];
        for (int i = 1; i < CACHE_SLOTS; i++) if (slots[i].stamp < victim->stamp) victim = &slots[i];
    }
    if (flush_slot(victim) != 0) return NULL;
    victim->valid = 1;
    victim->dirty = 0;
    victim->tx = tx;
    victim->ty = ty;
    victim->stamp = clock_tick++;
    size_t count = TILE_SIZE * TILE_SIZE;
    for (size_t i = 0; i < count; i++) victim->pixels[i] = NAN;
    char path[4096], dir[4096];
    tile_path(path, sizeof(path), dir, sizeof(dir), tx, ty);
    FILE *file = fopen(path, "rb");
    if (file) {
        fread(victim->pixels, sizeof(float), count, file);
        fclose(file);
    }
    return victim;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s xmin xmax ymin ymax resolution output_dir\n", argv[0]);
        return 2;
    }
    double xmin = strtod(argv[1], NULL), xmax = strtod(argv[2], NULL);
    double ymin = strtod(argv[3], NULL), ymax = strtod(argv[4], NULL);
    double resolution = strtod(argv[5], NULL);
    output_root = argv[6];
    const char *meta_path = argv[7];
    if (!(xmax > xmin && ymax > ymin && resolution > 0)) return 2;
    if (mkdir(output_root, 0755) != 0 && errno != EEXIST) return 3;
    for (int i = 0; i < CACHE_SLOTS; i++) {
        slots[i].pixels = (float *)malloc(TILE_SIZE * TILE_SIZE * sizeof(float));
        if (!slots[i].pixels) return 3;
    }
    size_t width = (size_t)ceil((xmax - xmin) / resolution);
    size_t height = (size_t)ceil((ymax - ymin) / resolution);
    fprintf(stderr, "MBES native grid %zux%zu at %.3f m\n", width, height, resolution);
    char *line = NULL;
    size_t capacity = 0;
    unsigned long long lines = 0, accepted = 0, outside = 0;
    while (getline(&line, &capacity, stdin) >= 0) {
        lines++;
        double x, y, z;
        char *p = parse_number(line, &x);
        if (!p || !(p = parse_number(p, &y)) || !(p = parse_number(p, &z))) continue;
        long col = (long)floor((x - xmin) / resolution + 1e-7);
        long row = (long)floor((ymax - y) / resolution + 1e-7);
        if (col < 0 || row < 0 || (size_t)col >= width || (size_t)row >= height) { outside++; continue; }
        int tx = (int)(col / TILE_SIZE), ty = (int)(row / TILE_SIZE);
        int lx = (int)(col % TILE_SIZE), ly = (int)(row % TILE_SIZE);
        TileSlot *slot = get_slot(tx, ty);
        if (!slot) return 4;
        slot->pixels[ly * TILE_SIZE + lx] = (float)z;
        slot->dirty = 1;
        accepted++;
        if (lines % 250000000ULL == 0) {
            fprintf(stderr, "MBES native: %llu million lines, %llu tile flushes\n", lines / 1000000ULL, tile_writes);
        }
    }
    free(line);
    for (int i = 0; i < CACHE_SLOTS; i++) {
        if (flush_slot(&slots[i]) != 0) return 5;
        free(slots[i].pixels);
    }
    FILE *meta = fopen(meta_path, "w");
    if (!meta) return 6;
    fprintf(meta,
        "{\n  \"width\": %zu,\n  \"height\": %zu,\n  \"xmin\": %.9g,\n  \"xmax\": %.9g,\n"
        "  \"ymin\": %.9g,\n  \"ymax\": %.9g,\n  \"resolution\": %.9g,\n"
        "  \"input_lines\": %llu,\n  \"accepted_points\": %llu,\n  \"outside_points\": %llu,\n"
        "  \"tile_flushes\": %llu\n}\n",
        width, height, xmin, xmax, ymin, ymax, resolution, lines, accepted, outside, tile_writes);
    fclose(meta);
    fprintf(stderr, "MBES native complete: lines=%llu accepted=%llu outside=%llu flushes=%llu\n", lines, accepted, outside, tile_writes);
    return 0;
}
