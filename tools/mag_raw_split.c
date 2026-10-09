#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CACHE_SLOTS 64
#define BUFFER_RECORDS 4096

typedef struct { float x; float y; float value; } Record;
typedef struct {
    int valid;
    int tx;
    int ty;
    uint64_t stamp;
    FILE *file;
    size_t used;
    Record records[BUFFER_RECORDS];
} FileSlot;

static const char *output_root;
static FileSlot slots[CACHE_SLOTS];
static uint64_t clock_tick = 1;
static unsigned long long file_count = 0;

static char *parse_number(char *p, double *value) {
    while (*p == ' ' || *p == '\t' || *p == ',') p++;
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    else if (*p == '+') p++;
    if ((*p < '0' || *p > '9') && *p != '.') return NULL;
    double result = 0.0;
    while (*p >= '0' && *p <= '9') { result = result * 10.0 + (*p - '0'); p++; }
    if (*p == '.') {
        p++; double place = 0.1;
        while (*p >= '0' && *p <= '9') { result += (*p - '0') * place; place *= 0.1; p++; }
    }
    if (*p == 'e' || *p == 'E') {
        p++; int exp_sign = 1;
        if (*p == '-') { exp_sign = -1; p++; } else if (*p == '+') p++;
        int exponent = 0;
        while (*p >= '0' && *p <= '9') { exponent = exponent * 10 + (*p - '0'); p++; }
        result *= pow(10.0, exp_sign * exponent);
    }
    *value = sign * result;
    return p;
}

static int flush_slot(FileSlot *slot) {
    if (!slot->valid || slot->used == 0) return 0;
    if (fwrite(slot->records, sizeof(Record), slot->used, slot->file) != slot->used) return -1;
    slot->used = 0;
    return 0;
}

static void close_slot(FileSlot *slot) {
    if (!slot->valid) return;
    flush_slot(slot);
    fclose(slot->file);
    slot->file = NULL;
    slot->valid = 0;
}

static FileSlot *get_slot(int tx, int ty) {
    FileSlot *victim = NULL;
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
        close_slot(victim);
    }
    char dir[4096], path[4096];
    snprintf(dir, sizeof(dir), "%s/%d", output_root, tx);
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) return NULL;
    snprintf(path, sizeof(path), "%s/%d.bin", dir, ty);
    int existed = access(path, F_OK) == 0;
    FILE *file = fopen(path, "ab");
    if (!file) return NULL;
    if (!existed) file_count++;
    victim->valid = 1; victim->tx = tx; victim->ty = ty; victim->stamp = clock_tick++;
    victim->file = file; victim->used = 0;
    return victim;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s xmin xmax ymin ymax tile_m output_dir meta.json\n", argv[0]);
        return 2;
    }
    double xmin = strtod(argv[1], NULL), xmax = strtod(argv[2], NULL);
    double ymin = strtod(argv[3], NULL), ymax = strtod(argv[4], NULL);
    double tile_m = strtod(argv[5], NULL);
    output_root = argv[6]; const char *meta_path = argv[7];
    if (!(xmax > xmin && ymax > ymin && tile_m > 0)) return 2;
    if (mkdir(output_root, 0755) != 0 && errno != EEXIST) return 3;
    int width_tiles = (int)ceil((xmax - xmin) / tile_m);
    int height_tiles = (int)ceil((ymax - ymin) / tile_m);
    fprintf(stderr, "MAG raw split %dx%d tiles at %.1f m\n", width_tiles, height_tiles, tile_m);
    char *line = NULL; size_t capacity = 0;
    unsigned long long lines = 0, accepted = 0, outside = 0;
    while (getline(&line, &capacity, stdin) >= 0) {
        lines++;
        double x, y, value;
        char *p = parse_number(line, &x);
        if (!p || !(p = parse_number(p, &y)) || !(p = parse_number(p, &value))) continue;
        int tx = (int)floor((x - xmin) / tile_m);
        int ty = (int)floor((ymax - y) / tile_m);
        if (tx < 0 || ty < 0 || tx >= width_tiles || ty >= height_tiles) { outside++; continue; }
        FileSlot *slot = get_slot(tx, ty);
        if (!slot) return 4;
        slot->records[slot->used++] = (Record){(float)(x - (xmin + tx * tile_m)), (float)((ymax - ty * tile_m) - y), (float)value};
        if (slot->used == BUFFER_RECORDS && flush_slot(slot) != 0) return 5;
        accepted++;
        if (lines % 50000000ULL == 0) fprintf(stderr, "MAG raw: %llu million lines\n", lines / 1000000ULL);
    }
    free(line);
    for (int i = 0; i < CACHE_SLOTS; i++) close_slot(&slots[i]);
    FILE *meta = fopen(meta_path, "w"); if (!meta) return 6;
    fprintf(meta,
        "{\n  \"xmin\": %.9g,\n  \"xmax\": %.9g,\n  \"ymin\": %.9g,\n  \"ymax\": %.9g,\n"
        "  \"tile_m\": %.9g,\n  \"width_tiles\": %d,\n  \"height_tiles\": %d,\n"
        "  \"input_lines\": %llu,\n  \"accepted_points\": %llu,\n  \"outside_points\": %llu,\n"
        "  \"new_files_this_run\": %llu\n}\n",
        xmin, xmax, ymin, ymax, tile_m, width_tiles, height_tiles, lines, accepted, outside, file_count);
    fclose(meta);
    fprintf(stderr, "MAG raw complete: lines=%llu accepted=%llu outside=%llu files=%llu\n", lines, accepted, outside, file_count);
    return 0;
}
