#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *name) {
    fprintf(stderr, "usage: %s <space|csv> xmin xmax ymin ymax resolution stride output_prefix\n", name);
}

int main(int argc, char **argv) {
    if (argc != 10) {
        usage(argv[0]);
        return 2;
    }
    int csv_mode = strcmp(argv[1], "csv") == 0;
    if (!csv_mode && strcmp(argv[1], "space") != 0) {
        usage(argv[0]);
        return 2;
    }
    double xmin = strtod(argv[2], NULL);
    double xmax = strtod(argv[3], NULL);
    double ymin = strtod(argv[4], NULL);
    double ymax = strtod(argv[5], NULL);
    double resolution = strtod(argv[6], NULL);
    unsigned long stride = strtoul(argv[7], NULL, 10);
    const char *prefix = argv[8];
    /* argv[9] is a required label used in progress output. */
    const char *label = argv[9];
    if (!(xmax > xmin && ymax > ymin && resolution > 0 && stride > 0)) {
        fprintf(stderr, "invalid grid arguments\n");
        return 2;
    }
    size_t width = (size_t)ceil((xmax - xmin) / resolution);
    size_t height = (size_t)ceil((ymax - ymin) / resolution);
    if (width == 0 || height == 0 || width > SIZE_MAX / height) {
        fprintf(stderr, "invalid grid dimensions\n");
        return 2;
    }
    size_t cells = width * height;
    float *sums = (float *)calloc(cells, sizeof(float));
    uint16_t *counts = (uint16_t *)calloc(cells, sizeof(uint16_t));
    if (!sums || !counts) {
        fprintf(stderr, "cannot allocate %zu cells (%s)\n", cells, strerror(errno));
        free(sums); free(counts);
        return 3;
    }

    fprintf(stderr, "%s grid %zux%zu, %.1f m, stride %lu\n", label, width, height, resolution, stride);
    char *line = NULL;
    size_t capacity = 0;
    ssize_t length;
    unsigned long long lines = 0, sampled = 0, accepted = 0, outside = 0, saturated = 0;
    double raw_min = INFINITY, raw_max = -INFINITY;
    while ((length = getline(&line, &capacity, stdin)) >= 0) {
        (void)length;
        lines++;
        if ((lines - 1) % stride != 0) continue;
        char *cursor = line;
        char *end = NULL;
        double x = strtod(cursor, &end);
        if (end == cursor) continue;
        cursor = end;
        if (csv_mode) {
            while (*cursor && *cursor != ',') cursor++;
            if (*cursor == ',') cursor++;
        }
        double y = strtod(cursor, &end);
        if (end == cursor) continue;
        cursor = end;
        if (csv_mode) {
            while (*cursor && *cursor != ',') cursor++;
            if (*cursor == ',') cursor++;
        }
        double value = strtod(cursor, &end);
        if (end == cursor || !isfinite(x) || !isfinite(y) || !isfinite(value)) continue;
        sampled++;
        if (value < raw_min) raw_min = value;
        if (value > raw_max) raw_max = value;
        long col = (long)floor((x - xmin) / resolution);
        long row = (long)floor((ymax - y) / resolution);
        if (col < 0 || row < 0 || (size_t)col >= width || (size_t)row >= height) {
            outside++;
            continue;
        }
        size_t index = (size_t)row * width + (size_t)col;
        if (counts[index] == UINT16_MAX) {
            saturated++;
            continue;
        }
        sums[index] += (float)value;
        counts[index]++;
        accepted++;
        if ((lines % 250000000ULL) == 0) {
            fprintf(stderr, "%s: %llu million input lines\n", label, lines / 1000000ULL);
        }
    }
    free(line);

    char sum_path[4096], count_path[4096], meta_path[4096];
    snprintf(sum_path, sizeof(sum_path), "%s.sum.f32", prefix);
    snprintf(count_path, sizeof(count_path), "%s.count.u16", prefix);
    snprintf(meta_path, sizeof(meta_path), "%s.meta.json", prefix);
    FILE *sum_file = fopen(sum_path, "wb");
    FILE *count_file = fopen(count_path, "wb");
    FILE *meta_file = fopen(meta_path, "w");
    if (!sum_file || !count_file || !meta_file) {
        fprintf(stderr, "cannot open output files (%s)\n", strerror(errno));
        if (sum_file) fclose(sum_file);
        if (count_file) fclose(count_file);
        if (meta_file) fclose(meta_file);
        free(sums); free(counts);
        return 4;
    }
    if (fwrite(sums, sizeof(float), cells, sum_file) != cells ||
        fwrite(counts, sizeof(uint16_t), cells, count_file) != cells) {
        fprintf(stderr, "failed to write grid output\n");
        fclose(sum_file); fclose(count_file); fclose(meta_file);
        free(sums); free(counts);
        return 5;
    }
    fprintf(meta_file,
        "{\n  \"label\": \"%s\",\n  \"width\": %zu,\n  \"height\": %zu,\n"
        "  \"xmin\": %.9g,\n  \"xmax\": %.9g,\n  \"ymin\": %.9g,\n  \"ymax\": %.9g,\n"
        "  \"resolution\": %.9g,\n  \"stride\": %lu,\n  \"input_lines\": %llu,\n"
        "  \"sampled_lines\": %llu,\n  \"accepted_samples\": %llu,\n  \"outside_samples\": %llu,\n"
        "  \"saturated_samples\": %llu,\n  \"raw_min\": %.9g,\n  \"raw_max\": %.9g\n}\n",
        label, width, height, xmin, xmax, ymin, ymax, resolution, stride,
        lines, sampled, accepted, outside, saturated, raw_min, raw_max);
    fclose(sum_file); fclose(count_file); fclose(meta_file);
    fprintf(stderr, "%s complete: lines=%llu sampled=%llu accepted=%llu outside=%llu range=[%.6g, %.6g]\n",
        label, lines, sampled, accepted, outside, raw_min, raw_max);
    free(sums); free(counts);
    return 0;
}
