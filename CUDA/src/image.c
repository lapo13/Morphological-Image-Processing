#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.h"


void load_image(const char* filename, matrix* img) {
     int width, height, channels;
     u_int8_t* data = stbi_load(filename, &width, &height, &channels, 1);
     if (!data) {
          fprintf(stderr, "Failed to load image: %s\n", stbi_failure_reason());
          exit(EXIT_FAILURE);
     }

     allocate_matrix(img, height, width);

     for (int i = 0; i < height; i++) {
          for (int j = 0; j < width; j++) {
               img->data[i][j] = data[i * width + j];
          }
     }
     stbi_image_free(data);
}

void save_image(const char* filename, matrix* img) {
     if (!stbi_write_png(filename, img->cols, img->rows, 1, img->data[0], img->cols)) {
          fprintf(stderr, "Failed to save image: %s\n", filename);
          exit(EXIT_FAILURE);
     }
}

void paste_image(matrix* dest, matrix* src, int row_offset, int col_offset) {
    if (dest->rows < src->rows + row_offset || dest->cols < src->cols + col_offset) {
        fprintf(stderr, "Error: Source image does not fit within destination image at the specified offset.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[i + row_offset][j + col_offset] = src->data[i][j];
        }
    }
}

void build_mosaic_image(matrix* dest, matrix* tile_buffer, const char** tile_paths, int grid_rows, int grid_cols) {
    load_image(tile_paths[0], tile_buffer);
    int tile_rows = tile_buffer->rows;
    int tile_cols = tile_buffer->cols;

    allocate_matrix(dest, tile_rows * grid_rows, tile_cols * grid_cols);

    paste_image(dest, tile_buffer, 0, 0);

    free_matrix(tile_buffer);

    for (int t = 1; t < grid_rows * grid_cols; t++) {
        int gi = t / grid_cols;
        int gj = t % grid_cols;
        load_image(tile_paths[t], tile_buffer);
        paste_image(dest, tile_buffer, gi * tile_rows, gj * tile_cols);
        free_matrix(tile_buffer);
    }
}


// ---------------------------------------------------------------------------
// Scansione del dataset: raccoglie ricorsivamente i .jpg sotto root e li ordina,
// cosi' che l'insieme di tile usato sia riproducibile fra esecuzioni.
// Portato da OpenMP/src/image.c per costruire gli stessi mosaici.
// ---------------------------------------------------------------------------

static int has_jpg_suffix(const char* name) {
    size_t n = strlen(name);
    return n > 4 && strcmp(name + n - 4, ".jpg") == 0;
}

static int push_path(char*** paths, int* count, int* cap, const char* path) {
    if (*count == *cap) {
        int new_cap = *cap ? *cap * 2 : 256;
        char** grown = (char**)realloc(*paths, new_cap * sizeof(char*));
        if (!grown) return 0;
        *paths = grown;
        *cap = new_cap;
    }
    char* copy = strdup(path);
    if (!copy) return 0;
    (*paths)[(*count)++] = copy;
    return 1;
}

static void scan_dir(const char* dir, char*** paths, int* count, int* cap) {
    DIR* d = opendir(dir);
    if (!d) return;

    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        if (entry->d_name[0] == '.') continue;   // salta . .. e file nascosti

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);

        struct stat st;
        if (stat(path, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            scan_dir(path, paths, count, cap);
        } else if (has_jpg_suffix(entry->d_name)) {
            if (!push_path(paths, count, cap, path)) break;
        }
    }
    closedir(d);
}

static int cmp_path(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

int scan_dataset(const char* root, char*** out_paths) {
    char** paths = NULL;
    int count = 0, cap = 0;

    scan_dir(root, &paths, &count, &cap);
    qsort(paths, count, sizeof(char*), cmp_path);

    *out_paths = paths;
    return count;
}

void free_dataset(char** paths, int count) {
    for (int i = 0; i < count; i++) free(paths[i]);
    free(paths);
}
