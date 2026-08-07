#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "logger.h"

void log_timings_csv(timing_entry* entries, int count) {
    time_t now = time(NULL);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm_info);

    char filename[64];
    snprintf(filename, sizeof(filename), "experiment_run/run_%s.csv", timestamp);

    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    fprintf(f, "operation,time_seconds\n");
    for (int i = 0; i < count; i++) {
        fprintf(f, "%s,%.6f\n", entries[i].operation, entries[i].seconds);
    }

    fclose(f);
    printf("Timing results written to %s\n", filename);
}
