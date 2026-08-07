#ifndef LOGGER_H
#define LOGGER_H

typedef struct {
    const char* operation;
    double seconds;
} timing_entry;

void log_timings_csv(timing_entry* entries, int count);

#endif
