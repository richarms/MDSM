#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "pthread.h"
#include "survey.h"

typedef struct {
    // Input parameters
    int nchans, nsamp, nthreads, iterations, maxiters;

    // Dedispersion parameters
    float startdm, dmstep;
    int ndms;

    // Input and output buffers memory pointers
    float** output_buffer;
   
    // Thread-specific info + synchronisation objects
    unsigned short stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;

    // Timing
    time_t start;

    // Output file
    FILE* output_file;

    // Survey parameters
    SURVEY *survey;

} OUTPUT_PARAMS;

void* process_output(void* output_params);

#endif
