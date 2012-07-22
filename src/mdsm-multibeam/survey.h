#ifndef SURVEY_H_
#define SURVEY_H_

#include "stdio.h"

#define NULLVALUE -999

typedef struct 
{
    // Beam parameters
    unsigned beam_id, gpu_id;
    float foff, fch1;
    
    // Dedispersion parameters
    float *dm_shifts;
    unsigned maxshift;

} BEAM;

typedef struct {

    // Data parameters
    unsigned nbeams, npols, nchans, nsamp, nbits;
    float    tsamp;

    // Beam parameters
    BEAM *beams;

    // Brute Force parameters
    float    lowdm, dmstep;
    unsigned tdms;

    // Timing parameters
    double timestamp;
    double blockRate;

    // Input file for standalone mode
    FILE *fp;

    // Output parameters
    char      fileprefix[80], basedir[120];
    unsigned  secs_per_file;
    char      use_pc_time, single_file_mode;

    // Detection parameters
    float *global_mean, *global_stddev;
    float detection_threshold;

    // Number of GPUs which are used
    unsigned num_threads;
    unsigned *gpu_ids;
    unsigned num_gpus;

} SURVEY;

#endif