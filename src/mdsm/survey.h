#ifndef SURVEY_H_
#define SURVEY_H_

#include "stdio.h"

typedef struct {

    float lowdm, highdm, dmstep, sub_dmstep;
    int binsize, ndms, ncalls, calldms, mean, stddev;

} SUBBAND_PASSES ;

typedef struct {

    // Data parameters
    unsigned int nsamp, nchans, tdms, maxshift, nbits, nsubs, npols;
    float tsamp, foff, fch1;
    
    // Switch between brute-froce & subband dedisp
    bool useBruteForce;

    // Brute Force parameters
	float lowdm, dmstep;

    // subband dedispersion paramters
    SUBBAND_PASSES *pass_parameters;
    unsigned num_passes, dedispSubbands;
    
    // Timing parameters
    double timestamp;
    double blockRate;

    // Input file for standalone mode
    FILE *fp;

    // Output parameters
    char fileprefix[80], basedir[120];
    unsigned secs_per_file;
    char use_pc_time, single_file_mode;

    // Number of GPUs which are used
    unsigned num_threads;
    unsigned *gpu_ids;
    unsigned num_gpus;

    // Folding parameters
    double period;
    
    // Actions performed on the GPU
    bool performChannelisation, performTranspose, performFolding;

} SURVEY;

#endif
