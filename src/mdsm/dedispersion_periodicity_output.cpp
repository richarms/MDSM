// MDSM stuff
#include "dedispersion_periodicity_output.h"
#include "dedispersion_output.h"
#include "unistd.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// C++ stuff
#include <cstdlib>
#include <iostream>

// Process periodicity output
void process_periodicity(float *buffer, FILE* outputFile, SURVEY *survey)
{
    // This will dump the data to disk, in the following format:
    // [BUFF1 .. (DM0)(DM1)...(DMN)][BUFF2...]
    int bins = survey -> period / survey -> tsamp;
    
    fwrite(buffer, sizeof(float), survey -> tdms * bins, outputFile);    
    fflush(outputFile);
}

// Process dedispersion output
void* process_periodicity_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    SURVEY *survey = params -> survey;
    int i, iters = 0, ret, loop_counter = 0, pnsamp = params -> survey -> nsamp;
    int ppnsamp = params -> survey-> nsamp;
    time_t start = params -> start, beg_read;
    double pptimestamp = 0, ptimestamp = 0;
    double ppblockRate = 0, pblockRate = 0;
    FILE *fp = NULL;
    int bins = survey -> period / survey -> tsamp;

    printf("%d: Started periodicity output thread\n", (int) (time(NULL) - start));

    // Create output file
    char pathName[256];
    strcpy(pathName, survey -> basedir);
    strcat(pathName, "/");
    strcat(pathName, survey -> fileprefix);
    strcat(pathName, "_periodicity");
    strcat(pathName, ".dat");
    fp = fopen(pathName, "wb");

    // Dump file header (4-float tuple)
    fwrite(&(survey -> tdms), sizeof(int), 1, fp);
    fwrite(&(survey -> dmstep), sizeof(float), 1, fp);
    fwrite(&(survey -> tsamp), sizeof(float), 1, fp);
    fwrite(&(survey -> period), sizeof(float), 1, fp);
    fwrite(&(bins), sizeof(int), 1, fp);
    fflush(fp);

    // Processing loop
    while (1) {

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD)) 
            { fprintf(stderr, "Error during input barrier synchronisation [output]\n"); exit(0); }

        // Process output
        if (loop_counter >= params -> iterations) {

            beg_read = time(NULL);
            process_periodicity(params -> output_buffer, fp, params -> survey);

            printf("%d: Processed output %d [periodicity]: %d\n", (int) (time(NULL) - start), loop_counter,
            												 (int) (time(NULL) - beg_read));
        }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during output barrier synchronisation [output]\n"); exit(0); }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [output]\n"); exit(0); } 

        // Update params
        ppnsamp = pnsamp;
        pnsamp = params -> survey -> nsamp;     
        pptimestamp = ptimestamp;
        ptimestamp = params -> survey -> timestamp;
        ppblockRate = pblockRate;
        pblockRate = params -> survey -> blockRate;    

        // Stopping clause
        if (((OUTPUT_PARAMS *) output_params) -> stop) {
            
            if (iters >= params -> iterations - 1) {
               
                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [output]\n"); exit(0); }

                for(i = 0; i < params -> maxiters - params -> iterations; i++) {
                    pthread_barrier_wait(params -> input_barrier);
                    pthread_barrier_wait(params -> output_barrier);
                }
                break;
            }
            else
                iters++;
        }

        // Release rw_lock
        if (pthread_rwlock_unlock(params -> rw_lock))
            { fprintf(stderr, "Error releasing rw_lock [output]\n"); exit(0); }

        loop_counter++;
    }   

    printf("%d: Exited gracefully [periodicity output]\n", (int) (time(NULL) - start));
    pthread_exit((void*) output_params);
}
