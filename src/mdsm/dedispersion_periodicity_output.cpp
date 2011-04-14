// MDSM stuff
#include "dedispersion_periodicity_output.h"
#include "dedispersion_output.h"
#include "unistd.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

// C++ stuff
#include <cstdlib>
#include <iostream>

// Global periodicity parameters 
float *profileBuffer; // DM * PERIOD * PROFILE_BINS
float *snr;           // DM vs PERIOD SNR values

void create_plot(FILE *fp, OUTPUT_PARAMS *params)
{
    SURVEY *survey = params -> survey;

    // Loop over all DMs
    for(unsigned i = 0; i < survey -> tdms; i++) {
        int shift = 0;
        
        // Loop over all periods
        for(unsigned j = 0; j < survey -> numPeriods; j++) {

            int bins = (survey -> pStart + (survey -> pStep * j)) / survey -> tsamp;
            float p_bar = FLT_MAX, sigma_bar = 0, max = 0;
            float mean[5] = {0, 0, 0, 0, 0};

            // Calculate off-pulse mean and stddev. This is done by splitting the profile
            // into 5 windows, calculate the mean of each, choosing the lowest mean and
            // then calculate the stddev within that window
            for(int k = 0; k < bins; k++) {
                float val = profileBuffer[i * params -> folded_size + shift + k];
                mean[k / (bins / 5)] += val;
                if (val > max) max = val;
            }
            
            // Select window with the least mean and store this as the off-pulse mean
            unsigned index = 0;            
            for(unsigned k = 0; k < 5; k++) {
                mean[k] /= bins / 5;
                if (mean[k] < p_bar) {
                    p_bar = mean[k];
                    index = k;
                }
            }

            // Calculate off-pulse stddev in window with least mean
            for(unsigned k = bins / 5 * index; k < bins / 5 * (index + 1); k++) {
                float val = profileBuffer[i * params -> folded_size + shift + k] - p_bar;
                sigma_bar += val * val;
            }
            sigma_bar = sqrt( (1.0 / (1.0 * bins - 1.0)) * sigma_bar);

            // Calculate the ratio of the peak power relative to the off-peak standard deviation
            snr[i * survey -> numPeriods + j] = (max - p_bar) / sigma_bar;

            // Add to shift for next period
            shift += bins;
        }
    }

    // If the output profile is required
    if (survey -> outputProfile != 0) {

        // Output profile
        unsigned int periodBin = round((survey -> outputProfile - survey -> pStart) / survey -> pStep);
        unsigned int dmBin     = round((survey -> outputDM - survey -> lowdm) / survey -> dmstep);
        unsigned int bins      = round((survey -> pStart + (survey -> pStep * periodBin)) / survey -> tsamp);
        
        int shift = 0;
        for(unsigned j = 0; j < periodBin; j++)
            shift += round((survey -> pStart + (survey -> pStep * j)) / survey -> tsamp);        

        printf("Dumping pulsar profile (%d bins) [%d-%d]\n", bins, periodBin, dmBin);
        fwrite(profileBuffer + dmBin * params -> folded_size + shift, sizeof(float), bins, fp);

    }
    else {
        // Dump snr surface to file
        fwrite(snr, sizeof(float), survey -> tdms * survey -> numPeriods, fp);
        fflush(fp);
        printf("SNR dumped\n");
    }
}

// Process periodicity output (accumulate profiles)
void process_periodicity(float *buffer, OUTPUT_PARAMS *params)
{
    SURVEY *survey = params -> survey;

    // Loop over all DM values
    for(unsigned i = 0; i < survey -> tdms; i++) {
        int shift = 0;
        
        // Loop over all periods
        for(unsigned j = 0; j < survey -> numPeriods; j++) {

            int bins = (survey -> pStart + (survey -> pStep * j)) / survey -> tsamp;

            // Add buffer profile to global profile
            for(int k = 0; k < bins; k++)
                profileBuffer[i * params -> folded_size + shift + k] 
                    += buffer[i * params -> folded_size + shift + k];

            // Add to shift for next period
            shift += bins;
        }
    }
}

// Process periodicity output
void* process_periodicity_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    SURVEY *survey = params -> survey;
    int i, iters = 0, ret, loop_counter = 0, pnsamp = params -> survey -> nsamp;
    int ppnsamp = params -> survey-> nsamp;
    time_t start = params -> start, beg_read;
    double pptimestamp = 0, ptimestamp = 0;
    double ppblockRate = 0, pblockRate = 0;
    long processed_samples = 0;
    FILE *fp = NULL;

    printf("%d: Started periodicity output thread\n", (int) (time(NULL) - start));

    // Initialise global periodicty variables
    profileBuffer = (float *) malloc(params -> folded_size * survey -> tdms * sizeof(float));
    snr           = (float *) malloc(survey -> numPeriods * survey -> tdms * sizeof(float));
    memset(profileBuffer, 0, params -> folded_size * survey -> tdms * sizeof(float));
    memset(snr, 0, survey -> numPeriods * survey -> tdms * sizeof(float));

    // Single output file mode, create file
    if (survey -> single_file_mode) {
        char pathName[256];
        strcpy(pathName, survey -> basedir);
        strcat(pathName, "/");
        strcat(pathName, survey -> fileprefix);
        strcat(pathName, "_periodicity");
        strcat(pathName, ".dat");
        fp = fopen(pathName, "w");
    }

    // Processing loop
    while (1) {

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD)) 
            { fprintf(stderr, "Error during input barrier synchronisation [output]\n"); exit(0); }

        // Process output
        if (loop_counter >= params -> iterations) {

            // Create new output file if required
            if (processed_samples == 0 && !(survey -> single_file_mode)) {
                char pathName[256];
                strcpy(pathName, survey -> basedir);
                strcat(pathName, "/");
                strcat(pathName, survey -> fileprefix);
                strcat(pathName, "_");

                // Format timestamp 
                struct tm *tmp;
                if (survey -> use_pc_time) {
                    time_t currTime = time(NULL);
                    tmp = localtime(&currTime);                    
                }
                else {
                    time_t currTime = (time_t) pptimestamp;
                    tmp = localtime(&currTime);
                }       

                char tempStr[30];
                strftime(tempStr, sizeof(tempStr), "%F_%T", tmp);

                strcat(pathName, tempStr);
                strcat(pathName, "_");
                sprintf(tempStr, "%d", survey -> secs_per_file);
                strcat(pathName, tempStr);
                strcat(pathName, "_periodicity"); 
                strcat(pathName, ".dat");

                fp = fopen(pathName, "w");
            }

            // Start processing periodicity
            beg_read = time(NULL);

            // Add buffer profile to flobal profile
            process_periodicity(params -> output_buffer, params);
            processed_samples += ppnsamp;

            // Check if time limit is reached
            if (processed_samples * ppblockRate > survey -> secs_per_file) {

                printf("%d: Creating periodicity plot\n", (int) (time(NULL) - start));

                // Output DM vs PERIOD plot
                create_plot(fp, params);

                // Reset the profiles
                memset(profileBuffer, 0, params -> folded_size * survey -> tdms * sizeof(float));   

                processed_samples = 0;
                if (!(survey -> single_file_mode))
                    fclose(fp);
                else
                    fseek(fp, 0, SEEK_SET);
            }
                
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
