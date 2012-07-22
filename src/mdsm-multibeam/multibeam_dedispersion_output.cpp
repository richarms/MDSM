// MDSM stuff
#include "multibeam_dedispersion_output.h"
#include "unistd.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// C++ stuff
#include <QFile>
#include <cstdlib>
#include <iostream>

// OpenMP
#include <omp.h>

// ========================== DETECTION FUNCTIONS ============================

// Process brute force dedispersion if that was chosen
void process_brute(FILE* output, float *buffer, SURVEY *survey, float mean, float stddev,
                   int read_nsamp, double timestamp, double blockRate, time_t start_time)
{
    unsigned int j, k, l;
    
    // Calculate the mean and standard deviation if not provided externally
    double localmean = 0, localrms = 0, tmp_stddev = 0;
    if (mean == NULLVALUE && stddev == NULLVALUE)
    {
        for(j = 0; j < read_nsamp * survey -> tdms; j++)
        {
            localmean  += buffer[j];
            tmp_stddev += pow(buffer[j], 2);
        }
        localmean /= read_nsamp * survey -> tdms;
        localrms = sqrt(tmp_stddev / (read_nsamp * survey -> tdms) - localmean * localmean);
    }
    else
        { localmean = mean; localrms = stddev; }
    printf("%d: Mean: %f, Stddev: %f\n", (int) (time(NULL) - start_time), localmean, localrms);
        
    // Subtract DM mean from all samples and apply threshold
    for (k = 0; k < survey -> tdms; k++) 
    {
        int index = k * survey -> nsamp;
        for(l = 0; l < survey -> nsamp - 4; l++) 
        {
            // Detection threshold sigma filter
            float dm = survey -> lowdm + k * survey -> dmstep;
            if (buffer[index + l] - localmean >= localrms * survey -> detection_threshold && dm > 1.0 )
                  fprintf(output, "%lf, %f, %f\n", timestamp + l * blockRate, dm, buffer[index + l] / localrms);
        }   
    }   

    fflush(output);
}

// Process dedispersion output
// Separate files are created per beam
void* process_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    SURVEY *survey = params -> survey;
    unsigned i, iters = 0, loop_counter = 0, pnsamp = params -> survey -> nsamp;
    int ppnsamp = params -> survey-> nsamp;
    time_t start = params -> start, beg_read;
    double pptimestamp = 0, ptimestamp = 0;
    double ppblockRate = 0, pblockRate = 0;
    float  *mean, *stddev;
//    long written_samples = 0;
    int ret;

    printf("%d: Started output thread\n", (int) (time(NULL) - start));

    // Set number of OpenMP threads
    omp_set_num_threads(params -> nthreads);

    // Create local copy of mean and stddev variables
    mean   = (float *) malloc(params -> nthreads * sizeof(float));
    stddev = (float *) malloc(params -> nthreads * sizeof(float));

    // Create file structures and files
    FILE *fp[survey -> nbeams];

    // Single output file mode, create file
    if (survey -> single_file_mode) 
        for (i = 0; i < survey -> nbeams; i++)
        {
            char beam_no[2];
            sprintf(beam_no, "%d", i);

            char pathName[256];
            strcpy(pathName, survey -> basedir);
            strcat(pathName, "/");
            strcat(pathName, "beam_");
            strcat(pathName, beam_no);
            strcat(pathName, "_");
            strcat(pathName, survey -> fileprefix);
            strcat(pathName, ".dat");
            fp[i] = fopen(pathName, "w");
        }

    // Processing loop
    while (1) 
    {
        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD)) 
            { fprintf(stderr, "Error during input barrier synchronisation [output]\n"); exit(0); }

        // Process output
        if (loop_counter >= params -> iterations) 
        {
            // Create new output file if required
//            if (written_samples == 0 && !(survey -> single_file_mode)) 
//            {
//                char pathName[256];
//                strcpy(pathName, survey -> basedir);
//                strcat(pathName, "/");
//                strcat(pathName, survey -> fileprefix);
//                strcat(pathName, "_");

//                // Format timestamp 
//                struct tm *tmp;
//                if (survey -> use_pc_time) {
//                    time_t currTime = time(NULL);
//                    tmp = localtime(&currTime);                    
//                }
//                else {
//                    time_t currTime = (time_t) pptimestamp;
//                    tmp = localtime(&currTime);
//                }       

//                char tempStr[30];
//                strftime(tempStr, sizeof(tempStr), "%F_%T", tmp);

//                strcat(pathName, tempStr);
//                strcat(pathName, "_");
//                sprintf(tempStr, "%d", survey -> secs_per_file);
//                strcat(pathName, tempStr);
//                strcat(pathName, ".dat");

//                fp = fopen(pathName, "w");
//            }

            beg_read = time(NULL);

            // Processed dedispersed time series (each beam processed by one OpenMP thread)
            #pragma omp parallel \
                shared (fp, params, survey, ppnsamp, pptimestamp, ppblockRate, start)
            {
                unsigned threadId = omp_get_thread_num();
                process_brute(fp[threadId], (params -> output_buffer)[threadId], survey, 
                              mean[threadId], stddev[threadId], ppnsamp, pptimestamp, 
                              ppblockRate, start);
            }

            printf("%d: Processed output %d [output]: %d\n", (int) (time(NULL) - start), loop_counter,
                   (int) (time(NULL) - beg_read));
            
//            if (!(survey -> single_file_mode)) 
//            {
//              written_samples += ppnsamp;
//              if (written_samples * ppblockRate > survey -> secs_per_file) 
//              {
//                  written_samples = 0;
//                  fclose(fp);
//              }
//            }
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
        for(i = 0; i < params -> nthreads; i++)
        {
            mean[i] = (survey -> global_mean)[i];
            stddev[i] = (survey -> global_stddev)[i];  
        }

        // Stopping clause
        if (((OUTPUT_PARAMS *) output_params) -> stop) {
            if (iters >= params -> iterations - 2) {
                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [output]\n"); exit(0); }

                for(i = 0; i < params -> maxiters - params -> iterations + 1; i++) {            
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

    printf("%d: Exited gracefully [output]\n", (int) (time(NULL) - start));
    pthread_exit((void*) output_params);
}
