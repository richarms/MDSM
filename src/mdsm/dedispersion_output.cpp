// MDSM stuff
#include "dedispersion_output.h"
#include "unistd.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// C++ stuff
#include <cstdlib>
#include <iostream>

// Calculate the median of five numbers for a median filter
float medianOfFive(float n1, float n2, float n3, float n4, float n5){
  float *a = &n1, *b = &n2, *c = &n3, *d = &n4, *e = &n5;
  float *tmp;

  // makes a < b and b < d
  if(*b < *a){
    tmp = a; a = b; b = tmp;
  }

  if(*d < *c){
    tmp = c; c = d; d = tmp;
  }

  // eleminate the lowest
  if(*c < *a){
    tmp = b; b = d; d = tmp; 
    c = a;
  }

  // gets e in
  a = e;

  // makes a < b and b < d
  if(*b < *a){
    tmp = a; a = b; b = tmp;
  }

  // eliminate another lowest
  // remaing: a,b,d
  if(*a < *c){
    tmp = b; b = d; d = tmp; 
    a = c;
  }

  if(*d < *a)
    return *d;
  else
    return *a;

}

// Process subband dedispersion if that has been chosen
void process_subband(float *buffer, FILE* output, SURVEY *survey, int read_nsamp, size_t size, double timestamp, double blockRate,
                     float* mean, float* rms )
{
    unsigned int i = 0, thread, k, t, ndms, nsamp, shift = 0;
    float startdm, dmstep;
    float localmean = 0, localrms = 0;

    std::cout << " First Mean from datablob: " << mean[0] << std::endl;
    std::cout << " First RMS from datablob: " << rms[0] << std::endl;

    for(thread = 0; thread < survey -> num_threads; thread++) {

        for(shift = 0, i = 0; i < survey -> num_passes; i++) {
	        int index_dm0 = size * thread + shift;

	        // ntimes is the number of samples over which to calculate the mean and rms
            // Calaculate parameters
            nsamp   = read_nsamp / survey -> pass_parameters[i].binsize;
            startdm = survey -> pass_parameters[i].lowdm + survey -> pass_parameters[i].sub_dmstep 
                      * (survey -> pass_parameters[i].ncalls / survey -> num_threads) * thread;
            dmstep  = survey -> pass_parameters[i].dmstep;
            ndms    = (survey -> pass_parameters[i].ncalls / survey -> num_threads) 
                      * survey -> pass_parameters[i].calldms;


            for (k = 0; k < ndms; k++) {

	        int index_dm = index_dm0 + k * nsamp;
                // Do we have useful values for the mean and rms?
                // If not, work something out for each DM
                if (mean == NULL) {

                    double total = 0.0, total2 = 0.0;
                    for (t = 0; t < nsamp; t++) {
                        total += buffer[index_dm + t];
                        total2 += pow(buffer[index_dm + t], 2);
                    }

                    localmean = total / nsamp;
                    localrms = sqrt(total2/nsamp - pow(localmean,2));
                }

                if ( k == 100)
                    printf(" Mean:%f StdDev:%f \n", localmean,localrms);

                for(t = 0; t < nsamp - 4; t++) {
                    float themedian, a, b, c, d, e, thisdm;
                    a = buffer[index_dm + t + 0];
                    b = buffer[index_dm + t + 1];
                    c = buffer[index_dm + t + 2];
                    d = buffer[index_dm + t + 3];
                    e = buffer[index_dm + t + 4];
                    themedian = medianOfFive (a, b, c, d, e );

                    // detection_threshold sigma filter
                    thisdm = startdm + k * dmstep;
                    if (mean == NULL)
                    {
                        if (themedian - localmean >= localrms * survey -> detection_threshold && thisdm > 1.0 )
                            
                            fprintf(output, "%lf, %f, %f\n", 
                                    timestamp + t * blockRate * survey -> pass_parameters[i].binsize,
                                    thisdm, themedian / localrms);
                    }
                    else 
                        if (themedian - mean[t / survey -> samplesPerChunk] >= 
                            rms[t / survey -> samplesPerChunk] * survey -> detection_threshold && thisdm > 1.0 )
                            
                            fprintf(output, "%lf, %f, %f\n", 
                                    timestamp + t * blockRate * survey -> pass_parameters[i].binsize,
                                    thisdm, themedian / rms[t / survey -> samplesPerChunk]);
                  }
              }
         }
    }
    shift += nsamp * ndms;
}

// Process brute force dedispersion if that was chosen
void process_brute(float *buffer, FILE* output, SURVEY *survey, int read_nsamp, size_t size, double timestamp, double blockRate, time_t start_time,
                   float* mean, float* rms)
{
    unsigned int j, k, l, iters, vals, mod_factor;
    float localmean = 0, localrms = 0;
	double total, total2, mean2;

    // Do we have useful values for the mean and rms?
    // If not, work something out for each DM
    if (mean == NULL) {
      
        // Calculate the total number of values
        vals = read_nsamp * survey -> tdms;
        mod_factor = vals < 32 * 1024 ? vals : 32 * 1024;
      
        // Calculate the mean
        iters = 0;
        mean2 = 0;
        localmean = 0;
        while(1) {
            total  = 0;
            total2 = 0;
            for(j = 0; j < mod_factor; j++){
               total += buffer[iters * mod_factor + j];
                total2 += pow(buffer[iters * mod_factor + j],2);
            }
          localmean += (total / j);
          mean2 += (total2/j);
          iters++;
          if (iters * mod_factor + j >= vals) break;
        }

        localmean /= iters;  // Mean for entire array
        localrms = sqrt(mean2/iters - localmean*localmean);
        printf("%d: Mean: %f, Stddev: %f\n", (int) (time(NULL) - start_time), localmean, localrms);
    }
        
    // Subtract dm mean from all samples and apply threshold
	unsigned thread;
	int thread_shift = survey -> tdms * survey -> dmstep / survey -> num_threads;

	for(thread = 0; thread < survey -> num_threads; thread++) {
        for (k = 0; k < survey -> tdms / survey -> num_threads; k++) {

            int index = size * thread + k * survey -> nsamp;
            for(l = 0; l < survey -> nsamp - 4; l++) {

                float themedian, a, b, c, d, e, thisdm;
                a = buffer[index + l + 0];
                b = buffer[index + l + 1];
                c = buffer[index + l + 2];
                d = buffer[index + l + 3];
                e = buffer[index + l + 4];
                themedian = medianOfFive (a, b, c, d, e );

                // detection_threshold sigma filter
                thisdm = survey -> lowdm + (thread_shift * thread) + k * survey -> dmstep;
                if (mean == NULL) {
                  if (themedian - localmean >= localrms * survey -> detection_threshold && thisdm > 1.0 )
                      fprintf(output, "%lf, %f, %f\n", 
                          timestamp + l * blockRate,
                          thisdm, themedian/localrms);

                }
                else
                    if (themedian - mean[l / survey -> samplesPerChunk] >= 
                            rms[l / survey -> samplesPerChunk] * survey -> detection_threshold && thisdm > 1.0 )

                    fprintf(output, "%lf, %f, %f, %f, %f  \n", 
                                  timestamp + l * blockRate, thisdm,
                                  (themedian - mean[l / survey -> samplesPerChunk])
                                  / rms[l / survey -> samplesPerChunk],
                                  mean[l / survey -> samplesPerChunk],
                                  rms[l / survey -> samplesPerChunk]);
            }
        }   
    }
    fflush(output);
}

// Process dedispersion output
void* process_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    SURVEY *survey = params -> survey;
    int i, iters = 0, ret, loop_counter = 0, pnsamp = params -> survey -> nsamp;
    int ppnsamp = params -> survey-> nsamp;
    time_t start = params -> start, beg_read;
    float* ppnoiseMean = NULL, *pnoiseMean = NULL;
    float* ppnoiseRMS = NULL, *pnoiseRMS = NULL;
    double pptimestamp = 0, ptimestamp = 0;
    double ppblockRate = 0, pblockRate = 0;
    long written_samples = 0;
    FILE *fp = NULL;

    printf("%d: Started output thread\n", (int) (time(NULL) - start));

    // Single output file mode, create file
    if (survey -> single_file_mode) {
        char pathName[256];
        strcpy(pathName, survey -> basedir);
        strcat(pathName, "/");
        strcat(pathName, survey -> fileprefix);
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
            if (written_samples == 0 && !(survey -> single_file_mode)) {
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
                strcat(pathName, ".dat");

                fp = fopen(pathName, "w");
            }

            beg_read = time(NULL);

            if (params -> survey -> useBruteForce)
              process_brute(params -> output_buffer, fp, params -> survey,  ppnsamp,
                            params -> dedispersed_size, pptimestamp, ppblockRate, start, ppnoiseMean, ppnoiseRMS);
            else {
              process_subband(params -> output_buffer, fp, params -> survey,  ppnsamp,
                              params -> dedispersed_size, pptimestamp, ppblockRate, ppnoiseMean, ppnoiseRMS);
            }
            printf("%d: Processed output %d [output]: %d\n", (int) (time(NULL) - start), loop_counter,
                   (int) (time(NULL) - beg_read));
            
            if (!(survey -> single_file_mode)) {
              written_samples += ppnsamp;
              if (written_samples * ppblockRate > survey -> secs_per_file) {
                written_samples = 0;
                fclose(fp);
              }
            }
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

        ppnoiseMean = pnoiseMean;
        pnoiseMean = params -> survey -> noiseMean;
        ppnoiseRMS = pnoiseRMS;
        pnoiseRMS = params -> survey -> noiseRMS;


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

    printf("%d: Exited gracefully [output]\n", (int) (time(NULL) - start));
    pthread_exit((void*) output_params);
}
