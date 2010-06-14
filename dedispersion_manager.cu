#include "dedispersion_thread.cu"
#include "dedispersion_output.cu"
#include "input_handler.c"
#include "file_handler.c"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"

#define TRUE 1

// Global parameters
float fch1, foff, tsamp, dmstep = 0.1, startdm = 0;
int nbits, nchans, nsamp = 0, tdms = 1000, nsubs = 32;
FILE *fp = NULL;
SURVEY *survey;


void report_error(char* description)
{
   fprintf(stderr, description);
   exit(0);
}

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;

    survey = process_parameter_file(NULL);
    
    while((fopen(argv[i], "rb")) != NULL) {
        if (fp != NULL) {
            fprintf(stderr, "Only one file can be processed!\n");
            exit(0);
        }
        
        fp = fopen(argv[i], "rb");
        FILE_HEADER *header = read_header(fp);
        nchans = header -> nchans; 
        tsamp = header -> tsamp;
        fch1 = header -> fch1; 
        foff = header -> foff;
        nbits = header -> nbits;
        i++;
    }

    while(i < argc) { 
       if (!strcmp(argv[i], "-tdms"))
           tdms = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-dmstep"))
           dmstep = atof(argv[++i]);
       else if (!strcmp(argv[i], "-startdm"))
           startdm = atof(argv[++i]);     
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);  
       else if (!strcmp(argv[i], "-nsubs"))
           nsubs = atoi(argv[++i]); 
       else
           report_error("Invalid parameter");
       i++;
    }

    survey -> nsamp = nsamp;
    survey -> nchans = nchans;
    survey -> tsamp = tsamp;
    survey -> fch1 = fch1;
    survey -> foff = foff;
    survey -> nsubs = nsubs;
    survey -> tdms = 0;

    for(i = 0; i < survey -> num_passes; i++)
        survey -> tdms += survey -> pass_parameters[i].ndms;
}

// Fill buffer with data (blocking call)
int generate_data(float* buffer, int nsamp, int nchans)
{
    for(int i = 0; i < nsamp * nchans; i++)
        buffer[i] = 1;

    return nsamp;
}

// Load from binary file
int get_data_file_binary(float *buffer, FILE *fp, int nsamp, int nchans)
{
    return read_block(fp, nbits, buffer, nsamp * nchans) / nchans;
}

// Query CUDA-capable devices
DEVICE_INFO** initialise_devices(int *num_devices)
{
    // Enumerate devices and create DEVICE_INFO list, storing device capabilities
    cutilSafeCall(cudaGetDeviceCount(num_devices));

    if (*num_devices <= 0) 
        report_error("No CUDA-capable device found");

    DEVICE_INFO **info = (DEVICE_INFO **) malloc( *num_devices * sizeof(DEVICE_INFO *));

    int orig_num = *num_devices, counter = 0;
    for(int i = 0; i < orig_num; i++) {
        cudaDeviceProp deviceProp;
        cutilSafeCall(cudaGetDeviceProperties(&deviceProp, i));
        
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            report_error("No CUDA-capable device found");
        else {
            if (deviceProp.totalGlobalMem < (long) 2 * 1024 * 1024 * 1024)
                *num_devices = *num_devices - 1;
            else {
                info[counter] = (DEVICE_INFO *) malloc(sizeof(DEVICE_INFO));
                info[counter] -> multiprocessor_count = deviceProp.multiProcessorCount;
                info[counter] -> constant_memory = deviceProp.totalConstMem;
                info[counter] -> shared_memory = deviceProp.sharedMemPerBlock;
                info[counter] -> register_count = deviceProp.regsPerBlock;
                info[counter] -> thread_count = deviceProp.maxThreadsPerBlock;
                info[counter] -> clock_rate = deviceProp.clockRate;
                info[counter] -> device_id = i;
                counter++;
            }
        }
    }

    if (*num_devices == 0)
        report_error("No CUDA-capable device found");

    *num_devices = 1;

    // OPTIONAL: Perform load-balancing calculations
    return info;
}

// Calculate number of samples which can be loaded at once
int calculate_nsamp(int maxshift, size_t *inputsize, size_t* outputsize)
{
    unsigned int i, input = 0, output = 0, chans = 0;

    for(i = 0; i < survey -> num_passes; i++) {
        input += nsubs * survey -> pass_parameters[i].ncalls / survey -> pass_parameters[i].binsize;
        output += survey -> pass_parameters[i].ndms / survey -> pass_parameters[i].binsize;
        chans += nchans / survey -> pass_parameters[i].binsize;
    }
    if (nsamp == 0) nsamp = ((1024 * 1024 * 1000) / (max(input, chans) + max(output, input))) - maxshift;

    // Round down nsamp to multiple of the largest binsize
    if (nsamp % survey -> pass_parameters[survey -> num_passes - 1].binsize != 0)
        nsamp -= nsamp % survey -> pass_parameters[survey -> num_passes - 1].binsize;

    // TODO: Correct maxshift calculation (when applied to input variable)
    *inputsize = (max(input, chans) * nsamp + maxshift * max(input, nchans)) * sizeof(float);  
    *outputsize = max(output, input) * (nsamp + maxshift) * sizeof(float);
    printf("Input size: %d MB, output size: %d MB\n", (int) (*inputsize / 1024 / 1024), (int) (*outputsize/1024/1024));

    return nsamp;
}

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
}

int main(int argc, char *argv[])
{
    // Initialise Generic variables
    int i, ret, ndms, maxshift, num_devices;
    time_t start = time(NULL), begin;

    // Initialise parameters
    process_arguments(argc, argv);

    // Initialise devices/thread-related variables
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

    pthread_t output_thread;
    DEVICE_INFO** devices = initialise_devices(&num_devices);
    pthread_t* threads = (pthread_t *) calloc(sizeof(pthread_t), num_devices);
    THREAD_PARAMS* threads_params = (THREAD_PARAMS *) malloc(num_devices * sizeof(THREAD_PARAMS));

    // Calculate temporary DM-shifts
    float *dmshifts = (float *) malloc(nchans * sizeof(float));
    for (i = 0; i < nchans; i++)
          dmshifts[i] = dmdelay(fch1 + (foff * i), fch1);

    // Calculate maxshift (maximum for all threads)
    // TODO: calculate proper maxshift
    maxshift = dmshifts[nchans - 1] * survey -> pass_parameters[survey -> num_passes - 1].highdm / tsamp;  

    // Calculate nsamp
    size_t *inputsize = (size_t *) malloc(sizeof(size_t));
    size_t *outputsize = (size_t *) malloc(sizeof(size_t));
   
    nsamp = calculate_nsamp(maxshift, inputsize, outputsize);
    survey -> nsamp = nsamp;

    // Initialise buffers and create output buffer (a separate buffer for each GPU output)
    // TODO: Need to realloc input buffer, due to extra buffer space in first iteration
    // TODO: Change to use all GPUs
    float* input_buffer = (float *) malloc(*inputsize);
    float** output_buffer = (float **) malloc(num_devices * sizeof(float *));
    for (i = 0; i < num_devices; i++)
        output_buffer[i] = (float *) malloc(*outputsize);

    // Log parameters
    printf("nchans: %d, nsamp: %d, tsamp: %f, foff: %f\n", nchans, nsamp, tsamp, foff);
    printf("ndms: %d, max dm: %f, maxshift: %d\n", survey -> tdms, survey -> pass_parameters[survey -> num_passes - 1].highdm, maxshift);

    // Thread-Synchronistion objects
    pthread_rwlock_t rw_lock = PTHREAD_RWLOCK_INITIALIZER;
    pthread_barrier_t input_barrier, output_barrier;

    if (pthread_barrier_init(&input_barrier, NULL, num_devices + 2))
        report_error("Unable to initialise input barrier\n");

    if (pthread_barrier_init(&output_barrier, NULL, num_devices + 2))
        report_error("Unable to initialise output barrier\n");

    // Create output params and output file
    OUTPUT_PARAMS output_params = {nchans, nsamp, num_devices, 2, 2, startdm, dmstep, survey -> tdms, output_buffer, 
                                   0, &rw_lock, &input_barrier, &output_barrier, start, fopen("output.dat", "w"), survey };

    // Create output thread 
    if (pthread_create(&output_thread, &thread_attr, process_output, (void *) &output_params))
        report_error("Error occured while creating output thread\n");

    // Create threads and assign devices
    for(i = 0; i < num_devices; i++) {

        // Create THREAD_PARAMS for thread, based on input data and DEVICE_INFO
        threads_params[i].iterations = 1;
        threads_params[i].maxiters = 2;
        threads_params[i].stop = 0;
        threads_params[i].nchans = nchans;
        threads_params[i].nsamp = nsamp;
        threads_params[i].tsamp = tsamp;
        threads_params[i].maxshift = maxshift;
        threads_params[i].binsize = 1;
        threads_params[i].output = output_buffer[i];
        threads_params[i].input = input_buffer;
        threads_params[i].ndms = ndms;
        threads_params[i].dmshifts = dmshifts;
        threads_params[i].startdm = startdm + (i * ndms * dmstep);
        threads_params[i].dmstep = dmstep;
        threads_params[i].thread_num = i;
        threads_params[i].device_id = devices[i] -> device_id;
        threads_params[i].rw_lock = &rw_lock;
        threads_params[i].input_barrier = &input_barrier;
        threads_params[i].output_barrier = &output_barrier;
        threads_params[i].start = start;
        threads_params[i].survey = survey;
        threads_params[i].inputsize = *inputsize;
        threads_params[i].outputsize = *outputsize;

         // Create thread (using function in dedispersion_thread)
         if (pthread_create(&threads[i], &thread_attr, dedisperse, (void *) &threads_params[i]))
             report_error("Error occured while creating thread\n");
    }
    
    // Main processing loop
    int data_read = 0, loop_counter = 0;
    while(TRUE) {

        // Wait input barrier
        ret = pthread_barrier_wait(&input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            report_error("Error during barrier synchronisation\n");         

        begin = time(NULL);
        if (loop_counter == 0)
            // First iteration, need to read size of buffer + maxshift
            data_read = get_data_file_binary(input_buffer, fp, nsamp + maxshift, nchans) - maxshift;
        else
            // Get next data set (buffer)
            data_read = get_data_file_binary(input_buffer, fp, nsamp, nchans);

        printf("%d: Read %d * 1024 samples: %d [%d]\n", (int) (time(NULL) - start),  data_read / 1024, 
                                                        (int) (time(NULL) - begin), loop_counter);  

        // Lock thread params through rw_lock
        if (pthread_rwlock_wrlock(&rw_lock))
            report_error("Error acquiring rw lock");

        // Wait output barrier
        ret = pthread_barrier_wait(&output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            report_error("Error during barrier synchronisation\n");    


        // Stopping clause
        if (data_read == 0) { 
            output_params.stop = 1;
            for(i = 0; i < num_devices; i++) 
                threads_params[i].stop = 1;

            // Release rw_lock
            if (pthread_rwlock_unlock(&rw_lock))
                report_error("Error releasing rw_lock\n");

            // Reach barriers maxiters times to wait for rest to process
            for(i = 0; i < 2 - 1; i++) {
                pthread_barrier_wait(&input_barrier);
                pthread_barrier_wait(&output_barrier);
            }  
            break;

        // Update thread params
        } else if (data_read < nsamp) {

          // Round down nsamp to multiple of the largest binsize
          if (data_read % survey -> pass_parameters[survey -> num_passes - 1].binsize != 0)
              data_read -= data_read % survey -> pass_parameters[survey -> num_passes - 1].binsize;

            output_params.nsamp = data_read;
            output_params.survey -> nsamp = data_read;
            for(i = 0; i < num_devices; i++) {
                threads_params[i].nsamp = data_read;
                threads_params[i].survey -> nsamp = data_read;
            }
        }

        // Release rw_lock
        if (pthread_rwlock_unlock(&rw_lock))
            report_error("Error releasing rw_lock\n");

        loop_counter++;
    }

    //Join all threads, making sure they had a clean cleanup
    void *status;
    for(i = 0; i < num_devices; i++)
        if (pthread_join(threads[i], &status))
            report_error("Error while joining threads\n");
    pthread_join(output_thread, &status);
    
    // Destroy attributes and synchronisation objects
    pthread_attr_destroy(&thread_attr);
    pthread_rwlock_destroy(&rw_lock);
    pthread_barrier_destroy(&input_barrier);
    pthread_barrier_destroy(&output_barrier);
    
    // Free memory
    for(i = 0; i < num_devices; i++) {
       free(output_buffer[i]);
       free(devices[i]);
    }

    free(output_buffer);
    free(threads_params);
    free(devices);
    free(input_buffer);
    free(dmshifts);
    free(threads);

    printf("%d: Finished Process\n", (int) (time(NULL) - start));
}

