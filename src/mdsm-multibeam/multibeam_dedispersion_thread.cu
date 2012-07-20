#include "multibeam_dedispersion_kernel.cu"
#include "multibeam_dedispersion_thread.h"

// ===================== CUDA HELPER FUNCTIONS ==========================

// Error checking function
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) _cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    _cudaCheckError( __FILE__, __LINE__ )

inline void _cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void _cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// List devices and assign to process
DEVICES* initialise_devices(SURVEY* survey)
{
	int num_devices;

    // Enumerate devices and create DEVICE_INFO list, storing device capabilities
    cutilSafeCall(cudaGetDeviceCount(&num_devices));

    if (num_devices <= 0)
        { fprintf(stderr, "No CUDA-capable device found"); exit(0); }

    // Create and populate devices object
    DEVICES* devices = (DEVICES *) malloc(sizeof(DEVICES));
    devices -> devices = (DEVICE_INFO *) malloc(num_devices * sizeof(DEVICE_INFO));
    devices -> num_devices = 0;
    devices -> minTotalGlobalMem = (1024 * 1024 * 16);

    int orig_num = num_devices, counter = 0;
    char useDevice = 0;
    for(int i = 0; i < orig_num; i++) 
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        useDevice = 0;
        
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            { fprintf(stderr, "No CUDA-capable device found"); exit(0); }
        else {

            // Check if device is in user specfied list, if any
            if (survey -> gpu_ids != NULL) {
                for(unsigned j = 0; j < survey -> num_gpus; j++)
                    if ((survey -> gpu_ids)[j] == i)
                        useDevice = 1;
            }
            else
                useDevice = 1;

            if (useDevice) 
            {
	            (devices -> devices)[counter].multiprocessor_count = deviceProp.multiProcessorCount;
	            (devices -> devices)[counter].constant_memory = deviceProp.totalConstMem;
	            (devices -> devices)[counter].shared_memory = deviceProp.sharedMemPerBlock;
	            (devices -> devices)[counter].register_count = deviceProp.regsPerBlock;
	            (devices -> devices)[counter].thread_count = deviceProp.maxThreadsPerBlock;
	            (devices -> devices)[counter].clock_rate = deviceProp.clockRate;
	            (devices -> devices)[counter].device_id = i;

	            if (deviceProp.totalGlobalMem / 1024 < devices -> minTotalGlobalMem)
		            devices -> minTotalGlobalMem = deviceProp.totalGlobalMem / 1024;

	            counter++;
                (devices -> num_devices)++;
            }
        }
    }

    if (devices -> num_devices == 0) 
        { fprintf(stderr, "No CUDA-capable device found"); exit(0); }

    return devices;
}



// Allocate memory-pinned input buffer
void allocateInputBuffer(float **pointer, size_t size)
{  CudaSafeCall(cudaMallocHost((void **) pointer, size, cudaHostAllocPortable));  }

// Allocate memory-pinned output buffer
void allocateOutputBuffer(float **pointer, size_t size)
{ CudaSafeCall(cudaMallocHost((void **) pointer, size, cudaHostAllocPortable)); }

// =================================== CUDA KERNEL HELPERS ====================================

// Cache-optimised brute force dedispersion algorithm on the GPU
void cached_brute_force(float *d_input, float *d_output, float *d_dmshifts, THREAD_PARAMS* params, 
                        cudaEvent_t event_start, cudaEvent_t event_stop, int maxshift)
{
    SURVEY *survey = params -> survey;

    int num_reg         = NUMREG;
    int divisions_in_t  = DIVINT;
    int divisions_in_dm = DIVINDM;
    int num_blocks_t    = (survey -> nsamp / (divisions_in_t * num_reg));
    int num_blocks_dm   = survey -> tdms / divisions_in_dm;

    float timestamp;       
    dim3 threads_per_block(divisions_in_t, divisions_in_dm);
    dim3 num_blocks(num_blocks_t,num_blocks_dm); 

    cudaEventRecord(event_start, 0);	

    cache_dedispersion<<< num_blocks, threads_per_block >>>
                      (d_output, d_input, d_dmshifts, survey -> nsamp, 
                       survey -> nchans, survey -> lowdm / survey -> tsamp, 
                       survey -> dmstep/survey -> tsamp, maxshift);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Brute-Force Dedispersion [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), params -> thread_num, timestamp);
}

// Perform median-filtering on dedispersed-time series
void apply_median_filter(float *d_input, THREAD_PARAMS* params, 
                         cudaEvent_t event_start, cudaEvent_t event_stop)
{
    SURVEY *survey = params -> survey;
    float timestamp;       

    cudaEventRecord(event_start, 0);	

    dim3 num_blocks(survey -> nsamp / MEDIAN_THREADS, survey -> tdms); 
    median_filter<<< num_blocks, MEDIAN_THREADS>>>(d_input, survey -> nsamp);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Median-Filtering [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), params -> thread_num, timestamp);
}

// =================================== CUDA CPU THREAD MAIN FUNCTION ====================================
void* dedisperse(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    BEAM beam = (params -> survey -> beams)[params -> thread_num];
    int i, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans;
    int loop_counter = 0, maxshift = beam.maxshift, iters = 0, tid = params -> thread_num;
    time_t start = params -> start;
    float *d_input, *d_output, *d_dmshifts;

    printf("%d: Started thread %d\n", (int) (time(NULL) - start), tid);

    // Initialise device
    CudaSafeCall(cudaSetDevice(beam.gpu_id));
    cudaSetDeviceFlags( cudaDeviceBlockingSync );

    // Allocate device memory and copy dmshifts and dmvalues to constant memory
    CudaSafeCall(cudaMalloc((void **) &d_input, params -> inputsize));
    CudaSafeCall(cudaMalloc((void **) &d_output, params -> outputsize));
    CudaSafeCall(cudaMalloc((void **) &d_dmshifts, nchans * sizeof(float)));
    CudaSafeCall(cudaMemcpy(d_dmshifts, beam.dm_shifts, nchans * sizeof(float), cudaMemcpyHostToDevice));

    // Set CUDA kernel preferences
    CudaSafeCall(cudaFuncSetCacheConfig(cache_dedispersion, cudaFuncCachePreferL1 ));
    CudaSafeCall(cudaFuncSetCacheConfig(median_filter, cudaFuncCachePreferShared ));

    // Initialise events / performance timers
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start);
    cudaEventCreateWithFlags(&event_stop, cudaEventBlockingSync); // Blocking sync when waiting for kernel launches

    // Thread processing loop
    while (1)
    {
        // Read input data into GPU memory
        if (loop_counter >= params -> iterations - 1) 
        {
            cudaEventRecord(event_start, 0);
            if (loop_counter == 1)
            {
                // First iteration, just copy maxshift spectra at the end of each channel (they
                // will be copied to the front of the buffer during the next iteration)
                for (i = 0; i < nchans; i++)
                    CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i + nsamp, 
                                                 params -> input + nsamp * i + (nsamp - maxshift), 
                                                 maxshift * sizeof(float), cudaMemcpyHostToDevice));

                CudaSafeCall(cudaThreadSynchronize());  // Wait for all copies
            }
            else 
            {
                // Copy maxshift to beginning of buffer (in each channel)
                for(i = 0; i < nchans; i++)
                    CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i, 
                                                 d_input + (nsamp + maxshift) * i + nsamp, 
                                                 maxshift * sizeof(float), cudaMemcpyDeviceToDevice));

                // Wait for maxshift copying to avoid data inconsistencies
                CudaSafeCall(cudaThreadSynchronize());

                // Copy nsamp from each channel to GPU (ignoring first maxshift samples)
                for(i = 0; i < nchans; i++)
                    CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i + maxshift, 
                                                 params -> input + nsamp * i,
                                                 nsamp * sizeof(float), cudaMemcpyHostToDevice));
            }

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data to GPU [Beam %d]: %f\n", (int) (time(NULL) - start), tid, timestamp);

            // Clear GPU output buffer
            CudaSafeCall(cudaMemset(d_output, 0, params -> outputsize));
        }

        // Wait input barrier
        int ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 1 [thread]\n"); exit(0); }

        // Perform Dedispersion
        if (loop_counter >= params -> iterations)
        {
            // Perform Dedispersion
		    cached_brute_force(d_input, d_output, d_dmshifts, params, event_start, event_stop, beam.maxshift);

            // Apply median-filter
            apply_median_filter(d_output, params, event_start, event_stop);
        }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 2 [thread]\n"); exit(0); }

        if (loop_counter >= params -> iterations) 
        { 
            // Collect and write output to host memory
            // TODO: Overlap this copy with the input part of this thread
            cudaEventRecord(event_start, 0);
            CudaSafeCall(cudaMemcpy( params -> output, d_output, 
            						 params -> dedispersed_size * sizeof(float),
                                     cudaMemcpyDeviceToHost));
            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data from GPU [Beam %d]: %f\n", (int) (time(NULL) - start), tid, timestamp);
        }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [thread]\n"); exit(0); }

        // Update params  
        nsamp = params -> survey -> nsamp;

        // Stopping clause
        if (((THREAD_PARAMS *) thread_params) -> stop) 
        {
            if (iters >= params -> iterations - 1)
            {
                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [thread]\n"); exit(0); }

                for(i = 0; i < params -> maxiters - params -> iterations ; i++) 
                {
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
            { fprintf(stderr, "Error releasing rw_lock [thread]\n"); exit(0); }

        loop_counter++;
    }   

    CudaSafeCall(cudaFree(d_output));
    CudaSafeCall(cudaFree(d_input));
    cudaEventDestroy(event_stop);
    cudaEventDestroy(event_start); 

    printf("%d: Exited gracefully %d\n", (int) (time(NULL) - start), tid);
    pthread_exit((void*) thread_params);
}
