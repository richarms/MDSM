#include "dedispersion_kernel.cu"
#include "dedispersion_thread.h"
#include "math.h"

FILE *fp = NULL;

// Folding arrays
int *pPrevDiff, *pShifts;

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
    for(int i = 0; i < orig_num; i++) {
        cudaDeviceProp deviceProp;
        cutilSafeCall(cudaGetDeviceProperties(&deviceProp, i));
        useDevice = 0;
        
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            { fprintf(stderr, "No CUDA-capable device found\n"); exit(0); }
        else if (deviceProp.totalGlobalMem / 1024 > 1024 * 3.5 * 1024) {

            // Check if device is in user specfied list, if any
            if (survey -> gpu_ids != NULL) {
                for(unsigned j = 0; j < survey -> num_gpus; j++)
                    if ((survey -> gpu_ids)[j] == i)
                        useDevice = 1;
            }
            else
                useDevice = 1;

            if (useDevice) {
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

// Perform subband dedispersion
void subband_dedispersion(float *d_input, float *d_output, THREAD_PARAMS* params, 
                          cudaEvent_t event_start, cudaEvent_t event_stop)
{
	// Declare function variables
    int maxshift = params -> maxshift, tid = params -> thread_num, num_threads = params -> num_threads;
    int i, j, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans;
    float tsamp = params -> survey -> tsamp;
    SURVEY *survey = params -> survey;
    time_t start = params -> start;
    float timestamp;

    // Define kernel thread configuration
    int blocksize_dedisp = 128; // gridsize_dedisp = 128, 
    dim3 gridDim_bin(128, (nchans / 128.0) < 1 ? 1 : nchans / 128.0);
    dim3 blockDim_bin(min(nchans, 128), 1);

    // Survey parameters
    int lobin = survey -> pass_parameters[0].binsize;
    int binsize, inshift, outshift, kernelBin;

	// ------------------------------------- Perform downsampling on GPU --------------------------------------
	// All input data is copied to all GPUs, so we need to perform binning on all of them

	cudaEventRecord(event_start, 0);
	binsize = lobin; inshift = 0, outshift = 0, kernelBin = binsize;
	for( i = 0; i < survey -> num_passes; i++) {

		if (binsize != 1) {        // if binsize is 1, no need to perform binning
			if (i == 0) {          // Original raw data not required, special case
				inplace_binning_kernel<<< gridDim_bin, blockDim_bin >>>(d_input, nsamp + maxshift, nchans, kernelBin);
				inplace_memory_reorganisation<<< gridDim_bin, blockDim_bin >>>(d_input, nsamp + maxshift, 
				                                                               nchans, kernelBin);
				cutilSafeCall( cudaMemset(d_input + (nsamp + maxshift) * nchans / binsize, 0,
										 ((nsamp + maxshift) * nchans - (nsamp + maxshift) * nchans / binsize) 
										  * sizeof(float)));
			} else {
				inshift = outshift;
				outshift += ( (nsamp + maxshift) * nchans) * 2 / binsize;
				binning_kernel<<< gridDim_bin, blockDim_bin >>>(d_input, (nsamp + maxshift) * 2 / binsize,
																nchans, kernelBin, inshift, outshift);
			}
		}

		binsize *= 2;
		kernelBin = 2;
	}

	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Processed Binning %d: %lf\n", (int) (time(NULL) - start), tid, timestamp);

	// --------------------------------- Perform subband dedispersion on GPU ---------------------------------
	cudaEventRecord(event_start, 0);

	// Handle dedispersion maxshift
	inshift = 0, outshift = 0;
	int ncalls, tempval = (int) (params -> dmshifts[(survey -> dedispSubbands - 1) * nchans / survey -> dedispSubbands]
						  * survey -> pass_parameters[survey -> num_passes - 1].highdm /
						  survey -> tsamp );
	float startdm;

	for( i = 0; i < survey -> num_passes; i++) {

		// Setup call parameters (ncalls is split among all GPUs)
		binsize = survey -> pass_parameters[i].binsize;
		ncalls = survey -> pass_parameters[i].ncalls / num_threads;
		startdm = survey -> pass_parameters[i].lowdm + survey -> pass_parameters[i].sub_dmstep * ncalls * tid;

		// Perform subband dedispersion
		opt_dedisperse_subband <<< dim3((nsamp + tempval) / binsize / blocksize_dedisp, ncalls), 
                                   blocksize_dedisp, blocksize_dedisp * survey -> dedispSubbands >>>
			    (d_output, d_input, (nsamp + tempval) / binsize, nchans, survey -> dedispSubbands,
			     startdm, survey -> pass_parameters[i].sub_dmstep,
			     tsamp * binsize, maxshift - tempval, inshift, outshift);

		outshift += (nsamp + tempval) * survey -> dedispSubbands * ncalls / binsize ;
		inshift += (nsamp + maxshift) * nchans / binsize;
	}

	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Processed Subband Dedispersion %d: %lf\n", (int) (time(NULL) - start), tid, timestamp);

	// Copy subband output as dedispersion input
	cutilSafeCall( cudaMemcpy(d_input, d_output, params -> outputsize, cudaMemcpyDeviceToDevice) );

	// ------------------------------------- Perform dedispersion on GPU --------------------------------------
	cudaEventRecord(event_start, 0);

	float dm = 0.0;
	inshift = outshift = 0;
	for (i = 0; i < survey -> num_passes; i++) {

		// Setup call parameters (ncalls is split among all GPUs)
		ncalls = survey -> pass_parameters[i].ncalls / num_threads;
		startdm = survey -> pass_parameters[i].lowdm + survey -> pass_parameters[i].sub_dmstep * ncalls * tid;
		binsize = survey -> pass_parameters[i].binsize;

		// Perform subband dedispersion for all subband calls
		for(j = 0; j < ncalls; j++) {

			dm = max(startdm + survey -> pass_parameters[i].sub_dmstep * j
				 - survey -> pass_parameters[i].calldms * survey -> pass_parameters[i].dmstep / 2, 0.0);

			opt_dedisperse_loop<<< dim3(nsamp / blocksize_dedisp, survey -> pass_parameters[i].calldms), 
                                   blocksize_dedisp, blocksize_dedisp >>>
				(d_output, d_input, nsamp / binsize, survey -> dedispSubbands,
				 tsamp * binsize, nchans /  survey -> dedispSubbands,
				 dm, survey -> pass_parameters[i].dmstep, tempval / binsize,
	  			 inshift, outshift);

			inshift += (nsamp + tempval) * survey -> dedispSubbands / binsize;
			outshift += nsamp * survey -> pass_parameters[i].calldms / binsize;
		}
	}

	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);

	printf("%d: Processed Deispersion %d: %lf\n", (int) (time(NULL) - start), tid, timestamp);
}

// Perform brute-froce dedisperion
void brute_force_dedispersion(float *d_input, float *d_output, THREAD_PARAMS* params, 
                              cudaEvent_t event_start, cudaEvent_t event_stop, int maxshift)
{
	// Define function variables;
    SURVEY *survey = params -> survey;
    float timestamp;

    // ------------------------------------- Perform dedispersion on GPU --------------------------------------
    cudaEventRecord(event_start, 0);
  
    // Optimised kernel
    opt_dedisperse_loop<<< dim3(512, survey -> tdms), 128, 128 >>>
			(d_output, d_input, survey -> nsamp, survey -> nchans,
			 survey -> tsamp, 1, survey -> lowdm, survey -> dmstep, maxshift, 0, 0);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Performed Brute-Force Dedispersion %d: %lf\n", (int) (time(NULL) - params -> start),
															          params -> thread_num, timestamp);
}

// Perform channelisation and intensity calculation on the GPU
void channelise(cufftComplex *d_input, float *d_output, THREAD_PARAMS* params,
                cudaEvent_t event_start, cudaEvent_t event_stop)
{
    // Define function variables;
    SURVEY *survey = params -> survey;
    unsigned chansPerSubband = survey -> nchans / survey -> nsubs,
             numSamples = survey -> nsamp + survey -> maxshift;
    float timestamp;

    // ------------------------------------- Perform Channelisation on GPU --------------------------------------
    cufftHandle plan;
    cufftPlan1d(&plan, chansPerSubband, CUFFT_C2C, numSamples * survey -> nsubs * survey -> npols);
       
    cudaEventRecord(event_start, 0);
    cufftExecC2C(plan, d_input, d_input, CUFFT_FORWARD); 
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Performed Channelisation in %d: %lf\n", (int) (time(NULL) - params -> start),
														       params -> thread_num, timestamp);

    // ------------------------------------- Calculate intensities on GPU --------------------------------------
                                
    // Calculate intensity and perform transpose in memory
    cudaEventRecord(event_start, 0);
    calculate_intensities<<<dim3(numSamples / 128, 1), 128>>>(d_input, d_output, numSamples, 
                                                              survey -> nsubs, chansPerSubband, survey -> npols);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Processed Intensities in %d: %lf\n", (int) (time(NULL) - params -> start),
														    params -> thread_num, timestamp);

    // Copy output to input
    cutilSafeCall( cudaMemcpy((float *) d_input, d_output, numSamples * survey -> nchans * sizeof(float), 
                              cudaMemcpyDeviceToDevice) );     

    // Temporary dump of channelised data
    if (0) {
        cudaEventRecord(event_start, 0);
        transposeDiagonal<<<dim3(numSamples/TILE_DIM, survey -> nsubs * chansPerSubband / TILE_DIM), 
                            dim3(TILE_DIM, BLOCK_ROWS) >>>
                            ((float *) d_input, d_output, numSamples, survey -> nsubs * chansPerSubband);
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        printf("%d: Performed transpose in: %lfms\n", (int) (time(NULL) - params -> start), timestamp);

        cudaMemcpy(params -> output, d_output, numSamples * survey -> nchans * sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(params -> output, sizeof(float), numSamples * survey -> nchans, fp);
        printf("Dumped channelised data to disk...\n");
        exit(-1);
    }       

    // Destroy plan                              
    cufftDestroy(plan);
}

// Separate, transpose and extract polarisations
void transpose(float *d_input, float *d_output, THREAD_PARAMS* params,
                cudaEvent_t event_start, cudaEvent_t event_stop)
{   
    // Define function variables;
    SURVEY *survey = params -> survey;
    float timestamp;
    unsigned chansPerSubband = survey -> nchans / survey -> nsubs,
             numSamples = (survey -> nsamp + survey -> maxshift) * chansPerSubband;
												   
    // -------------------------------- Separate Polarisations ---------------------------------
    cudaEventRecord(event_start, 0);
    seperateXYPolarisations<<<dim3(numSamples/survey -> nsubs, 1), survey -> nsubs >>>
                              (d_input, d_output, numSamples, survey -> nsubs);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Separated X and Y polarisations in: %lf\n", (int) (time(NULL) - params -> start), timestamp);
    
    // -------------------------------- Transpose Polarisations ---------------------------------
    cudaEventRecord(event_start, 0);
    transposeDiagonal<<<dim3(survey -> nsubs/TILE_DIM, numSamples / TILE_DIM), 
                        dim3(TILE_DIM, BLOCK_ROWS) >>>
                        (d_output, d_input, survey -> nsubs, numSamples);
    transposeDiagonal<<<dim3(survey -> nsubs/TILE_DIM, numSamples / TILE_DIM), 
                        dim3(TILE_DIM, BLOCK_ROWS) >>>
                        (d_output + numSamples * survey -> nsubs, d_input + numSamples * survey -> nsubs, 
                         survey -> nsubs, numSamples);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed transpose in: %lfms\n", (int) (time(NULL) - params -> start), timestamp);
           
    // Copy input back to output (to keep the input buffer as the largest)
    cutilSafeCall( cudaMemcpy(d_output, d_input, numSamples * survey -> nsubs * survey -> npols * sizeof(float),
                              cudaMemcpyDeviceToDevice));

    // -------------------------------- Extract Polarisations ---------------------------------
    cudaEventRecord(event_start, 0);
    expandValues<<<dim3(2048, 1), survey -> nsubs >>>
                       ((short *) d_output, d_input, numSamples * survey -> nsubs * survey -> npols);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Expanded polarisations in: %lfms\n", (int) (time(NULL) - params -> start), timestamp); 
}

// Clip data
void clip(float *d_input, float *d_means, THREAD_PARAMS* params, cudaEvent_t event_start, cudaEvent_t event_stop)
{
    SURVEY *survey = params -> survey;
    float *means = (float *) malloc(survey -> nsubs * sizeof(float));
    int samples = (survey -> nchans / survey -> nsubs) * (survey -> nsamp + survey -> maxshift);
    float timestamp;

    cudaEventRecord(event_start, 0);
    rfi_clipping<<<dim3(survey -> nsubs), 512 >>>(d_input, d_means, samples, survey -> nsubs);

    // Copy means to host memory
    cudaMemcpy(means, d_means, sizeof(float) * survey -> nsubs, cudaMemcpyDeviceToHost);

    // Calculate mean of means
    double meanOfMeans;
    for(unsigned i = 0; i < survey -> nsubs; i++)
        meanOfMeans += means[i];
    meanOfMeans /= (survey -> nsubs * 1.0);

    // Apply subband excision
//    for(unsigned i = 0; i < survey -> nsubs; i++)
//        if (means[i] < meanOfMeans / 2) {
//            cudaMemset(d_input + i * samples, 0, samples * sizeof(float)); // TODO: need to set this as meanOfMeans...
//            printf("Performed excision on subband %d\n", i);
//        }

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Clippied data in: %lfms\n", (int) (time(NULL) - params -> start), timestamp);
    
}

// Fold data
void fold(float *d_input, float *d_output, int *d_pShifts, THREAD_PARAMS* params, 
          cudaEvent_t event_start, cudaEvent_t event_stop, int counter)
{
	// Define function variables;
    SURVEY *survey = params -> survey;
    float timestamp;

    // ------------------------------------- Perform folding on GPU --------------------------------------

    cudaEventRecord(event_start, 0);
    // Calculate shifts for intra-buffer calls
    for(unsigned i = 0; i < survey -> numPeriods; i++) {
        float period = survey -> pStart + survey -> pStep * i;
        float nbins = period / survey -> tsamp;
        pShifts[i] = pPrevDiff[i] == 0 ? 0 : (int) (nbins - pPrevDiff[i]); 

        //TODO FIX THIS BIJATCH
        pPrevDiff[i] = survey -> nsamp - pShifts[i] - round(floor((survey -> nsamp - pShifts[i]) / nbins) * nbins);
    }

    // Copy shifts to GPU memory
    cutilSafeCall( cudaMemcpy(d_pShifts, pShifts, survey -> numPeriods * sizeof(int), cudaMemcpyHostToDevice));

    // Fold on GPU
    multi_fold <<<dim3(survey -> tdms / survey -> num_threads, survey -> numPeriods), 128 >>>
                 (d_input, d_output, survey -> nsamp, survey -> tsamp, survey -> pStart, 
                 survey -> pStep, params -> folded_size, d_pShifts);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Folded data in %d: %lf\n", (int) (time(NULL) - params -> start),
															   params -> thread_num, timestamp);
}

// Dedispersion algorithm
void* dedisperse(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    int i, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans;
    int ret, loop_counter = 0, maxshift = params -> maxshift, iters = 0, tid = params -> thread_num;
    time_t start = params -> start;
    SURVEY *survey = params -> survey;
    float *d_input, *d_output, *d_means;
    int *d_pShifts;

    printf("%d: Started thread %d\n", (int) (time(NULL) - start), tid);

    // Initialise device, allocate device memory and copy dmshifts and dmvalues to constant memory
    cutilSafeCall( cudaSetDevice(params -> device_id));
    cudaSetDeviceFlags( cudaDeviceBlockingSync );

    cutilSafeCall( cudaMalloc((void **) &d_input, params -> inputsize));
    cutilSafeCall( cudaMalloc((void **) &d_output, params -> outputsize));
    cutilSafeCall( cudaMemcpyToSymbol(dm_shifts, params -> dmshifts, nchans * sizeof(nchans)) );

    if (survey -> performClipping)
        cutilSafeCall ( cudaMalloc((void **) &d_means, survey -> nsubs * sizeof(float)) );

    if (survey -> performFolding)
        cutilSafeCall ( cudaMalloc((void **) &d_pShifts, survey -> numPeriods * sizeof(int)));

    // Temporary output file
    fp = fopen("channelisedOutput.dat", "wb");

    // Initialise events / performance timers
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start);
    cudaEventCreateWithFlags(&event_stop, cudaEventBlockingSync); // Blocking sync when waiting for kernel launches

    // Initialise folding arrays if required
    if (survey -> performFolding) {
        pPrevDiff = (int *) malloc(survey -> numPeriods * sizeof(int));
        pShifts = (int *) malloc(survey -> numPeriods * sizeof(int));
        memset(pPrevDiff, 0, survey -> numPeriods * sizeof(int));
        memset(pShifts, 0, survey -> numPeriods * sizeof(int));
    }  

    // Thread processing loop
    while (1) {

        if (loop_counter >= params -> iterations) {

            // Read input data into GPU memory
            cudaEventRecord(event_start, 0);
            cudaMemset(d_input, 0, params -> inputsize);

            // If performing channelisation, input data will contain npols polarisations with complex 32-bit values
            if (survey -> performTranspose)
                cutilSafeCall( cudaMemcpy(d_input, params -> input, survey -> npols * sizeof(float) *
                                         (nsamp + maxshift) * nchans, cudaMemcpyHostToDevice) );  
            
            // If performing channelisation, input data will contain npols polarisations with complex 64-bit values
            else if (survey -> performChannelisation)
                cutilSafeCall( cudaMemcpy(d_input, params -> input, survey -> npols * sizeof(cufftComplex) *
                                         (nsamp + maxshift) * nchans, cudaMemcpyHostToDevice) );

            // Dedispersing data only
            else
                cutilSafeCall( cudaMemcpy(d_input, params -> input, (nsamp + maxshift) * nchans * sizeof(float), 
                                          cudaMemcpyHostToDevice) );
    
            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data to GPU %d: %f\n", (int) (time(NULL) - start), tid, timestamp);
        }

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 1 [thread]\n"); exit(0); }

        // Perform operations on GPU data
        if (loop_counter >= params -> iterations){

            // Perform matrix transpose if required
            if (survey -> performTranspose)
                transpose(d_input, d_output, params, event_start, event_stop);

            // Perform channelisation and calculate intensities if required
            if (survey -> performChannelisation)
                channelise((cufftComplex*) d_input, d_output, params, event_start, event_stop);

            // Perform RFI clipping if required
            if (survey -> performClipping)
                clip(d_input, d_means, params, event_start, event_stop);
                
            // Perform subbands dedispersion or brute force dedispersion    
        	if (survey -> useBruteForce)
        		brute_force_dedispersion(d_input, d_output, params, event_start, event_stop, maxshift);
        	else
        		subband_dedispersion(d_input, d_output, params, event_start, event_stop);

            // Fold output if required
            if (survey -> performFolding)
        		fold(d_output, d_input, d_pShifts, params, event_start, event_stop, loop_counter - params -> iterations);
        }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 2 [thread]\n"); exit(0); }

        if(loop_counter >= params -> iterations) { 

            // Collect and write output to host memory
            cudaEventRecord(event_start, 0);

            // Copy dedispersed data
            cutilSafeCall(cudaMemcpy( params -> output, d_output, 
        						      params -> dedispersed_size * sizeof(float),
                                      cudaMemcpyDeviceToHost) );

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);

            // Copy folded data
            if (survey -> performFolding) {
                cudaEventRecord(event_start, 0);
                cutilSafeCall(cudaMemcpy( params -> folded_output, d_input, 
                						  params -> folded_size * survey -> tdms / survey -> num_threads * sizeof(float),
                                          cudaMemcpyDeviceToHost) );
                cudaEventRecord(event_stop, 0);
                cudaEventSynchronize(event_stop);
                cudaEventElapsedTime(&timestamp, event_start, event_stop);
            }
        }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [thread]\n"); exit(0); }

        // Update params  
        nsamp = params -> survey -> nsamp;

        // Stopping clause
        if (((THREAD_PARAMS *) thread_params) -> stop) {

            if (iters >= params -> iterations - 1) {  

                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [thread]\n"); exit(0); }

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
            { fprintf(stderr, "Error releasing rw_lock [thread]\n"); exit(0); }

        loop_counter++;
    }   

    cutilSafeCall( cudaFree(d_output));
    cutilSafeCall( cudaFree(d_input));
    cudaEventDestroy(event_stop);
    cudaEventDestroy(event_start); 

    printf("%d: Exited gracefully %d\n", (int) (time(NULL) - start), tid);
    pthread_exit((void*) thread_params);
}
