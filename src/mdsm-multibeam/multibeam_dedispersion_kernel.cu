#ifndef DEDISPERSE_KERNEL_H_
#define DEDISPERSE_KERNEL_H_

#include <cutil_inline.h>
#include "cache_brute_force.h"

// ---------------------- level_one_cache_with_accumulators_brute_force -------------------------------------
//TODO: optimise access to dm_shifts
__global__ void cache_dedispersion(float *output, float *input, float *dm_shifts, 
                                   const int nsamp, const int nchans, const float mstartdm, 
                                   const float mdmstep, const int maxshift)
{
	int   shift;	
	float local_kernel_t[NUMREG];

	int t  = blockIdx.x * NUMREG * blockDim.x  + threadIdx.x;
	
	// Initialise the time accumulators
	for(int i = 0; i < NUMREG; i++) local_kernel_t[i] = 0.0f;

	float shift_temp = mstartdm + ((blockIdx.y * blockDim.y + threadIdx.y) * mdmstep);
	
	// Loop over the frequency channels.
    for(int c = 0; c < nchans; c++) 
    {
		// Calculate the initial shift for this given frequency
		// channel (c) at the current despersion measure (dm) 
		// ** dm is constant for this thread!!**
		shift = (c * (nsamp + maxshift) + t) + __float2int_rz (dm_shifts[c] * shift_temp);
		
        #pragma unroll
		for(int i = 0; i < NUMREG; i++) {
			local_kernel_t[i] += input[shift + (i * DIVINT) ];
		}
	}

	// Write the accumulators to the output array. 
    #pragma unroll
	for(int i = 0; i < NUMREG; i++) {
		output[((blockIdx.y * DIVINDM) + threadIdx.y)* nsamp + (i * DIVINT) + (NUMREG * DIVINT * blockIdx.x) + threadIdx.x] = local_kernel_t[i];
	}
}

// -------------------- 1D Median Filter -----------------------
__global__ __device__ void median_filter(float *input, const int nsamp)
{
    // Declare shared memory array to hold local kernel samples
    // Should be (blockDim.x+width floats)
    __shared__ float local_kernel[MEDIAN_THREADS + MEDIAN_WIDTH - 1];

    // Loop over sample blocks
    for(unsigned s = threadIdx.x + blockIdx.x * blockDim.x; 
                 s < nsamp; 
                 s += blockDim.x * gridDim.x)
    {
        // Value associated with thread
        unsigned index = blockIdx.y * nsamp + s;
        unsigned wing  = MEDIAN_WIDTH / 2;

        // Load sample associated with thread into shared memory
        local_kernel[threadIdx.x + wing] = input[index];

        // Synchronise all threads        
        __syncthreads();

        // Load kernel wings into shared memory, handling boundary conditions
        // (for first and last wing elements in time series)
        if (s >= wing && s < nsamp - wing)
        {
            // Not in boundary, choose threads at the edge and load wings
            if (threadIdx.x < wing)   // Load wing element at the beginning of the kernel
                local_kernel[threadIdx.x] = input[blockIdx.y * nsamp + blockIdx.x * blockDim.x - (wing - threadIdx.x)];
            else if (threadIdx.x >= blockDim.x - wing)  // Load wing elements at the end of the kernel
                local_kernel[threadIdx.x + MEDIAN_WIDTH - 1] = input[index + wing];
        }

        // Handle boundary conditions (ignore end of buffer for now)
        else if (s < wing && threadIdx.x < wing + 1)   
            // Dealing with the first items in the input array
            local_kernel[threadIdx.x] = local_kernel[wing];
        else if (s > nsamp - wing && threadIdx.x == blockDim.x / 2)
            // Dealing with last items in the input array
            for(unsigned i = 0; i < wing; i++)
                local_kernel[MEDIAN_THREADS + wing + i] = local_kernel[nsamp - 1];

        // Synchronise all threads and start processing
        __syncthreads();

        // Load value to local registers median using "moving window" in shared memory 
        // to avoid bank conflicts
        float median[MEDIAN_WIDTH];
        for(unsigned i = 0; i < MEDIAN_WIDTH; i++)
            median[i] = local_kernel[threadIdx.x + i];

        // Perform partial-sorting on median array
        for(unsigned i = 0; i < wing + 1; i++)    
            for(unsigned j = i; j < MEDIAN_WIDTH; j++)
                if (median[j] < median[i])
                    { float tmp = median[i]; median[i] = median[j]; median[j] = tmp; }

        // We have our median, store to global memory
        input[index] = median[wing];
    }
}

// ------------ Calculate Mean and Standard Deviation -------------------
__global__ void mean_stddev(float *input, float2 *stddev, const int nsamp)
{
    // Declare shared memory to store temporary mean and stddev
    __shared__ float local_mean[MEAN_NUM_THREADS];
    __shared__ float local_stddev[MEAN_NUM_THREADS];

    // Initialise shared memory
    local_stddev[threadIdx.x] = 0;
    local_mean[threadIdx.x]   = 0;

    // Synchronise threads
    __syncthreads();

    // Loop over samples
    for(unsigned s = threadIdx.x + blockIdx.x * blockDim.x; 
                 s < nsamp; 
                 s += blockDim.x * gridDim.x)
    {
        float val = input[s];
        local_stddev[threadIdx.x] += (val * val);
        local_mean[threadIdx.x]   += val; 
    }

    // Synchronise threads
    __syncthreads();

    // Use reduction to calculate block mean and stddev
	for (unsigned i = MEAN_NUM_THREADS / 2; i >= 1; i /= 2)
	{
		if (threadIdx.x < i)
		{
            local_stddev[threadIdx.x] += local_stddev[threadIdx.x + i];
            local_mean[threadIdx.x]   += local_mean[threadIdx.x + i];
		}
		
		__syncthreads();
	}

    // Finally, return temporary standard deviation value
    if (threadIdx.x == 0)
    {
        float2 vals = { local_mean[0], local_stddev[0] };
        stddev[blockIdx.x] = vals;
    }
}

#endif
