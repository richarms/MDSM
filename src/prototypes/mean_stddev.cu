#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "time.h"

#define NUM_THREADS 512
#define MEDIAN_WIDTH 5
#define NUM_BLOCKS  32

int nsamp = 262144 * 32;

__global__ void mean_stddev(float *input, float2 *stddev, const int nsamp)
{
    // Declare shared memory to store temporary mean and stddev
    __shared__ float local_mean[NUM_THREADS];
    __shared__ float local_stddev[NUM_THREADS];

    // Initialise shared memory
    local_mean[threadIdx.x] = local_stddev[threadIdx.x] = 0;

    // Synchronise threads
    __syncthreads();

    // Loop over samples
    for(unsigned s = threadIdx.x + blockIdx.x * blockDim.x; 
                 s < nsamp; 
                 s += blockDim.x * gridDim.x)
    {
        float val = input[s];
        local_mean[threadIdx.x] += val;
        local_stddev[threadIdx.x] += val * val;
    }

    // Use reduction to calculate block mean and stddev
	for (unsigned i = NUM_THREADS / 2; i >= 1; i /= 2)
	{
		if (threadIdx.x < i)
		{
            local_mean[threadIdx.x] += local_mean[threadIdx.x + i];
            local_stddev[threadIdx.x] += __powf(local_stddev[threadIdx.x + i], 2);
		}
		
		__syncthreads();
	}

    // Finally, return temporary standard deviation value
    float2 def = {1,2};
    if (threadIdx.x == 0)
    {
        float2 vals = { local_mean[0], local_stddev[0] };
        stddev[blockIdx.x] = def;//vals;
 printf("%d done %f %f\n", blockIdx.x, vals.x, vals.y);
    }
}

int main(int argc, char *argv[])
{
	float  *input, *d_input;
    float2 *stddev, *d_stddev;
	int i, j;

	// Initialise
	cudaSetDevice(0);
	cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
	
	// Allocate and generate buffers
    cudaMallocHost((void **) &input, nsamp * sizeof(float));
    cudaMallocHost((void **) &stddev, NUM_BLOCKS * sizeof(float2));

    for(j = 0; j < nsamp; j++)
        input[j] = 5;

	printf("Number of samples: %d\n", nsamp);

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start); 
	cudaEventCreate(&event_stop); 

	// Allocate GPU memory and copy data
    cudaEventRecord(event_start, 0);
	cudaMalloc((void **) &d_input, nsamp * sizeof(float) );
	cudaMalloc((void **) &d_stddev, NUM_BLOCKS * sizeof(float) );
    cudaMemcpy(d_input, input, nsamp * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

	// Call kernel
	cudaEventRecord(event_start, 0);
    printf("blocks: %d, threads: %d\n", NUM_BLOCKS, NUM_THREADS);
    mean_stddev<<<NUM_BLOCKS, NUM_THREADS>>>(d_input, d_stddev, nsamp);
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Calculated mean and standard deviation: %lf\n", timestamp);

	// Get output 
	cudaMemcpy(stddev, d_stddev, NUM_BLOCKS * sizeof(float2), cudaMemcpyDeviceToHost);
    printf("Copied back results\n");

    // Calculate final mean and standard deviation
    float mean = 0, std = 0;
    for(i = 0; i < NUM_BLOCKS; i++)
    {
        printf("%f %f\n", stddev[i].x, stddev[i].y);
        mean += stddev[i].x;
        std  += pow(stddev[i].y, 2);
    }
    mean /= nsamp;
    std   = sqrt((std / nsamp)- mean * mean);

    printf("Mean: %f, stddev: %f\n", mean, std);
}
