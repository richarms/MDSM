#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cutil_inline.h>

// ---------------------- Optimised Dedispersion Loop  ------------------------------
__global__ void fold(float *input, float *output, int nsamp, float tsamp,
                     float startP, float deltaP, int dmShift, int *shifts)
{
    // Calculate total shift to store output for block period
    int outputShift = 0;
    for(unsigned i = 0; i < blockIdx.y; i++)
        outputShift += (int) ((startP + (deltaP * i)) / tsamp);

    // Calculate parameters for block period
    float bins = (startP + (deltaP * blockIdx.y)) / tsamp;
    int values = floorf(nsamp / bins);
    float shift = shifts[blockIdx.y];

    // Fold (bin per thread)
    for(unsigned b = threadIdx.x;
                 b < bins;
                 b += blockDim.x)
    {
        float val = 0;
        for(unsigned s = 0; s < values; s ++)
            val += input[(int) (blockIdx.x * (nsamp + shift) + shift + s * (bins) + b)];
       
        output[blockIdx.x * dmShift + outputShift + b] = val / values;
    }

}

// -------------------------- Main Program -----------------------------------

float period = 64, tsamp = 0.0000128;
int nsamp = 1024*64, tdms = 2;
int gridsize = 128, blocksize = 128;
float startP = 0, endP = 0, deltaP = 0;
int nP;

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-tdms"))
           tdms = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-gridsize"))
           gridsize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-blocksize"))
           blocksize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-tsamp"))
           tsamp = atof(argv[++i]);
       else if (!strcmp(argv[i], "-startP"))
           startP = atof(argv[++i]);
       else if (!strcmp(argv[i], "-endP"))
           endP = atof(argv[++i]);
       else if (!strcmp(argv[i], "-deltaP"))
           deltaP = atof(argv[++i]);
       i++;
    }

    if (startP == 0)
        startP = tsamp * 256;

    if (deltaP == 0)
        deltaP = tsamp;

    if (endP == 0)
        endP = startP + deltaP * 256;

    nP = (int) ((endP - startP) / tsamp);
}

int main(int argc, char *argv[])
{
    float *input, *output, *d_input, *d_output;
    int *pShifts, *d_pShifts;
    int i, j;

    process_arguments(argc, argv);

    printf("nsamp: %d, tdms: %d, tsamp: %f, startP: %f, deltaP: %f, endP: %f, nP: %d\n",
           nsamp, tdms, tsamp, startP, deltaP, endP, nP);

    // Allocate and initialise arrays
    input =  (float *) malloc( tdms * nsamp * sizeof(float));
    for(i = 0; i < tdms; i++)
        for(j = 0; j < nsamp; j++)
            input[i *nsamp + j] = i + 1;
         

    // Initialise CUDA stuff
    cutilSafeCall( cudaSetDevice(1));
    cudaEvent_t event_start, event_stop;
    float timestamp, kernelTime;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    // Callculate periodicty parameters and shifts (dummy)
    int pMem = 0;   
    for(i = 0; i < nP; i++)
        pMem += (int) ((startP + (deltaP * i)) / tsamp);

    pShifts = (int*) malloc(nP * sizeof(int));
    memset(pShifts, 0, nP * sizeof(int));

    output = (float *) malloc( tdms * pMem * sizeof(float));

    printf("Memory required for DMs: %.2f MB\n", tdms * nsamp * sizeof(float) / 1024.0 / 1024.0);
    printf("Memory required for periods: %.2f MB\n", tdms * pMem * sizeof(float) / 1024.0 / 1024.0);

    // Allocate CUDA memory
    cutilSafeCall( cudaMalloc((void **) &d_input, tdms * nsamp * sizeof(float)));
    cutilSafeCall( cudaMalloc((void **) &d_output, pMem * tdms * sizeof(float)));
    cutilSafeCall( cudaMalloc((void **) &d_pShifts, nP * sizeof(int)));
    
    time_t start = time(NULL);

    // Copy input to GPU
    cudaEventRecord(event_start, 0);
    cutilSafeCall( cudaMemcpy(d_input, input, tdms * nsamp * sizeof(float), cudaMemcpyHostToDevice) );    
    cutilSafeCall( cudaMemcpy(d_pShifts, pShifts, nP * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf\n", timestamp);

    cudaEventRecord(event_start, 0);
    fold<<<dim3(tdms, nP), blocksize>>>(d_input, d_output, nsamp, tsamp, startP, deltaP, pMem, d_pShifts);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&kernelTime, event_start, event_stop);
    printf("Folded in: %lf\n", kernelTime);

    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    cutilSafeCall( cudaMemcpy(output, d_output, tdms * pMem * sizeof(float), cudaMemcpyDeviceToHost) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied from GPU in: %lf\n", timestamp);

    printf("Total time: %d\n", (int) (time(NULL) - start));

    // Compute FLOPS
    printf("GPU Performance: %f\n", ((nsamp * tdms) / (1073741 * kernelTime)) * nP * 7);
}

