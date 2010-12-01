#include <cutil.h>
#include <cufft.h>
#include <sys/time.h>

// ---------------------- Intensities Calculation Loop  ------------------------------

__global__ __device__ void calculate_intensities(cufftComplex *inbuff, float *outbuff, int nsamp, 
                                                 int nsubs, int nchans, int npols)
{
    unsigned s, c, p;
    
    for(s = threadIdx.x + blockIdx.x * blockDim.x;
        s < nsamp;
        s += blockDim.x * gridDim.x)
    {
        // Loop over all channels
        for(c = 0; c < nsubs; c++) {
              
            float intensity = 0;
            cufftComplex tempval;
                
            // Loop over polarisations
            for(p = 0; p < npols; p++) {

                // Square real and imaginary parts
                tempval = inbuff[p * nsubs * nsamp + c * nsamp + s] ;
                intensity += tempval.x * tempval.x + tempval.y * tempval.y;
            }

            // Store in output buffer
            outbuff[(c * nchans + s % nchans) * (nsamp / nchans) + s / nchans ] = intensity;
        }
    }
}

// One thread block per original subband (nsamp = nsamp * chansPerSubband)
__global__ __device__ void calculate_intensities2(cufftComplex *input, float *output, float *finalOutput, 
                                                  float *means, int nsamp, int nsubs, int nchans, int npols)
{
    extern __shared__ float2 tempSums[];
    float mean, stddev;   // Store as registers to avoid bank conflicts in shared memory
    int chansPerSubband = nchans / nsubs;

    // Initial setup
    tempSums[threadIdx.x].x = 0;
    tempSums[threadIdx.x].y = 0;

    for(unsigned s =  threadIdx.x; 
                 s <  nsamp; 
                 s += blockDim.x)
    {
        float intensity = 0;
        cufftComplex tempval;

        // Square real and imaginary parts of both polarisations
        tempval = input[blockIdx.x * nsamp + s];
        intensity += tempval.x * tempval.x + tempval.y * tempval.y;
        tempval = input[nsamp * nsubs + blockIdx.x * nsamp + s];
        intensity += tempval.x * tempval.x + tempval.y * tempval.y;

        tempSums[threadIdx.x].x += intensity;
        tempSums[threadIdx.x].y += intensity * intensity;
    
        // Store in output buffer
        output[blockIdx.x * nsamp + s] = tempval.x;//intensity;
    }

    // synchronise threads
    __syncthreads();

    // TODO: use reduction to optimise this part
    if (threadIdx.x == 0) {
        for(unsigned i = 1; i < blockDim.x; i++) {
            tempSums[0].x += tempSums[i].x;
            tempSums[0].y += tempSums[i].y;
        }

        // Calculate mean and stddev
        mean   = tempSums[0].x / nsamp;
        stddev = sqrtf((tempSums[0].y - nsamp * mean * mean) / nsamp);
        means[blockIdx.x] = mean;
    }

    // Synchronise threads
    __syncthreads();  

    // Clip RFI within the subbands
    for(unsigned s =  threadIdx.x; 
                 s <  nsamp;    
                 s += blockDim.x)
    {
        float val = output[blockIdx.x * nsamp + s];
        finalOutput[blockIdx.x * nsamp + s] = 0;
//        val = ( fabs(val - mean) <= stddev * 4) ? val : mean;

        // Distribute channels acorss array
//        finalOutput[blockIdx.x * nsamp + s] = val;
         finalOutput[s]=val;//(blockIdx.x * (s % chansPerSubband)) * nsamp/chansPerSubband + s/chansPerSubband ] = val;
    }
}

// ------------------------------------------------------------------------------------

int nchans = 2048, nsamp = 32768, nsubs = 512, npols = 2;
int gridsize = 128, blocksize = 512;

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nchans"))
           nchans = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsubs"))
           nsubs = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-npol"))
           npols = atoi(argv[++i]);
      else if (!strcmp(argv[i], "-blocksize"))
           blocksize = atoi(argv[++i]);
       i++;
    }
}

// -------------------------- Main Program -----------------------------------

int main(int argc, char *argv[]) 
{
    // Initialise stuff
    process_arguments(argc, argv);
    cudaEvent_t event_start, event_stop;

    // Initialise CUDA stuff
    cudaSetDevice(2);
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 
    float timestamp;
    
    cufftComplex *d_input, *input;
    float *output, *d_output, *means, *d_means;
    int chansPerSubband = nchans/nsubs;
    
    printf("nsamp: %d, nsubs: %d, nchans: %d, npols: %d, chansPerSubband: %d\n", 
                                     nsamp, nsubs, nchans, npols, chansPerSubband);
    printf("Total size: %.2f MB\n", (sizeof(cufftComplex) * nsubs * nsamp * npols + 
                                     sizeof(float) * nsubs * nsamp + 
                                     sizeof(float) * nsubs) / (float) (1024*1024));
    // Initialise data 
    input  = (cufftComplex *) malloc(nsubs * nsamp * npols * sizeof(cufftComplex));
    output = (float *) malloc(nsubs * nsamp * sizeof(float));
    means  = (float *) malloc(nsubs * sizeof(float));

    // Enable for Intensities check
    for(unsigned p = 0; p < npols; p++)
        for(unsigned k = 0; k < nsubs; k++)
            for(unsigned j = 0; j < nsamp; j++)
            {
                input[p*nsamp*nsubs + k * nsamp + j].x = j;
                input[p*nsamp*nsubs + k * nsamp + j].y = j;
            }

    // Allocate and transfer data to GPU (nsamp * nchans * npols)
    cudaMalloc((void **) &d_input,  sizeof(cufftComplex) * nsubs * nsamp * npols);
    cudaMalloc((void **) &d_output, sizeof(float) * nsubs * nsamp);
    cudaMalloc((void **) &d_means,  sizeof(float) * nsubs);
    cudaMemset(d_input, 0, sizeof(cufftComplex) * nsubs * nsamp * npols);
    cudaMemset(d_output, 0,sizeof(float) * nsubs * nsamp);
    
    cudaEventRecord(event_start, 0);
    cudaMemcpy(d_input, input, sizeof(cufftComplex) * nsubs * nsamp * npols, cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied input to GPU in: %lfms\n", timestamp);

    // Calculate intensity
    cudaEventRecord(event_start, 0);
    calculate_intensities2<<<dim3(nsubs, 1), blocksize, blocksize >>>
                          (d_input, d_output, (float *) d_input, d_means, nsamp, nsubs, nchans, npols);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Processed Intensities in: %lfms\n", timestamp);

    // Copy means to host memory
    cudaMemcpy(means, d_means, sizeof(float) * nsubs, cudaMemcpyDeviceToHost);
//    for(unsigned i = 0; i < nsubs; i++)
//        printf("Mean %d: %f\n", i, means[i]);

    // Calculate mean of means
    double meanOfMeans;
    for(unsigned i = 0; i < nsubs; i++)
        meanOfMeans += means[i];
    meanOfMeans /= (nsubs * 1.0);
    printf("Mean of means: %lf\n", meanOfMeans);

    // Subband excision
    cudaEventRecord(event_start, 0);

//    for(unsigned i = 0; i < nsubs; i++)
//        if (means[i] > 2*meanOfMeans)
//            cudaMemset(d_output + i * nsamp, 0, nsamp);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Subband RFI excision in: %lfms\n", timestamp);

    // Get result
    cudaMemcpy(output, d_input, sizeof(float) * nsubs * nsamp, cudaMemcpyDeviceToHost);
    for(unsigned i = 0; i < nchans; i++)
        for(unsigned k = 0; k < nsamp / chansPerSubband; k++)
//            if (output[i * nsamp / chansPerSubband + k] != 2 * npols *i*i) {
                printf("%d.%d = %f\n", i, k, output[i * nsamp / chansPerSubband + k]);            
//                exit(0);
//            }
   
   // Clean up
   cudaFree(d_input);
   cudaFree(d_output);
}
