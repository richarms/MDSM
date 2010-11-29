#include <cutil.h>
#include <cufft.h>
#include <sys/time.h>

// ---------------------- Intensities Calculation Loop  ------------------------------

// One thread block per original subband (nsamp = nsamp * chansPerSubband)
__global__ void calculate_intensities(cufftComplex *input, float *output, float *means,
                                      int nsamp, int nsubs, int npols)
{
    extern __shared__ float2 tempSums[];
    float mean, stddev;   // Store as registers to avoid bank conflicts in shared memory

    // Initial setup
    tempSums[threadIdx.x].x = 0;
    tempSums[threadIdx.x].y = 0;

    for(unsigned s = blockIdx.x * nsamp + threadIdx.x; 
                 s < (blockIdx.x + 1) * nsamp; 
                 s += blockDim.x)

    {
        float intensity = 0;
        cufftComplex tempval;
                
        // Loop over polarisations
        for(unsigned p = 0; p < npols; p++) {

            // Square real and imaginary parts
            tempval = input[p * nsubs * nsamp + s] ;
            intensity += tempval.x * tempval.x + tempval.y * tempval.y;
        }

        tempSums[threadIdx.x].x += intensity;
        tempSums[threadIdx.x].y += intensity * intensity;
    
        // Store in output buffer
        output[s] = intensity;
    }

    // synchronise threads
//    __syncthreads();
//    for(unsigned i = 0; i < blockDim.x; i++) {
//        tempSums[0].x += tempSums[i].x;
//        tempSums[0].y += tempSums[i].y;
//    }

//    // Calculate mean and stddev
//    mean   = tempSums[0].x / nsamp;
//    stddev = sqrtf((tempSums[0].y - nsamp * mean * mean) / nsamp);
//    means[blockIdx.x] = mean;

//    // Synchronise threads
//    __syncthreads();
//  
//    // Clip RFI within the subbands
//    for(unsigned s = blockIdx.x * nsamp + threadIdx.x; 
//                 s < (blockIdx.x + 1) * nsamp; 
//                 s += blockDim.x)
//    {
//        float val = output[s];
//        val = (fabs(val - mean) > stddev * 4) ? mean : val;

//        // Distribute channels acorss array
//        output[s] = val;
//    }
}

// ------------------------------------------------------------------------------------

int nchans = 2048, nsamp = 32768, nsubs = 512, npols = 2;
int gridsize = 128, blocksize = 128;

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
    cudaSetDevice(0);
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 
    float timestamp;
    
    cufftComplex *d_input, *input;
    float *output, *d_output, *means, *d_means;
    
    printf("nsamp: %d, nsubs: %d, nchans: %d, npols: %d\n", nsamp, nsubs, nchans, npols);
    printf("Input size: %f MB\n", nchans * nsamp * npols * sizeof(cufftComplex) / (float) (1024*1024));

    // Initialise data 
    input  = (cufftComplex *) malloc(nchans * nsamp * npols * sizeof(cufftComplex));
    output = (float *) malloc(nchans * nsamp * npols * sizeof(float));
    means  = (float *) malloc(nsubs * sizeof(float));

    // Enable for Intensities check
    for(unsigned p = 0; p < npols; p++)
        for(unsigned s = 0; s < nsubs; s++) {
            for(unsigned c = 0; c < nsamp * nchans/nsubs; c++)
                input[p*nsamp*nchans + s*nsamp*nchans/nsubs + c].x = 
                input[p*nsamp*nchans + s*nsamp*nchans/nsubs + c].y = s;
            }
    

    // Allocate and transfer data to GPU (nsamp * nchans * npols)
    cudaMalloc((void **) &d_input,  sizeof(cufftComplex) * nchans * nsamp * npols);
    cudaMalloc((void **) &d_output, sizeof(float) * nchans * nsamp * npols);
    cudaMalloc((void **) &d_means,  sizeof(float) * nchans);
    
    cudaEventRecord(event_start, 0);
    cudaMemcpy(d_input, input, sizeof(cufftComplex) * nchans * nsamp * npols, cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied input to GPU in: %lfms\n", timestamp);

    // Calculate intensity
    cudaEventRecord(event_start, 0);
    calculate_intensities<<<dim3(nsubs, 1), blocksize, nsubs >>>
                          (d_input, d_output, d_means, nsamp * nchans/nsubs, nsubs, npols);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Processed Intensities in: %lfms\n", timestamp);

   // Get result
   cudaMemcpy(output, d_output, sizeof(float) * nchans * nsamp, cudaMemcpyDeviceToHost);
    for(unsigned s = 0; s < nsubs; s++)
        for(unsigned c = 0; c < nsamp * nchans/nsubs; c++)
            if (output[s*nsamp*nchans/nsubs + c] != 4*c*c) {
                printf("Invalid...: %d.%d = %f\n", s,c, output[s*nsamp*nchans/nsubs + c]);
                exit(0);
             }
	   
   // Clean up
   cudaFree(d_input);
   cudaFree(d_output);
}
