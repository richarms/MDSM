#include <cutil.h>
#include <cufft.h>
#include <sys/time.h>

// ---------------------- Intensities Calculation Loop  ------------------------------
__global__ void calculate_intensities(cufftComplex *inbuff, float *outbuff, int nsamp, 
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

int nchans = 4, nsamp = 32768, nsubs = 512, npols = 2;
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
    unsigned i, j, k;

    // Initialise CUDA stuff
    cudaSetDevice(0);
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 
    float timestamp;
    
    cufftHandle plan;
    cufftComplex *d_input, *input;
    float *output, *d_output;
    
    printf("nsamp: %d, nsubs: %d, nchans: %d, npols: %d\n", nsamp, nsubs, nchans, npols);

    // Initialise data 
    input = (cufftComplex *) malloc(nsubs * nsamp * npols * sizeof(cufftComplex));
    output = (float *) malloc(nsubs * nsamp * sizeof(float));
    for(unsigned p = 0; p < npols; p++)
        for(k = 0; k < nsubs; k++)
            for (i = 0; i < nsamp / nchans; i++) 
                for(j = 0; j < nchans; j++)
                {
                    input[p * nsubs * nsamp + k * nsamp + i * nchans + j].x = i;
                    input[p * nsubs * nsamp + k * nsamp + i * nchans + j].y = i;
                }

   // Allocate and transfer data to GPU (nsamp * nchans * npols)
   cudaMalloc((void **) &d_input, sizeof(cufftComplex) * nsubs * nsamp * npols);
   cudaMalloc((void **) &d_output, sizeof(float) * nsubs * nsamp);
   
   cudaEventRecord(event_start, 0);
   cudaMemcpy(d_input, input, sizeof(cufftComplex) * nsubs * nsamp * npols, cudaMemcpyHostToDevice);
   cudaEventRecord(event_stop, 0);
   cudaEventSynchronize(event_stop);
   cudaEventElapsedTime(&timestamp, event_start, event_stop);
   printf("Copied input to GPU in: %lfms\n", timestamp);

   // Create plan
   cufftPlan1d(&plan, nchans, CUFFT_C2C, nsubs * npols * (nsamp / nchans));
   printf("%d\n", nsubs * npols * (nsamp / nchans));

   // Execute FFT on GPU
   cudaEventRecord(event_start, 0);
   cufftExecC2C(plan, d_input, d_input, CUFFT_FORWARD);
   cudaEventRecord(event_stop, 0);
   cudaEventSynchronize(event_stop);
   cudaEventElapsedTime(&timestamp, event_start, event_stop);
   printf("Processed 1D FFT in: %lfms\n", timestamp);
   
   // Calculate intensity and perform transpose in memory
   cudaEventRecord(event_start, 0);
   calculate_intensities<<<dim3(nsamp / blocksize, 1), blocksize>>>(d_input, d_output, nsamp, nsubs, nchans, npols);
   cudaEventRecord(event_stop, 0);
   cudaEventSynchronize(event_stop);
   cudaEventElapsedTime(&timestamp, event_start, event_stop);
   printf("Processed Intensities in: %lfms\n", timestamp);

   // Get result
   cudaMemcpy(output, d_output, sizeof(float) * nsubs * nsamp, cudaMemcpyDeviceToHost);
//   for(i = 0; i < nchans * nsubs; i++)
//       for(k = 0; k < nsamp / nchans; k++)
//               if (output[i * nsamp / nchans + k] != 2 * npols * k*k) 
//                   printf("%d = %f\n", k, output[i * nsamp / nchans + k]);
//   printf("\n");
   
   // Clean up
   cufftDestroy(plan);
   cudaFree(d_input);
}
