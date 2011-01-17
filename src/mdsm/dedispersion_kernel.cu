#ifndef DEDISPERSE_KERNEL_H_
#define DEDISPERSE_KERNEL_H_

#include <cutil_inline.h>

//#define FERMI

// Stores temporary shift values
__device__ __constant__ float dm_shifts[16384];

// Stores output value computed in inner loop for each sample
//#ifdef FERMI
//	__device__ __shared__ float localvalue[8192];
//#else
	__device__ __shared__ float localvalue[1024];
//#endif

#define TILE_DIM    16
#define BLOCK_ROWS  16

// -------------------------- Optimised Dedispersion Loop -----------------------------------
__global__ void opt_dedisperse_loop(float *outbuff, float *buff, int nsamp, int nchans, float tsamp,
                                int chanfactor, float startdm, float dmstep, int maxshift, int inshift, int outshift)
{
    // Dynamic shared memory, amount specified in kernel configuration
    extern __shared__ float shared[];

    float shift_temp = (startdm + blockIdx.y * dmstep) / tsamp;
    int c, s;
    
    // Dedispersing over loop of samples (1 thread = 1+ samples)
    for (s = threadIdx.x + blockIdx.x * blockDim.x; 
         s < nsamp; 
         s += blockDim.x * gridDim.x) {

        // Clear shared memory
        shared[threadIdx.x] = 0;
     
        // Loop over all channels, calucate shift and sum for current sample
        for(c = 0; c < nchans; c++) {
            int shift = c * (nsamp + maxshift) + floor(dm_shifts[c * chanfactor] * shift_temp);
            shared[threadIdx.x] += buff[inshift + shift + s];
        }

        // Store output
        outbuff[outshift + blockIdx.y * nsamp + s] = shared[threadIdx.x];
    }
}

// -------------------------- The Dedispersion Loop -----------------------------------
__global__ void dedisperse_loop(float *outuff, float *buff, int nsamp, int nchans, float tsamp,
                                int chanfactor, float startdm, float dmstep, int inshift, int outshift)
{
    int samp, s, c, indx, soffset;
    float shift_temp;

    /* dedispersing loop over all samples in this buffer */
    s = threadIdx.x + blockIdx.x * blockDim.x;
    shift_temp = (startdm + blockIdx.y * dmstep) / tsamp; 

    for (samp = 0; s + samp < nsamp; samp += blockDim.x * gridDim.x) {
        soffset = (s + samp);
        
        /* clear array element for storing dedispersed subband */
        localvalue[threadIdx.x] = 0.0;

        /* loop over the channels */
        for (c = 0; c < nchans; c ++) {
            indx = (soffset + (int) (dm_shifts[c * chanfactor] * shift_temp)) * nchans + c;
            localvalue[threadIdx.x] += buff[inshift + indx];
        }

        outuff[outshift + blockIdx.y * nsamp + soffset] = localvalue[threadIdx.x];
    }
}

// -------------------------- The Subband Dedispersion Loop -----------------------------------
__global__ void dedisperse_subband(float *outbuff, float *buff, int nsamp, int nchans, int nsubs, 
                                   float startdm, float dmstep, float tsamp, int inshift, int outshift)
{
    int samp, s, c, indx, soffset, sband, tempval, chans_per_sub = nchans / nsubs;
    float shift_temp;

    s = threadIdx.x + blockIdx.x * blockDim.x;
    shift_temp = (startdm + blockIdx.y * dmstep) / tsamp;

    // dedispersing loop over all samples in this buffer
    for (samp = 0; s + samp < nsamp; samp += blockDim.x * gridDim.x) {
        soffset = (s + samp);       

        // loop over the subbands
        for (sband = 0; sband < nsubs; sband++) {  

            // Clear array element for storing dedispersed subband
            localvalue[threadIdx.x * nsubs + sband] = 0.0;

            // Subband channels are shifted to sample location of the highest frequency
            tempval = (int) (dm_shifts[sband * chans_per_sub] * shift_temp); 

            // Add up channels within subband range
            for (c = (sband * chans_per_sub); c < (sband + 1) * chans_per_sub; c++) {
                indx = (soffset + (int) (dm_shifts[c] * shift_temp - tempval)) * nchans + c;
                localvalue[threadIdx.x * nsubs + sband] += buff[inshift + indx];
            }

            // Store values in global memory
            outbuff[outshift + blockIdx.y * nsamp * nsubs + soffset * nsubs + sband] = 
                    localvalue[threadIdx.x * nsubs + sband];
        }
    }
}

// -------------------------- The Optimised Subband Dedispersion Loop -----------------------------------
__global__ void opt_dedisperse_subband(float *outbuff, float *buff, int nsamp, int nchans, int nsubs, 
                                   float startdm, float dmstep, float tsamp, int maxshift, int inshift, int outshift)
{
    extern __shared__ float shared[];
    
    int s, c, shift, sband, tempval, chans_per_sub = nchans / nsubs;
    float shift_temp = (startdm + blockIdx.y * dmstep) / tsamp;

    // dedispersing loop over all samples in this buffer
    for (s = threadIdx.x + blockIdx.x * blockDim.x; 
         s < nsamp; 
         s += blockDim.x * gridDim.x) {

        // loop over the subbands
        for (sband = 0; sband < nsubs; sband++) {  

            // Clear array element for storing dedispersed subband
            shared[threadIdx.x * nsubs + sband] = 0.0;

            // Subband channels are shifted to sample location of the highest frequency
            tempval = dm_shifts[sband * chans_per_sub] * shift_temp; 

            // Add up channels within subband range
            for (c = (sband * chans_per_sub); c < (sband + 1) * chans_per_sub; c++) {
                shift = dm_shifts[c] * shift_temp - tempval;
                shared[threadIdx.x * nsubs + sband] += buff[inshift + c * (nsamp + maxshift) + shift + s];
            }

            // Store values in global memory
            outbuff[outshift + blockIdx.y * nsamp * nsubs + sband * nsamp + s] = shared[threadIdx.x * nsubs + sband];
        }
    }
}

// ----------------------------- Channel Binnig Kernel --------------------------------
__global__ void channel_binning_kernel(float *input, int nchans, int binsize)
{
    int b, c, channel;
    float total;

    channel = threadIdx.x + blockDim.x * blockIdx.x;
    for(c = 0; c + channel < nchans; c += gridDim.x * blockDim.x) {

        localvalue[threadIdx.x] = input[c + channel];
    
        __syncthreads();
 
        if (threadIdx.x % binsize == 0) {       
            total = 0;
            for(b = 0; b < binsize; b++)       
               total += localvalue[threadIdx.x + b];
            input[c + channel] = total;
        }

       __syncthreads();
    }
}

// --------------------------- Data binning kernel ----------------------------
__global__ void binning_kernel(float *input, int nsamp, int nchans, int binsize, int inshift, int outshift)
{
    int b, c, channel, shift;

    // Loop over all values (nsamp * nchans)
    shift = threadIdx.x + blockIdx.y * (nchans / gridDim.y);
    channel = shift + blockDim.x * gridDim.y * blockIdx.x;
    for(c = 0; c + channel < nsamp * nchans; c += gridDim.x * blockDim.x * gridDim.y * binsize) {

        // Load data from binsize samples into shared memory
        localvalue[threadIdx.x] = 0;

        for(b = 0; b < binsize; b++)
            localvalue[threadIdx.x] += input[inshift + c + blockIdx.x * gridDim.y * 
                                             blockDim.x * binsize + nchans * b + shift];
 
        // Copy data to global memory
        input[outshift + channel + c/binsize] = localvalue[threadIdx.x] / binsize;
    }
}

// --------------------------- In-place data binning kernel ----------------------------
__global__ void inplace_binning_kernel(float *input, int nsamp, int nchans, int binsize)
{
    int b, c, channel, shift;

    // Loop over all values (nsamp * nchans)
    shift = threadIdx.x + blockIdx.y * (nchans / gridDim.y);
    channel = shift + blockIdx.x * gridDim.y * blockDim.x * binsize;
    for(c = 0; c + channel < nsamp * nchans; c += gridDim.x * blockDim.x * gridDim.y * binsize) {

        // Load data from binsize samples into shared memory
        localvalue[shift] = 0;

        for(b = 0; b < binsize; b++)
            localvalue[shift] += input[c + blockIdx.x * gridDim.y * blockDim.x * 
                                       binsize + nchans * b + shift];

        // Copy data to global memory
        input[c +  channel] = localvalue[shift] / binsize;
    }
}

__global__ void inplace_memory_reorganisation(float *input, int nsamp, int nchans, int binsize)
{
    int c, channel, shift;

    // Loop over all values (nsamp * nchans)
    shift = threadIdx.x + blockIdx.y * (nchans / gridDim.y);
    channel = shift + blockDim.x * gridDim.y * blockIdx.x;
    for(c = 0; c + channel < nsamp * nchans; c += gridDim.x * blockDim.x * gridDim.y * binsize) {

        // Load data from binsize samples into shared memory
        localvalue[shift] = input[c + blockIdx.x * gridDim.y * blockDim.x * binsize + shift];
 
        // Copy data to global memory
        input[channel + c/binsize] = localvalue[shift];
    }
}

// ---------------------- Intensities Calculation Loop  ------------------------------
__global__ void calculate_intensities(cufftComplex *inbuff, float *outbuff, int nsamp, 
                                      int nsubs, int nchans, int npols)
{
    /// Loop over all samples
    for(unsigned s = threadIdx.x + blockIdx.x * blockDim.x;
                 s < nsamp;
                 s += blockDim.x * gridDim.x)
    {
        // Loop over all subbands
        for(unsigned c = 0; c < nsubs; c++) {

            // Loop over all channels
            for(unsigned chan  = 0; chan < nchans; chan++) {
              
                float intensity = 0;
                cufftComplex tempval;
                    
                // Loop over polarisations
                for(unsigned p = 0; p < npols; p++) {

                    // Square real and imaginary parts
                    tempval = inbuff[c*nchans*nsamp + s*nchans + chan];
                    intensity += tempval.x * tempval.x + tempval.y * tempval.y;
                }

                // Store in output buffer (relocate channels)
                 outbuff[(nchans*nsubs-1 - (c * nchans + chan)) * nsamp + s] = intensity * 0.1;
            }
        }
    }
}

// ---------------------- Polarisations Kernels ------------------------------

// Separate X and Y polarisation into separate buffers (Blocksize == nsubs)
__global__ void seperateXYPolarisations(float *input, float *output, int nsamp, 
                                        int nchans)
{
    // Assign each thread block to one sample
    for(unsigned s = blockIdx.x; 
                 s < nsamp;
                 s += gridDim.x)  {
                 
        // Load X polarisation and save in output   
        output[s * nchans + threadIdx.x] = input[s * nchans * 2 + threadIdx.x];

        // Load Y polarisation and save in output   
        output[nsamp * nchans + s * nchans + threadIdx.x] = input[s * nchans * 2 + nchans + threadIdx.x];
    }
}

// Expand polarisations from 16-bit complex to 32-bit complex
__global__ void expandValues(short *input, float *output, int nvalues)
{
    // Assign each thread block to one value
    for(int s = threadIdx.x + blockIdx.x * blockDim.x; 
            s < nvalues;
            s += gridDim.x * blockDim.x)  {
                 
        // Load polarisations and save in output (REAL AND IMAGINARY ARE INVERTED!!)
          output[s * 2]     = (float) input[s * 2 + 1];
          output[s * 2 + 1] = (float) input[s * 2];
    }
}

// ---------------------- Diagonal Matrix Transpose ------------------------------
__global__ void transposeDiagonal(float *idata, float *odata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    // do diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }    

    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;  
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex) * height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
          tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
      odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
}

// ---------------------- Folding kernel  ------------------------------
__global__ void fold(float *input, float *output, int nsamp, float tsamp,
                    float period, int shift)
{
    int bins = period / tsamp;

    for(unsigned b = threadIdx.x;
                 b < bins;
                 b += blockDim.x)
    {
        float val = 0;
        for(unsigned s = 0; s < nsamp / bins; s ++)
            val += input[blockIdx.x * (nsamp + shift) + shift + s * bins + b];
         output[blockIdx.x * bins + b] = val / (nsamp / bins);
    }
}

// ---------------------- RFI Clipping kernel --------------------------
// One thread block per original subband (nsamp = nsamp * chansPerSubband)
__global__ __device__ void rfi_clipping(float *input, float *means, int nsamp, int nsubs)
{
    __shared__ float2 tempSums[1024];
    float mean, stddev;   // Store as registers to avoid bank conflicts in shared memory

    // Initial setup
    tempSums[threadIdx.x].x = 0;
    tempSums[threadIdx.x].y = 0;

    for(unsigned s =  threadIdx.x; 
                 s <  nsamp; 
                 s += blockDim.x)
    {
        // Calculate partial sums
        float intensity = input[blockIdx.x * nsamp + s];
        tempSums[threadIdx.x].x += intensity;
        tempSums[threadIdx.x].y += intensity * intensity;
    }

    // synchronise threads
    __syncthreads();

    // TODO: use reduction to optimise this part
    if (threadIdx.x == 0) {
        double val1 = 0, val2 = 0;
        for(unsigned i = 0; i < blockDim.x; i++) {
            val1 += tempSums[i].x;
            val2 += tempSums[i].y;
        }

        // Calculate mean and stddev
        mean   = val1 / nsamp;
        stddev = sqrtf((val2 - nsamp * mean * mean) / nsamp);
        means[blockIdx.x] = mean;

        // Store mean and stddev in tempSums
        tempSums[0].x = mean;
        tempSums[0].y = stddev;
    }

    // Synchronise threads
    __syncthreads();  
    mean = tempSums[0].x;
    stddev = tempSums[0].y;

    // Clip RFI within the subbands
    for(unsigned s =  threadIdx.x; 
                 s <  nsamp;    
                 s += blockDim.x)
    {
        float val = input[blockIdx.x * nsamp + s];
        float tempval = fabs(val - mean);
        if (tempval >= stddev * 4 || (tempval - stddev * 4) > 0 || tempval == 0)
            val = mean;

        __syncthreads();
        input[blockIdx.x * nsamp + s] = val;
    }
}

#endif
