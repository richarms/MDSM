#include <cutil.h>
#include <cufft.h>
#include <sys/time.h>

#define NX 2 * 512 * 8192
#define NY 4 * 1024
#define BATCH 2

int main() {

    // Initialise stuff
    cudaEvent_t event_start, event_stop;
    struct timeval start, stop;
    struct timezone tzp;
    int i;

    // Initialise CUDA stuff
    cudaSetDevice(0);
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 
    float timestamp;

    // ==================== 1D FFT ==============================

    cufftHandle plan;
    cufftComplex *d_data, *data;

    // Initialise data 
    data = (cufftComplex *) malloc(NX * BATCH * sizeof(cufftComplex));
    for(i = 0; i < NX * BATCH; i++) {
        data[i].x = 1.0f;
        data[i].y = 1.0f;
    }

   // Allocate and transfer data to GPU
  cudaMalloc((void **) &d_data, sizeof(cufftComplex) * NX * BATCH);
   cudaEventRecord(event_start, 0);
   cudaMemcpy(d_data, data, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
   cudaEventSynchronize(event_stop);
   cudaEventElapsedTime(&timestamp, event_start, event_stop);
   printf("Copied input to GPU in: %lfms\n", timestamp);

   // Create plan
   cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);

   // Execute FFT
   cudaEventRecord(event_start, 0);
   cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
   cudaEventRecord(event_stop, 0);
   cudaEventSynchronize(event_stop);
   cudaEventElapsedTime(&timestamp, event_start, event_stop);
   printf("Processed 1D FFT in: %lfms\n", timestamp);
 
   // Get result
   cudaMemcpy(data, d_data, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyDeviceToHost);

   // Clean up
   cufftDestroy(plan);
   cudaFree(d_data);

 // ==================== 2D FFT ==============================

//    // Initialise data 
//    realloc(data, NX * BATCH * sizeof(cufftComplex));
//    for(i = 0; i < NX * BATCH; i++) {
//        data[i].x = 1.0f;
//        data[i].y = 1.0f;
//    }

//   // Allocate and transfer data to GPU
//   cudaMalloc((void **) &d_data, sizeof(cufftComplex) * NX * BATCH);
//   cudaMemcpy(d_data, data, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyHostToDevice);

//   // Create plan
//   cufftPlan2d(&plan, NY, NY, CUFFT_C2C);

//   // Execute FFT
//   gettimeofday(&start, &tzp);
//   cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
//   cudaThreadSynchronize();
//   gettimeofday(&stop, &tzp);
//   if (start.tv_usec > stop.tv_usec) {
//       stop.tv_usec += 1000000;
//       stop.tv_sec--;
//   }
//   printf("Processed 2D FFT in %0.2fms [%d, %d]\n", (stop.tv_usec - start.tv_usec) / 1000.0f, NY, NY);
// 
//   // Get result
//   cudaMemcpy(data, d_data, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyDeviceToHost);

   // Clean up
   cufftDestroy(plan);
   cudaFree(d_data);

}
