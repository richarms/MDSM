#include "stdio.h"
#include "stdlib.h"
#include "cutil.h"
#include <iostream>
#include <omp.h>

#include "openmp_cuda.h"

using namespace std;

#define elements     10000
#define multiplier   2 

int **d_input;
unsigned* deviceIDs;

// #################### CUDA MODULE ####################
CudaModule::CudaModule(unsigned numDevices):
    _numDevices(numDevices)
{ }

CudaModule::~CudaModule()
{ }

// Executes the kernel
void CudaModule::execute()
{
    // OpenMP magic
    
    #pragma omp parallel num_threads(_numDevices)
    {
        // Get thread number
        unsigned threadNum = omp_get_thread_num();
        
        // Associate thread with one GPU
        cudaSetDevice(deviceIDs[threadNum]);

        // Execute the kernel
        kernel(_numDevices, threadNum);

        #pragma omp critical
        cout << "Live from thread " << threadNum << " using gpu " << deviceIDs[threadNum] << endl;
       
    }
}

// #################### TEST MODULE ####################
TestModule::TestModule(unsigned numDevices) : CudaModule(numDevices)
{ }

TestModule::~TestModule()
{  }

void TestModule::run()
{ 
    // Execute kernel
    execute();
}

// TestModule kernel call (cannot be called directly)
void TestModule::kernel(unsigned numDevices, unsigned id)
{
     multiplyKernel<<<32,32>>>(d_input[id], elements, multiplier);
}

// ######################## MAIN #######################

unsigned initialise_devices()
{
	int num_devices;

    // Enumerate devices
    cudaGetDeviceCount(&num_devices);

    if (num_devices <= 0)
        { fprintf(stderr, "No CUDA-capable device found"); exit(0); }

    // Allocate device ids array
    deviceIDs = (unsigned *) malloc(num_devices * sizeof(unsigned));
    unsigned counter = 0;

    for(unsigned i = 0; i < num_devices; i++) 
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            { fprintf(stderr, "No CUDA-capable device found\n"); exit(0); }

        else if (deviceProp.totalGlobalMem / 1024 > 1024 * 1024) 
        {
            deviceIDs[counter] = i;
            counter++;
        }
    }

    if (counter == 0) 
        { fprintf(stderr, "No CUDA-capable device found"); exit(0); }

    deviceIDs = (unsigned *) realloc(deviceIDs, counter * sizeof(unsigned));

    return counter;
}

int main()
{
    // Get device count
    unsigned deviceCount;
    deviceCount = initialise_devices();
    printf("Found %d GPUs\n", deviceCount);

    // Create input array
    int nItems = elements * deviceCount;
    int *buffer = (int *) malloc(nItems * sizeof(nItems));
    d_input = (int **) malloc(deviceCount * sizeof(int *));

    // Initialise input array
    for(unsigned i = 0; i < nItems; i++)
        buffer[i] = i;

    // Allocate buffer on all GPUs and copy input array
    for(unsigned i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(deviceIDs[i]);
        cudaMalloc((void **) d_input[i], elements * sizeof(int));
        cudaMemcpy(d_input[i], &buffer[elements * i], elements * sizeof(int), cudaMemcpyHostToDevice);
    }

    printf("Initialised everything\n");

    // Create Test Module and run kernels
    TestModule module(deviceCount);
    module.run();

    // Copy data back to buffer
    for(unsigned i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(deviceIDs[i]);
        cudaMemcpy(&buffer[elements * i], d_input[i], elements * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Initialise input array
    for(unsigned i = 0; i < nItems; i++)
        if (buffer[i] == i * multiplier && buffer[i] != 0) {
            printf("Incorrect! %d = %d\n", buffer[i], i * multiplier);
            exit(0);
        }
    
}

