// #################### CUDA MODULE ####################
class CudaModule
{
    public:
        CudaModule(unsigned numDevices);
        ~CudaModule();

    public:
        virtual void kernel(unsigned numDevices, unsigned id) = 0;

    protected:
        void execute();
        unsigned  _numDevices;

};

// #################### TEST MODULE ####################
class TestModule: public CudaModule
{
    public:
        TestModule(unsigned numDevices);
        ~TestModule();

    public:
        void run();
        virtual void kernel(unsigned numDevices, unsigned id);
};


// #################### CUDA KERNEL ####################
void __global__ __device__ multiplyKernel(int *buffer, int elements, int multiplier)
{
    for(unsigned i =  threadIdx.x + blockIdx.x * blockDim.x;
                 i <  elements;
                 i += blockDim.x * gridDim.x)

        buffer[i] *= 0;
}
