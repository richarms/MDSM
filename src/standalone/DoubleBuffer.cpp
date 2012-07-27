#include "DoubleBuffer.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"
#include "string.h"
#include "time.h"
#include "sys/time.h"
#include "Types.h"

using namespace TYPES;

#define PWR(X,Y) X*X + Y*Y

// Class constructor
DoubleBuffer::DoubleBuffer(unsigned nbeams, unsigned nchans, unsigned nsamp) 
    : _nbeams(nbeams), _nchans(nchans), _nsamp(nsamp)
{
    // Initialise buffers
    // TODO: Allocate these with CUDA
    _buffer[0] =  (float *) malloc(nbeams * nsamp * nchans * sizeof(float));
    _buffer[1] =  (float *) malloc(nbeams * nsamp * nchans * sizeof(float));

    _readBuff = _samplesBuffered = _fullBuffers = 0;
    _writeBuff = 1; _writingHeap = 0;
    _counter = 0;
    
    printf("============== Buffers Initialised - read: %d, write: %d ==============\n", _readBuff, _writeBuff);
}


// Set timing variables
void DoubleBuffer::setTimingVariables(double timestamp, double blockrate)
{
    _timestamp = timestamp;
    _blockrate = blockrate;
}

// Populate writer parameters (called from network thread)
char *DoubleBuffer::setHeapParameters(unsigned nchans, unsigned nsamp)
{
    _heapChans = nchans;
    _heapNsamp = nsamp;

    // Allocate heap buffer
    _heapBuffer = (char *) malloc(_nbeams * nchans * nsamp * sizeof(float));
    _localHeap = (char *) malloc(_nbeams * nchans * nsamp * sizeof(float));
    return _heapBuffer;
}

// Lock buffer segment for reading
float *DoubleBuffer::prepareRead(double *timestamp, double *blockrate)
{
    // Busy wait for enough data, for now
    while (_fullBuffers < 1)
        sleep(0.001);
        
    // Data available (TODO: why are we mutex locking here?)
    _readMutex.lock();
    *timestamp = _timestamp + _blockrate * _nsamp * _counter;
    *blockrate = _blockrate;
    _readMutex.unlock();
    return _buffer[_readBuff];
}

// Read is ready
void DoubleBuffer::readReady()
{
    // Mutex buffer control
    _readMutex.lock();

    // Mark buffer as empty
    _fullBuffers--;
    _readMutex.unlock();
    printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
}

// Check whether heap can be written to buffer
void DoubleBuffer::writeHeap()
{
    // Busy wait for writer to finish current heap, for now
    while (_writingHeap != 0)
        sleep(0.001);
    
    // Buffer not writing, set local heap buffer
    memcpy(_localHeap, _heapBuffer, _nbeams * _heapChans * _heapNsamp * sizeof(char));

    // Notify running thread that a new heap is available
    _writeMutex.lock();
    _writingHeap = 1;
    _writeMutex.unlock();

    // Done, return to network thread to continue reading data
}

// Write data to buffer
void DoubleBuffer::run()
{
    // Infinite loop which read heap data from network thread and write it to buffer
    while(true)
    {
        // Wait for buffer to be read
        while(_fullBuffers == 2)
            sleep(0.01);

        // Wait for heap data to become available
        while (_writingHeap != 1)
            sleep(0.001);

        // Set writing heap
        _writeMutex.lock();
        _writingHeap = 2;
        _writeMutex.unlock();

        // Store incoming data into current writable double buffer
        short *complexData = (short *) _localHeap;
        for(unsigned b = 0; b < _nbeams; b++)
            for(unsigned c = 0; c < _heapChans; c++)
                for(unsigned s = 0; s < _heapNsamp; s++)
                {
                    unsigned bufferIndex = b * _nchans * _nsamp + c * _nsamp + s + _samplesBuffered;
                    unsigned complexIndex = 2*(b * _heapChans * _heapNsamp + c * _heapNsamp + s);
                    _buffer[_writeBuff][bufferIndex] = PWR(_heapBuffer[complexIndex], _heapBuffer[complexIndex+1]);
                }

        // Increment sample count
        _samplesBuffered += _heapNsamp;

        // Dealing with a new heap, check if buffer is already full
        if (_samplesBuffered == _nsamp)
        {
            // Check if reading buffer has been read
            while(_fullBuffers == 1)
                sleep(0.001);
        
            // Lock critical section with mutex, and swap buffers
            _readMutex.lock();
            _samplesBuffered = 0;
            _fullBuffers++;
            unsigned temp = _writeBuff;
            _writeBuff = _readBuff;
            _readBuff = temp;
            _counter++;
            printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
            _readMutex.unlock();
        }

        // Finished writing heap
        _writeMutex.lock();
        _writingHeap = 0;
        _writeMutex.unlock();
    }
}
