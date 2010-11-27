#include "DoubleBuffer.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"
#include "string.h"
#include "Types.h"

using namespace TYPES;

// Class constructor
DoubleBuffer::DoubleBuffer(unsigned nsamp, unsigned nchans, unsigned npols) 
    : _nsamp(nsamp), _nchans(nchans), _npols(npols)
{
    // Initialise buffers
    _buffer = (float **) malloc(2);
    _buffer[0] =  (float *) malloc(nsamp * nchans * npols * sizeof(float));
    _buffer[1] =  (float *) malloc(nsamp * nchans * npols * sizeof(float));

    _readBuff = _writePtr = _fullBuffers = 0;
    _writeBuff = 1;
    _samplesBuffered = nsamp;
    _buffLen = nsamp * nchans * npols * sizeof(float);
}

// Lock buffer segment for reading
float *DoubleBuffer::prepareRead()
{
    // Busy wait for enough data, for now
    while (_fullBuffers < 1)
        sleep(0.001);
        
    // Data available
    return _buffer[_readBuff];
}

// Read is ready
void DoubleBuffer::readReady()
{
    // Mutex buffer control
    _mutex.lock();
    _fullBuffers--;
    _mutex.unlock();
    printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
}

// Write data to buffer
void DoubleBuffer::writeData(unsigned nsamp, unsigned nchans, float* data, bool interleavedMode)
{   
    // Wait for buffer to be read
    while(_fullBuffers == 2)
        sleep(0.001);

    if (interleavedMode) {
    
        // Store one time spectrum for all channels at a time
        for(unsigned i = 0; i < nsamp; i++) {
            for(unsigned j = 0; j < _npols; j++)
                for(unsigned k = 0; k < nchans; k++)
                    _buffer[_writeBuff][j * _nsamp * _nchans + _writePtr * nchans + k] = 
                        data[i * _npols * nchans + k * _npols + j];
                
            // Check if writing buffer is now full
            if (_writePtr++ > _samplesBuffered) {
            
                // Check if reading buffer has been read
                while(_fullBuffers == 1)
                    sleep(0.001);
            
                // Lock critical section with mutex, and swap buffers
                _mutex.lock();
                _writePtr = 0;
                _fullBuffers++;
                unsigned temp = _writeBuff;
                _writeBuff = _readBuff;
                _readBuff = temp;
                _mutex.unlock();
                
                printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
            }
         }
    }
    else
       { } // TODO: Implement
}

