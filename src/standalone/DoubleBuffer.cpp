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
    _sampledBuffered = nsamp;
    _buffLen = nsamp * nchans * npols * sizeof(float);
}

// Lock buffer segment for reading
float *DoubleBuffer::prepareRead()
{
    // Busy wait for enough data, for now
    while (_fullBuffers < 1)
        sleep(0.2);
        
    // Data available
    return _buffer[_readBuff];
}

// Read is ready
void DoubleBuffer::readReady()
{
    // Mutex buffer control
    _mutex.lock();
    _fullBuffers--;
    printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
    _mutex.unlock();
}

// Write data to buffer
unsigned DoubleBuffer::writeData(unsigned nsamp, unsigned nchans, float* data, bool interleavedMode)
{   
    while(_fullBuffers == 2)
        ; // Wait for buffer to be read

    if (interleavedMode) {
    
        for(unsigned i = 0; i < nsamp; i++) {
            memcpy(_buffer[_writeBuff] + _writePtr * _nchans * _npols, 
                   data, nsamp * nchans * _npols * sizeof(float));
                
            // Check if writing buffer is full
            if (_writePtr++ > _sampledBuffered) {
            
                // Check if reading buffer has been read
                while(_fullBuffers == 1) {
                    sleep(0.001);
                     // wait for buffer to be read
                 }
            
                _mutex.lock();
                
                _writePtr = 0;
                _fullBuffers++;
                
                // Swap buffers
                unsigned temp = _writeBuff;
                _writeBuff = _readBuff;
                _readBuff = temp;
                printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
                _mutex.unlock();
            }
         }
    }
    else
       { } // TODO: Implement
        
        return 1;
}

