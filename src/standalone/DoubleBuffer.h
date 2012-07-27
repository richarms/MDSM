// A 2D circular buffer for buffering input data (time vs frequency)
// Data stored as complex 16-floats OR 32-bit floats

#ifndef DoubleBuffer_H
#define DoubleBuffer_H

#include <QMutex>
#include <QThread>
#include <Types.h>

class DoubleBuffer: public QThread 
{

    public:
        DoubleBuffer(unsigned nbeams, unsigned nchans, unsigned nsamp);
        ~DoubleBuffer() { }    
        
        // Notify that a read request has finished
        void readReady();

        // Check whether heap can be written to buffer
        void writeHeap();

        // Wait for a buffer to become available
        float *prepareRead(double *timestamp, double *blockrate);

        // Populate writer parameters (called from network thread)
        char *setHeapParameters(unsigned nchans, unsigned nsamp);
    
        // Set timing variable (called from network thread)
        void  setTimingVariables(double timestamp, double blockrate);

        // Return heap buffer allocated here
        char  *getHeapBuffer();

        // Infinite thread loop to populate buffers
        virtual void run();    

    private:
        // Double buffer
        float     *_buffer[2];

        // Heap buffer
        char      *_heapBuffer, *_localHeap;   
    
        unsigned  _nbeams, _nchans, _nsamp, _readBuff, _writeBuff, _counter;
        unsigned  _fullBuffers, _samplesBuffered, _writingHeap;
        double    _timestamp,_blockrate;
        unsigned  _heapChans, _heapNsamp;
        
        QMutex    _readMutex, _writeMutex; 
};

#endif // DoubleBuffer_H
