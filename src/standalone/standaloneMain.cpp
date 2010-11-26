#include "DoubleBuffer.h"
#include "UdpChunker.h"
#include "Types.h"

#include <QCoreApplication>

#include "dedispersion_manager.h"
#include "survey.h"

#include <stdio.h>
#include <stdlib.h>

#define SQR(x) (x * x)

// Global arguments
unsigned chansPerSubband = 1, sampPerPacket = 1, subsPerPacket = 512, 
         sampSize = 16, port = 10000, nPols = 2, sampPerSecond = 97656;

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;
        
    if (i != 2) {
        printf("MDSM needs observation file!\n");
        exit(-1);
    }
    
    while(i < argc) {
       if (!strcmp(argv[i], "-chansPerSubband"))
           chansPerSubband = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-sampPerPacket"))
           sampPerPacket = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-subsPerPacket"))
           subsPerPacket = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-sampSize"))
           sampSize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-sampSize"))
           sampSize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-port"))
           port = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nPols"))
           nPols = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-sampPerSecond"))
           sampPerSecond = atoi(argv[++i]);
       i++;
    }
}

// Main method
int main(int argc, char *argv[])
{
    unsigned  dataRead = 0;
    float     *inputBuffer;
    SURVEY    *survey;
    
     // Create mait QCoreApplication instance
    QCoreApplication app(argc, argv);

    // Process arguments
    process_arguments(argc, argv);   
    
    // Initialise MDSM
    survey = processSurveyParameters(argv[1]);
    inputBuffer = initialiseMDSM(survey);
    
    // Initialise Circular Buffer
    DoubleBuffer doubleBuffer(survey -> nsamp, survey -> nchans, nPols);
    
    // Initialise UDP Chunker. Data is now being read
    UDPChunker chunker(port, sampPerPacket, subsPerPacket, nPols, sampPerSecond, sampSize);
    
    chunker.setDoubleBuffer(&doubleBuffer);
    chunker.start();
    chunker.setPriority(QThread::TimeCriticalPriority);
    
     // ==================== First iteration removed out of loop... ==============
    // Get pointer to next buffer
    float *udpBuffer = doubleBuffer.prepareRead();
    
    // Process stokes parameters inplace
    TYPES::i16complex *complexData = reinterpret_cast<TYPES::i16complex *>(udpBuffer);
        
    for(unsigned i = 0; i < survey -> maxshift; i++)
        for(unsigned j = 0; j < survey -> nchans; j++) {
            TYPES::i16complex X = complexData[(survey -> nsamp - survey -> maxshift + i) 
                                               * survey -> nchans * nPols + j * nPols];
            TYPES::i16complex Y = complexData[(survey -> nsamp - survey -> maxshift + i)
                                               * survey -> nchans * nPols + j * nPols + 1];
            inputBuffer[j * (survey -> nsamp + survey -> maxshift) + i] = 
                SQR(X.real()) + SQR(X.imag()) + SQR(Y.real()) + SQR(Y.imag()); 
        }
        
    dataRead += survey -> maxshift;
    doubleBuffer.readReady();
    // ========================  END OF FIRST ITERATION  ====================

    // Start main processing loop
    while(true) {
        
        // Get pointer to next buffer
        float *udpBuffer = doubleBuffer.prepareRead();
        
        // Process stokes parameters inplace
        TYPES::i16complex *complexData = reinterpret_cast<TYPES::i16complex *>(udpBuffer);

	    // Load nsamp values, offset by maxshift
        for(unsigned i = 0; i < survey -> nsamp; i++)
            for(unsigned j = 0; j < survey -> nchans; j++) {
                TYPES::i16complex X = complexData[i * survey -> nchans * nPols + j * nPols];
                TYPES::i16complex Y = complexData[i * survey -> nchans * nPols + j * nPols + 1];
                inputBuffer[j * (survey -> nsamp + survey -> maxshift) + i + survey -> maxshift] = 
                    SQR(X.real()) + SQR(X.imag()) + SQR(Y.real()) + SQR(Y.imag());              
            }
            
        doubleBuffer.readReady();
        dataRead += survey -> nsamp;
        
        printf("Processed Intensities %d %d\n", survey -> nsamp, survey -> nchans);
        
        // Call MDSM for dedispersion
        unsigned int samplesProcessed;
	    next_chunk(dataRead, samplesProcessed);
	    if (!start_processing(dataRead)) {
	        printf("MDSM stopped....\n");
	    }
	    
        dataRead = 0;
    } 

}
