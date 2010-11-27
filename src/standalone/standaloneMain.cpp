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
unsigned sampPerPacket = 1, subsPerPacket = 512, 
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
       if (!strcmp(argv[i], "-sampPerPacket"))
           sampPerPacket = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-subsPerPacket"))
           subsPerPacket = atoi(argv[++i]);
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
    unsigned  chansPerSubband, samples, shift;
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
    chansPerSubband = survey -> nchans / survey -> nsubs;
    samples = survey -> nsamp * chansPerSubband;
    shift = survey -> maxshift * chansPerSubband;
    
    // Initialise Circular Buffer
    DoubleBuffer doubleBuffer(survey -> nsamp * chansPerSubband, survey -> nsubs, nPols);
    
    // Initialise UDP Chunker. Data is now being read
    UDPChunker chunker(port, sampPerPacket, subsPerPacket, nPols, sampPerSecond, sampSize);
    
    // Temporary store for maxshift
    float *maxshift = (float *) malloc(shift * survey -> nsubs * survey -> npols * sizeof(float) * 2);
    
    chunker.setDoubleBuffer(&doubleBuffer);
    chunker.start();
    chunker.setPriority(QThread::TimeCriticalPriority);
    
    printf("nsub: %d\n", survey -> nsubs);
    
     // ==================== First iteration removed out of loop... ==============
    // Get pointer to next buffer
    float *udpBuffer = doubleBuffer.prepareRead();
    
    // Process stokes parameters inplace
    TYPES::i16complex *complexData = reinterpret_cast<TYPES::i16complex *>(udpBuffer);
    memcpy(maxshift, complexData + (samples - shift) * nPols * survey -> nsubs, 
                     shift * survey -> nsubs * survey -> npols * sizeof(float) * 2);
    
//    for(unsigned i = 0; i < shift; i++)
//        for(unsigned j = 0; j < survey -> nsubs; j++) {
//            TYPES::i16complex val = complexData[(samples - shift + i) * nPols * survey -> nsubs + j * nPols];
            
//            maxshift[2 * (j * shift + i)]      = val.real();
//            maxshift[2 * (j * shift + i) + 1]  = val.imag();
            
//            printf("1. %d %d %d %d\n",i, j, (samples - shift + i) * nPols * survey -> nsubs + j * nPols,
//                                            2 * (j * shift + i));
            
//            val = complexData[(samples - shift + i) * nPols * survey -> nsubs + j * nPols + 1];
//                                                 
//            maxshift[2 * (shift * survey -> nsubs + j * shift + i)]      = val.real();
//            maxshift[2 * (shift * survey -> nsubs + j * shift + i) + 1]  = val.imag();
 
//             printf("2. %d %d %d %d\n",i, j,
//                          (samples - shift + i) * nPols * survey -> nsubs + j * nPols + 1,
//                           2 * (shift * survey -> nsubs + j * shift + i));

//        }

    dataRead += survey -> maxshift * chansPerSubband;
    doubleBuffer.readReady();
    // ========================  END OF FIRST ITERATION  ====================

    // Start main processing loop
    while(true) {
        
        // Get pointer to next buffer
        float *udpBuffer = doubleBuffer.prepareRead();
        
        // Process stokes parameters inplace
        TYPES::i16complex *complexData = reinterpret_cast<TYPES::i16complex *>(udpBuffer);
        
        // Copy maxshift to buffer
        memcpy(inputBuffer, maxshift, shift * survey -> nsubs * survey -> npols * sizeof(float) * 2);
        
        // Copy UDP data to buffer
        memcpy(inputBuffer + shift * survey -> nsubs * survey -> npols, complexData, 
               samples * survey -> nsubs * survey -> npols * sizeof(float));
        
        // Copy new maxshift
	    memcpy(maxshift, complexData + (samples - shift) * nPols * survey -> nsubs, 
               shift * survey -> nsubs * survey -> npols * sizeof(float));
            
        doubleBuffer.readReady();
        dataRead += survey -> nsamp;
        
        printf("Calling MDSM\n");
        
        // Call MDSM for dedispersion
        unsigned int samplesProcessed;
	    next_chunk(dataRead, samplesProcessed);
	    if (!start_processing(dataRead)) {
	        printf("MDSM stopped....\n");
	    }
	    
        dataRead = 0;
    } 

}
