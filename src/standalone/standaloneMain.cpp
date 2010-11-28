#include "DoubleBuffer.h"
#include "UdpChunker.h"
#include "Types.h"

#include <QCoreApplication>

#include "dedispersion_manager.h"
#include "survey.h"

#include <stdio.h>
#include <stdlib.h>

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
    unsigned  chansPerSubband, samples, shift, memSize;
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
    memSize = survey -> npols * survey -> nsubs * sizeof(float);
    
    // Initialise Circular Buffer
    DoubleBuffer doubleBuffer(survey -> nsamp * chansPerSubband, survey -> nsubs, nPols);
    
    // Initialise UDP Chunker. Data is now being read
    UDPChunker chunker(port, sampPerPacket, subsPerPacket, nPols, sampPerSecond, sampSize);
    
    // Temporary store for maxshift
    float *maxshift = (float *) malloc(shift * survey -> nsubs * survey -> npols * sizeof(float) * 2);
    
    chunker.setDoubleBuffer(&doubleBuffer);
    chunker.start();
    chunker.setPriority(QThread::TimeCriticalPriority);
    
    // ======================== Store first maxshift =======================
    // Get pointer to next buffer
    float *udpBuffer = doubleBuffer.prepareRead();
       
    // Copy first maxshift to temporary
    memcpy(maxshift, udpBuffer + (samples - shift) * nPols * survey -> nsubs, 
                     shift * survey -> nsubs * survey -> npols * sizeof(float));
    // =====================================================================

    // Start main processing loop
    while(true) {
        
        // Get pointer to next buffer
        float *udpBuffer = doubleBuffer.prepareRead();
        
        // Copy maxshift to buffer
        memcpy(inputBuffer, maxshift, shift * memSize);
        
        // Copy UDP data to buffer
        memcpy(inputBuffer + shift * survey -> nsubs * survey -> npols, udpBuffer, samples * memSize);
               
        // Copy new maxshift
	    memcpy(maxshift, udpBuffer + (samples - shift) * nPols * survey -> nsubs, shift * memSize);
	                
        doubleBuffer.readReady();
        
        // Call MDSM for dedispersion
        unsigned int samplesProcessed;
	    next_chunk(shift + samples, samplesProcessed);
	    if (!start_processing(survey -> nsamp + survey -> maxshift)) {
	        printf("MDSM stopped....\n");
	    }
    } 

}
