#include "multibeam_dedispersion_output.h"
#include "multibeam_dedispersion_thread.h"
#include "unistd.h"

// QT stuff
#include <QFile>
#include <QStringList>
#include <QDomElement>

// Function macros
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

// Forward declarations
extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICES* call_initialise_devices(SURVEY *survey);
extern "C" void call_allocateInputBuffer(float **pointer, size_t size);
extern "C" void call_allocateOutputBuffer(float **pointer, size_t size);

// Global parameters
SURVEY  *survey;  // Pointer to survey struct
DEVICES *devices; // Pointer to devices struct

pthread_attr_t thread_attr;
pthread_t  output_thread;
pthread_t* threads;

pthread_rwlock_t rw_lock = PTHREAD_RWLOCK_INITIALIZER;
pthread_barrier_t input_barrier, output_barrier;

THREAD_PARAMS* threads_params;
OUTPUT_PARAMS output_params;

unsigned long *inputsize, *outputsize;
float** output_buffer;
float* input_buffer;

int loop_counter = 0;
bool outSwitch = true;
unsigned pnsamp, ppnsamp;

time_t start = time(NULL), begin;

#include <iostream>

// ================================== C++ Stuff =================================
// Process observation parameters
SURVEY* processSurveyParameters(QString filepath)
{
    QDomDocument document("observation");

    QFile file(filepath);
    if (!file.open(QFile::ReadOnly | QFile::Text))
        throw QString("Cannot open observation file '%1'").arg(filepath);

    // Read the XML configuration file into the QDomDocument.
    QString error;
    int line, column;
    if (!document.setContent(&file, true, &error, &line, &column)) {
        throw QString("Config::read(): Parse error "
                "(Line: %1 Col: %2): %3.").arg(line).arg(column).arg(error);
    }

    QDomElement root = document.documentElement();
    if( root.tagName() != "observation" )
        throw QString("Invalid root elemenent observation parameter xml file, should be 'observation'");

    // Get the root element of the observation file
    QDomNode n = root.firstChild();

    // Initalise survey object
    survey = (SURVEY *) malloc(sizeof(SURVEY));

    // Count number of pass tags
    int nbeams = 0;
    while(!n.isNull()) 
    {
        if (QString::compare(n.nodeName(), QString("beams"), Qt::CaseInsensitive) == 0) 
        {
            n = n.firstChild();
            while(!n.isNull()) 
            {
                nbeams++;
                n = n.nextSibling();
            }
        }
        n = n.nextSibling();
    }
    survey -> nbeams = nbeams;
    survey -> beams = (BEAM *) malloc(nbeams * sizeof(BEAM));
    nbeams = 0;

    // Assign default values
    survey -> nsamp = 0;
    survey -> nbits = 0;
    survey -> gpu_ids = NULL;
    survey -> num_gpus = 0;
    survey -> detection_threshold = 5.0;
    strcpy(survey -> fileprefix, "output");
    strcpy(survey -> basedir, ".");
    survey -> secs_per_file = 600;
    survey -> use_pc_time = 1;
    survey -> single_file_mode = 0;

    // Start parsing observation file and generate survey parameters
    n = root.firstChild();
    while( !n.isNull() )
    {
        QDomElement e = n.toElement();
        if( !e.isNull() )
        {
            if (QString::compare(e.tagName(), QString("dm"), Qt::CaseInsensitive) == 0) {
                survey -> lowdm = e.attribute("lowDM").toFloat();
                survey -> tdms = e.attribute("numDMs").toUInt();
                survey -> dmstep = e.attribute("dmStep").toFloat();
            }
            else if (QString::compare(e.tagName(), QString("channels"), Qt::CaseInsensitive) == 0) {
                survey -> nchans = e.attribute("nchans").toUInt();
                survey -> npols  = e.attribute("npols").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("timing"), Qt::CaseInsensitive) == 0)
                survey -> tsamp = e.attribute("tsamp").toFloat();
            else if (QString::compare(e.tagName(), QString("samples"), Qt::CaseInsensitive) == 0)
			   survey -> nsamp = e.attribute("number").toUInt();
            else if (QString::compare(e.tagName(), QString("detection"), Qt::CaseInsensitive) == 0)
			   survey -> detection_threshold = e.attribute("threshold").toFloat();
            else if (QString::compare(e.tagName(), QString("output"), Qt::CaseInsensitive) == 0) {
                char *temp = e.attribute("filePrefix", "output").toUtf8().data();
			    strcpy(survey -> fileprefix, temp);
                temp = e.attribute("baseDirectory", ".").toUtf8().data();
                strcpy(survey -> basedir, temp);
                survey -> secs_per_file = e.attribute("secondsPerFile", "600").toUInt();
                survey -> use_pc_time = e.attribute("usePCTime", "1").toUInt();
                survey -> single_file_mode = e.attribute("singleFileMode", "0").toUInt();
            }
        
            // Check if user has specified GPUs to use
            else if (QString::compare(e.tagName(), QString("gpus"), Qt::CaseInsensitive) == 0) {
			    QString gpus = e.attribute("ids");
                QStringList gpuList = gpus.split(",", QString::SkipEmptyParts);
                survey -> gpu_ids = (unsigned *) malloc(sizeof(unsigned) * gpuList.count());
                for(int i = 0; i < gpuList.count(); i++)
                    (survey -> gpu_ids)[i] = gpuList[i].toUInt();
                survey -> num_gpus = gpuList.count();
            }

            // Read beam parameters
            else if (QString::compare(e.tagName(), QString("beams"), Qt::CaseInsensitive) == 0)
            {
                // Process list of beams
                if (survey -> nbeams == 0)
                    continue;

                QDomNode beam = n.firstChild();
                while (!beam.isNull())
                {
                    e = beam.toElement();    
                    survey -> beams[nbeams].beam_id = e.attribute("beamId").toUInt();
                    survey -> beams[nbeams].fch1 = e.attribute("topFrequency").toFloat();
                    survey -> beams[nbeams].foff = e.attribute("frequencyOffset").toFloat();

                    beam = beam.nextSibling();
                    nbeams++;
                }
            }
        }
        n = n.nextSibling();
    }
    return survey;
}

// ==================================  C Stuff ==================================

inline float dmdelay(float F1, float F2) 
{  return (4148.741601 * ((1.0 / F1 / F1) - (1.0 / F2 / F2))); }

// Calculate number of samples which can be loaded at once (calls appropriate method)
int calculate_nsamp(int maxshift, size_t *inputsize, size_t* outputsize, unsigned long int memory)
{
    if (survey -> nsamp == 0)
    	survey -> nsamp = ((memory * 1000 * 0.99 / sizeof(float)) - maxshift * survey -> nchans) 
                           / (survey -> nchans + survey -> tdms);

    *inputsize = (survey -> nsamp + maxshift) * survey -> nchans * sizeof(float);
    *outputsize = survey -> nsamp * survey -> tdms * sizeof(float);
    printf("Memory Required (per beam): Input = %d MB; Output = %d MB\n", 
            (int) (*inputsize / 1024 / 1024), (int) (*outputsize/1024/1024));

	return survey -> nsamp;
}

// Initliase MDSM parameters, return poiinter to input buffer where
// input data will be stored
float* initialiseMDSM(SURVEY* input_survey)
{
    unsigned i, j, k;

    // Initialise survey
    survey = input_survey;

    // Initialise devices
    devices = call_initialise_devices(input_survey);

    // Calculate temporary DM-shifts, maxshift and nsamp per beam
    unsigned greatest_maxshift = 0;
    for (i = 0; i < survey -> nbeams; i++)
    {
        // Calculate temporary shifts
        BEAM *beam = &(survey -> beams[i]);
        beam -> dm_shifts = (float *) malloc(survey -> nchans * sizeof(float));
        for (j = 0; j < survey -> nchans; j++)
            beam -> dm_shifts[j] = dmdelay(beam -> fch1 + (beam -> foff * j), beam ->fch1);

        // Calculate maxshift
        float high_dm    = survey -> lowdm + survey -> dmstep * (survey -> tdms - 1);
        int maxshift     = beam -> dm_shifts[survey -> nchans - 1] * high_dm / survey -> tsamp;

        greatest_maxshift = ( maxshift > greatest_maxshift) 
                            ? maxshift : greatest_maxshift;
    }

    // TEMPORARY: ASSIGN ALL BEAMS SAME MAXSHIFT
    for(i = 0; i < survey -> nbeams; i++)
    {
        BEAM *beam = &(survey -> beams[i]);
        beam -> maxshift = greatest_maxshift;
    }

    printf("============================================================================\n");

    // Calculate global nsamp for all beams
    size_t inputsize, outputsize;
    survey -> nsamp = calculate_nsamp(greatest_maxshift, &inputsize, &outputsize, devices -> minTotalGlobalMem);

    // Check if nsamp is greater than maxshift
    if (greatest_maxshift > survey -> nsamp)
    {
        fprintf(stderr, "Number of samples (%d) must be greater than maxshift (%d)\n", survey -> nsamp, greatest_maxshift);
        exit(-1);
    }

    // Allocate GPUs to beams (split beams among GPUs, one beam cannot be processed on more than 1 GPU)
    for(i = 0; i < survey -> nbeams; i++)
        survey -> beams[i].gpu_id = (survey -> gpu_ids)[i % survey -> num_gpus];

    // Calculate output dedispersion size
    size_t outsize = outputsize / sizeof(float);

    // Create memory-pinned CPU buffers (holds input for all beams)
//    call_allocateInputBuffer(&input_buffer, survey -> nbeams * survey -> nsamp * survey -> nchans * sizeof(float));

    // Output buffer (one for each beam)
//    output_buffer = (float **) malloc(survey -> nbeams * sizeof(float *));
//    for(i = 0; i < survey -> nbeams; i++)
//        call_allocateOutputBuffer(&(output_buffer[i]), outsize * sizeof(float));

    input_buffer = (float *) malloc(survey -> nbeams * survey -> nsamp * survey -> nchans * sizeof(float));
    output_buffer = (float **) malloc(survey -> nbeams * sizeof(float *));
    for (i=0; i < survey -> nbeams; i++)
        output_buffer[i] = (float *) malloc(outsize * sizeof(float));

    // Allocate mean and stddev buffer in survey
    survey -> global_stddev = (float *) malloc(survey -> nbeams * sizeof(float));
    survey -> global_mean = (float *) malloc(survey -> nbeams * sizeof(float));

    // Log parameters
    printf("Observation Params: ndms = %d, maxDM = %f\n", survey -> tdms, 
                survey -> lowdm + survey -> dmstep * (survey -> tdms - 1));

    printf("Observation Params: nchans = %d, nsamp = %d, tsamp = %f\n", 
            survey -> nchans, survey -> nsamp, survey -> tsamp);

    printf("Beam Params:\n");
    for(i = 0; i < survey -> nbeams; i++)
    {
        BEAM beam = survey -> beams[i];
        printf("    Beam %d: fch1 = %f, foff = %f, maxshift = %d, gpuID = %d\n", 
                    beam.beam_id, beam.fch1, beam.foff, beam.maxshift, beam.gpu_id);
    }

    printf("============================================================================\n");

    // Initialise processing threads
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
    survey -> num_threads = survey -> nbeams;
    threads = (pthread_t *) calloc(sizeof(pthread_t), survey -> num_threads);
    threads_params = (THREAD_PARAMS *) malloc(survey -> num_threads * sizeof(THREAD_PARAMS));

    // Initialise barriers
    if (pthread_barrier_init(&input_barrier, NULL, survey -> num_threads + 2))
        { fprintf(stderr, "Unable to initialise input barrier\n"); exit(0); }

    if (pthread_barrier_init(&output_barrier, NULL, survey -> num_threads + 2))
        { fprintf(stderr, "Unable to initialise output barrier\n"); exit(0); }

    // Create output params and output file
    output_params.nthreads = survey -> num_threads;
    output_params.iterations = 3;
    output_params.maxiters = 2;
    output_params.output_buffer = output_buffer;
    output_params.dedispersed_size = outsize;
    output_params.stop = 0;
    output_params.rw_lock = &rw_lock;
    output_params.input_barrier = &input_barrier;
    output_params.output_barrier = &output_barrier;
    output_params.start = start;
    output_params.survey = survey;

    // Create output thread 
    if (pthread_create(&output_thread, &thread_attr, process_output, (void *) &output_params))
        { fprintf(stderr, "Error occured while creating output thread\n"); exit(0); }

    // Create threads and assign devices
    for(k = 0; k < survey -> num_threads; k++) {

        // Create THREAD_PARAMS for thread, based on input data and DEVICE_INFO
        // Input and output buffer pointers will point to beginning of beam
        threads_params[k].iterations = 2;
        threads_params[k].maxiters = 2;
        threads_params[k].stop = 0;
        threads_params[k].dedispersed_size = outsize;
        threads_params[k].output = output_buffer[k];
        threads_params[k].input = &input_buffer[survey -> nchans * survey -> nsamp * k];
        threads_params[k].thread_num = k;
        threads_params[k].num_threads = survey -> num_threads;
        threads_params[k].rw_lock = &rw_lock;
        threads_params[k].input_barrier = &input_barrier;
        threads_params[k].output_barrier = &output_barrier;
        threads_params[k].start = start;
        threads_params[k].survey = survey;
        threads_params[k].inputsize = inputsize;
        threads_params[k].outputsize = outputsize;

         // Create thread (using function in dedispersion_thread)
         if (pthread_create(&threads[k], &thread_attr, call_dedisperse, (void *) &threads_params[k]))
            { fprintf(stderr, "Error occured while creating thread\n"); exit(0); }
    }

    // Wait input barrier (for dedispersion_manager, first time)
    int ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n");  exit(0); }

    return input_buffer;
}

// Cleanup MDSM
void tearDownMDSM()
{
    unsigned  k;

    // Join all threads, making sure they had a clean cleanup
    void *status;
    for(k = 0; k < survey -> nbeams; k++)
        if (pthread_join(threads[k], &status))
            { fprintf(stderr, "Error while joining threads\n"); exit(0); }
    pthread_join(output_thread, &status);
    
    // Destroy attributes and synchronisation objects
    pthread_attr_destroy(&thread_attr);
    pthread_rwlock_destroy(&rw_lock);
    pthread_barrier_destroy(&input_barrier);
    pthread_barrier_destroy(&output_barrier);
    
    // Free memory
    free(devices -> devices);
    free(devices);

    free(threads_params);
    free(threads);
        
    free(survey -> global_mean);
    free(survey -> global_stddev);

    // TODO: Clear pinned memory and beams 

    printf("%d: Finished Process\n", (int) (time(NULL) - start));
}

// Process one data chunk
float **next_chunk_multibeam(unsigned int data_read, unsigned &samples, double timestamp = 0, double blockRate = 0)
{   
    unsigned k;

    printf("%d: Read %d * 1024 samples [%d]\n", (int) (time(NULL) - start), data_read / 1024, loop_counter);  

    // Lock thread params through rw_lock
    if (pthread_rwlock_wrlock(&rw_lock))
        { fprintf(stderr, "Error acquiring rw lock"); exit(0); }

    // Wait output barrier
    int ret = pthread_barrier_wait(&output_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    ppnsamp = pnsamp;
    pnsamp = data_read;

    // Stopping clause (handled internally)
    if (data_read == 0) { 
        output_params.stop = 1;
        for(k = 0; k < survey -> nbeams; k++) 
            threads_params[k].stop = 1;

        // Release rw_lock
        if (pthread_rwlock_unlock(&rw_lock))
            { fprintf(stderr, "Error releasing rw_lock\n"); exit(0); }

        // Return n-1 buffer
        samples = ppnsamp;
        return output_buffer;

    }

    //  Update timing parameters
    survey -> timestamp = timestamp;
    survey -> blockRate = blockRate;

    // Release rw_lock
    if (pthread_rwlock_unlock(&rw_lock))
        { fprintf(stderr, "Error releasing rw_lock\n"); exit(0); }

    if (loop_counter >= 1) {
    	samples = ppnsamp;
    	return output_buffer;
    }
    else {
    	samples = -1;
    	return NULL;
    }
}

// Start processing next chunk
int start_processing(unsigned int data_read) {

	// Stopping clause must be handled here... we need to return buffered processed data
	if (data_read == 0 && outSwitch)
        outSwitch = false;
	else if (data_read == 0 && !outSwitch)
		return 0;

	// Wait input barrier (since input is being handled by the calling host code
    int ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    return ++loop_counter;
}