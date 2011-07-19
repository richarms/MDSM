#include "MdsmPipeline.h"
#include "DedispersedDataWriter.h"
#include "WeightedSpectrumDataSet.h"
#include <iostream>

MdsmPipeline::MdsmPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

MdsmPipeline::~MdsmPipeline()
{ }

// Initialise the pipeline
void MdsmPipeline::init()
{
    // Create modules
    mdsm = (MdsmModule *) createModule("MdsmModule");
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    dedispersedData = (DedispersedTimeSeriesF32*) createBlob("DedispersedTimeSeriesF32");
    weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Request remote data
    requestRemoteData("TimeSeriesDataSetC32");
}

// Run the pipeline
void MdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    timeSeries = (TimeSeriesDataSetC32*) remoteData["TimeSeriesDataSetC32"];

    if (timeSeries -> size() == 0) {
        std::cout << "Reached end of stream" << std::endl;
        for (unsigned i = 0; i < 2; i++) { // NOTE: Too dependent on MDSM's internal state
            std::cout << "Processing extra step " << i << std::endl;
            mdsm->run(stokes, dedispersedData);
            dataOutput(dedispersedData, "DedispersedTimeSeriesF32");
            stop();
        }
    }

    // Output raw data
//    stokesGenerator -> run(timeSeries, stokes);
//    dataOutput(stokes, "TimeSeriesDataSetC32");

//    // Run modules
    ppfChanneliser->run(timeSeries, spectra);
                    stokesGenerator->run
(spectra,stokes);                                                                  
//    rfiClipper->run(stokes);

    // Perform dedispersion
//    mdsm->run(stokes, dedispersedData);

//    // Output channelised data
//    // dataOutput(stokes, "DedispersedDataWriter");

//    // Output dedispersed data
//    // dataOutput(dedispersedData, "DedispersedTimeSeriesF32");

    _iteration++;
    if (_iteration % 1000 == 0)
    std::cout << "Iteration: " << _iteration << std::endl;
}
