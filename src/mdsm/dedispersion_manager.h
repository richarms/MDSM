#ifndef DEDISPERSION_MANAGER_H_
#define DEDISPERSION_MANAGER_H_

#include "survey.h"
#include <QString>

SURVEY *processSurveyParameters(QString filepath);
float* initialiseMDSM(SURVEY* input_survey);
float* next_chunk(unsigned int data_read, unsigned &samples, double timestamp = 0, double blockRate = 0,
                  float* mean = NULL, float* rms = NULL);
int start_processing(unsigned int data_read);
void tearDownMDSM();

#endif // DEDISPERSION_MANAGER_H_
