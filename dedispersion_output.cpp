// MDSM stuff
#include "dedispersion_output.h"
#include "unistd.h"
#include "math.h"

// QT stuff
#include <QString>
#include <QtSql>

// C++ stuff
#include <iostream>

// Calaculate the mean and standard deviation for the data
void mean_stddev(float **buffer, SURVEY *survey, int read_nsamp)
{
    int i, j, iters, vals, mod_factor = 32 * 1024, shift = 0;
    double total;
    float mean = 0, stddev = 0;

    for(i = 0; i < survey -> num_passes; i++) {
        vals = read_nsamp / survey -> pass_parameters[i].binsize * survey -> pass_parameters[i].ndms;

        // Split value calculation in "kernels" to avoid overflows        

        // Calculate the mean
        iters = 0;
        while(1) {
            total  = 0;
            for(j = 0; j < mod_factor; j++)
                total += buffer[0][shift + iters * mod_factor + j];
            mean += (total / j);
            
            iters++;
            if (iters * mod_factor + j >= vals) break;
        }
        mean /= iters;  // Mean for entire array

        // Calculate standard deviation
        iters = 0;
        while(1) {
            total = 0;
            for(j = 0; j < mod_factor; j++)
                total += pow(buffer[0][shift + iters * mod_factor + j] - mean, 2);
             stddev += (total / j);

             iters++; 
             if (iters * mod_factor + j <= vals) break;
        }
        stddev = sqrt(stddev / iters);

        // Store mean and stddev values in survey
        survey -> pass_parameters[i].mean = mean;
        survey -> pass_parameters[i].stddev = stddev;

        shift += vals;
    }
}

// Apply mean and stddev to apply thresholding
void process(float **buffer, FILE* output, SURVEY *survey, int loop_counter, int read_nsamp)
{
    int i = 0, k, l, ndms, nsamp, shift = 0; 
    float temp_val, startdm, dmstep, mean, stddev; 

   // Start database transaction
    QSqlDatabase::database().transaction();

    // Create insert query with value placeholders
    QSqlQuery query;
    query.prepare("INSERT INTO events (sample, dm, snr) VALUES (?, ?, ?)");

    for(i = 0; i < survey -> num_passes; i++) {

        nsamp   = read_nsamp / survey -> pass_parameters[i].binsize;
        startdm = survey -> pass_parameters[i].lowdm;
        dmstep  = survey -> pass_parameters[i].dmstep;
        ndms    = survey -> pass_parameters[i].ndms;
        mean    = survey -> pass_parameters[i].mean;
        stddev  = survey -> pass_parameters[i].stddev;

        // Subtract dm mean from all samples and apply threshold
        for (k = 1; k < ndms; k++)
            for(l = 0; l < nsamp; l++) {
                temp_val = buffer[0][shift + k * nsamp + l] - mean;

                if (temp_val >= (stddev * 4) ) {
                      fprintf(output, "%d, %f, %f\n", loop_counter * survey -> nsamp + l * survey -> pass_parameters[i].binsize, 
                                                       startdm + k * dmstep, temp_val);
                      query.addBindValue(loop_counter * survey -> nsamp + l * survey -> pass_parameters[i].binsize);
                      query.addBindValue(startdm + k * dmstep);
                      query.addBindValue(temp_val);
                      query.exec();
                }
            }

        shift += nsamp * ndms;
    }
 
    // Commit transaction
    QSqlDatabase::database().commit();
}

// Create database connection
bool connectDatabase(QString username, QString password, QString dbName)
{
    QSqlDatabase db = QSqlDatabase::addDatabase("QPSQL7");
    db.setHostName("localhost");
    db.setDatabaseName(dbName);
    db.setUserName(username);
    db.setPassword(password);
    return db.open();
}

// Process dedispersion output
void* process_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    int i, iters = 0, ret, loop_counter = 0, pnsamp = params -> nsamp, ppnsamp = params -> nsamp;
    time_t start = params -> start, beg_read;

    // Allocate enough stack space
    printf("%d: Started output thread\n", (int) (time(NULL) - start));

    // Connect to MDSM database
    if (connectDatabase("lessju", "arcadia10", "MDSM"))
        printf("%d: Connected successfuly to MDSM database\n", (int) (time(NULL) - start));

    // TEMPORARY
    QSqlQuery query;
    query.exec("DELETE FROM events");

    // Processing loop
    while (1) {

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD)) 
            { fprintf(stderr, "Error during input barrier synchronisation [output]\n"); exit(0); }

        // Process output
        if (loop_counter >= params -> iterations) {
            beg_read = time(NULL);
            mean_stddev(params -> output_buffer, params -> survey, ppnsamp);
            process(params -> output_buffer, params -> output_file, params -> survey, loop_counter - params -> iterations, ppnsamp);
            printf("%d: Processed output %d [output]: %d\n", (int) (time(NULL) - start), loop_counter, 
                                                             (int) (time(NULL) - beg_read));
        }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during output barrier synchronisation [output]\n"); exit(0); }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [output]\n"); exit(0); } 

        // Update params
        ppnsamp = pnsamp;
        pnsamp = params -> nsamp;         

        // Stopping clause
        if (((OUTPUT_PARAMS *) output_params) -> stop) {
            
            if (iters >= params -> iterations - 1) {
               
                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [output]\n"); exit(0); }

                for(i = 0; i < params -> maxiters - params -> iterations; i++) {
                    pthread_barrier_wait(params -> input_barrier);
                    pthread_barrier_wait(params -> output_barrier);
                }
                break;
            }
            else
                iters++;
        }

        // Release rw_lock
        if (pthread_rwlock_unlock(params -> rw_lock))
            { fprintf(stderr, "Error releasing rw_lock [output]\n"); exit(0); }

        loop_counter++;
    }   

    printf("%d: Exited gracefully [output]\n", (int) (time(NULL) - start));
    pthread_exit((void*) output_params);
}