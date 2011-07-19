#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sched.h>
#include <netdb.h>
#include <fcntl.h>

#include "UdpHeader.h"

unsigned initialiseReader(unsigned sampPerPacket, unsigned subsPerPacket, unsigned nPols, unsigned sampPerSecond, 
                          unsigned sampSize, char *IP, int pipeId, unsigned port) ;
    
void run();
