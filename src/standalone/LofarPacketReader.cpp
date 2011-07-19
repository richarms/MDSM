#include "LofarPacketReader.h"
#include "Types.h"

#if defined(__x86_64__) && !defined(__NR_recvmmsg)
#define __NR_recvmmsg    299
#endif


// Global arguments
unsigned _nPackets;
unsigned _samplesPerPacket;
unsigned _subbandsPerPacket;
unsigned _nrPolarisations;
unsigned _samplesPerSecond;
unsigned _startTime;
unsigned _startBlockid;
unsigned _packetSize;
unsigned _sampleSize;

UDPPacket emptyPacket;

// Socket stuff

// Socket stuff
struct sockaddr_in  _serverAddress;
struct hostent *    _host;
int                 _socket;

const int           batch_size = 8;
char                *buf[batch_size];
struct iovec        iovecs[batch_size][1];
struct mmsghdr      mmsgs[batch_size];
struct sockaddr     addr[batch_size];

// Pipe to buffering thread
int _pipeId;

unsigned initialiseReader(unsigned sampPerPacket, unsigned subsPerPacket, unsigned nPols, unsigned sampPerSecond, 
                          unsigned sampSize, char *IP, int pipeId, unsigned port)
{   
    // Get configuration options
    _startTime = _startBlockid = 0;

    _samplesPerPacket = sampPerPacket; 
    _subbandsPerPacket = subsPerPacket;
    _nrPolarisations = nPols;
    _samplesPerSecond = sampPerSecond;
    _sampleSize = sampSize;

    _packetSize = _subbandsPerPacket * _samplesPerPacket * _nrPolarisations;

    size_t headerSize = sizeof(struct UDPPacket::Header);
    switch (sampSize)
    {
        case 4:
            _packetSize = _packetSize * sizeof(TYPES::i4complex) + headerSize;
            break;
        case 8:
            _packetSize = _packetSize * sizeof(TYPES::i8complex) + headerSize;
            break;
        case 16:
            _packetSize = _packetSize * sizeof(TYPES::i16complex) + headerSize;
            break;
    }

    _host = (struct hostent *) gethostbyname(IP);

    // Create socket
    if ((_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    { perror("Socket Error"); exit(1); }

    _serverAddress.sin_family = AF_INET;
    _serverAddress.sin_port = htons(port);
    _serverAddress.sin_addr = *((struct in_addr *) _host -> h_addr);
    bzero(&(_serverAddress.sin_zero), 8);

    // Bind socket
    if (bind(_socket, (struct sockaddr *) &_serverAddress, sizeof(struct sockaddr)) == -1)
    { perror("Bind Error"); exit(1); }

    // Set socket options
	int rcvbuf = 16 * 1048576;
	if (setsockopt(_socket, SOL_SOCKET, SO_RCVBUF,
				   &rcvbuf, sizeof(rcvbuf)) != 0)
		perror("SO_RCVBUF");

	rcvbuf = 0;
	socklen_t optlen = sizeof(rcvbuf);
	if (getsockopt(_socket, SOL_SOCKET, SO_RCVBUF,
				   &rcvbuf, &optlen))
		perror("SO_RCVBUF");

	fprintf(stdout, "Using receive buffer size %d\n", rcvbuf);

    // Initialise batch receive mode buffers
	for (int j = 0; j < batch_size; j++)
	{
        buf[j]                          = (char *) malloc(sizeof(_packetSize));
		iovecs[j][0].iov_base           = buf[j];
		iovecs[j][0].iov_len            = _packetSize;
		mmsgs[j].msg_hdr.msg_iov        = iovecs[j];
		mmsgs[j].msg_hdr.msg_iovlen     = 1;
		mmsgs[j].msg_hdr.msg_name       = &addr[j];
		mmsgs[j].msg_hdr.msg_namelen    = sizeof(addr[j]);	
		mmsgs[j].msg_hdr.msg_control    = 0;
		mmsgs[j].msg_hdr.msg_controllen = 0;
		mmsgs[j].msg_hdr.msg_flags      = 0;
		mmsgs[j].msg_len                = 0;
	}

    // Set pipe stuff
    _pipeId      = pipeId;

    return _packetSize - sizeof(struct UDPPacket::Header);
}

// Empty OS Packet Buffer
void emptyOSPacketBuffer()
{   
    unsigned i = 0;
    char tempBuffer[_packetSize];

    for(i = 0; i < 1e5; i++) {
        socklen_t addr_len = sizeof(struct sockaddr);
        if (recvfrom(_socket, tempBuffer, _packetSize, 0,
                    (struct sockaddr *) &_serverAddress, &addr_len) <= 0) {
            printf("Error while receiving UDP Packet");
            continue;
        }
    }
}

// Read in packets and pipe them to the buffering thread
void run()
{
    // Initialise receiving thread
    unsigned prevSeqid = _startTime;
    unsigned prevBlockid = _startBlockid;
    UDPPacket emptyPacket, currPacket;
    long lost = 0, caught = 0;
    
    unsigned long counter = 0;

    // Empty OS buffer to reduce risk of generating lost packets in the beginning
    emptyOSPacketBuffer();

    // Generate an empty packet to replace lost ones
    size_t packetDataSize = _packetSize - sizeof(struct UDPPacket::Header);
    memset((void*) emptyPacket.data, 0, packetDataSize);
    emptyPacket.header.nrBeamlets = _subbandsPerPacket;
    emptyPacket.header.nrBlocks   = _samplesPerPacket;
    emptyPacket.header.timestamp  = 0;
    emptyPacket.header.blockSequenceNumber = 0;

    // Read in packets, forever
    while(true) {

        // Read UDP packet from socket
//        socklen_t addr_len = sizeof(struct sockaddr);
//        if (recvfrom(_socket, reinterpret_cast<char *>(&currPacket), _packetSize, 0, 
//                    (struct sockaddr *) &_serverAddress, &addr_len) <= 0) {
//            printf("Error while receiving UDP Packet\n");
//            continue;
//        }

        // Alternative batch datagram reading mode
        unsigned ndatagrams = syscall(__NR_recvmmsg, _socket, mmsgs, batch_size, 0, 0);
    
        for(unsigned bCount = 0; bCount < ndatagrams; bCount++) {
    
            currPacket = *reinterpret_cast<UDPPacket *>(buf[bCount]);

            unsigned seqid, blockid;

            // TODO: Check for endianness
            seqid   = currPacket.header.timestamp;
            blockid = currPacket.header.blockSequenceNumber;

            // First time next has been run, initialise startTime and startBlockId
            if (counter == 0 && _startTime == 0) {
                prevSeqid = _startTime = _startTime == 0 ? seqid : _startTime;
                prevBlockid = _startBlockid = _startBlockid == 0 ? blockid : _startBlockid;
            }

            // Sanity check in seqid. If the seconds counter is 0xFFFFFFFF,
            // the data cannot be trusted (ignore)
            if (seqid == ~0U || prevSeqid + 10 < seqid)
                continue;

            // Check that the packets are contiguous. Block id increments by no_blocks
            // which is defined in the header. Blockid is reset every interval (although
            // it might not start from 0 as the previous frame might contain data from this one)
            unsigned totBlocks = _samplesPerSecond / _samplesPerPacket;
            unsigned lostPackets = 0, diff = 0;

            diff =  (blockid >= prevBlockid) ? (blockid - prevBlockid) : (blockid + totBlocks - prevBlockid);

            // Duplicated packets... ignore
            if (diff < _samplesPerPacket)
                continue;

            // Missing packets
            else if (diff > _samplesPerPacket) {
                lostPackets = (diff / _samplesPerPacket) - 1;  // -1 since it includes the received packet as well
                fprintf(stderr, "==================== Generated %u empty packets =====================\n", lostPackets);
            }

            // Generate lostPackets empty packets, if any
            unsigned packetCounter = 0;
            for (packetCounter = 0; packetCounter < lostPackets; ++packetCounter)
                 write(_pipeId, emptyPacket.data, packetDataSize);
            lost += packetCounter;

            // Write received packet
            write(_pipeId, currPacket.data, packetDataSize);
            prevSeqid = seqid;
            prevBlockid = blockid;
        
            counter += packetCounter + 1;
            caught++;
            
//            if (counter % 100000 == 0)
//                printf("====================== Received 100000 packets ==================== %ld %ld\n", caught, lost);
        }
    }
} 
