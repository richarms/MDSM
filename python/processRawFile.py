from matplotlib import pyplot as plt
import matplotlib.cm as cm
from math import ceil
import numpy as np
import struct
import os, sys

# Store global parameters
args = { 'nsamp'  : 65536,
         'nchans' : 256,
         'nbits'  : 16}

# Create main figure
fig = plt.figure()
fig.subplots_adjust(left=0.2, wspace=0.2)

def read_data():
    """ Read data from the file and store as a numpy matrix """
    f = open(args['filename'])

    if args['nbits'] == 8:
        mode = 'B'
    elif args['nbits'] == 16:
        mode = 'h'
    elif args['nbits'] == 32:
        mode = 'f'
    else:
        print args['nbits'] + " bits not supported"
        exit()

    data = f.read(args['nsamp'] * args['nchans'] * args['nbits'] / 8)
    data = np.array(struct.unpack( args['nsamp'] * args['nchans'] * mode, data ))
    return np.reshape(data, (args['nsamp'], args['nchans']))

def plot_bandpass(data, ax):
    """ Plot bandpass """

    x = range(args['nchans'])
    if 'fch1' in args.keys() and 'foff' in args.keys():
        x = np.arange(args['fch1'], args['fch1'] - (args['foff'] * args['nchans']), -args['foff'])

    ax.plot(x[::-1], np.sum(data, axis=0), 'r')
    ax.grid(True)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('dB')
    ax.set_title('Bandpass plot')

    with open('bandpass.txt', 'w') as f:
        np.sum(data, axis=0).tofile(f, '\n')

def plot_dedispersed(data, ax):
    """ Dedisperse data with a given DM """

    dedisp = lambda f1, f2, dm, tsamp: 4148.741601 * (f1**-2 - f2**-2) * dm / tsamp
    shifts = [dedisp(args['fch1'] + diff, args['fch1'], args['dm'], args['tsamp']) 
             for diff in -np.arange(0, args['foff'] * args['nchans'], args['foff'])]

    # Roll each subband by its shift to remove dispersion
    for i in range(args['nchans']):
        data[:,i] = np.roll(data[:,i], -int(shifts[i]))
    dedispersed = np.sum(data, 1)

    x = np.arange(0, args['tsamp'] * int(args['nsamp'] - ceil(max(shifts))), args['tsamp'])
    ax.plot(x, dedispersed[:np.size(x)])
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power')
    ax.set_title('Dedispersed Time Series (DM %.3f)' % args['dm'])
 
    return dedispersed

def plot_profile(data, ax):
    """ Fold the data with a given period """

    bins = args['period'] / args['tsamp']
    int_bins = int(bins)
    profile = np.zeros(int_bins)

    for i in range(int_bins):
        for j in range(int(args['nsamp'] / bins)):
            profile[i] += data[int(j * bins + i)]

    x = np.arange(0, args['tsamp'] * int_bins, args['tsamp'])
    ax.plot(x[:np.size(profile)], profile)
    ax.grid(True)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power')
    ax.set_title('Folded Time Series (Period %.3fms)' % args['period'])

    return profile

if __name__ == "__main__":

    # Process command-line arguments
    if len(sys.argv) < 3:
        print "Not enough arguments!"
        print "python processRawFile.py filename nsamp=x nchans=x nbits=x dm=x tsamp=x period=x"

    args['filename'] = sys.argv[1]

    for item in sys.argv[2:]:
        ind = item.find('=')
        if ind > 0:
            args[item[:ind]] = eval(item[ind + 1:])

    # Read data
    print "Reading data..."
    data = read_data()

    # Generate plots
    print "Generating bandpass"
    plot_bandpass(data, fig.add_subplot(223))

    print "Dedispersing time series"
 #   dedispersed = plot_dedispersed(data, fig.add_subplot(211))

    print "Folding time series"
 #   profile = plot_profile(dedispersed, fig.add_subplot(224))
    
    plt.show()
