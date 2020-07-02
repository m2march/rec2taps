import argparse
import os
import sys
import m2.rec2taps
import logging
from m2.rec2taps import defaults
from m2.rec2taps import errors

def rec2taps():
    parser = argparse.ArgumentParser(
        description=('Obtain tap times from a recording file synchronized '
                     'to a provided stimuli file. In the context of '
                     '"Simple and cheap setup for measuring timed responses '
                     'to auditory stimuli" (Miguel et. al. 2020).')
    )
    
    parser.add_argument('stimuli_file', type=str,
                        help='audio file of the stimuli')
    parser.add_argument('recording_file', type=str,
                        help=('audio file of the experiment recording. Should '
                              'have two channels, one with a loopback '
                              'recording of the stimuli and another one with '
                              'the signal from the input device.')
                       )
    parser.add_argument('-d', dest='distance',
                        type=int, default=defaults.DEFAULT_DISTANCE, 
                        help='Minimum distance (in ms) between detected peaks')
    parser.add_argument('-p', dest='prominence',
                        type=float, default=defaults.DEFAULT_PROMINENCE,
                        help=('Minimum prominence of the detected peaks '
                              '(in multiples of the input signal std).'))
    parser.add_argument('-v', dest='verbose',
                        action='store_true', 
                        help=('Enables printing standard information.'))
    args = parser.parse_args()

    if not os.path.isfile(args.stimuli_file):
        print('{} does not refer to a file.'.format(args.stimuli_file))
        sys.exit()
    if not os.path.isfile(args.recording_file):
        print('{} does not refer to a file.'.format(args.recording_file))
        sys.exit()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, 
                            format="Debug: {message}",
                            style='{')

    try:
        peaks = m2.rec2taps.extract_peaks(args.stimuli_file,
                                          args.recording_file,
                                          args.distance,
                                          args.prominence)
    except errors.Rec2TapsError as r2te:
        print(r2te, file=sys.stderr)
        sys.exit()

    for p in peaks:
        print(p)

if __name__ == '__main__':
    main()
