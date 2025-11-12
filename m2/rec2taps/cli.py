import sys
import logging
logging.basicConfig(filename=sys.stderr)
logging.getLogger().setLevel(logging.INFO)

import argparse
import os
import m2.rec2taps
import re
from m2.rec2taps import defaults
from m2.rec2taps import errors

FILE_CHANNEL_RE = re.compile(r'([\w\.]+):(\d+)')

def rec2taps():
    parser = argparse.ArgumentParser(
        description=('Obtain tap times from a recording file synchronized '
                     'to a provided stimuli file. In the context of '
                     '"Simple and cheap setup for measuring timed responses '
                     'to auditory stimuli" (Miguel et. al. 2020).')
    )
    
    parser.add_argument('stimuli', type=str, metavar='stimuli_file(:channel)',
                        help='audio file of the stimuli')
    parser.add_argument('recording', type=str, 
                        metavar='recording_file(:loopback_channel)',
                        help=('audio file of the experiment recording. Should '
                              'have two channels, one with a loopback '
                              'recording of the stimuli and another one with '
                              'the signal from the input device.')
                       )
    parser.add_argument('tapping_files', type=str,
                        metavar='tapping_files:channel',
                        nargs='*',
                        help=('(optional) sequence of files and channel of interest'
                              'from which to extract taps synchronized to the stimuli')
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
    parser.add_argument('-D', '--debug_plot', dest='debug_plot',
                        const=defaults.DEFAULT_DEBUG_PLOT, 
                        default=None, nargs='?',
                        help=('Enables outputting a debug plot overlaying '
                              'peaks with the recording signal. Can receive '
                              'an argument to define the output filename.'))
    parser.add_argument('-i', dest='invert_input',
                        default=False, action='store_true',
                        help=('Enables outputting a debug plot overlaying '
                              'peaks with the recording signal. Can receive '
                              'an argument to define the output filename.'))
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.debug('Input paramters:' + str([args.stimuli, args.recording, args.tapping_files]))
    if args.stimuli.find(':') > -1 or len(args.tapping_files) > 0:
        logging.info('Processing files in INDIVIDUAL CHANNEL MODE')

        can_proceed = True

        for s in [args.stimuli, args.recording] + args.tapping_files:
            if FILE_CHANNEL_RE.match(s) is None:
                logging.error(f' {s} does not have the expected format `file_name:channel_number`')
                can_proceed = False

        if len(args.tapping_files) == 0:
            logging.error(' No tapping files were provided.')
            can_proceed = False

        if not can_proceed:
            logging.error(' Files were not properly input. Stopping.')
            sys.exit()

        stimuli_file, stimuli_channel = FILE_CHANNEL_RE.match(args.stimuli).groups()
        loopback_file, loopback_channel = FILE_CHANNEL_RE.match(args.recording).groups()
        tapping_files_channels = [FILE_CHANNEL_RE.match(t).groups() for t in args.tapping_files]

        audio_files = [stimuli_file, loopback_file] + [
            f for f, c in tapping_files_channels
        ]

        for file in audio_files:
            if not os.path.isfile(file):
                logging.error(f' Audio file could not be found: {file}')
                can_proceed = False
        
        if not can_proceed:
            logging.error(' Files were not properly input. Stopping.')
            sys.exit()
    

    else:
        logging.info('Processing files in STIMULI-RECORDING FILE MODE')
        if not os.path.isfile(args.stimuli):
            print('{} does not refer to a file.'.format(args.stimuli))
            sys.exit()
        if not os.path.isfile(args.recording):
            print('{} does not refer to a file.'.format(args.recording))
            sys.exit()

        try:
            peaks = m2.rec2taps.extract_peaks(args.stimuli,
                                              args.recording,
                                              args.distance,
                                              args.prominence,
                                              args.debug_plot,
                                              args.invert_input
                                             )
        except errors.Rec2TapsError as r2te:
            print(r2te, file=sys.stderr)
            sys.exit()

        for p in peaks:
            print(p)

if __name__ == '__main__':
    main()
