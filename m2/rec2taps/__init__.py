import logging
import numpy as np
from scipy.signal import find_peaks, fftconvolve
from scipy.io import wavfile
from m2.rec2taps import errors
from m2.rec2taps.defaults import DEFAULT_DISTANCE, DEFAULT_PROMINENCE

CAN_DEBUG_PLOT = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    CAN_DEBUG_PLOT = True
except ImportError as ie:
    logging.warn(('Matplotlib is not available. If --debug_plot flag is used '
                  'no plot will be produced.'))
except Exception as e:
    logging.error(
        ('An error occured while loading matplotlib: {}'.format(str(e)))
    )


def prominence_amp(data, prominence=DEFAULT_PROMINENCE):
    prominence_amp = data.std() * prominence
    return prominence_amp


def rectify(data, prominence_amp):
    rect_ys = data.copy()
    rect_ys[data < prominence_amp] = 0
    return rect_ys


def numpy_peaks(data, sr, distance=DEFAULT_DISTANCE,
                prominence=DEFAULT_PROMINENCE):
    '''
    Obtains peaks using scipy find_peaks adjusted to our FSR data.
    
    Params:
        data: 1d-array of signal values
        sr: int indicating sample rate
        distance: minimun distance in ms between peaks
	prominence: minimun prominence as multiple of signal standard
		    deviation
    '''
    prominence_a = prominence_amp(data, prominence)
    rect_ys = rectify(data, prominence_a)
    distance = distance * sr / 1000
    peaks, props = find_peaks(rect_ys, prominence=prominence_a,
                              distance=distance)
    return peaks


def best_crosscorrelation(signal_a, channel_a, signal_b, channel_b):
    '''
    Correlates both signals and return max crosscorr value and position.

    Args:
        signal_a, signal_b: signals to be cross correlated as 2d array
        channel_a, channel_b: channels to be used from each signal

    Returns:
        dictionary with best crosscorr value and position of the value. E.g.:

        { 'argmax': 12540, 'max': 45.6 }

    '''
    if (signal_a[:, channel_a].shape[0] < signal_b[:, channel_b].shape[0]):
        raise errors.SignalTooShortForConvolution()

    cc = fftconvolve(signal_a[:, channel_a],
                     list(reversed(signal_b[:, channel_b])),
                     'valid')
    return {
        'argmax': np.argmax(cc),
        'max': np.max(cc)
    }


def best_channel_crosscorrelation(stimulus_signal, recording_signal):
    '''
    Returns indexes and lag of the channels that best correlate the signals.

    The function compares stimulus and recording channels assuming one of the
    channels from the recording was recorded as loopback of another of the
    channels from the stimulus. 

    It returns an index for each signal indicating the channels that best
    cross correlate between both. Additionally, it returns the
    cross-correlation lag between said signals.

    The functions assumes boths signals have equal sample rate.

    Args:
        stimulus_signal: 2d array with the signal time series from the stimulus
            audio
        recording_signal: 2d array with the signal time series from the
            recording audio

    Returns:
        Returns 3 elements as a tuple:
            stimulus loopback channel (0 or 1)
            recording loopback channel (0 or 1)
            delay between stimulus to recorded loopback in samples
    '''
    corrs = [
        [
            best_crosscorrelation(recording_signal, ri, stimulus_signal, si)
            for ri in [0, 1]
        ]
        for si in [0, 1]
    ]
    max_cor_idx = np.argmax([[c['max'] for c in x] for x in corrs])

    row = max_cor_idx // 2
    col = max_cor_idx % 2
    return (row, col, corrs[row][col]['argmax'])


def extract_peaks(stimulus_file, recording_file, 
                  distance=DEFAULT_DISTANCE, 
                  prominence=DEFAULT_PROMINENCE,
                  debug_plot=None,
                  invert_input_signal=False,
                 ):
    '''
    Extracts peaks from recording file synchronized to the stimulus.

    The function extracts peaks from the recording file considering it has two
    channels, one with the loopback of the stimulus and another one with the
    recording of the input device.

    To select the channel from the recording file, it uses the one that has the
    lowest cross-correlation with any channel of the stimulus file.

    The cross-correlation of the other channel in the recording file is used to
    find the lag between stimulus and recording to offset the peaks found to the
    start of the stimulus.

    The function also requires the stimulus file to have the same sample rate
    as the recording file.

    Params:
        stimulus_file: path to the stimulus audio file
        recording_file: path to the recording audio file
        distance: minimum distance in ms between detected peaks
        prominence: minimum prominence of detected peaks in multiples of the
            input recording signal in standard deviation
        debug_plot: if not None, string with file path to output a debug plot
            of the detected peaks
        invert_input_signal: if not True, input signal from recording_file
            is inverted (* -1)

    Returns:
        1d array of peaks in ms relative to the beginning of the stimulus
        signal
    '''
    logging.debug(('Obtaining peaks for {} synched to {} with params '
                   'distance={} and prominence={}').format(
                       recording_file, stimulus_file, distance, prominence))

    stimulus_sr, stimulus_signal = wavfile.read(stimulus_file)
    recording_sr, recording_signal = wavfile.read(recording_file)

    if (stimulus_sr != recording_sr):
        raise errors.UnequalSampleRate(stimulus_file, recording_file,
                                       stimulus_sr, recording_sr)

    try:
        si, ri, lag_s = best_channel_crosscorrelation(stimulus_signal,
                                                      recording_signal)
    except errors.SignalTooShortForConvolution as r2te:
        ne = errors.StimuliShorterThanRecording(stimulus_file,
                                                recording_file)
        raise ne from r2te

    logging.debug(('Obtaining lag from recording to '
                   'stimulus using channels {} and {} '
                   'from stimulus and recording audios (resp.)').format(
        si, ri))

    lag = lag_s / recording_sr * 1000
    logging.debug(('Recording is delayed {} ms from the stimulus').format(lag))

    fsr_signal = recording_signal[:, 1-ri]
    
    fsr_signal = fsr_signal * (1 - 2 * int(invert_input_signal))

    peaks = numpy_peaks(fsr_signal, recording_sr, distance, prominence)

    recording_peaks = (np.array(peaks) / recording_sr * 1000)


    if debug_plot is not None and CAN_DEBUG_PLOT:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(fsr_signal.shape[0]) / recording_sr * 1000,
                 fsr_signal, color='C2')
        ymin, ymax = plt.ylim()
        plt.yticks([])
        plt.vlines(recording_peaks, ymin, ymax, color='C1')
        plt.xlabel('time (ms)')
        plt.ylabel('amplitude')
        plt.savefig(debug_plot)

    return recording_peaks - lag
