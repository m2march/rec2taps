import logging
import numpy as np
from scipy.signal import find_peaks, fftconvolve
from scipy.io import wavfile
from m2.rec2taps import errors


def prominence_amp(data, prominence=1.5):
    prominence_amp = data.std() * prominence
    return prominence_amp


def rectify(data, prominence_amp):
    rect_ys = data.copy()
    rect_ys[data < prominence_amp] = 0
    return rect_ys


def numpy_peaks(data, sr, distance=100, prominence=1.5):
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
        print(signal_a[:, channel_a].shape[0],
              signal_b[:, channel_b].shape[0])
        raise SignalTooShortForConvolution()

    cc = fftconvolve(signal_a[:, channel_a], 
                     list(reversed(signal_b[:, channel_b])), 
                     'valid')
    return {
        'argmax': np.argmax(cc),
        'max': np.max(cc)
    }


def best_channel_crosscorrelation(stimuli_signal, recording_signal):
    '''
    Returns indexes and lag of the channels that best correlate the signals.

    The function compares stimuli and recording channels assuming one of the
    channels from the recording was recorded as loopback of another of the
    channels from the stimuli. 

    It returns an index for each signal indicating the channels that best
    cross correlate between both. Additionally, it returns the
    cross-correlation lag between said signals.

    The functions assumes boths signals have equal sample rate.

    Args:
        stimuli_signal: 2d array with the signal time series from the stimuli
            audio
        recording_signal: 2d array with the signal time series from the
            recording audio

    Returns:
        Returns 3 elements as a tuple:
            stimuli loopback channel (0 or 1)
            recording loopback channel (0 or 1)
            delay between stimuli to recorded loopback in samples
    '''
    corrs = [
        [
            best_crosscorrelation(recording_signal, ri, stimuli_signal, si)
            for ri in [0, 1]
        ]
        for si in [0, 1]
    ]
    max_cor_idx = np.argmax([[c['max'] for c in x] for x in corrs])

    row = max_cor_idx // 2
    col = max_cor_idx % 2
    return (row, col, corrs[row][col]['argmax'])


def extract_peaks(stimuli_file, recording_file, distance, threshold):
    '''
    Extracts peaks from recording file synchronized to the stimuli.

    The function extracts peaks from the recording file considering it has two
    channels, one with the loopback of the stimuli and another one with the
    recording of the input device.

    To select the channel from the recording file, it uses the one that has the
    lowest cross-correlation with any channel of the stimuli file.

    The cross-correlation of the other channel in the recording file is used to
    find the lag between stimuli and recording to offset the peaks found to the
    start of the stimuli.

    The function also requires the stimuli file to have the same sample rate
    as the recording file.

    Params:
        stimuli_file: path to the stimuli audio file
        recording_file: path to the recording audio file
        distance: minimum distance in ms between detected peaks
        prominence: minimum prominence of detected peaks in multiples of the
            input recording signal in standard deviation

    Returns:
        1d array of peaks in ms relative to the beginning of the stimulus
        signal
    '''
    stimuli_sr, stimuli_signal = wavfile(stimuli_file)
    recording_sr, recording_signal = wavfile(recording_file)

    if (stimuli_sr != recording_sr):
        raise UnequalSampleRate(stimuli_file, recording_file, stimuli_sr,
                                recording_sr)

    si, ri, lag_s = best_channel_crosscorrelation(stimuli_signal,
                                                  recording_signal)
    
    logging.debug(('Obtaining lag from recording to '
                   'stimuli using channels {} and {} '
                   'from stimuli and recording audios (resp.)').format(
                   si, ri))

    lag = lag_s / sr * 1000
    logging.debug(('Recording is delayed {} ms from the stimuli').format(lag))

    peaks = numpy_peaks(recording_signal, recording_sr,
                        distance, prominence)

    return (np.array(peaks) / sr * 1000) - lag
