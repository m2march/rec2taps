import pytest
import numpy as np
import m2.rec2taps as rec2taps
import os.path
from scipy.io import wavfile

STI_FILE = os.path.join(os.path.dirname(__file__), 'stim.wav')
REC_FILE = os.path.join(os.path.dirname(__file__), 'rec.wav')
STI_ALT_FILE = os.path.join(os.path.dirname(__file__), 'stim_alt.wav')
SR = 48000


def base_lag(sti_file, rec_file):
    sr, sti_data = wavfile.read(sti_file)
    _, rec_data = wavfile.read(rec_file)
    cc = rec2taps.best_crosscorrelation(sti_data, 0, rec_data, 0)
    return cc['argmax'] * 1000 / SR


@pytest.fixture(scope='module')
def stim_base_lag():
    return base_lag(REC_FILE, STI_FILE)


@pytest.fixture(scope='module')
def stim_data():
    _, sti_data = wavfile.read(STI_FILE)
    return sti_data

@pytest.fixture(scope='module')
def stim_alt_data():
    _, sti_data = wavfile.read(STI_ALT_FILE)
    return sti_data

@pytest.fixture(scope='module')
def rec_data():
    _, rec_data = wavfile.read(REC_FILE)
    return rec_data


def lag_signal(signal, lag, sr):
    'Lags signal by lag (in ms)'
    lag_s = int(lag * sr / 1000)
    if lag < 0:
        signal = signal[-lag_s:, :]
    if lag > 0:
        signal = np.concatenate([np.zeros((lag_s, 2)),
                                 signal])
    return signal



LAGS = [0, -50, -10, 10, 50, 107]

@pytest.mark.parametrize('lag', LAGS)
@pytest.mark.parametrize('inverted', [False, True])
@pytest.mark.parametrize('stereo_unequal', [False, True, 'inverted'])
def test_best_channel_crosscorrelation(stim_data, stim_alt_data, rec_data, 
                                       lag, inverted, stereo_unequal,
                                       stim_base_lag):
    print(stim_base_lag)
    # Stim
    if stereo_unequal != False:
        l = 0 if stereo_unequal != 'inverted' else 1
        print('unequal:', l)
        stim_subdata = stim_data[:, 0]
        alt_stim_subdata = stim_alt_data[:, 0]
        max_width = max(len(stim_subdata), len(alt_stim_subdata))
        stim_subdata_pad = np.pad(stim_subdata, 
                                  (0, max_width - len(stim_subdata)),
                                  'constant')
        alt_stim_subdata_pad = np.pad(alt_stim_subdata,
                                      (0, max_width - len(alt_stim_subdata)),
                                      'constant')
        all_data = [stim_subdata_pad, alt_stim_subdata_pad]
        stim_data_processed = np.array([
            all_data[l], all_data[1-l]
        ]).T
    else:
        stim_data_processed = stim_data.copy()

    # Rec
    l = 0 if not inverted else 1
    rec_data_processed = np.array([rec_data[:, l], rec_data[:, 1-l]]).T
    rec_data_processed = lag_signal(rec_data_processed, lag, SR)

    # Expected
    stim_channel = {
        True: 0,
        False: None,
        'inverted': 1
    }[stereo_unequal]
    rec_channel = 0 if not inverted else 1

    r = rec2taps.best_channel_crosscorrelation(stim_data_processed,
                                               rec_data_processed)

    if stim_channel is not None:
        assert r[0] == stim_channel

    assert r[1] == rec_channel
    assert abs((r[2] * 1000 / SR) - (stim_base_lag + lag)) < 1
