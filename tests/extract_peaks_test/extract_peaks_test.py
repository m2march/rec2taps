import pytest
import numpy as np
import m2.rec2taps
from m2.rec2taps.defaults import DEFAULT_DISTANCE, DEFAULT_PROMINENCE
from m2.rec2taps.defaults import DEFAULT_DEBUG_PLOT


@pytest.mark.parametrize('invert', [False, True])
def test_invert(mocker, invert):
    numpy_peaks_mock = mocker.patch('m2.rec2taps.numpy_peaks')

    fsr_signal = np.array([[1, 2, 3]]).T

    mocker.patch('scipy.io.wavfile.read',
                 mocker.MagicMock(side_effect=[(2, None),
                                               (2, fsr_signal)])
                )
    mocker.patch('m2.rec2taps.best_channel_crosscorrelation',
                 mocker.MagicMock(side_effect=[(0, 1, 0)])
                )

    m2.rec2taps.extract_peaks(None, None, invert_input_signal=invert)

    if invert:
        expected_arg = fsr_signal[:, 0] * -1
    else:
        expected_arg = fsr_signal[:, 0]

    print(numpy_peaks_mock.call_args_list)
    assert len(numpy_peaks_mock.call_args_list) == 1
    assert (numpy_peaks_mock.call_args_list[0][0][0] == expected_arg).all()


def test_peaks_inverted():
    rec_peaks = m2.rec2taps.extract_peaks('stim.wav', 'rec.wav')
    inverted_peaks = m2.rec2taps.extract_peaks('stim.wav', 'rec-i.wav')
    corrected_peaks = m2.rec2taps.extract_peaks('stim.wav', 'rec-i.wav',
                                                invert_input_signal=True)
    assert (rec_peaks != inverted_peaks)
    assert (rec_peaks == corrected_peaks).all()
