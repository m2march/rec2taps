import sys
import io
import pytest
import m2.rec2taps
import numpy as np
from m2.rec2taps import defaults
from m2.rec2taps.cli import rec2taps


def test_default_arguments(mocker):
    mocker.patch('m2.rec2taps.extract_peaks')
    mocker.patch('sys.argv', ['exec', 'sti', 'rec'])
    mocker.patch('os.path.isfile', lambda x: True)

    rec2taps()

    m2.rec2taps.extract_peaks.assert_called_once_with(
        'sti', 'rec', defaults.DEFAULT_DISTANCE, defaults.DEFAULT_PROMINENCE)


def test_passed_arguments(mocker):
    mocker.patch('m2.rec2taps.extract_peaks')
    mocker.patch('sys.argv', ['exec', 'sti', 'rec', '-d', '50', '-p', '3'])
    mocker.patch('os.path.isfile', lambda x: True)

    rec2taps()

    m2.rec2taps.extract_peaks.assert_called_once_with(
        'sti', 'rec', 50, 3)


def test_error_different_sr(mocker):
    mocker.patch('sys.argv', ['exec', 'sti', 'rec'])
    mocker.patch('os.path.isfile', lambda x: True)
    mocker.patch('scipy.io.wavfile.read',
                 mocker.MagicMock(side_effect=[('1', None),
                                               ('2', None)]))
    stdout_mock = mocker.patch('sys.stdout', new_callable=io.StringIO)
    stderr_mock = mocker.patch('sys.stderr', new_callable=io.StringIO)

    with pytest.raises(SystemExit):
        rec2taps()

    assert stdout_mock.getvalue() == ''
    assert stderr_mock.getvalue() == ('sti and rec do not have the same '
                                      'sample rate (1 != 2)\n')


def test_error_shorter_sti(mocker):
    mocker.patch('sys.argv', ['exec', 'sti', 'rec'])
    mocker.patch('os.path.isfile', lambda x: True)
    mocker.patch('scipy.io.wavfile.read',
                 mocker.MagicMock(side_effect=[(None, np.zeros((10, 2))),
                                               (None, np.zeros((8, 2)))]))
    stdout_mock = mocker.patch('sys.stdout', new_callable=io.StringIO)
    stderr_mock = mocker.patch('sys.stderr', new_callable=io.StringIO)

    with pytest.raises(SystemExit):
        rec2taps()

    assert stdout_mock.getvalue() == ''
    assert stderr_mock.getvalue() == ('Stimuli file (sti) is shorter than '
                                      'recording file (rec).\n')


def test_peaks_output(mocker):
    mocker.patch('sys.argv', ['exec', 'sti', 'rec'])
    mocker.patch('os.path.isfile', lambda x: True)
    mocker.patch('m2.rec2taps.extract_peaks', lambda *x : [1, 2, 3])
    stdout_mock = mocker.patch('sys.stdout', new_callable=io.StringIO)
    stderr_mock = mocker.patch('sys.stderr', new_callable=io.StringIO)

    rec2taps()

    assert stderr_mock.getvalue() == ''
    assert stdout_mock.getvalue() == '1\n2\n3\n'
