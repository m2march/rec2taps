This utility extract tap times from a input device recording synchronized to
the start time of an stimuli device under the assumption that the recording has
one channel where a loopback from the stimuli was recorded.

This tool was written as part of a proposed setup for recording tap times
presented in "Simple and cheap setup for measuring timed responses to auditory
stimuli" (Miguel et. al. 2020).

The tool works as follows: according the proposed experimental setup, a channel
from the output is looped back into the recording device and recorded along
with the signal of the input device where the participant performs the tapping.
The tool takes as input the stimuli presented and the recording and analyses
the signals to obtain tap times relative to the beginning of the simuli. This
requires the tool to find which channel of the stimuli is also recorded in at
which file of the recording.  With this information, the tool can recognize
which channel in the recording corresponds with the the stimuli and which is
the input. The lag between the stimuli start and its start in the recording is
found by maximizing the crosscorrelation between the loopback and the original
signal (in the looped-back channel).

Nextn, the input signal is analyze to obtain peaks. Peaks are found according 
to the algorithm in `scipy.signal.find_peaks` by detectecting peaks with 
a minimum prominence and distance between the peaks. These parameters can 
be configured (see `rec2taps -h`).
