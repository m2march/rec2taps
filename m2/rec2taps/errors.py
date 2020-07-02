class Rec2TapsError(ValueError):
    pass


class UnequalSampleRate(Rec2TapsError):
    'Stimuli file and the recording file have unequal sample rate'

    def __init__(self, stimuli_file, recording_file, stimuli_sr, recording_sr):
        self.stimuli_file = stimuli_file
        self.recording_file = recording_file
        self.stimuli_sr = stimuli_sr
        self.recording_sr = recording_sr

        super().__init__(('{} and {} do not have the same sample rate '
                          '({} != {})').format(stimuli_file, recording_file,
                                               stimuli_sr, recording_sr))


class SignalTooShortForConvolution(Rec2TapsError):
    pass


class StimuliShorterThanRecording(Rec2TapsError):
    'Stimuli signal is shorter than recording signal'

    def __init__(self, stimuli_file, recording_file):
        super().__init__(('Stimuli file ({}) is shorter than recording file '
                          '({}).').format(stimuli_file, recording_file))
