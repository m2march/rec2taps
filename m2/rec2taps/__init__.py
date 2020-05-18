from scipy.signal import find_peaks

def numpy_peaks(data, sr, distance=100, prominence=2):
    '''
    Obtains peaks using scipy find_peaks adjusted to our FSR data.
    
    Params:
        data: 1d-array of signal values
        sr: int indicating sample rate
        distance: minimun distance in ms between peaks
	prominence: minimun prominence as multiple of signal standard
		    deviation
    '''
    prominence_amp = data.std() * prominence
    rect_ys = data.copy()
    rect_ys[data < prominence_amp] = 0
    distance = distance * sr / 1000
    peaks, props = find_peaks(rect_ys, prominence=prominence_amp, 
                              distance=distance)
    return peaks
