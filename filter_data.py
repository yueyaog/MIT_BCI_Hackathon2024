from pylsl import StreamInlet, resolve_stream
import time
import numpy as np
import matplotlib.pyplot as plt

def hpf(data, cutoff=20, fs=250):
    """Implement a high-pass filter for each of the 7 channels."""
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    # perform filter on each channel
    filtered_data = np.array([filtfilt(b, a, data[:, i]) for i in range(data.shape[1])]).T
    return filtered_data

import numpy as np
import scipy.signal

# Assuming we have a signal array called 'signal' and a sampling rate called 'fs'

def detect_spikes(signal, fs, threshold_factor=3.5, window_size=50):
    # Preprocess with a bandpass filter
    high = 1 / (fs / 2)  # High-pass filter at 1 Hz to remove DC offset
    low = 300 / (fs / 2)  # Low-pass filter to remove high-frequency noise
    b, a = scipy.signal.butter(1, [high, low], btype='band')
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    
    # Thresholding
    noise_std = np.std(filtered_signal[:window_size])
    threshold = noise_std * threshold_factor
    
    # Detection of onset and offset
    spike_onset_indices = []
    spike_offset_indices = []
    
    in_spike = False
    for i, value in enumerate(filtered_signal):
        if not in_spike and value > threshold:
            spike_onset_indices.append(i)
            in_spike = True
        elif in_spike and value < threshold:
            spike_offset_indices.append(i)
            in_spike = False
            
    # Convert indices to time
    spike_onsets = np.array(spike_onset_indices) / fs
    spike_offsets = np.array(spike_offset_indices) / fs
    
    return spike_onsets, spike_offsets, filtered_signal

def detect_spikes(data):
    # Use the function to detect spikes
    spike_onsets, spike_offsets, filtered_signal = detect_spikes(signal, fs)
    # Check the result
    spike_onsets, spike_offsets

def main():
    # read the data to check if it was saved correctly
    data_read = np.load('data/data_jaw2.npy', allow_pickle=True).item()
    print(data_read['data'].shape)
    # apply hpf to data
    data_read['data'] = hpf(data_read['data'])
    # plot all channels
    figure = plt.figure(figsize=(10, 7))
    for i in range(7):
        plt.subplot(7, 1, i+1)
        plt.plot(data_read['timestamps'], data_read['data'][:, i])
        plt.title(f'Channel {i+1}', fontsize=8)  # Reduce the font size
        plt.xlabel('Time (s)', fontsize=8)  # Reduce the font size
        plt.ylabel('Amplitude', fontsize=8)  # Reduce the font size
        plt.tick_params(axis='both', which='major', labelsize=6)  # Reduce the tick label size
    plt.show()

if __name__ == '__main__':
    main()
