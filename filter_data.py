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

def detect_jaw_clench(data, threshold=0.5):
    """Detect jaw clenching based on the amplitude of the data."""
    #

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
