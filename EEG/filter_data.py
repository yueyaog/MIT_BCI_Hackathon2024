import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class RealTimeSpikeDetector:
    def __init__(self, fs, threshold_factor=5, window_size=40):
        self.fs = fs
        self.threshold_factor = threshold_factor
        self.window_size = window_size
        self.buffer = np.zeros(window_size)
        self.threshold = None
        self.previous_value = 0
    
    def process_frame(self, frame):
        # Update buffer
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = frame
        # apply hpf
        buffer_hpf = hpf(self.buffer, cutoff=10, fs=self.fs)
        
        noise_std = np.std(buffer_hpf)
        self.threshold = max(noise_std * self.threshold_factor,100)
        
        # Check if the current frame crosses the threshold
        if self.buffer[-1] > self.threshold:
            return 1  # Spike detected
        else:
            return 0  # No spike detected


# Define the frame generator based on the saved data.
def frame_generator(data, frame_duration_ms, fs):
    frame_size = int((frame_duration_ms / 1000) * fs)  # Calculate frame size in samples
    num_frames = len(data) // frame_size

    for i in range(num_frames):
        # Yield a frame of data
        yield np.mean(data[i*frame_size:(i+1)*frame_size,:7])


def hpf(data, cutoff=20, fs=250):
    """Implement a high-pass filter for one channel."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    # perform filter on each channel
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def detect_spikes(signal, fs, threshold_factor=2, window_size=20):
    
    # Thresholding
    noise_std = np.std(signal[:window_size])
    threshold = noise_std * threshold_factor
    
    # Detection of onset and offset
    spike_onset_indices = []
    spike_offset_indices = []
    
    in_spike = False
    for i, value in enumerate(signal):
        if not in_spike and value > threshold:
            spike_onset_indices.append(i)
            in_spike = True
        elif in_spike and value < threshold:
            spike_offset_indices.append(i)
            in_spike = False
            
    # Convert indices to time
    spike_onsets = np.array(spike_onset_indices) / fs
    spike_offsets = np.array(spike_offset_indices) / fs
    
    return spike_onsets, spike_offsets, signal

def main():
    # read the data to check if it was saved correctly
    data_read = np.load('MIT_BCI_Hackathon2024/EEG/data/data_jaw2.npy', allow_pickle=True).item()
    print(data_read['data'].shape)

    # After loading and preprocessing your data
    fs = 250  # Your sampling rate
    frame_duration_ms = 4  # Your frame duration in milliseconds

    # Initialize your real-time spike detector with the appropriate parameters
    spike_detector = RealTimeSpikeDetector(fs, threshold_factor=3, window_size=40)

    # Use the frame generator with your actual data
    gen = frame_generator(data_read['data'], frame_duration_ms, fs)


    # Now process each frame as it comes in
    real_time_decisions = []
    for frame in gen:
        decision = spike_detector.process_frame(frame)
        real_time_decisions.append(decision)
    
    # spike_onsets, spike_offsets, filtered_signal = detect_spikes(data_read['data'][:,0], 250)
    # plot all channels
    figure = plt.figure(figsize=(10, 7))
    for i in range(7):
        plt.subplot(7, 1, i+1)
        plt.plot(data_read['timestamps'], data_read['data'][:, i])
        # plot vertical lines at real time decisions
        plt.vlines(np.where(real_time_decisions)[0]/250, -200, 200, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Channel {i+1}', fontsize=8)  # Reduce the font size
        plt.xlabel('Time (s)', fontsize=8)  # Reduce the font size
        plt.ylabel('Amplitude', fontsize=8)  # Reduce the font size
        plt.tick_params(axis='both', which='major', labelsize=6)  # Reduce the tick label size
    plt.show()

if __name__ == '__main__':
    main()
