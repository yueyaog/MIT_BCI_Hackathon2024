
"""
  Example program to show how to read a 
  multi-channel time series from LSL.
"""

from pylsl import StreamInlet, resolve_stream
import numpy as np
from scipy.signal import iirnotch, lfilter
import time
import matplotlib.pyplot as plt


def filter_sample(sample, freq=60, q=30):
    """Implement a 60Hz notch filter for each of the 7 channels."""
    fs = 500.0  # Sample frequency (Hz)
    w0 = freq / (fs / 2)  # Normalized Frequency
    # Design notch filter
    b, a = iirnotch(w0, q)
    filtered_sample = np.apply_along_axis(lambda x: lfilter(b, a, x), 0, sample)
    return filtered_sample

def get_new_data(inlet):
    # Placeholder for your data retrieval logic
    sample, timestamp = inlet.pull_sample()
    filtered_sample = filter_sample(sample)
    return filtered_sample

def update_lines(frame, lines, data_stream):
    timestamp, data = next(data_stream)
    for line, channel_data in zip(lines, data):
        x, y = line.get_data()
        x = np.append(x, timestamp)
        y = np.append(y, channel_data)
        # Update the line's data and keep the last N points for a rolling plot
        N = 100  # Adjust N based on how many data points you want to display at once
        line.set_data(x[-N:], y[-N:])
    return lines

def collect_data(inlet, interval=0.01, duration=1.0):
    """Collect data for a specified duration with a given interval between data points."""
    num_points = int(duration / interval)
    timestamps = np.linspace(0, duration, num_points, endpoint=False)
    data = np.array([get_new_data(inlet) for _ in range(num_points)])
    return timestamps, data

def plot_data(timestamps, data):
    """Plot the collected EEG data for all 7 channels."""
    plt.clf()
    for i in range(7):
        plt.subplot(7, 1, i+1)
        plt.plot(timestamps, data[:, i])
        plt.title(f'Channel {i+1}', fontsize=8)  # Reduce the font size
        plt.xlabel('Time (s)', fontsize=8)  # Reduce the font size
        plt.ylabel('Amplitude', fontsize=8)  # Reduce the font size
        plt.tick_params(axis='both', which='major', labelsize=6)  # Reduce the tick label size
    plt.subplots_adjust(hspace=0.4)  # Adjust the spacing between subplots
    plt.tight_layout()
    plt.pause(0.001)

def main():
    plt.ion()
    figure = plt.figure(figsize=(10, 7))
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    looking = True
    while(looking):
      streams = resolve_stream('type', 'EEG')
      # Iterate through streams
      print(f"Found {len(streams)} streams")
      print("---------------")
      for stream in streams:
          sid = stream.name()[-2:]
          print(stream.name())
          if sid == "76":
              looking = False
              inlet = StreamInlet(stream)
              print("Found the headset!")
              break
    while True:
        start_time = time.time()
        timestamps, data = collect_data(inlet)
        print(data)
        plot_data(timestamps, data)
        time.sleep(max(0, 0.5 - (time.time() - start_time)))  # Adjust sleep to ensure a 1-second cycle


if __name__ == '__main__':
    main()
