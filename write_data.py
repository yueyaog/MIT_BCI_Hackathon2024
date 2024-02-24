from pylsl import StreamInlet, resolve_stream
import time
import numpy as np
import matplotlib.pyplot as plt


def get_new_data(inlet):
    # Placeholder for your data retrieval logic
    sample, timestamp = inlet.pull_sample()
    return sample

def collect_data(inlet, interval=0.01, duration=1.0):
    """Collect data for a specified duration with a given interval between data points."""
    num_points = int(duration / interval)
    timestamps = np.linspace(0, duration, num_points, endpoint=False)
    data = np.array([get_new_data(inlet) for _ in range(num_points)])
    return timestamps, data

def main():
    # first resolve an EEG stream on the lab network
    # print("looking for an EEG stream...")
    # looking = True
    # while(looking):
    #   streams = resolve_stream('type', 'EEG')
    #   # Iterate through streams
    #   print(f"Found {len(streams)} streams")
    #   print("---------------")
    #   for stream in streams:
    #       sid = stream.name()[-2:]
    #       print(stream.name())
    #       if sid == "76":
    #           looking = False
    #           inlet = StreamInlet(stream)
    #           print("Found the headset!")
    #           break
    # timestamps, data = collect_data(inlet, interval=0.004, duration=60.0)
    # data_save = {
    #     'data': data,
    #     'timestamps': timestamps
    # }
    # # Save the data to a file
    # np.save('data_eyeblink.npy', data_save)

    # read the data to check if it was saved correctly
    data_read = np.load('data_eyeblink.npy', allow_pickle=True).item()
    print(data_read['data'].shape)
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
