"""
Control robot using data from the EEG sensor.
"""

from pylsl import StreamInlet, resolve_stream
import numpy as np
from scipy.signal import iirnotch, lfilter,butter, filtfilt
from queue import Queue
import time
from duckietown.sdk.robots.duckiebot import DB21J


# General params
RUNTIME_SECONDS = 10

# EEG-related params
EEG_STREAM_ID = "76"
EEG_SAMPLES_BUFFER_SIZE = 40
FILTER_FREQ = 60
FILTER_Q = 30
FILTER_FS = 500.0  # Sample frequency (Hz)

# Robot-related params
SIMULATED_ROBOT_NAME = "map_0/vehicle_0"
REAL_ROBOT_NAME = "rover"
BASE_SPEED = 0.1

def hpf(data, cutoff=20, fs=250):
    """Implement a high-pass filter for one channel."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    # perform filter on each channel
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def lpf(data, cutoff=20, fs=250):
    """Implement a low-pass filter for one channel."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    # perform filter on each channel
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def is_eeg_gesture_left(eeg_data):
    eeg_data = np.array(eeg_data.queue)
    mean_data = np.mean(eeg_data[:,:7], axis=0)
    filtered_data = hpf(mean_data, cutoff=10, fs=250)
    noise_std = np.std(filtered_data)
    threshold = max(noise_std * 3, 100)
    return np.sum(filtered_data>threshold)>1


def is_eeg_gesture_right(eeg_data):
    n_samples_interval = 4
    eeg_data = np.array(eeg_data.queue)
    mean_data = np.mean(eeg_data[:,:7], axis=0)
    filtered_data = lpf(mean_data, cutoff=10, fs=250)
    noise_std = np.std(filtered_data)
    threshold = max(noise_std * 3, 100)
    last_sample_peak = filtered_data[-n_samples_interval]>threshold
    return np.sum(mean_data>threshold)==1 and last_sample_peak


def find_eeg_inlet_stream(stream_id):
    print("looking for an EEG stream...")
    while True:
        streams = resolve_stream('type', 'EEG')
        # Iterate through streams
        print(f"Found {len(streams)} streams")
        print("---------------")
        for stream in streams:
            sid = stream.name()[-2:]
            print(stream.name())
            if sid == EEG_STREAM_ID:
                print("Found the headset!")
                return StreamInlet(stream)


def filter_sample(sample, filter_b, filter_a):
    return np.apply_along_axis(lambda x: lfilter(filter_b, filter_a, x), 0, sample)


def main():
    inlet = find_eeg_inlet_stream(EEG_STREAM_ID)
    eeg_samples_buffer = Queue(maxsize=EEG_SAMPLES_BUFFER_SIZE)

    # Design notch filter
    FILTER_W0 = FILTER_FREQ / (FILTER_FS / 2)  # Normalized Frequency
    filter_b, filter_a = iirnotch(FILTER_W0, FILTER_Q)

    robot: DB21J = DB21J(SIMULATED_ROBOT_NAME, simulated=True)  # change accordingly
    # robot: DB21J = DB21J(REAL_ROBOT_NAME, simulated=False)  # change accordingly
    robot.motors.start()
    # robot.camera.start()

    start_time = time.time()
    while time.time() - start_time < RUNTIME_SECONDS:
            
        sample, timestamp = inlet.pull_sample()
        eeg_samples_buffer.put(sample)

        if eeg_samples_buffer.full():
            # EEG input (boolean)
            eeg_control_left = is_eeg_gesture_left(eeg_samples_buffer)
            eeg_control_right = is_eeg_gesture_right(eeg_samples_buffer)

            # Camera input (if needed)
            # camera_data = robot.camera.capture(block=True)

            # Set speeds
            speed_left = BASE_SPEED
            speed_right = BASE_SPEED
            if eeg_control_left:
                speed_left = 0
            if eeg_control_right:
                speed_right = 0

            # Motor control
            robot.motors.publish((speed_left, speed_right))

    robot.motors.stop()
    # robot.camera.stop()
    print("Stopped.")


if __name__ == '__main__':
    main()
