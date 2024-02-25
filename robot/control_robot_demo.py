"""
Control robot using data from the EEG sensor.
"""

from pylsl import StreamInlet, resolve_stream
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
import time
from duckietown.sdk.robots.duckiebot import DB21J


# General params
RUNTIME_SECONDS = 30

# EEG-related params
EEG_STREAM_ID = "76"
EEG_SAMPLES_BUFFER_SIZE = 40

# Robot-related params
RUN_IN_SIMULATION = True
# RUN_IN_SIMULATION = False
SIMULATED_ROBOT_NAME = "map_0/vehicle_0"
REAL_ROBOT_NAME = "rover"
BASE_SPEED = 0.3
STEERING_DECREASE_FACTOR = 0.1


def hpf(data, cutoff=20, fs=250):
    """Implement a high-pass filter for one channel."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    # perform filter on each channel
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def lpf(data, cutoff=20, fs=3000):
    """Implement a low-pass filter for one channel."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    # perform filter on each channel
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def is_eeg_gesture_left(eeg_data):
    n_samples_interval = 4
    eeg_data = np.array(eeg_data.queue)
    mean_data = np.mean(eeg_data[:,:7], axis=1)
    filtered_data = lpf(mean_data, cutoff=10, fs=250)
    noise_std = np.std(filtered_data)
    print(noise_std)
    threshold = max(noise_std * 2, 100)
    # threshold = 10
    # last_sample_peak = filtered_data[-n_samples_interval] > threshold
    # is_gesture = np.sum(mean_data > threshold) == 1 and last_sample_peak
    peaks, _ = find_peaks(filtered_data, height=threshold, prominence=threshold)
    is_gesture = len(filtered_data[peaks])>1
    if is_gesture:
        print(f"Left gesture: {filtered_data} {threshold} {len(filtered_data[peaks])}")
    return is_gesture



def is_eeg_gesture_right(eeg_data):
    n_samples_interval = 4
    eeg_data = np.array(eeg_data.queue)
    mean_data = np.mean(eeg_data[:,:7], axis=1)
    filtered_data = lpf(mean_data, cutoff=10, fs=250)
    noise_std = np.std(filtered_data)
    print(noise_std)
    threshold = max(noise_std * 2, 100)
    # threshold = 10
    # last_sample_peak = filtered_data[-n_samples_interval] > threshold
    # is_gesture = np.sum(mean_data > threshold) == 1 and last_sample_peak
    peaks, _ = find_peaks(filtered_data, height=threshold, prominence=threshold)
    is_gesture = len(filtered_data[peaks])==1
    if is_gesture:
        print(f"Right gesture: {filtered_data} {threshold} {len(filtered_data[peaks])}")
    return is_gesture
    # return False


def find_eeg_inlet_stream(stream_id):
    print("Looking for an EEG stream...")
    while True:
        streams = resolve_stream('type', 'EEG')
        # Iterate through streams
        print(f"Found {len(streams)} EEG streams")
        print("---------------")
        for stream in streams:
            sid = stream.name()[-2:]
            print(stream.name())
            if sid == stream_id:
                print(f"Found the headset with ID {sid}!")
                return StreamInlet(stream)


def main():
    inlet = find_eeg_inlet_stream(EEG_STREAM_ID)
    eeg_samples_buffer = deque(maxlen=EEG_SAMPLES_BUFFER_SIZE)

    if RUN_IN_SIMULATION:
        robot = DB21J(SIMULATED_ROBOT_NAME, simulated=RUN_IN_SIMULATION)
    else:
        robot = DB21J(REAL_ROBOT_NAME, simulated=RUN_IN_SIMULATION)
    robot.motors.start()

    print('Starting main loop...')
    counter = 0
    start_time = time.time()
    while time.time() - start_time < RUNTIME_SECONDS:
            
        # print(f'Get sample {eeg_samples_buffer.qsize()}')
        sample, timestamp = inlet.pull_sample()
        # print(f'Get sample {eeg_samples_buffer.qsize()}')
        eeg_samples_buffer.append(sample)

        # print(f'Got sample, buffer size {eeg_samples_buffer.qsize()}')

        if eeg_samples_buffer.full():

            # print(f'Got enough samples, ...')
            eeg_control_left = is_eeg_gesture_left(eeg_samples_buffer)
            eeg_control_right = is_eeg_gesture_right(eeg_samples_buffer)
            # print(f'Left: {eeg_control_left}, Right: {eeg_control_right}')

            if counter <= 0:
                if eeg_control_left or eeg_control_right:
                    counter = 100

                # Set speeds
                speed_left = BASE_SPEED
                speed_right = BASE_SPEED
                if eeg_control_left:
                    speed_left *= STEERING_DECREASE_FACTOR
                if eeg_control_right:
                    speed_right *= STEERING_DECREASE_FACTOR

                # Motor control
                # print(f'SPEED: Left: {speed_left}, Right: {speed_right}')
                robot.motors.publish((speed_left, speed_right))
                
            else:
                counter -= 1
                robot.motors.publish((speed_left, speed_right))

        # time.sleep(0.01)

    robot.motors.stop()
    print(f"Robot stopped after {RUNTIME_SECONDS} seconds.")


if __name__ == '__main__':
    main()
