"""
Control robot using data from the EEG sensor.
"""
from pylsl import StreamInlet, resolve_stream
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
import time
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.types import LEDsPattern


# General params
RUNTIME_SECONDS = 120

# EEG-related params
EEG_STREAM_ID = "76"
EEG_SAMPLES_BUFFER_SIZE = 40

# Robot-related params
RUN_IN_SIMULATION = True
RUN_IN_SIMULATION = False
SIMULATED_ROBOT_NAME = "map_0/vehicle_0"
REAL_ROBOT_NAME = "rover"
BASE_SPEED = 0.4
STEERING_DECREASE_FACTOR_LEFT = 0.05
STEERING_DECREASE_FACTOR_RIGHT = 0.02
# BASE_SPEED = 0.35
# STEERING_DECREASE_FACTOR_LEFT = 0.05
# STEERING_DECREASE_FACTOR_RIGHT = 0.02
STEERING_MEMORY_NUM_SAMPLES = 100
COLOR_OFF = (0, 0, 0, 0.0)
COLOR_AMBER = (1, 0.7, 0, 1.0)
COLOR_RED = (1, 0, 0, 1.0)
COLOR_GREEN = (0, 1, 0, 1.0)
LED_pattern_left = LEDsPattern(front_left=COLOR_GREEN, rear_left=COLOR_GREEN, front_right=COLOR_OFF, rear_right=COLOR_OFF)
LED_pattern_right = LEDsPattern(front_left=COLOR_OFF, rear_left=COLOR_OFF, front_right=COLOR_GREEN, rear_right=COLOR_GREEN)
LED_pattern_straight_none = LEDsPattern(front_left=COLOR_OFF, rear_left=COLOR_OFF, front_right=COLOR_OFF, rear_right=COLOR_OFF)
LED_pattern_straight_both = LEDsPattern(front_left=COLOR_RED, rear_left=COLOR_RED, front_right=COLOR_RED, rear_right=COLOR_RED)


def hpf(data, cutoff=20, fs=250):
    """Implement a high-pass filter for one channel."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def lpf(data, cutoff=20, fs=3000):
    """Implement a low-pass filter for one channel."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def is_eeg_gesture_left(eeg_data):
    eeg_data = np.array(eeg_data)
    mean_data = np.mean(eeg_data[:,:7], axis=1)
    filtered_data = hpf(mean_data, cutoff=10, fs=250)
    noise_std = np.std(filtered_data)
    threshold = max(noise_std * 2, 100)
    peaks, _ = find_peaks(filtered_data, height=threshold, prominence=threshold/2)
    is_gesture = len(filtered_data[peaks]) > 1
    if is_gesture:
        print(f"Left gesture: {filtered_data} {threshold} {len(filtered_data[peaks])}")
    return is_gesture


def is_eeg_gesture_right(eeg_data):
    eeg_data = np.array(eeg_data)
    mean_data = np.mean(eeg_data[:,:7], axis=1)
    filtered_data = lpf(mean_data, cutoff=10, fs=250)
    noise_std = np.std(filtered_data)
    threshold = max(noise_std * 2, 100)
    peaks, _ = find_peaks(filtered_data, height=threshold, prominence=threshold/4)
    is_gesture = len(filtered_data[peaks]) == 1
    if is_gesture:
        print(f"Right gesture: {filtered_data} {threshold} {len(filtered_data[peaks])}")
    return is_gesture


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
    robot.lights.start()
    robot.motors.start()

    print('Starting main loop...')
    steering_memory_samples_counter = 0
    start_time = time.time()
    while time.time() - start_time < RUNTIME_SECONDS:
            
        # print(f'Getting sample {eeg_samples_buffer.qsize()}')
        sample, timestamp = inlet.pull_sample()
        # print(f'Got sample {eeg_samples_buffer.qsize()}')
        eeg_samples_buffer.append(sample)
        # print(f'Got sample, buffer size {eeg_samples_buffer.qsize()}')

        if len(eeg_samples_buffer) == EEG_SAMPLES_BUFFER_SIZE:
            # print(f'Got enough samples, ...')
            eeg_control_left = is_eeg_gesture_left(eeg_samples_buffer)
            eeg_control_right = is_eeg_gesture_right(eeg_samples_buffer)
            # print(f'Detected gestures: Left: {eeg_control_left}, Right: {eeg_control_right}')
            if eeg_control_left and eeg_control_right:
                print(f'----- Both gestures detected!')

            if steering_memory_samples_counter <= 0:
                # if eeg_control_left != not eeg_control_right:
                if eeg_control_left or eeg_control_right:
                    steering_memory_samples_counter = STEERING_MEMORY_NUM_SAMPLES

                # Set speeds
                speed_left = BASE_SPEED
                speed_right = BASE_SPEED
                if eeg_control_left:
                    speed_left *= STEERING_DECREASE_FACTOR_LEFT
                if eeg_control_right:
                    speed_right *= STEERING_DECREASE_FACTOR_RIGHT                
            else:
                steering_memory_samples_counter -= 1

            # LED control based on speeds
            if speed_left < speed_right:
                robot.lights.publish(LED_pattern_left)
            elif speed_left > speed_right:
                robot.lights.publish(LED_pattern_right)
            elif eeg_control_left and eeg_control_right:
                robot.lights.publish(LED_pattern_straight_both)
            else:
                robot.lights.publish(LED_pattern_straight_none)

            # Motor control
            # print(f'SPEED: Left: {speed_left}, Right: {speed_right}')
            robot.motors.publish((speed_left, speed_right))

        # time.sleep(0.01)

    robot.motors.stop()
    robot.lights.stop()
    print(f"Robot stopped after {time.time() - start_time} seconds.")

if __name__ == '__main__':
    main()
