import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('tkagg')
rng = np.random.default_rng()

fs = 250
window = int(fs * 0.5)
x1 = np.load('data_jaw2.npy', allow_pickle=True).item()
x2 = np.load('data_eyeblink.npy', allow_pickle=True).item()
x1 = x1['data'][:, 1]
x2 = x2['data'][:, 1]
fig, ax = plt.subplots(2, 2)
for i in range(0, min(len(x1) - window, len(x2) - window), 25):
    print(i)
    ax[0,0].clear()
    ax[0,1].clear()
    xi1 = x1[i:i + window]
    xi2 = x2[i:i + window]
    ax[0, 0].plot(xi1)
    f, t, Zxx = signal.stft(xi2, fs, nperseg=1000)
    ax[0, 1].plot(xi2)
    f, t, Zxx = signal.stft(xi1, fs, nperseg=1000)
    ax[1, 0].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    ax[0, 0].set_title('jaw')
    f, t, Zxx = signal.stft(xi2, fs, nperseg=1000)
    ax[1, 1].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    ax[0, 1].set_title('eyeblink')
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    plt.pause(1e-3)
    plt.savefig(f"result/{i/250}s.png")
plt.show()