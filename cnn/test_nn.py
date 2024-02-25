import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
from train_nn import Net
from torch import stft
import torch
from collections import deque

matplotlib.use('tkagg')
rng = np.random.default_rng()



class RealTimeDetector:
    def __init__(self, model, fs=250, window_dt=0.5, n_channels=11, device="cpu"):
        self.device = torch.device(device)
        window_dn = int(fs * window_dt)  # 0.5s
        self.model = model.to(self.device)
        self.model.eval()

    def inference(self, frame):
        frame = torch.tensor(frame, requires_grad=False).to(self.device)
        specs = stft(frame, 50, hop_length=5, return_complex=True).real.unsqueeze(0).float()
        result = torch.argmax(self.model(specs)).item()
        category = ["nothing", "jaw", "eye_blink"]
        print(category[result])

if __name__=="__main__":
    model = Net()
    ckpt = torch.load('EEG/ckpt/ckpt_20.pt')
    model.load_state_dict(ckpt)
    detector = RealTimeDetector(model)

    test_data = np.load('traindata_2min.npy', allow_pickle=True).item()
    # test_data = np.load('EEG/data/data_eyeblink.npy', allow_pickle=True).item()
    test_data = test_data['data']
    fs = 250
    n_channels = 7
    channels = [0,1,2,3,4,5,6]
    n_plot_window = int(1*fs)
    n_data_window = int(0.5*fs)
    plot_data_window = deque([np.zeros(n_channels)]*n_plot_window,maxlen=n_plot_window)
    test_data_window = deque([np.zeros(n_channels)]*n_data_window, maxlen=n_data_window)
    # fig, axes = plt.subplots(n_channels,1)
    fig, axes = plt.subplots(1,1)
    for i in range(n_data_window, len(test_data)):
        test_data_window.append(test_data[i][channels])
        detector.inference(np.array(test_data_window).transpose())
        plot_data_window.append(test_data[i][channels])

        # for j in range(n_channels):
        #     axes[j].clear()
        #     axes[j].plot(np.array(plot_data_window)[:,j])
        #  #    axes[j].text(0.5, 0.5,"hahaahahah",horizontalalignment='center',
        #  # verticalalignment='center')
        #     # axes[j].xaxis.set_major_locator(MultipleLocator(500))
        #     # axes[j].xaxis.set_minor_locator(MultipleLocator(25))
        #     axes[j].tick_params(which='both', width=2)
        #     axes[j].tick_params(which='major', length=7)
        #     axes[j].tick_params(which='minor', length=4, color='r')
        if i%10 ==0:
            axes.clear()
            axes.plot(np.array(plot_data_window)[:,0])
            axes.tick_params(which='both', width=2)
            axes.tick_params(which='major', length=7)
            axes.tick_params(which='minor', length=4, color='r')
            plt.tight_layout()
            plt.title('test')
            plt.pause(0.01)