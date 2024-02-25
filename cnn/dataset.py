import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from torch import stft
import torch
from torch.utils.data import Dataset
# from torch.nn.functional import one_hot
matplotlib.use('tkagg')
rng = np.random.default_rng()

jaw_labels = [[1540, 1610],
              [2770, 2900],
              [4025, 4125],
              [5310, 5410],
              [6525, 6625],
              [7775, 7900],
              [9050, 9150],
              [10275, 10400],
              [11525, 11650],
              [12770, 12900],
              [14030, 14130]]

eye_labels = [[1540, 1725],
              [2375, 2825],
              [3625, 4200],
              [4935, 5575],
              [6150, 6655],
              [7425, 7825],
              [8575, 9150],
              [9900, 10400],
              [11175, 11650],
              [12425, 13100],
              [14900, 15000]]


class CustomDataset(Dataset):
    def __init__(self):
        jaw_data_path = 'data_jaw2.npy'
        jaw_data, jaw_label = self.get_data(jaw_data_path, jaw_labels)
        eye_data_path = 'data_eyeblink.npy'
        eye_data, eye_label = self.get_data(eye_data_path, eye_labels)
        self.specs = torch.concatenate([jaw_data, eye_data])
        self.labels =torch.concatenate([jaw_label*1, eye_label*2]).long()
        print("Done init.")

    def get_data(self, data_path, labels, channels=[0,1,2,3,4,5,6],fs=250, window_dt=0.5, clip_dt=0.1):
        window_dn = int(fs * window_dt)  # 0.5s
        clip_dn = int(fs * clip_dt)
        x = np.load(data_path, allow_pickle=True).item()
        x = torch.tensor(x['data'][:,channels]).transpose(1, 0)  # [channels, len]
        n_channels, data_len = x.shape[0], x.shape[1]
        clip_start = torch.arange(0, data_len - window_dn, clip_dn)
        clip_indices = torch.arange(window_dn).unsqueeze(0).repeat(len(clip_start), 1) + clip_start.unsqueeze(1)
        x_clips = x[:, clip_indices].permute(1, 0, 2)  # nclip, channels, window_len
        nclip = len(x_clips)
        # for x_clip in x_clips:
        x_clips = x_clips.contiguous().view(-1, window_dn)
        specs = stft(x_clips, 50, hop_length=5, return_complex=True).real
        specs = specs.view(nclip, n_channels, specs.shape[-2], specs.shape[-1]).float()
        labels_range = torch.concatenate(([torch.arange(l[0], l[1]) for l in labels]))
        labels = torch.any(torch.isin(clip_indices, labels_range), dim=-1).float()
        return specs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx]


if __name__ == "__main__":
    dataset = CustomDataset()
    x, y = dataset[0]
    print(x, y)
