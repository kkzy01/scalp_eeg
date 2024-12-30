### define dataset and dataloader (including collate fn)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyedflib
from scipy.signal import butter, filtfilt
import os
from torch.nn.utils.rnn import pad_sequence
import pickle

## transform function 
def bandpass(x, lowcut,highcut,fs):
    b, a = butter(4, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return filtfilt(b, a, x)

class EEG_Dataset(Dataset):
    def __init__(self, files, labels, lowcut, highcut, fs=256, transform = bandpass):
        """
        Initializes the EDFDataset.
        
        Args:
            file_paths (list): List of EDF file paths.
            lowcut (float): Low cutoff frequency for bandpass filter.
            highcut (float): High cutoff frequency for bandpass filter.
            fs (int): Sampling frequency of the signal.
            order (int): Order of the filter.
        """
        self.file_paths = files
        self.labels = labels
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.transform = transform

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single EDF file.
        
        Args:
            idx (int): Index of the file in the file list.
        
        Returns:
            torch.Tensor: Bandpass-filtered EEG signal.
        """
        file_path = self.file_paths[idx]
        x = []
        edf_reader = pyedflib.EdfReader(file_path)
        for chn in range(18):
            x.append(edf_reader.readSignal(chn))
        x = np.array(x) # shape of (channel, time)
        edf_reader.close()
        if self.transform:
            x = self.transform(x,self.lowcut,self.highcut,self.fs)
        id = file_path.split('/')[-1].split('.')[0]
        label = self.labels.loc[self.labels['id']==id]['labels'].item()
        
        if len(label)<x.shape[1]:
            x = x[:,:len(label)]
        x_ = x.copy()
        signal = torch.from_numpy(x_).float()
        label = torch.tensor(label,dtype=torch.long)
        return signal, label, id
    

def eeg_collate_fn(batch):
    """
    Collate function for handling variable-length signals and labels.
    
    Args:
        batch (list): A list of tuples where each tuple contains:
            - signal (torch.Tensor): EEG signals of shape (n_channels, seq_len).
            - label (torch.Tensor): Corresponding labels of shape (seq_len,).
    
    Returns:
        dict: A dictionary containing:
            - "signals": Padded signals of shape (batch_size, n_channels, max_seq_len).
            - "labels": Padded labels of shape (batch_size, max_seq_len).
            - "lengths": Original lengths of each signal in the batch.
    """
    signals, labels, id = zip(*batch)

    # Pad signals along the sequence length dimension
    padded_signals = pad_sequence([s.transpose(0, 1) for s in signals], batch_first=True)
    padded_signals = padded_signals.transpose(1, 2)  # (batch_size, n_channels, max_seq_len)

    # Pad labels to match the length of padded signals
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    # Original lengths of the signals
    lengths = torch.tensor([s.shape[1] for s in signals], dtype=torch.long)

    return {
        "signals": padded_signals,  # (batch_size, n_channels, max_seq_len)
        "labels": padded_labels,    # (batch_size, max_seq_len)
        "lengths": lengths          # (batch_size,)
    }
def eeg_binary_collate_fn(batch):
    def seq_length_(p):
        return p[0].shape[1]

    def target_length_(p):
        return len(p[1])

    pad_id = 0
    seq_lengths = torch.IntTensor([len(s[0][0]) for s in batch])
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(1)
    max_target_size = len(max_target_sample)

    batch_size = len(batch)
    eeg_type_size = len(batch[0][0])

    seqs = torch.zeros(batch_size, max_seq_size, eeg_type_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_id)
    ids = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        id = sample[2]
        ids.append(id)
        seq_length = tensor.size(1)
        tensor = tensor.permute(1,0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    return seqs, targets, ids

### test
'''folder_path = "/mnt/sauce/littlab/users/kuangzy/scalp_eeg/edf_files_seizure_end"
edf_files = []
print('prepare data...')
for root, dirs, files in os.walk(folder_path):
    for f in files:
        edf_files.append(os.path.join(root, f))

with open('/mnt/sauce/littlab/users/kuangzy/scalp_eeg/eeg_label_seizure_end.pkl', 'rb') as file: 
    labels = pickle.load(file) 

dataset = EEG_Dataset(edf_files, labels, 3, 15)

# Create the DataLoader with the custom collate function
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=eeg_binary_collate_fn)

# Iterate through the DataLoader
for idx, batch in enumerate(data_loader):
    if idx>0:
        break
    m = batch[1].permute(1,0)
    m_temp = m[0:0+256*4]
    
    seiz_count = torch.count_nonzero(m_temp, dim=0)
    
    print(seiz_count)
    n, _ = torch.max(m_temp,0)
    print(n)
    n[seiz_count<500]=0
    print(n)'''