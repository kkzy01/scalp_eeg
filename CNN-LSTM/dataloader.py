from torch.utils.data import Dataset
import numpy as np
import torch

"""
 All of the examples are for the case of non-overlapping sliding windows, for the sake of simplicity
 need tweak implementation slightly for overlapping sliding windows
 *** remember to vectorize the function, avoid for loop if possible in __getitem__() ***
"""


class SwavDataset(Dataset):
    def __init__(self, data, data_augmentation=False):

        self.data = data  # list(recording1, recording2, .....)
        self.data_augmentation = data_augmentation
        self.transform = ...  # transform function to spectrogram
        self.sampling_rate = 256
        self.sliding_window_size = 256

        self.per_recording_length = int(np.floor([len(dt) / self.sliding_window_size for dt in self.data]))
        self.cumulative_recording_lengths = np.cumsum(self.per_recording_length)

        self.compute_index_look_up_table()

        self.augmentation = ...

    def __len__(self):
        return np.sum(
            self.per_recording_length)  # get the sum of maximum number of sliding windows from each recoding, for non-overlapping sliding windows

    def compute_index_look_up_table(self):
        # helper function to get the correct sliding window from the index
        # general idea, given some global index, find the correct recording from a list of recordings (your input data),
        # and find the correct slice from the correct recording
        index_look_up_table = {}
        for i in np.sum(self.per_recording_length):
            # some implementation_here
            ...
            index_look_up_table[i] = [correct_recording_index_in_self_data, within_recording_start_index,
                                      within_recording_stop_index]

        self.index_look_up_table = index_look_up_table

    def slice_helper(self, data, index):
        recording_idx, start_index, stop_index = self.index_look_up_table[index]
        recording = data[recording_idx]
        return recording[start_index:stop_index]

    def __getitem__(self, idx):

        data_clip = self.slice_helper(self.data, idx)

        if self.data_augmentation:
            data_clip = self.augmentation(data_clip)

        data_clip = self.transform(data_clip)

        # remember to convert to rgb image, resize to correct shape and to torch tensor

        return data_clip  # output_shape : features * sliding_window_size for example (190*256)


class LSTMDataset(Dataset):
    def __init__(self, data, label, data_augmentation=False):
        self.data = data  # list(recording1, recording2, .....)
        self.label = label
        self.data_augmentation = data_augmentation
        self.transform = ...  # transform function to spectrogram\
        self.label_transform = ...  # transform function for labels
        self.sampling_rate = 256
        self.LSTM_max_length = 200
        self.sliding_window_size = 256
        self.segment_length = self.sliding_window_size * self.LSTM_max_length

        self.per_recording_length = np.ceil(
            int(np.floor([len(dt) / self.sliding_window_size for dt in self.data])) / self.LSTM_max_length)
        # because before, we are throwing away sliding windows that are less than 1 second, but now we want to PRESERVE short segments that is less than LSTM_max_length, again for non-overlapping short segments
        self.cumulative_recording_lengths = np.cumsum(self.per_recording_length)

        self.compute_index_look_up_table()

        self.augmentation = ...  # data augmentation function

    def __len__(self):
        return np.sum(self.per_recording_length)

    def compute_index_look_up_table(self):
        # similar as above, but this time find the correct segment
        # need to consider the less than edge cases of LSTM_max_length
        ...
        self.index_look_up_table = index_look_up_table

    def slice_helper(self, data, index):
        recording_idx, start_index, stop_index = self.index_look_up_table[index]
        recording = data[recording_idx]
        segment = recording[start_index:stop_index]
        return segment.reshape(...)  # return size: length_of_segment * features * sliding_window_size

    def __getitem__(self, idx):
        data_segment = self.slice_helper(self.data, idx)

        if self.data_augmentation:
            data_segment = self.augmentation(data_segment)

        data_segment = self.transform(data_segment)
        label_segment = self.label_transform(self.label)


        # remember to convert to rgb image, resize to correct shape and to torch tensor

        # output_shape: data_segment =  length_of_segment * features * sliding_window_size
        # output_shape: label_segment =  length_of_segment

        return data_segment, label_segment.to(dtype=torch.long)


# to be called in dataloader, very important to handle variable length case
def collate_fn(batch):
    info = list(zip(*batch))
    data = info[0]
    label = info[1]
    sequence_len = [dt.size(0) for dt in data]

    return torch.concat(data), torch.concat(label), sequence_len


# for dataloader
train_dataset = LSTMDataset(train_data, train_label, data_augmentation=False)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=16 # critical to performance, but each worker need its own RAM, so the memory usage will be 16 times the num_workers = 1S
)
