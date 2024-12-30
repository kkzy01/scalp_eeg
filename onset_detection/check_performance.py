import torch
import numpy as np
import pickle
from get_trainer import sliding_window
import torch.optim as optim
from torch.utils.data import DataLoader
from get_dataset import EEG_Dataset, eeg_binary_collate_fn
from Res_LSTM import CNN2D_LSTM_V8_4
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')

with open('/mnt/sauce/littlab/users/kuangzy/scalp_eeg/eeg_label.pkl', 'rb') as file: 
    labels = pickle.load(file)
with open('/mnt/sauce/littlab/users/kuangzy/scalp_eeg/test_edf.pkl', 'rb') as f:
    test = pickle.load(f)
edf_test = []
for f in test:
    f_ = f.replace('edf_files_seizure_end','edf_files')
    edf_test.append(f_)

test_dataset = EEG_Dataset(edf_test, labels,lowcut=3,highcut=15)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory= True, collate_fn=eeg_binary_collate_fn, num_workers= 8)

model = CNN2D_LSTM_V8_4(device=device, batch_size=1,num_layers=2)
model.to(device)
criterion = nn.CrossEntropyLoss(reduction = 'mean')
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
model.eval()
for i, test_batch in enumerate(test_dataloader):
        test_x, test_y, ids = test_batch
        #print(test_x.shape, test_y.shape)
        test_x, test_y = test_x.to(device), test_y.to(device)
        pred_y = sliding_window(0, test_x, test_y, 4, 1, 0.75, 256, model, device, scheduler, optimizer, criterion, type='test')
        print(pred_y)
        if i >0:
             break