import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from get_dataset import EEG_Dataset, eeg_binary_collate_fn
from Res_LSTM import CNN2D_LSTM_V8_4
import os
import pickle
import random
from get_trainer import sliding_window
from logger import TrainingLogger, ModelSaver

# logger setting
logger = TrainingLogger(file_name="/mnt/sauce/littlab/users/kuangzy/scalp_eeg/onset_detection/training_log.csv")
model_saver = ModelSaver(save_dir="/mnt/sauce/littlab/users/kuangzy/scalp_eeg/onset_detection/checkpoints", best_model_name="best_model.pth")
best_val_loss = float("inf")

# prepare data
print('prepare data...')
folder_path = "/mnt/sauce/littlab/users/kuangzy/scalp_eeg/edf_files_seizure_end"
edf_files = []
for root, dirs, files in os.walk(folder_path):
    for f in files:
        edf_files.append(os.path.join(root, f))
with open('/mnt/sauce/littlab/users/kuangzy/scalp_eeg/eeg_label_seizure_end.pkl', 'rb') as file: 
    labels = pickle.load(file)
print('Total number of EEG data with seizures:', len(edf_files))
edf_train = random.sample(edf_files, int(len(edf_files)*0.7))
other = [f for f in edf_files if f not in edf_train]
edf_val = random.sample(other, int(len(other)*0.5))
edf_test = [f for f in other if f not in edf_val]
with open('test_edf.pkl', 'wb') as file: 
    pickle.dump(edf_test, file)
print('Number of Training data:', len(edf_train))
print('Number of Validation data:', len(edf_val))
print('Number of Testing data:', len(edf_test))

train_dataset = EEG_Dataset(edf_train, labels,lowcut=3,highcut=15)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory= True, collate_fn=eeg_binary_collate_fn, num_workers= 8)

val_dataset = EEG_Dataset(edf_val, labels,lowcut=3,highcut=15)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory= True, collate_fn=eeg_binary_collate_fn, num_workers= 8)

# training params
if torch.cuda.is_available():
    device = torch.device('cuda')
batch_size = 32
epoch = 30
lr_scheduler = 'CosineAnnealing'

model = CNN2D_LSTM_V8_4(device=device, batch_size=batch_size,num_layers=2)
model.to(device)
criterion = nn.CrossEntropyLoss(reduction = 'mean')
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
one_epoch_iter_num = len(train_dataloader)
iteration_num = epoch * one_epoch_iter_num

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)



start = time.time()
pbar = tqdm(total=iteration_num, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
iteration = 0
print(' ')
print('start training...')

for epoch in range(1,epoch+1):
    train_loss =[]
    val_loss = []
    print(' ')
    print(f'--------------------------EPOCH {epoch}----------------------------------')
    for train_batch in train_dataloader:
        model.train()
        pbar.update(1)
        train_x, train_y, ids = train_batch
        train_x, train_y = train_x.to(device), train_y.to(device)
        iteration += 1
        model, iter_loss = sliding_window(iteration, train_x, train_y, 4, 1, 0.75, 256, 
                                          model, device, scheduler, optimizer, criterion, type='train')
        print(f'loss of iteration {iteration} : {np.mean(iter_loss)}')
        train_loss.append(np.mean(iter_loss))
        
    scheduler.step()
    for val_batch in val_dataloader:
        model.eval()
        val_x, val_y, ids = val_batch
        val_x, val_y = val_x.to(device), val_y.to(device)
        model, val_iter_loss = sliding_window(iteration, val_x, val_y, 4, 1, 0.75, 256, 
                                          model, device, scheduler, optimizer, criterion, type='val')
        val_loss.append(np.mean(val_iter_loss))
    
    logger.log(epoch, np.mean(train_loss), np.mean(val_loss))
    is_best = np.mean(val_loss) < best_val_loss
    if is_best:
        best_val_loss = np.mean(val_loss)
    model_saver.save_checkpoint(model, epoch, is_best)

    print(f'---------------------SUMMARY OF EPOCH {epoch}-----------------------------')
    print(f'average loss in training: {np.mean(train_loss)}')
    print(f'average loss in validation: {np.mean(val_loss)}')
    print('---------------------------------------------------------------------------')

logger.print_summary()
