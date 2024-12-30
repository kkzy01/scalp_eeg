## function that takes in current recording data, and use sliding windows to train the model iteratively

import numpy as np
import torch
import math

def sliding_window(iteration, x, y, win_len, win_shift, threshold, fs, 
                   model, device, scheduler, optimizer, criterion, type):
    '''
    :param x, y: eeg signal and label
    :param win_len, win_shift: window length and window shift in seconds
    :param type: 'train', 'val' or 'test'
    '''
    batch_size = min(x.size(0), y.size(0))
    x = x[:batch_size]
    y = y[:batch_size]
    x = x.permute(1,0,2) #now shape [length, batch_size, channel]
    y = y.permute(1,0) #now shape [length, batch]

    model.init_state(device, batch_size)
    shift_num = math.ceil((x.shape[0]-win_len*fs)/float(win_shift*fs))+1
    pred_y = []
    iter_loss = []
    for i in range(0,shift_num):
        slice_start = i*win_shift*fs
        slice_end = slice_start+win_len*fs
        temp_x = x[slice_start:slice_end].permute(1,0,2)
        temp_y = y[slice_start:slice_end]
        seiz_count = torch.count_nonzero(temp_y,dim=0)
        #print(temp_y)
        target,_ = torch.max(temp_y,0)
        #print(target)
        target[seiz_count<threshold*win_len*fs] = 0
        target = torch.round(target).type(torch.LongTensor)
        target = target.to('cuda')
        if type == 'train':
            optimizer.zero_grad()
            logits = model(temp_x).float()
            logits.to(device)
            loss = criterion(logits,target)
            iter_loss.append(torch.mean(loss).item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step(iteration)
        elif type == 'val':
            with torch.no_grad():
                logits = model(temp_x).float()
                logits.to(device)
                loss = criterion(logits,target)
                iter_loss.append(torch.mean(loss).item())
        elif type == 'test':
            with torch.no_grad():
                logits = model(temp_x).float()
                #logits.to(device)
                #print(logits)
                loss = criterion(logits,target)
                iter_loss.append(torch.mean(loss).item())
                proba = torch.nn.functional.softmax(logits, dim=1)
                if proba[0,1] > 0.5:
                    pred_y.append(1)
                else:
                    pred_y.append(0)
    if type == 'train' or type == 'val':
        return model, np.mean(iter_loss)
    else:
        return pred_y
        
