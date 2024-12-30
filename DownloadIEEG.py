import numpy as np
import pandas as pd
from ieeg.auth import Session
import os
from numbers import Number
import pyedflib
import warnings
import pickle
warnings.filterwarnings("ignore")

def get_iEEG_data(
    username,
    password_bin_file,
    iEEG_filename,
    start_time_usec,
    stop_time_usec,
    select_electrodes=None,
):

    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec
    with open(password_bin_file, "r") as f:
        s = Session(username, f.read())
    ds = s.open_dataset(iEEG_filename)
    all_channel_labels = ds.get_channel_labels()

    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    try:
        data = ds.get_data(start_time_usec, duration, channel_ids)
    except:
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6
        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = ds.get_data(clip_start, clip_size, channel_ids)
            else:
                data = np.concatenate(
                    ([data, ds.get_data(clip_start, clip_size, channel_ids)]), axis=0
                )
            clip_start = clip_start + clip_size
        data = np.concatenate(
            ([data, ds.get_data(clip_start, stop_time_usec - clip_start, channel_ids)]),
            axis=0,
        )

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate
    return df, fs

def save_data(save_path1, save_path2, clips,idx,select_channels):
    # save path 1: seizure end
    # save path 2: normal
    file_name = clips['ieeg_file_name'].iloc[idx]
    temp = clips.iloc[idx]
    clip_start,clip_end = temp['clip_start'],temp['clip_end']
    file_name = temp['ieeg_file_name']
    seizure_on = temp['onset_time']
    if temp['offset_time_1'] > seizure_on:
        seizure_off = temp['offset_time_1']
    elif temp['offset_time_2'] > seizure_on:
        seizure_off = temp['offset_time_2']
    else: seizure_off = temp['offset_time_3']
    if clip_end<seizure_off:
        clip_end = seizure_off
    id = temp['admission_id']+'_{}'.format(idx+1)
    edf_file_name = f"{id}.edf"
    # seizure end
    df1, fs1 = get_iEEG_data(username = 'joie1',
            password_bin_file = 'ieeglogin.bin',
            iEEG_filename = file_name,
            start_time_usec = clip_start*1e6,
            stop_time_usec = seizure_off*1e6,
            select_electrodes = select_channels)
    labels1 = [0]*int((seizure_on-clip_start)*fs1)+[1]*int((seizure_off-seizure_on)*fs1)
    if len(labels1)<df1.shape[0]:
        labels1 = labels1+[1]*(df1.shape[0]-len(labels1))
    edf_file_path1 = os.path.join(save_path1, edf_file_name)
    # normal
    df2, fs2 = get_iEEG_data(username = 'joie1',
            password_bin_file = 'ieeglogin.bin',
            iEEG_filename = file_name,
            start_time_usec = clip_start*1e6,
            stop_time_usec = clip_end*1e6,
            select_electrodes = select_channels)
    labels2 = [0]*int((seizure_on-clip_start)*fs2)+[1]*int((seizure_off-seizure_on)*fs2)+[0]*int((clip_end-seizure_off)*fs2)
    if len(labels2)<df2.shape[0]:
        labels2 = labels2+[0]*(df2.shape[0]-len(labels2))
    edf_file_path2 = os.path.join(save_path2, edf_file_name)
    if len(labels1) != df1.shape[0]:
        print(f'warning: {file_name} has wrong labeling!!')
    else:
        # Save the EDF file
        with pyedflib.EdfWriter(edf_file_path1, len(select_channels), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
            channel_info = [{'label': ch, 'dimension': 'uV', 'sample_rate': fs1, 'physical_min': np.min(df1), 'physical_max': np.max(df1), 'digital_min': -32768, 'digital_max': 32767} for ch in select_channels]
            f.setSignalHeaders(channel_info)
            f.writeSamples(df1.to_numpy().T)
        with pyedflib.EdfWriter(edf_file_path2, len(select_channels), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
            channel_info = [{'label': ch, 'dimension': 'uV', 'sample_rate': fs2, 'physical_min': np.min(df2), 'physical_max': np.max(df2), 'digital_min': -32768, 'digital_max': 32767} for ch in select_channels]
            f.setSignalHeaders(channel_info)
            f.writeSamples(df2.to_numpy().T)
        print(f"Saved EDF file: {edf_file_name}")
    return labels1, labels2, id

data_dir = '/mnt/sauce/littlab/users/kuangzy/scalp_eeg/seizure_annotations_UEO_revised.csv'
save_path1 = '/mnt/sauce/littlab/users/kuangzy/scalp_eeg/edf_files_seizure_end'
save_path2 = '/mnt/sauce/littlab/users/kuangzy/scalp_eeg/edf_files_normal'
seizure_data = pd.read_csv(data_dir)
select_channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']

patient = list(set(seizure_data['admission_id']))
seizure_num = []
for i in patient:
    seizure_num.append(list(seizure_data['admission_id']).count(i))

patient_clip_num = {
    'admission_id': patient,
    'number of clips': seizure_num
}
labels1 = pd.DataFrame(columns=['id', 'labels'])
labels2 = pd.DataFrame(columns=['id', 'labels'])
j = 0
for i in range(len(patient)):
    clips = seizure_data[seizure_data['admission_id'] == patient[i]]
    for idx in range(seizure_num[i]):
        label1, label2, id = save_data(save_path1, save_path2,clips=clips,idx=idx,select_channels=select_channels)
        labels1.loc[j] = [id, label1]
        labels2.loc[j] = [id, label2]
        j+=1

with open('eeg_label_seizure_end.pkl', 'wb') as file: 
    pickle.dump(labels1, file)
with open('eeg_label_normal', 'wb') as file:
    pickle.dump(labels2, file)

print('Save Label File')