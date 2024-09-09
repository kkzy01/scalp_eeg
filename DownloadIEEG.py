import numpy as np
import pandas as pd
from ieeg.auth import Session
import re
import matplotlib.pyplot as plt
# from keras.models import model_from_json
# from tensorflow.keras.optimizers import Adam
import os
# import pickle
import gc
from numbers import Number
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from scipy.signal import butter, sosfilt, resample

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