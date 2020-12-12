import numpy as np
import scipy.io as sio
import mne
import h5py
import math
from pymatreader import read_mat
from feature_extract import extract_ecg_features, extract_abd_features


def load_signal(path):
    """
    :param path:
    :return:
    """
    try:
        # for old version of matlab file
        mat = sio.loadmat(path)
        Fs = 200  # sampling rate is not available in the old version, we assume it's 200Hz here

        channel_names = mat['hdr']['signal_labels'][0]
        # find ABD and ECG channels
        ecg_channel_id = [i for i in range(len(channel_names)) if 'ECG' in channel_names[i][0].upper() or 'EKG' in channel_names[i][0].upper()][0]
        abd_channel_id = [i for i in range(len(channel_names)) if 'ABD' in channel_names[i][0].upper()][0]

        # get actual signal
        ecg_signal = mat['s'][ecg_channel_id]
        abd_signal = mat['s'][abd_channel_id]

    except Exception as ex:
        # for new version of matlab file
        with h5py.File(path, 'r') as f:
            # read sampling rate
            Fs = f['recording']['samplingrate'][()][0,0]

            # read hdr, which is channel names
            refs = f['hdr']['signal_labels'][()]
            channel_names = []
            for ii in range(len(refs)):
                this_hdr = f[refs[ii, 0]][()].flatten()
                this_hdr = ''.join([chr(this_hdr[i]) for i in range(len(this_hdr))])
                channel_names.append(this_hdr)

            # find ABD and ECG channels
            ecg_channel_id = [i for i in range(len(channel_names)) if 'ECG' in channel_names[i].upper()][0]
            abd_channel_id = [i for i in range(len(channel_names)) if 'ABD' in channel_names[i].upper()][0]

            # get actual signal
            ecg_signal = f['s'][:,ecg_channel_id]
            abd_signal = f['s'][:,abd_channel_id]

    return ecg_signal, abd_signal, Fs


def load_label(path):
    """
    :param path:
    :return
    """
    try:
        # for old version of matlab file
        mat = sio.loadmat(path)
        sleep_stage = mat['stage']
        # sometimes sleep_stage is a 2D array where one dimension is 1
        # in that case, we can flatten it to 1D array
        sleep_stage = sleep_stage.flatten()

    except Exception as ex:
        # for new version of matlab file

        with h5py.File(path, 'r') as f:
            # read sampling rate
            sleep_stage = f['stage'][()]
        sleep_stage = sleep_stage.flatten()

    return sleep_stage



# load signal and label
#signal_path = r'Z:\Datasets_ConvertedData\sleeplab\natus_data\Jones~ Terrenc_a07dec2b-9f27-4b59-a376-2c673b47b9ec\Signal_Jones~ Terrenc_a07dec2b-9f27-4b59-a376-2c673b47b9ec.mat'
#label_path = r'Z:\Datasets_ConvertedData\sleeplab\natus_data\Jones~ Terrenc_a07dec2b-9f27-4b59-a376-2c673b47b9ec\Labels_Jones~ Terrenc_a07dec2b-9f27-4b59-a376-2c673b47b9ec.mat'
#signal_path = r'C:\Users\xiaoy\Desktop\Signal_Jones~ Terrenc_a07dec2b-9f27-4b59-a376-2c673b47b9ec.mat'
#label_path = r'C:\Users\xiaoy\Desktop\Labels_Jones~ Terrenc_a07dec2b-9f27-4b59-a376-2c673b47b9ec.mat'

signal_path = r'C:\Users\xiaoy\Desktop\Signal_TwinData10_392.mat'
label_path = r'C:\Users\xiaoy\Desktop\Labels_TwinData10_392.mat'

ecg_signal, abd_signal, Fs = load_signal(signal_path)
sleep_stage = load_label(label_path)
T = len(ecg_signal)
# ecg_signal.shape = (T,)
# abd_signal.shape = (T,)
# sleep_stage.shape = (T,)

# preprocessing: notch filtering
notch_freq = 60  # [Hz]
ecg_signal = mne.filter.notch_filter(ecg_signal, Fs, notch_freq)
abd_signal = mne.filter.notch_filter(abd_signal, Fs, notch_freq)

# preprocessing: band-pass filtering
lfreq = 0  # [Hz]
hfreq = 30  # [Hz]
ecg_signal = mne.filter.filter_data(ecg_signal, Fs, lfreq, hfreq)
abd_signal = mne.filter.filter_data(abd_signal, Fs, lfreq, hfreq)

# segment into 270s epochs with 30s step
epoch_time = 270 #[s]
epoch_step = 30  #[s]
ecg_epochs = []
abd_epochs = []
end = 0
i = 0
while end<T:
    start = 30*i  #[s]
    end = 270+30*i  #[s]

    epoch = ecg_signal[int(start*Fs):int(end*Fs)]   # ecg_signal.shape=(T,)
    # epoch.shape = (Tepoch,)   # define Tepoch = int(end*Fs)-int(start*Fs)
    ecg_epochs.append(epoch)
    
    epoch = abd_signal[int(start*Fs):int(end*Fs)]
    abd_epochs.append(epoch)
    i = i+1
    
# what is ecg_epochs?
# it is a list with Nepoch elements, where each element is one epoch, which is a numpy array

# convert to 2d numpy array
ecg_epochs = np.array(ecg_epochs)
# ecg_epochs.shape = (#epoch, Tepoch)
abd_epochs = np.array(abd_epochs)
Nepoch = len(ecg_epochs)

# compute feature from every epoch
ecg_features = []
abd_features = []

for i in range(Nepoch):
    ecg_feature, ecg_feature_names = extract_ecg_features(ecg_epochs[i], Fs)
    abd_feature, abd_feature_names = extract_abd_features(abd_epochs[i], Fs)
    ecg_features.append(ecg_feature)
    abd_features.append(abd_feature)

sio.savemat('features.mat',
            {'ecg_features': ecg_features,
             'abd_features': abd_features,
             'ecg_feature_names':ecg_feature_names,
             'abd_feature_names':abd_feature_names,})
