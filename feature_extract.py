import mne
import neurokit2 as nk
import numpy as np
from scipy.stats import linregress
from biosppy.signals import ecg

def extract_ecg_features(ecg_epoch, fs):
    '''
    extract ecg features from \ecg_epoch\
    :param ecg_epoch:
    :return:
    '''
    #detect R peaks
    # Find peaks
    peaks, info = nk.ecg_peaks(ecg_epoch, sampling_rate=fs)
    R_peaks = info['ECG_R_Peaks']
    # Compute HRV indices
    hrv = nk.hrv(peaks, sampling_rate=fs, show=True)
    feature_names = list(hrv.columns)
    features = hrv.values.flatten()
    """
    # compute entropy
    sampen = nk.entropy_sample(ecg_epoch)

    # combine sample entropy to existing ones
    features = np.r_[features, sampen]
    feature_names.append('SampEn')
    """

    # cardiopulmonary coupling
    # ????

    return features, feature_names

"""
    R_peak = biosppy.signals.ecg.christove_segmenter(signal = ecg_epoch, sampling_rate = fs)
    # low frequency
    if sampling_rate <
    return low_frequency

    # high frequency
    if sampling_rate >
    return high_frequency

    # very low frequency
    if sampling_rate <
    return very_low_frequency
"""


def band_power(psd, freq, low, high, return_db=True):
    band_psd = psd[(freq>low) & (freq<high)]
    bp = band_psd.sum()*(freq[1]-freq[0])
    if return_db:
        bp = 10*np.log10(bp)
    return bp


def extract_abd_features(abd_epoch, fs):
    '''
    extract abd features from \abd_epoch\
    :param abd_epoch:
    :param fs:
    :return:
    '''

    # Clean signal
    cleaned = nk.rsp_clean(abd_epoch, sampling_rate=fs)

    # Extract peaks
    df, peaks_dict = nk.rsp_peaks(cleaned)
    peaks_dict = nk.rsp_fixpeaks(peaks_dict)

    rsp_rate = nk.rsp_rate(cleaned, peaks_dict, sampling_rate=fs)
    rrv = nk.rsp_rrv(rsp_rate, peaks_dict, sampling_rate=fs)

    feature_names = list(rrv.columns)
    features = rrv.values.flatten()

    peaks = peaks_dict['RSP_Peaks']
    amplitude = cleaned[peaks_dict['RSP_Peaks']]

    # compute the following features

    # group 1: features not based on spectrum (temporal features, time domain)
    # respiratory rate variability (RRV), or peak-to-peak interval standard deviation
    #rrv = np.std(np.diff(peaks)/fs)
    # envelope mean
    env_mean = amplitude.mean()
    # envelope standard deviation
    env_std = amplitude.std()

    # group 2: features based on spectrum (spectral features,  frequency domain)

    # convert signal into spectrum
    psd, freq = mne.time_frequency.psd_array_multitaper(
                        abd_epoch, fs, fmin=0, fmax=1,
                        bandwidth=0.01, normalization='full')

    """
    # high band power in db
    bp_high_db = band_power(psd, freq, 0.1, 1)
    # low band power in db
    bp_low_db = band_power(psd, freq, 0.01, 0.1)
    # high/low
    bp_high_low_ratio = bp_high_db / bp_low_db
    # spectrogram kurtosis
    spec_peakness =
    """
    # spectrum slope - 1/f
    # alpha is negative of the slope from lineer regression between log f and log psd
    #y = np.log(psd)
    #x = np.log(freq)
    #alpha = -linregress(x, y).slope

    """
    # group 3: complexity (nonlinear domain)

    # sample entropy of the waveform
    entropy = nk.entropy_sample(abd_epoch)

    # poincare plot

    # hurst exponent

    # lypnov exponent

    # detrended fluctuation analysis
    """

    features = np.r_[features, env_mean, env_std]#, alpha]
    feature_names = np.r_[feature_names, ['env_mean', 'env_std']]#, '1/f alpha'

    return features, feature_names
