__version__ = "1.1"

from pathlib import Path
from pybaselines.whittaker import arpls
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
from .downloader import download_data
from .evaluation import *

def remove_baseline(x):
    x = np.array(x, dtype=float)
    try:
        return x - arpls(x)[0]
    except:
        print('Baseline removal failed')
        return None

def anormaly_imputation(sers, window_size=5, threshold=2.0):
    """
    Detect and impute anomalies in SERS spectrum.
    
    Args:
        sers (array-like): Input SERS spectrum
        window_size (int, optional): Size of local window for median calculation. Defaults to 5.
        threshold (float, optional): Threshold multiplier for standard deviation. Defaults to 2.0.
    
    Returns:
        array-like: SERS spectrum with anomalies imputed, or None if input is None
    """
    if sers is None:
        return None
    sers = np.array(sers, copy=True)
    std = np.std(sers)
    for j in range(window_size, len(sers) - window_size):
        local_median = np.median(sers[j - window_size:j + window_size + 1])
        if sers[j] > local_median + threshold * std:
            sers[j] = local_median
    return sers

def blape(signal, original_wn, target_wn, sigma=25, is_baseline_removed=False, eps=0.25, anomaly_imputation=True):
    """
    Calculate BLAPE (Baseline-removed Laplacian Peak Enhancement) for Raman spectra.
    
    Args:
        signal (array-like): Input spectrum
        original_wn (array-like): Original wavenumber values
        target_wn (array-like): Target wavenumber values for interpolation
        sigma (float, optional): Standard deviation for Gaussian smoothing. Defaults to 25.
        is_baseline_removed (bool, optional): Whether baseline is already removed. Defaults to False.
        eps (float, optional): Small value to avoid blow up. Defaults to 0.25.
        anomaly_imputation (bool, optional): Whether to apply anomaly imputation before processing. Defaults to True.
    
    Returns:
        array-like: BLAPE processed spectrum interpolated to target wavenumbers
    """
    signal = np.array(signal)
    
    if len(signal.shape) > 1 and signal.shape[0] > 1:
        results = []
        for single_signal in signal:
            if anomaly_imputation:
                single_signal = anormaly_imputation(single_signal)
            if not is_baseline_removed:
                single_signal = remove_baseline(single_signal)
            
            laplacian = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
            peaks = -np.convolve(gaussian_filter1d(single_signal, sigma), laplacian, 'same')/(gaussian_filter1d(single_signal, sigma)+eps*np.mean(np.abs(single_signal)))
            peaks[peaks<0] = 0
            peaks = peaks[10:-10]
            peaks = np.power(peaks/max(peaks) if max(peaks) > 0 else peaks, 1)
            interpolated = np.interp(target_wn, original_wn[10:-10], peaks)
            results.append(interpolated)
        return np.array(results)
    else:
        if len(signal.shape) > 1:
            signal = signal.flatten()
        
        if anomaly_imputation:
            signal = anormaly_imputation(signal)
        if not is_baseline_removed:
            signal = remove_baseline(signal)
            
        laplacian = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
        peaks = -np.convolve(gaussian_filter1d(signal, sigma), laplacian, 'same')/(gaussian_filter1d(signal, sigma)+eps*np.mean(np.abs(signal)))
        peaks[peaks<0] = 0
        peaks = peaks[10:-10]
        peaks = np.power(peaks/max(peaks) if max(peaks) > 0 else peaks, 1)
        interpolated = np.interp(target_wn, original_wn[10:-10], peaks)
        return interpolated

def read_data(path='data'):
    raw_path = Path(path) / 'raw'
    baseline_path = Path(path) / 'baseline_removed'
    
    files_raw = []
    if raw_path.exists():
        files_raw = list(raw_path.glob('*.csv'))
        print(f"Found {len(files_raw)} raw SERS files")
    else:
        print(f"No raw SERS files found in {raw_path}")
    
    files_baseline_removed = []
    if baseline_path.exists():
        files_baseline_removed = list(baseline_path.glob('*.csv'))
        print(f"Found {len(files_baseline_removed)} baseline-removed files")
    else:
        print(f"No baseline-removed files found in {baseline_path}")
        
    codes = set([file.stem for file in files_raw] + [file.stem for file in files_baseline_removed])
    data = {code: {'code': code} for code in codes}
    
    total_samples = 0
    for file in files_raw:
        code = file.stem
        csv = pd.read_csv(file)
        data[code]['signal'] = csv.iloc[:, 1:].values.T
        data[code]['wavenumbers'] = csv.iloc[:, 0].values
        total_samples += data[code]['signal'].shape[0]
    print(f"Total raw samples: {total_samples}")

    total_samples = 0
    for file in files_baseline_removed:
        code = file.stem
        csv = pd.read_csv(file)
        data[code]['baseline_removed'] = csv.iloc[:, 1:].values.T
        data[code]['wavenumbers'] = csv.iloc[:, 0].values
        total_samples += data[code]['baseline_removed'].shape[0]
    print(f"Total baseline-removed samples: {total_samples}")
        
    return data

def get_common_wavenumber_range(data, num_points=1000):
    wn_from = max([d['wavenumbers'].min() for d in data.values()])
    wn_to = min([d['wavenumbers'].max() for d in data.values()])
    target_wn = np.linspace(wn_from, wn_to, num_points)
    return target_wn

if __name__ == "__main__":
    download_data()
