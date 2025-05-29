__version__ = "0.3.2"

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

def blape(signal, original_wn, target_wn, sigma=25, is_baseline_removed=False, eps=0.25):
    """
    Calculate BLAPE (Baseline-removed Laplacian Peak Enhancement) for Raman spectra.
    
    Args:
        signal (array-like): Input spectrum
        original_wn (array-like): Original wavenumber values
        target_wn (array-like): Target wavenumber values for interpolation
        sigma (float, optional): Standard deviation for Gaussian smoothing. Defaults to 25.
        is_baseline_removed (bool, optional): Whether baseline is already removed. Defaults to False.
        eps (float, optional): Small value to avoid blow up. Defaults to 0.25.
    
    Returns:
        array-like: BLAPE processed spectrum interpolated to target wavenumbers
    """
    signal = np.array(signal)
    
    if len(signal.shape) > 1 and signal.shape[0] > 1:
        # batch processing
        results = []
        for single_signal in signal:
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
        # single sample processing
        if len(signal.shape) > 1:
            signal = signal.flatten()
            
        laplacian = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
        if not is_baseline_removed:
            signal = remove_baseline(signal)
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