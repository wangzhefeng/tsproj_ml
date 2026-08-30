# -*- coding: utf-8 -*-

# ***************************************************
# * File        : statistical_features.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-14
# * Version     : 1.0.040517
# * Description : 统计特征（实验性，未接入主流程）
# ***************************************************


__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

def extract_statistical_features(y):
     def calculate_entropy(y):
         # Ensure y is positive and normalized
         y = np.abs(y)
         y_sum = np.sum(y)
 
         # Avoid division by zero
         if y_sum == 0:
             return 0
 
         # Normalize the signal
         p = y / y_sum
 
         # Calculate entropy
         entropy_value = -np.sum(p * np.log2(p + 1e-12))  # Add a small value to avoid log(0)
 
         return entropy_value
     # Remove the mean to center the data
     y_centered = y - np.mean(y)
     y = y+np.max(np.abs(y))*10**-4
 
     # Statistical features
     mean_value = np.mean(y)
     variance_value = np.var(y)
     skewness_value = skew(y)
     kurtosis_value = kurtosis(y)
     autocorrelation_value = np.correlate(y_centered, y_centered, mode='full')[len(y) - 1] / len(y)
     quantiles = np.percentile(y, [25, 50, 75])
     entropy_value = calculate_entropy(y)  # Add a small value to avoid log(0)
 
     # Create a dictionary of features
     statistical_dict = {
         'Mean': mean_value,
         'Variance': variance_value, 
         'Skewness': skewness_value, 
         'Kurtosis': kurtosis_value, 
         'Autocorrelation': autocorrelation_value, 
         'Quantile_25': quantiles[0], 
         'Quantile_50': quantiles[1], 
         'Quantile_75': quantiles[2], 
         'Entropy': entropy_value
    }
 
     # Convert to DataFrame for easy visualization and manipulation
     statistical_data = pd.DataFrame([statistical_dict])
 
     return statistical_dict, statistical_data

def extract_peaks_and_valleys(y, N=10):
    """
    https://docs.scipy.org/doc/scipy/reference/signal.html#peak-finding

    Args:
        y (_type_): _description_
        N (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    # Find peaks and valleys
    peaks, _ = find_peaks(y)
    valleys, _ = find_peaks(-y)

    # Combine peaks and valleys
    all_extrema = np.concatenate((peaks, valleys))
    all_values = np.concatenate((y[peaks], -y[valleys]))

    # Sort by absolute amplitude (largest first)
    sorted_indices = np.argsort(-np.abs(all_values))
    sorted_extrema = all_extrema[sorted_indices]
    sorted_values = all_values[sorted_indices]

    # Select the top N extrema
    top_extrema = sorted_extrema[:N]
    top_values = sorted_values[:N]

    # Pad with zeros if fewer than N extrema are found
    if len(top_extrema) < N:
        padding = 10 - len(top_extrema)
        top_extrema = np.pad(top_extrema, (0, padding), 'constant', constant_values=0)
        top_values = np.pad(top_values, (0, padding), 'constant', constant_values=0)

    # Prepare the features
    features = []
    for i in range(N):
        features.append(top_values[i])
        features.append(top_extrema[i])

    # Create a dictionary of features
    feature_dict = {f'peak_{i+1}': features[2*i] for i in range(N)}
    feature_dict.update({f'loc_{i+1}': features[2*i+1] for i in range(N)})

    return feature_dict, pd.DataFrame([feature_dict])
