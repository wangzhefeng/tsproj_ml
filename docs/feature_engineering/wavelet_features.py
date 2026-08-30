# -*- coding: utf-8 -*-

# ***************************************************
# * File        : wavelet_features.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-14
# * Version     : 1.0.040516
# * Description : 小波特征（实验性，未接入主流程）
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
import pywt

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

def extract_wavelet_features(y, wavelet='db4', level=3, num_features=5):
    # Remove the mean
    y = y - np.mean(y)

    # Perform the Discrete Wavelet Transform
    coeffs = pywt.wavedec(y, wavelet, level=level)

    # Flatten the list of coefficients into a single array
    coeffs_flat = np.hstack(coeffs)

    # Get the absolute values of the coefficients
    coeffs_abs = np.abs(coeffs_flat)

    # Find the indices of the largest coefficients
    largest_coeff_indices = np.flip(np.argsort(coeffs_abs))[0:num_features]

    # Extract the largest coefficients as features
    top_coeffs = coeffs_flat[largest_coeff_indices]

    # Generate feature names for the wavelet features
    feature_keys = ['Wavelet Coeff ' + str(i+1) for i in range(num_features)]

    # Create a dictionary for the features
    wavelet_dict = {
        feature_keys[i]:top_coeffs[i] 
        for i in range(num_features)
    }

    # Create a DataFrame for the features
    wavelet_data = pd.DataFrame(top_coeffs).T
    wavelet_data.columns = feature_keys

    return wavelet_dict, wavelet_data
