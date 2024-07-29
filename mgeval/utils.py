# coding:utf-8
"""utils.py
Include distance calculation for evaluation metrics
"""
import sys
import os
import glob
import math
import sklearn
import numpy as np
from scipy import stats, integrate
from sklearn.decomposition import PCA


def categorize_tone(pitch_class, chord):
    CHORD_TONES = 1
    COLOR_TONES = 2/3
    APPROACH_TONES = 1/3
    OTHER_TONES = 0
    
    colored = [(2 + chord[0]) % 12, (6 + chord[0]) % 12, (9 + chord[0]) % 12, (11 + chord[0]) % 12]
    if pitch_class in chord:
        return CHORD_TONES
    if (pitch_class + chord[0]) % 12 in colored:
        return COLOR_TONES
    for interval in [1, -1]:
        if (pitch_class + chord[0] + interval) % 12 in chord + colored:
            return APPROACH_TONES
    return OTHER_TONES
    


def count_n_consecutive_values(arr, n):
    if n <= 1:
        return len(arr) - 1
    count = 0
    consecutive_count = 1  # Start count at 1 for the first element
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            consecutive_count += 1
            if consecutive_count == n:
                count += 1
        else:
            consecutive_count = 1
    return count


def find_closest_value(hist_list, x):
    hist_array = np.array(hist_list)
    differences = np.abs(hist_array - x)
    closest_index = np.argmin(differences)
    return hist_array[closest_index], closest_index


# Calculate overlap between the two PDF
def overlap_area(A, B):
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    return integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)), np.min((np.min(A), np.min(B))), np.max((np.max(A), np.max(B))), limit=100)[0]


def apply_pca(data, n_components=None):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data.reshape(-1, 1)).flatten()


# Calculate KL distance between the two PDF
def kl_dist(A, B, num_sample=1000):
    A += 1e-6
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)
    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))


def c_dist(A, B, mode='None', normalize=0):
    c_dist = np.zeros(len(B))
    for i in range(0, len(B)):
        if mode == 'None':
            c_dist[i] = np.linalg.norm(A - B[i])
        elif mode == 'EMD':
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm='l1')[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm='l1')[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            c_dist[i] = stats.wasserstein_distance(A_, B_)

        elif mode == 'KL':
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm='l1')[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm='l1')[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            B_[B_ == 0] = 0.00000001
            c_dist[i] = stats.entropy(A_, B_)
    return c_dist
