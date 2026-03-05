import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def pattern_match(signal, template,method):
    '''
    
    '''
    
    if len(signal) <= len(template):
       raise ValueError('len(signal) must be greater the len(template)')
    
    window_size = len(template)
    similarity_list =[]
    for i in range(len(signal)- window_size + 1): # sliding window
        similarity_list.append(calculate_similarity(signal[i:i+window_size], template,method))
    
    return (np.argmin(similarity_list))
       

def calculate_similarity(time_series_A, time_series_B, method):

    if method =='euclidean':
        return (euclidean_distance(time_series_A, time_series_B))
    elif method == 'dtw':
        return (dtw_distance(time_series_A, time_series_B))
    elif method == 'cosine':
        return (cosine_similarity(time_series_A, time_series_B))
    else:
        raise NotImplementedError ('Not implemented' )
      
        

def euclidean_distance(time_series_A, time_series_B):
    return np.sqrt(np.sum((time_series_A - time_series_B)**2))

# Computing the DTW distance between the two time-series data sets
def dtw_distance(time_series_A, time_series_B):
    n = len(time_series_A)
    m = len(time_series_B)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Computing the cost by using the above mathematical formula &
            # Finding the absolute difference between two values
            cost = abs(time_series_A[i-1] - time_series_B[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                          dtw_matrix[i, j-1],
                                          dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m]

def cosine_similarity(A, B):
    # The time-series data sets should be normalized.
    A_norm = (A - np.mean(A)) / np.std(A)
    B_norm = (B - np.mean(B)) / np.std(B)
 
    # Determining the dot product of the normalized time series data sets.
    dot_product = np.dot(A_norm, B_norm)
 
    # Determining the Euclidean norm for each normalized time-series data collection.
    norm_A = np.linalg.norm(A_norm)
    norm_B = np.linalg.norm(B_norm)
 
    # Calculate the cosine similarity of the normalized time series data 
    # using the dot product and Euclidean norms. setse-series data set
    cosine_sim = dot_product / (norm_A * norm_B)
 
    return cosine_sim