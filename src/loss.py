import numpy as np


def loss(arr1, arr2):
    d1 = distance_values(arr1, arr2) 
    return d1

def loss_pop(arr1, arr2):
    d1 = distance_values_pop(arr1, arr2)
    return d1

def distance_values(arr1, arr2):
    return np.absolute(arr1 - arr2)

def distance_values_pop(arr1, arr2):
    return np.absolute(arr1 - arr2)