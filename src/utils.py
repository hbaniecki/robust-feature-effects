import numpy as np


def check_early_stopping(df, epsilon=1e-4, stop_iter=10):
    #:# did not find a better result in the last `stop_iter` iterations 
    if len(df['loss']) > stop_iter:
        relative_change = (np.min(df['loss'][:-stop_iter]) - np.min(df['loss'][-stop_iter:])) / np.max(df['loss'])
        return relative_change < epsilon
    else:
        return False