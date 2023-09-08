import h5py
import numpy as np
with h5py.File('/root/dacon/LinearityIQA/data/KonIQ-10kinfo.mat', 'r') as f:
    a = (f['subjective_scores_old'][0])
    print(a)
    print(len(a))