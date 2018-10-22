import KNN
import numpy as np


def generate_samples(sample1, sample2):
    dif = sample2 - sample1
    gap = np.random.rand()
    return sample1 + gap*dif


def oversample_data(samples, smote_proc, k):
    smote_proc = int(len(samples) * int(smote_proc/100))

    nn = []
    synthetic = []

    for i in range(len(samples)):
        subset = np.copy(samples)
        np.delete(subset, i)

        nn.append(KNN.find_closest_neighbors(samples[i], subset, k))

    while smote_proc > 0:
        rands_ind = np.random.randint(0, len(samples))
        randnn_ind = np.random.randint(0, k)
        synthetic.append(generate_samples(samples[rands_ind], nn[rands_ind][randnn_ind]))
        smote_proc -= 1

    return synthetic

#test
#print(oversample_data(np.array([[1,2,3], [3,4,5], [1,4,2], [2,1,3], [1,1,1], [2,7,9]]), 100, 3))

