import math
import numpy as np


def get_distance(sample1, sample2, dist_type='L2'):
    assert len(sample1) == len(sample2)
    num_of_features = len(sample1)
    dist = 0
    if dist_type == 'L2':
        for i in range(num_of_features):
            dist += (sample1[i] - sample2[i]) ** 2
        dist = math.sqrt(dist)

    return dist


def find_closest_neighbors(input, samples, k):
    distances = list(np.zeros(len(samples)))
    for i in range(len(samples)):
        distances[i] = (get_distance(input, samples[i]), i)
    sorted(distances, key=lambda x: x[0])
    closest_nn = []
    for distance in distances[:k]:
        closest_nn.append(samples[distance[1]])

    return closest_nn
