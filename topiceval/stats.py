"""
This module implements some entropy related calculations.
"""

from __future__ import print_function
from __future__ import division

from scipy import stats
from scipy.spatial import distance
import numpy as np


def jaccard_similarity(set1, set2):
    return len(set1 & set2)/len(set1 | set2)


def cosine_similarity(vec1, vec2):
    return 1 - distance.cosine(vec1, vec2)


def hellinger_distance(vec1, vec2):
    return np.linalg.norm(np.sqrt(vec1) - np.sqrt(vec2)) / np.sqrt(2)


def get_entropy(distribution):
    return stats.entropy(distribution)


def get_entropies(distributions):
    return [get_entropy(distribution) for distribution in distributions]


def get_entropy_stats(distributions):
    entropies = get_entropies(distributions)
    return np.mean(entropies), np.mean(entropies)


def almost_equal(a, b):
    return round(a - b, 7) == 0


def smooth_distribution(distribution):
    epsilon = (1/len(distribution))/100
    nonzeros = len(distribution[distribution != 0.])
    zeros = len(distribution) - nonzeros
    epsilon_frac = epsilon*zeros/nonzeros
    smoothed_distribution = []
    for elem in distribution:
        if elem != 0.:
            smoothed_distribution.append(elem - epsilon_frac)
        else:
            smoothed_distribution.append(epsilon)
    assert (almost_equal(sum(smoothed_distribution), sum(distribution))), "Discrepancy in smoothing of distribution"
    return smoothed_distribution


def build_entropy_matrix(distributions):
    smooth_distributions = [smooth_distribution(distribution) for distribution in distributions]
    dimension = len(smooth_distributions)
    entropy_matrix = np.zeros((dimension, dimension))
    for topic1 in range(dimension):
        entropy_matrix[topic1, :] = np.array([stats.entropy(smooth_distributions[topic1], smooth_distributions[topic2])
                                              for topic2 in range(dimension)])
        for topic2 in range(dimension):
            assert (np.isfinite(entropy_matrix[topic1, topic2])), "nan values in entropy matrix"
    return entropy_matrix
