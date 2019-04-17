"""Utilities for initial values."""

import numpy as np
# import torch

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MiniBatchKMeans


def calc_median_of_pairwise_distance(
        dataloader, key=None, num_minibatches=None):
    """Calculate median of pairwise Eucilidian distances.

    Calculate median of each minibatch.
    Parameters:
        key: function
            Item getter function for dataloader, which returns
            2-d numpy array.
        num_minibatches: int
            The number of minibatches to compute minibatch-
            wise medians.
            If not given, all minibatches are used.
    """
    median_list = list()

    for i, data in enumerate(dataloader):
        if num_minibatches is not None and i > num_minibatches - 1:
            break

        if key is None:
            value = data
        else:
            value = key(data)

        D = euclidean_distances(value)
        batch_median = np.median(D[np.tril_indices(len(D), -1)])

        median_list.append(batch_median)

    return np.median(median_list)


def calc_centroid(num_centroids, dataloader, key=None,
                  num_kmeans_iters=1):
    """Calculate centroid of input features.

    Perform minibatch K-means and obtain centroids
    Parameters:
        num_centroids: int
        key: function
            Item getter function for dataloader, which returns
            2-d numpy array.
        num_kmeans_iters: int
            The number of iterations for minibatch K-means.
    """
    kmeans = MiniBatchKMeans(n_clusters=num_centroids)

    for i in range(num_kmeans_iters):
        for j, data in enumerate(dataloader):
            if key is None:
                value = data
            else:
                value = key(data)

            kmeans.partial_fit(value)

    return kmeans.cluster_centers_
