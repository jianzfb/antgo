# -*- coding: UTF-8 -*-
# @Time    : 2019/1/23 3:13 PM
# @File    : kcenter_greedy.py
# @Author  : jian<jian@mltalker.com>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from .sampling_def import SamplingMethod



class kCenterGreedy(SamplingMethod):
  def __init__(self, X, metric='euclidean'):
    self.name = 'kcenter'
    self.features = self.flatten_X(X)
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.features.shape[0]

  def update_distances(self, cluster_centers, already_selected, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, already_selected, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """
    # init
    self.update_distances(None, None, only_new=False, reset_dist=True)

    new_batch = []
    for _ in range(N):
      if self.min_distances is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected

      self.update_distances([ind], already_selected, only_new=True, reset_dist=False)
      new_batch.append(ind)

    print('Maximum distance from cluster centers is %0.2f'% max(self.min_distances))
    return new_batch
