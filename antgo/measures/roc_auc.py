from __future__ import unicode_literals
from __future__ import division

import numpy as np
import unittest
import json


def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    Parameters
    ----------
    x : list of numbers, numpy array
    Returns
    -------
    score : list of numbers
            The tied rank f each element in x
    """
    sorted_x = sorted(zip(x, range(len(x))))
    r = [0 for k in x]

    cur_val = sorted_x[0][0]
    offset = 0
    for i in range(len(sorted_x)):
        if sorted_x[i][0] > -float('inf'):
            cur_val = sorted_x[i][0]
            offset = i
            break

    last_rank = 0
    for i in range(len(sorted_x)):
        if sorted_x[i][0] <= -float('inf'):
            continue

        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            ioffset = i - offset
            for j in range(last_rank, ioffset):
                r[sorted_x[j + offset][1]] = float(last_rank+1+ioffset)/2.0
            last_rank = ioffset

        if i == len(sorted_x)-1:
            ioffset = i - offset
            for j in range(last_rank, ioffset+1):
                r[sorted_x[j + offset][1]] = float(last_rank+ioffset+2)/2.0
    return r


def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC) (only for binary classification task)
    References: http://cs.ru.nl/~tomh/onderwijs/dm/dm_files/roc_auc.pdf
    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.
    Returns
    -------
    score : double
            The mean squared error between actual and posterior
    """
    # transform to array
    if type(actual) == list:
        actual = np.array(actual)
    if type(posterior) == list:
        posterior = np.array(posterior)

    if actual.shape[0] == 0 or posterior.shape[0] == 0:
        return None

    # transform to array(int)
    if actual.dtype != int:
        actual = actual.astype(dtype=int)

    # absorb some error
    if np.max(actual) > 1:
        actual[np.where(actual > 1)[0]] = 1
    if len(posterior.shape) > 1 and posterior.shape[1] > 1:
        posterior = posterior[:, 0]
    posterior = np.reshape(posterior, [posterior.size])

    r = tied_rank(posterior.tolist())
    det_num = len(np.where(posterior > -float('inf'))[0])
    det_positive_index = [i for i in range(len(actual)) if actual[i] == 1 and posterior[i] > -float('inf')]
    det_num_positive = len(det_positive_index)
    num_positive = len(np.where(actual == 1)[0])    # detected positive and missed positive
    det_num_negative = det_num - det_num_positive

    det_sum_positive = sum([r[i] for i in det_positive_index])
    auc = ((det_sum_positive - det_num_positive * (det_num_positive + 1)/2.0) /
           (det_num_negative * num_positive + 0.000001))
    return auc


def roc(actual,posterior):
    '''
    compute receiver-operater characteristic curve(ROC) for binary classification
    Parameters
    ----------
    actual: list
            ground truth label
    posterior: list
            predicated score for positive label

    Returns
    -------
    x: false positive rate
    y: true positive rate
    '''
    # transform to array
    if type(actual) == list:
        actual = np.array(actual)
    if type(posterior) == list:
        posterior = np.array(posterior)

    if actual.shape[0] == 0 or posterior.shape[0] == 0:
        return None

    # transform to array(int)
    if actual.dtype != int:
        actual = actual.astype(dtype=int)

    # absorb some error
    if np.max(actual) > 1:
        actual[np.where(actual > 1)[0]] = 1
    if len(posterior.shape) > 1 and posterior.shape[1] > 1:
        posterior = posterior[:, 0]
    posterior = np.reshape(posterior, [posterior.size])

    #
    num = actual.shape[0]
    positive_num = len(np.where(actual == 1)[0])
    negative_num = num - positive_num

    true_positive_num = 0.0
    false_positive_num = 0.0

    sorted_x = sorted(zip(posterior, range(len(posterior))), reverse=True)
    fp_thres = np.array(range(11)) * 0.1

    fp = []
    tp = []

    for i in range(len(sorted_x)):
        if posterior[sorted_x[i][1]] <= -float("inf"):
            continue

        if actual[sorted_x[i][1]] == 1:
            true_positive_num += 1.0
        else:
            false_positive_num += 1.0

        # false positive rate
        negative_num = negative_num if negative_num > 0 else negative_num + 0.0000000001
        fp_rate = false_positive_num / float(negative_num)
        # true positive rate
        positive_num = positive_num if positive_num > 0 else positive_num + 0.0000000001
        tp_rate = true_positive_num / float(positive_num)

        fp.append(fp_rate)
        tp.append(tp_rate)

    inds = np.searchsorted(fp, fp_thres, side='left')
    x = [0 for _ in range(len(inds))]
    y = [0 for _ in range(len(inds))]
    try:
        for ri, pi in enumerate(inds):
            if pi == len(tp):
                y[ri] = tp[pi - 1]
                x[ri] = fp[pi - 1]
            else:
                y[ri] = tp[pi]
                x[ri] = fp[pi]
    except:
        pass

    xy = np.array([x, y])
    return xy.transpose()
