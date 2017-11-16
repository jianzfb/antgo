#encoding=utf-8
from __future__ import unicode_literals
from __future__ import division

import numpy as np

def cross_entropy(actual,predicated_score,confidence_interval=0.95):
    '''
    compute cross entropy (for multi-classification task and binary-classification task)
    Parameters
    ----------
    actual: list or ndarray
            ground truth label
    predicated_s: list or ndarray
            predicated score for every label

    Returns
    -------

    '''
    # type consistent
    if type(actual) == list:
        actual = np.array(actual, dtype=np.int32)
    if type(predicated_score) == list:
        predicated_score = np.array(predicated_score)

    if actual.dtype != np.int32:
        actual = actual.astype(dtype=np.int32)

    # shape consistent
    assert (len(actual.shape) == 1)
    assert (len(predicated_score.shape) == 2)
    assert (actual.shape[0] == predicated_score.shape[0])

    # class number
    C_N = predicated_score.shape[1]
    assert(C_N >= 2)
    # sample number
    S_N = predicated_score.shape[0]

    # normalize predicated score
    ss = np.sum(predicated_score,axis=1)
    predicated_s = np.reshape(predicated_score,[S_N,1,C_N]) / np.reshape(ss,[S_N,1,1])
    predicated_s = np.reshape(predicated_s,[S_N,C_N])
    loss = 0.0

    # cross entropy
    for n in range(C_N):
        index = np.where(actual == n)[0]
        loss += sum(-np.log(predicated_s[index,n]))

    return loss / float(S_N)

def multi_accuracy(actual,predicated_score):
    '''
    compute classification accuracy(for multi-classification task and binary-classification task)
    Parameters
    ----------
    actual: list or ndarray
            ground truth list
    predicated_score: list or ndarray
            predicated score for every label

    Returns
    -------
    '''
    # type consistent
    if type(actual) == list:
        actual = np.array(actual,dtype=np.int32)
    if type(predicated_score) == list:
        predicated_score = np.array(predicated_score)

    if actual.shape[0] == 0 or predicated_score.shape[0] == 0:
        return None

    if actual.dtype != np.int32:
        actual = actual.astype(dtype=np.int32)

    # shape consistent
    assert(len(actual.shape) == 1)
    assert(len(predicated_score.shape) == 2)
    assert(actual.shape[0] == predicated_score.shape[0])

    #predicated label
    max_index = np.argmax(predicated_score,axis=1)
    index = np.where(max_index == actual)
    return len(index[0]) / float(actual.shape[0])


def multi_accuracy_labels(actual, predicated_label):
    '''
    compute classification accuracy(for multi-classification task and binary-classification task)
    Parameters
    ----------
    actual: list or ndarray
            ground truth list
    predicated_score: list or ndarray
            predicted label list
    Returns
    -------
    '''
    # type consistent
    if type(actual) == list:
        actual = np.array(actual, dtype=np.int32)
    if type(predicated_label) == list:
        predicated_label = np.array(predicated_label)

    if actual.shape[0] == 0 or predicated_label.shape[0] == 0:
        return None

    if actual.dtype != np.int32:
        actual = actual.astype(dtype=np.int32)

    # shape consistent
    assert (len(actual.shape) == 1)
    assert (len(predicated_label.shape) == 1)

    # predicated label
    index = np.where(predicated_label == actual)
    return len(index[0]) / float(actual.shape[0])

def per_class_accuracy(actual,predicated_score):
    '''
    compute mean per-class classification accuracy(for multi-classification and binary-classification task)
    It's just for imbalance data
    Parameters
    ----------
    actual: list or ndarray
            ground truth label
    preicated_score: list or ndarray
            predicated score for every label
    Returns
    -------
    return mean classification accuracy
    '''
    # type consistent
    if type(actual) == list:
        actual = np.array(actual, dtype=np.int32)
    if type(predicated_score) == list:
        predicated_score = np.array(predicated_score)

    if actual.shape[0] == 0 or predicated_score.shape[0] == 0:
        return None

    if actual.dtype != np.int32:
        actual = actual.astype(dtype=np.int32)

    # shape consistent
    assert (len(actual.shape) == 1)
    assert (len(predicated_score.shape) == 2)
    assert (actual.shape[0] == predicated_score.shape[0])

    # class number
    c_num = predicated_score.shape[1]

    max_arg = np.argmax(predicated_score,axis=1)
    per_class_accuracy_array = np.ones(c_num)
    for c_index in range(c_num):
        index = np.where(actual == c_index)[0]
        predicated_label = max_arg[index]
        per_class_accuracy_array[c_index] = len(np.where(predicated_label == c_index)[0]) / float(len(index))

    return per_class_accuracy_array.tolist()