#encoding=utf-8
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import unittest

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k. (only for Information Retrieval Task)
    This function computes the average prescision at k between two lists of
    items.
    References: http://nlp.stanford.edu/IR-book/resource/htmledition/evaluation-of-ranked-retrieval-results-1.resource
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
             A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def s_apk(actual,predicated_s,k=10):
    '''
    Computes the average precision at k. (only for Information Retrieval Task)
    This function computes the average prescision at k between two lists of
    items.
    References: http://nlp.stanford.edu/IR-book/resource/htmledition/evaluation-of-ranked-retrieval-results-1.resource

    Parameters
    ----------
    actual: list
            ground truth label list
    predicated_s: list
            predicated score list
    k: int
            The maximum number of predicted elements
    Returns
    -------
    score : double
            he average precision at k over the input lists
    '''
    if type(actual) == list:
        actual = np.array(actual)
    if type(predicated_s) == list:
        predicated_s = np.array(predicated_s)

    if actual.dtype != int:
        actual = actual.astype(int)

    #absorb some error
    if np.max(actual) > 1:
        actual[np.where(actual > 1)[0]] = 1
    if len(predicated_s.shape) > 1 and predicated_s.shape[1] > 1:
        predicated_s = predicated_s[:, 0]
    if len(predicated_s.shape) > 1:
        predicated_s = predicated_s.flatten()

    positive_actual = np.where(actual == 1)[0]
    sorted_predicated_s = sorted(zip(predicated_s,range(predicated_s.shape[0])),reverse=True)
    sorted_predicated = [n for m,n in sorted_predicated_s]

    return apk(positive_actual.tolist(),sorted_predicated,k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def vmap(label, predicted_s):
    '''
    References
    ----------
    PASCAL VOC CHALLENGE (for binary classification problem)

    Parameters
    ----------
    label: A list of ground truth label (0 or 1)
    predicted_s: A list of predicted elements score
    Returns
    -------
    return mean average precision (for binary classification problem)
    '''
    if type(label) == list:
        label = np.array(label)
    if type(predicted_s) == list:
        predicted_s = np.array(predicted_s)

    if label.shape[0] == 0 or predicted_s.shape[0] == 0:
        return None

    # transform to array(int)
    if label.dtype != int:
        actual = label.astype(dtype=int)

    # absorb some error
    if np.max(label) > 1:
        label[np.where(label > 1)[0]] = 1
    if len(predicted_s.shape) > 1 and predicted_s.shape[1] > 1:
        predicted_s = predicted_s[:, 0]
    if len(predicted_s.shape) > 1:
        predicted_s = predicted_s.flatten()

    #data number
    num = label.size
    assert(label.size == predicted_s.size)

    #sort descend
    index = sorted(range(num), key=lambda k: predicted_s[k], reverse=True)

    label = label[index]
    predicted_s = predicted_s[index]

    positive_num = len(np.where(label == 1)[0])
    num_hits = 0.0

    #recall_thres = np.array(range(11)) * 0.1
    recall_thres = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
    rc = []
    pr = []
    for i in range(label.size):
        if predicted_s[i] <= -float("inf"):
            continue

        if label[i] == 1:
            num_hits += 1.0
            #recall
            recall = num_hits / positive_num

            rc.append(recall)
            pr.append(num_hits / (i + 1.0))

    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

    inds = np.searchsorted(rc, recall_thres, side='left')
    q = [-1 for _ in range(len(inds))]
    try:
        for ri, pi in enumerate(inds):
            if pi == len(pr):
                q[ri] = pr[pi - 1]
            else:
                q[ri] = pr[pi]
    except:
        pass

    return float(np.mean(q))

#unit test
class TestAveragePrecision(unittest.TestCase):
    def test_apk(self):
        apk_val = apk(range(1, 6), [6, 4, 7, 1, 2], 2)
        self.assertAlmostEqual(apk_val,0.25)

        actual = [0,1,1,1,1,1,0,0]
        predicted_s = [0.1,0.5,0.4,0.3,0.7,0.2,0.8,0.6]

        self.assertAlmostEqual(s_apk(actual,predicted_s,2),0.25)
        self.assertAlmostEqual(map(actual, predicted_s), 0.270995670995671)

if __name__ == "__main__":
    unittest.main()