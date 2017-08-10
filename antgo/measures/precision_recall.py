from __future__ import unicode_literals
from __future__ import division

import numpy as np
import unittest


def pr(actual,predicated_s):
    '''
    computing precision recall curve
    what's difference between ROC and precision-recall curve?
    ROC curve plot True Positive Rate Vs. False Positive Rate;
    Whereas, PR curve plot Precision Vs. Recall.Particularly,
    if true negative is not much valuable to the problem,
    or negative examples are abundant.
    Then, PR-curve is typically more appropriate.
    For example, if the class is highly imbalanced and
    positive samples are very rare, then use PR-curve.
    One example may be fraud detection, where non-fraud
    sample may be 10000 and fraud sample may be below 100.
    In other cases, ROC curve will be more helpful.
    (only for binary classificaiton or information retrieval task)
    Parameters
    ----------
    actual: list
            ground truth label [1,0,0,1.....]
    predicated_s: list
            predicated score [0.5,0.2,....]

    Returns
    -------
    recall(x): recall list
    precision(y): precision list
    '''
    if type(actual) == list:
        actual = np.array(actual)
    if type(predicated_s) == list:
        predicated_s = np.array(predicated_s)

    if actual.shape[0] == 0 or predicated_s.shape[0] == 0:
        return None

    # transform to array(int)
    if actual.dtype != int:
        actual = actual.astype(dtype=int)

    # absorb some error
    if np.max(actual) > 1:
        actual[np.where(actual > 1)[0]] = 1
    if len(predicated_s.shape) > 1 and predicated_s.shape[1] > 1:
        predicated_s = predicated_s[:,0]
    if len(predicated_s.shape) > 1:
        predicated_s = predicated_s.flatten()

    sorted_x = sorted(zip(predicated_s, range(predicated_s.shape[0])), reverse=True)

    p_num = len(np.where(actual == 1)[0])
    precision = []
    recall = []
    num_hits = 0.0
    re_thres = np.array(range(11)) * 0.1

    for i in range(len(sorted_x)):
        if predicated_s[sorted_x[i][1]] <= -float("inf"):
            continue

        if actual[sorted_x[i][1]] == 1:
            num_hits += 1.0

            cur_recall = num_hits / float(p_num)
            cur_precision = num_hits / (i + 1.0)

            recall.append(cur_recall)
            precision.append(cur_precision)

    inds = np.searchsorted(recall, re_thres, side='left')
    pr = [-1 for _ in range(len(inds))]
    rc = [-1 for _ in range(len(inds))]
    try:
        for ri, pi in enumerate(inds):
            if ri > 0 and pi == inds[ri - 1]:
                continue

            if pi == len(precision):
                pr[ri] = precision[pi - 1]
                rc[ri] = recall[pi - 1]
            else:
                pr[ri] = precision[pi]
                rc[ri] = recall[pi]
    except:
        pass

    pr = [_ for _ in pr if _ > 0]
    rc = [_ for _ in rc if _ > 0]

    pr = np.array([rc, pr])
    return pr.transpose()


def pr_f1(actual,predicated_s, k = 10):
    '''
    summarizing the precision-recall curve with fixed k
    Parameters
    ----------
    actual: list
            ground truth label
    predicated_s: list
            predicated score
    k: int
            The maximum number of predicted elements

    Returns
    -------
    F1 value
    '''
    if type(actual) == list:
        actual = np.array(actual)
    if type(predicated_s) == list:
        predicated_s = np.array(predicated_s)

    if actual.shape[0] == 0 or predicated_s.shape[0] == 0:
        return None

    # transform to array(int)
    if actual.dtype != int:
        actual = actual.astype(dtype=int)

    # absorb some error
    if np.max(actual) > 1:
        actual[np.where(actual > 1)[0]] = 1
    if len(predicated_s.shape) > 1 and predicated_s.shape[1] > 1:
        predicated_s = predicated_s[:,0]
    if len(predicated_s.shape) > 1:
        predicated_s = predicated_s.flatten()

    sorted_x = sorted(zip(predicated_s, range(predicated_s.shape[0])), reverse=True)

    p_num = len(np.where(actual == 1)[0])
    top_k = min(k, len(sorted_x))

    num_hits = 0.0
    for i in range(top_k):
        if actual[sorted_x[i][1]] == 1:
            num_hits += 1.0

    precision = num_hits / (k + 0.000001)
    recall = num_hits / (p_num + 0.000001)
    F1 = 2.0 * (precision * recall) / (precision + recall + 0.000001)
    return F1

def dcg_k(r,k,method = 0):
    '''
    Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Parameters
    ----------
    r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    k: Number of results to consider
    method: 0 or 1

    Returns
    -------
    Discounted cumulative gain
    '''
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            #standard definition
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            #used in Kaggle
            return np.sum((np.power(2,r) - 1.0) / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_k(r,k,method=0):
    '''
    Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Parameters
    ----------
    r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    k: Number of results to consider
    method: 0 or 1

    Returns
    -------
    Normalized discounted cumulative gain
    '''
    dcg_max = dcg_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_k(r, k, method) / dcg_max

class TestPR(unittest.TestCase):
    def test_func(self):
        actual = [0, 1, 1, 1]
        predicated_s = [0.3, 0.1, 0.2, 0.4]

        recall, precision = pr(actual, predicated_s)
        np.testing.assert_array_almost_equal(recall, [0.3333333333333333, 0.6666666666666666, 1.0])
        np.testing.assert_array_almost_equal(precision, [1.0, 0.6666666666666666, 0.75])

        f1_v = pr_f1(actual, predicated_s, k=2)
        self.assertAlmostEqual(f1_v, 2.0 * ((1.0 / 2.0) * (1.0 / 3.0)) / (1.0 / 2.0 + 1.0 / 3.0))

        r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        # dcg test
        self.assertAlmostEqual(dcg_k(r, 1), 3.0)
        self.assertAlmostEqual(dcg_k(r, 2), 5.0)

        # ndcg test
        r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        self.assertAlmostEqual(ndcg_k(r, 4), 0.77509868485972222)

if __name__ == "__main__":
    unittest.main()