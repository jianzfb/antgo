#encoding=utf-8
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import unittest

def binary_c_stats(actual,predicated):
    '''
    computing true positive rate, false positive rate,
    true negative rate and false negative rate for binary classification problem
    Parameters
    ----------
    actual: list
            ground truth label
    predicated: list
            predicated label
    Returns
    -------
    TP,FN,TN,FP
    '''
    # transform to array
    if type(actual) == list:
        actual = np.array(actual)
    if type(predicated) == list:
        predicated = np.array(predicated)

    if actual.shape[0] == 0 or predicated.shape[0] == 0:
        return None

    # transform to array(int)
    if actual.dtype != int:
        actual = actual.astype(dtype=int)

    # absorb some error
    if np.max(actual) > 1:
        actual[np.where(actual > 1)[0]] = 1
    if len(predicated.shape) > 1 and predicated.shape[1] > 1:
        predicated = predicated[:,0]
    predicated = np.reshape(predicated,[predicated.size])
    if predicated.dtype != int:
        predicated = predicated.copy()
        predicated[np.where(predicated <= 0.5)[0]] = 0.0
        predicated[np.where(predicated > 0.5)[0]] = 1.0
        predicated = predicated.astype(dtype=int)

    true_index = np.where(actual == 1)[0]
    true_index_num = len(true_index)
    true_positive = len(np.where(predicated[true_index] == 1)[0])
    false_negative = len(true_index) - true_positive

    negative_index = np.where(actual == 0)[0]
    negative_index_num = len(negative_index)
    true_negative = len(np.where(predicated[negative_index] == 0)[0])
    false_positive = len(negative_index) - true_negative

    return true_positive / (true_index_num + 0.000001),\
           false_negative / (true_index_num + 0.000001),\
           true_negative / (negative_index_num + 0.000001),\
           false_positive / (negative_index_num + 0.000001)

def binary_c_stats2(actual, predicated):
    '''
    regular statistics about binary classification task
    Parameters
    ----------
    actual: list
            ground truth label
    predicated: list
            predicated label
    Returns
    -------
    accuracy: (corrected positive + corrected negative) / all samples
    precision: corrected positive / predicated positives
    recall: corrected positive / all positives
    F1-measure:  
    '''
    #transform to array
    if type(actual) == list:
        actual = np.array(actual)
    if type(predicated) == list:
        predicated = np.array(predicated)

    if actual.shape[0] == 0 or predicated.shape[0] == 0:
        return None

    #transform to array(int)
    if actual.dtype != int:
        actual = actual.astype(dtype=int)

    #absorb some error
    if np.max(actual) > 1:
        actual[np.where(actual > 1)[0]] = 1
    if len(predicated.shape) > 1 and predicated.shape[1] > 1:
        predicated = predicated[:,0]
    predicated = np.reshape(predicated,[predicated.size])

    if predicated.dtype != int:
        predicated = predicated.copy()
        predicated[np.where(predicated <= 0.5)[0]] = 0.0
        predicated[np.where(predicated > 0.5)[0]] = 1.0
        predicated = predicated.astype(dtype=int)

    num = actual.shape[0]
    true_index = np.where(actual == 1)[0]
    true_index_num = len(true_index)
    true_positive = len(np.where(predicated[true_index] == 1)[0])
    false_negative = true_index_num - true_positive

    negative_index = np.where(actual == 0)[0]
    negative_index_num = len(negative_index)
    true_negative = len(np.where(predicated[negative_index] == 0)[0])
    false_positive = negative_index_num - true_negative

    accuracy = float(true_positive + true_negative) / (num + 0.000001)
    precision = float(true_positive) / (true_positive + false_positive + 0.000001)
    recall = float(true_positive) / (true_positive + false_negative + 0.000001)
    F1 = float(2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative + 0.000001)

    return accuracy,precision,recall,F1

class TestBC(unittest.TestCase):
    def test(self):
        a = [1, 0, 0, 1, 1, 1, 0, 1]
        b = [0, 0, 0, 1, 1, 0, 1, 1]

        accuracy, precision, recall, F1 = binary_c_stats2(a, b)
        self.assertAlmostEqual(accuracy,5.0/8.0)
        self.assertAlmostEqual(precision,3.0/4.0)
        self.assertAlmostEqual(recall,3.0/5.0)
        self.assertAlmostEqual(F1,6.0/(2*3.0+1+2))

        TP, FN, TN, FP = binary_c_stats(a, b)
        self.assertAlmostEqual(TP, 3.0 / 5.0)
        self.assertAlmostEqual(FN, 2.0 / 5.0)
        self.assertAlmostEqual(TN, 2.0 / 3.0)
        self.assertAlmostEqual(FP, 1.0 / 3.0)

if __name__ == "__main__":
    unittest.main()