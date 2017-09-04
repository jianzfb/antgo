#encoding=utf-8
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import unittest

def mape(actual_s,predicated_s):
    '''
    MAPE(median absolute percentage error) could be subjective to outliers(for Regresion Task)
    RMSE has some problems, which is sensitive to large outliers. MAPE maybe a good select.
    Parameters
    ----------
    actual_s: list
            ground truth value
    predicated_s: list
            predicated value

    Returns
    -------
    error
    '''
    if type(actual_s) == list:
        actual_s = np.array(actual_s, dtype=np.float32)
    if type(predicated_s) == list:
        predicated_s = np.array(predicated_s, dtype=np.float32)

    if len(predicated_s.shape) > 1 and predicated_s.shape[1] > 1:
        predicated_s = predicated_s[:,0]
    #reshape
    predicated_s = np.reshape(predicated_s,[predicated_s.size])

    diff = actual_s - predicated_s
    actual_s_eps = actual_s + 0.000000001   #avoid 0
    v = np.median(np.abs(diff / actual_s_eps))

    return v


def almost_correct(actual_s,predicated_s,X):
    '''
    It compute the percent of estimates that differ from the true value by
    no more than X%
    Parameters
    ----------
    actual_s: list
            ground truth value
    predicated_s: list
            predicated value
    X:  double
            percentage

    Returns
    -------
    score: double
            the percent of estimates within X% of the true values

    '''
    if type(actual_s) == list:
        actual_s = np.array(actual_s, dtype=np.float32)
    if type(predicated_s) == list:
        predicated_s = np.array(predicated_s, dtype=np.float32)

    #absorb some error
    if len(predicated_s.shape) > 1 and predicated_s.shape[1] > 1:
        predicated_s = predicated_s[:,0]
    #reshape
    predicated_s = np.reshape(predicated_s,[predicated_s.size])

    diff = actual_s - predicated_s
    actual_s_eps = actual_s + 0.000000001  # avoid 0

    diff = np.abs(diff/actual_s_eps)
    num = actual_s.shape[0]
    return float(len(np.where(diff <= X)[0])) / float(num)

class TestMAPEandAC(unittest.TestCase):
    def test(self):
        actual_s = [0.1,0.2,0.1,0.8,0.3]
        predicated_s = [0.12,0.1,0.11,0.74,0.04]
        self.assertAlmostEqual(mape(actual_s,predicated_s),0.2)
        self.assertAlmostEqual(almost_correct(actual_s, predicated_s, 0.1), 0.4)

if __name__ == "__main__":
    unittest.main()