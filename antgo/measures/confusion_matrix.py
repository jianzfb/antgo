from __future__ import unicode_literals
from __future__ import division

import numpy as np
import unittest

def confusion_matrix(actual,predicated_score):
  '''
  compute confusion matrix (for muti-classification task and binary-classification task)
  Parameters
  ----------
  actual: list or ndarray
          ground truth label list
  predicted_score: list or ndarray
          predicted score (for every label)

  Returns
  -------
  return confusion matrix
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

  # label number
  class_num = predicated_score.shape[1]

  # confusion matrix
  predicated = np.argmax(predicated_score, axis=1)
  cm = np.zeros((class_num,class_num))
  num = len(actual)
  for i in range(num):
    cm[actual[i],predicated[i]] += 1

  return cm

def compute_confusion_matrix(actual, predict_label, class_num):
  if type(actual) == list:
    actual = np.array(actual, dtype=np.int32)

  if type(predict_label) == list:
    predict_label = np.array(predict_label)

  cm = np.zeros((class_num, class_num))
  for c_i in range(class_num):
    c_i_pos = np.where(actual == c_i)
    c_i_pos_num = len(c_i_pos[0])
    c_i_predict = predict_label[c_i_pos]
    for to_i in range(class_num):
      to_i_num = len(np.where(c_i_predict == to_i)[0])
      cm[c_i, to_i] = to_i_num

    cm[c_i, :] = cm[c_i, :] / float(c_i_pos_num)

  cm = cm / float(len(actual))
  return cm

#unit test
class TestConfusionMatrix(unittest.TestCase):
  def test_cm(self):
    actual = [0,0,1,2,1,2]
    predicated_score = [[0.8,0.1,0.1],[0.3,0.5,0.2],[0.1,0.5,0.4],[0.2,0.2,0.6],[0.2,0.3,0.5],[0.2,0.1,0.7]]
    cm = confusion_matrix(actual,predicated_score)
    np.testing.assert_array_almost_equal(cm.flatten(),np.array([1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,2.0]))

if __name__ == "__main__":
    unittest.main()