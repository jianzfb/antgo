# -*- coding: UTF-8 -*-
# @Time    : 2018/12/17 4:58 PM
# @File    : bayesian.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import random
import time
from copy import deepcopy
from functools import total_ordering
try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import math
from sklearn import gaussian_process
from antgo.automl.suggestion.metric import *
import functools
# import matplotlib.pyplot as plt


class BayesianOptimizer(object):
    """

    gpr: A GaussianProcessRegressor for bayesian optimization.
    """
    def __init__(self, t_min, metric, kernel_lambda, beta):
        self.t_min = t_min
        self.metric = metric
        self.gp = gaussian_process.GaussianProcessRegressor()

        self.beta = beta
        self.y = []
        self.is_ok = False

    def fit(self, x_queue, y_queue):
        self.gp.fit(x_queue, y_queue)
        self.y.extend(y_queue)

        self.is_ok = True

    def optimize_acq(self, searcher, timeout=60 * 60 * 24):
        start_time = time.time()
        elem_class = Elem
        if self.metric.higher_better():
            elem_class = ReverseElem

        # Initialize the priority queue.
        pq = queue.PriorityQueue()
        for metric_value in self.y:
            pq.put(elem_class(metric_value, None))

        t = 1.0
        t_min = self.t_min
        alpha = 0.9
        opt_acq = self._get_init_opt_acq_value()
        opt_model = None
        opt_model_x = None
        remaining_time = timeout
        while not pq.empty() and t > t_min and remaining_time > 0:
            elem = pq.get()
            # simulated annealing
            if self.metric.higher_better():
                temp_exp = min((elem.metric_value - opt_acq) / t, 1.0)
            else:
                temp_exp = min((opt_acq - elem.metric_value) / t, 1.0)
            ap = math.exp(temp_exp)
            if ap >= random.uniform(0, 1):
                for model_x, model in searcher():
                    # UCB acquisition function
                    temp_acq_value = self.acq(np.expand_dims(model_x, 0))[0]
                    pq.put(elem_class(temp_acq_value, model))

                    if self._accept_new_acq_value(opt_acq, temp_acq_value):
                        opt_acq = temp_acq_value
                        opt_model = model
                        opt_model_x = model_x

            t *= alpha
            remaining_time = timeout - (time.time() - start_time)

        if remaining_time < 0:
            raise TimeoutError

        return opt_model_x, opt_acq, opt_model

    def acq(self, x):
        mean, std = self.gp.predict(x,return_std=True)
        if self.metric.higher_better():
            return mean + self.beta * std
        return mean - self.beta * std

    def _get_init_opt_acq_value(self):
        if self.metric.higher_better():
            return -np.inf
        return np.inf

    def _accept_new_acq_value(self, opt_acq, temp_acq_value):
        if temp_acq_value > opt_acq and self.metric.higher_better():
            return True
        if temp_acq_value < opt_acq and not self.metric.higher_better():
            return True
        return False


@total_ordering
class Elem:
    def __init__(self, metric_value, graph):
        self.graph = graph
        self.metric_value = metric_value

    def __eq__(self, other):
        return self.metric_value == other.metric_value

    def __lt__(self, other):
        return self.metric_value < other.metric_value


class ReverseElem(Elem):
    def __lt__(self, other):
        return self.metric_value > other.metric_value


if __name__ == '__main__':
    class AA(object):
        def __init__(self):
            pass

        def random(self, num):
            return [(np.array([random.random() * 2 - 1]), None)]

    target_func = lambda x: x**2

    bo = BayesianOptimizer(0.000000001, Loss, 0.1, 2.576)
    x = [np.array([random.random()*2-1]) for _ in range(100)]
    y = [target_func(xi[0]) for xi in x]
    bo.fit(x, y)

    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter([xi[0] for xi in x], y, c='b')

    aa = AA()

    suggestion_val_list = []
    suggestion_gt_list = []
    for _ in range(100):
        suggestion_val, suggestion_score_predicted, _ = bo.optimize_acq(functools.partial(aa.random, 1))
        gt_val = target_func(suggestion_val)
        suggestion_val_list.append(suggestion_val)
        suggestion_gt_list.append(gt_val)
        print('predict %f gt %f'%(suggestion_score_predicted, gt_val))

    plt.scatter(suggestion_val_list, suggestion_gt_list, c='r')
    plt.show()