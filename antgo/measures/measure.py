from __future__ import unicode_literals
from __future__ import division
from antgo.measures.objdect_task import *
from antgo.measures.multic_task import *
from antgo.measures.regression_task import *
from antgo.measures.segmentation_task import *
from antgo.measures.matting_task import *
from antgo.html.html import *
from antgo.measures.significance import *
import copy


class AntMeasures():
    def __init__(self, task):
        self.task = task

    def _supported_measures(self):
        if self.task.task_type is not None:
            if self.task.task_type == 'CLASSIFICATION':
                return [AntAccuracyMultiC(self.task),
                        AntConfusionMatrixMultiC(self.task)]
            elif self.task.task_type == 'OBJECT-DETECTION':
                return [AntVOCDet(self.task),
                        AntCOCODet(self.task),
                        AntROCandAUCDet(self.task),
                        AntPRDet(self.task),
                        AntAPRFDet(self.task),
                        AntTFTFDet(self.task)]
            elif self.task.task_type == 'REGRESSION':
                return [AntMAPERegression(self.task),
                        AntAlmostCRegression(self.task)]
            elif self.task.task_type == 'RETRIEVAL':
                return []
            elif self.task.task_type == 'SEGMENTATION':
                return [AntPixelAccuracySeg(self.task),
                        AntMeanAccuracySeg(self.task),
                        AntMeanIOUSeg(self.task),
                        AntFrequencyWeightedIOUSeg(self.task),
                        AntMeanIOUBoundary(self.task)]
            elif self.task.task_type == 'INSTANCE-SEGMENTATION':
                return []
            elif self.task.task_type == 'MATTING':
                return [AntSADMatting(self.task),
                        AntMSEMatting(self.task),
                        AntGradientMatting(self.task)]

            return []
        else:
            # return all
            return []

    def measures(self, measure_names=None):
        all_measures = self._supported_measures()
        if measure_names is not None:
            if type(measure_names) != list:
                measure_names = [measure_names]

            applied_measures = []
            for measure in all_measures:
                if measure.name in measure_names:
                    applied_measures.append(measure)
            return applied_measures
        else:
            return all_measures


def multi_repeats_measures_statistic(multi_statistics, method='repeated-holdout'):
  # time
  series_elapsed_time = []
  series_elapsed_time_per_sample = []

  # cpu
  series_cpu_mean_usage = []
  series_cpu_median_usage = []
  series_cpu_max_usage = []

  # memory
  series_mem_mean_usage = []
  series_mem_median_usage = []
  series_mem_max_usage = []

  # measures
  series_measures = []
  cpu_model = None
  title = ""

  for statistic_result_index, statistic_result in enumerate(multi_statistics):
    for ant_name, ant_statistic in statistic_result.items():
      title = ant_name
      if 'time' in ant_statistic:
        series_elapsed_time.append(ant_statistic['time']['elapsed_time'])
        if 'elapsed_time_per_sample' in ant_statistic['time']:
          series_elapsed_time_per_sample.append(ant_statistic['time']['elapsed_time_per_sample'])

      if 'cpu' in ant_statistic:
        if 'cpu_mean_usage' in ant_statistic['cpu']:
          series_cpu_mean_usage.append(ant_statistic['cpu']['cpu_mean_usage'])
        if 'cpu_median_usage' in ant_statistic['cpu']:
          series_cpu_median_usage.append(ant_statistic['cpu']['cpu_median_usage'])
        if 'cpu_max_usage' in ant_statistic['cpu']:
          series_cpu_max_usage.append(ant_statistic['cpu']['cpu_max_usage'])

        if 'mem_mean_usage' in ant_statistic['cpu']:
          series_mem_mean_usage.append(ant_statistic['cpu']['mem_mean_usage'])
        if 'mem_median_usage' in ant_statistic['cpu']:
          series_mem_median_usage.append(ant_statistic['cpu']['mem_median_usage'])
        if 'mem_max_usage' in ant_statistic['cpu']:
          series_mem_max_usage.append(ant_statistic['cpu']['mem_max_usage'])
        if 'cpu_model' in ant_statistic['cpu']:
          cpu_model = ant_statistic['cpu']['cpu_model']

      if 'gpu' in ant_statistic:
        pass

      if 'measure' in ant_statistic:
        for measure_index, measure_statistic in enumerate(ant_statistic['measure']):
          if statistic_result_index == 0:
            series_measures.append(copy.deepcopy(measure_statistic))

            for per_value in series_measures[measure_index]['statistic']['value']:
              if per_value['type'] == 'SCALAR':
                # (1) shape - scalar
                # (2) shape - [scalar, scalar, ...]
                per_value['value'] = [per_value['value']]
              elif per_value['type'] == 'CURVE':
                # (1) shape - [np.array.tolist(),np.array.tolist(),...]
                per_value['value'] = [per_value['value']]
              elif per_value['type'] == 'MATRIX':
                # (1) shape - Matrix(np.array)
                per_value['value'] = [per_value['value']]
            #
            continue

          # combine others
          final_measure_statistic = series_measures[measure_index]
          for per_value_index, per_value in enumerate(measure_statistic['statistic']['value']):
            if per_value['type'] == 'SCALAR':
              # (1) shape - scalar
              # (2) shape - [scalar, scalar, ...]
              final_measure_statistic['statistic']['value'][per_value_index]['value'].append(per_value['value'])
            elif per_value['type'] == 'CURVE':
              # (1) shape - [np.array,np.array,...]
              final_measure_statistic['statistic']['value'][per_value_index]['value'].append([a for a in per_value['value']])
            elif per_value['type'] == 'MATRIX':
              # (1) shape - Matrix
              final_measure_statistic['statistic']['value'][per_value_index]['value'].append(per_value['value'])


  multi_statistics = {}
  # time
  if len(series_elapsed_time) > 0:
    multi_statistics['time'] = {}
    series_elapsed_time_mean = np.mean(series_elapsed_time)
    series_elapsed_time_std = np.std(series_elapsed_time)
    multi_statistics['time']['elapsed_time'] = series_elapsed_time_mean
    multi_statistics['time']['elapsed_time_interval'] = series_elapsed_time_std

  if len(series_elapsed_time_per_sample) > 0:
    series_elapsed_time_mean_per_sample = np.mean(series_elapsed_time_per_sample)
    series_elapsed_time_std_per_sample = np.std(series_elapsed_time_per_sample)
    multi_statistics['time']['elapsed_time_per_sample'] = series_elapsed_time_mean_per_sample
    multi_statistics['time']['elapsed_time_per_sample_interval'] = series_elapsed_time_std_per_sample

  # cpu
  if cpu_model is not None:
    multi_statistics['cpu'] = {}
    multi_statistics['cpu']['cpu_model'] = cpu_model

    series_cpu_mean_usage_mean = np.mean(series_cpu_mean_usage)
    series_cpu_mean_usage_std = np.std(series_cpu_mean_usage)
    multi_statistics['cpu']['cpu_mean_usage'] = series_cpu_mean_usage_mean
    multi_statistics['cpu']['cpu_mean_usage_interval'] = series_cpu_mean_usage_std

    series_cpu_median_usage_mean = np.mean(series_cpu_median_usage)
    series_cpu_median_usage_std = np.std(series_cpu_median_usage)
    multi_statistics['cpu']['cpu_median_usage'] = series_cpu_median_usage_mean
    multi_statistics['cpu']['cpu_median_usage_interval'] = series_cpu_median_usage_std

    series_cpu_max_usage_mean = np.mean(series_cpu_max_usage)
    series_cpu_max_usage_std = np.std(series_cpu_max_usage)
    multi_statistics['cpu']['cpu_max_usage'] = series_cpu_max_usage_mean
    multi_statistics['cpu']['cpu_max_usage_interval'] = series_cpu_max_usage_std

    # memory
    series_mem_mean_usage_mean = np.mean(series_mem_mean_usage)
    series_mem_mean_usage_std = np.std(series_mem_mean_usage)
    multi_statistics['cpu']['mem_mean_usage'] = series_mem_mean_usage_mean
    multi_statistics['cpu']['mem_mean_usage_interval'] = series_mem_mean_usage_std

    series_mem_median_usage_mean = np.mean(series_mem_median_usage)
    series_mem_median_usage_std = np.std(series_mem_median_usage)
    multi_statistics['cpu']['mem_median_usage'] = series_mem_median_usage_mean
    multi_statistics['cpu']['mem_median_usage_interval'] = series_mem_median_usage_std

    series_mem_max_usage_mean = np.mean(series_mem_max_usage)
    series_mem_max_usage_std = np.std(series_mem_max_usage)
    multi_statistics['cpu']['mem_max_usage'] = series_mem_max_usage_mean
    multi_statistics['cpu']['mem_max_usage_interval'] = series_mem_max_usage_std

  # measure
  for measure in series_measures:
    for per_value in measure['statistic']['value']:
      if per_value['type'] == 'SCALAR':
        scalar_array = np.array(per_value['value'])
        if len(scalar_array.shape) == 1:
          per_value['value'] = np.mean(scalar_array)
          if method == 'bootstrap':
            per_value['interval'] = bootstrap_direct_confidence_interval(scalar_array)
          else:
            per_value['interval'] = np.std(scalar_array)
        else:
          assert(len(scalar_array.shape) == 2)
          per_value['value'] = np.mean(scalar_array,axis=0).tolist()
          per_value['interval'] = np.std(scalar_array,axis=0).tolist()
      elif per_value['type'] == 'CURVE':
        # shape (N,curves_num,point_num,2)
        scalar_array = np.array(per_value['value'])
        assert(len(scalar_array.shape) == 4)
        x,y = np.split(scalar_array,2,axis=3)
        x = np.squeeze(x,axis=3)
        xt = np.transpose(x,[1,2,0])
        x_mean = np.mean(xt,axis=2)

        y = np.squeeze(y,axis=3)
        yt = np.transpose(y,[1,2,0])
        y_mean = np.mean(yt,axis=2) # (curves_num,point_num)
        y_std = np.std(yt,axis=2)   # (curves_num,point_num)

        # shape (curves_num,point_num,2)
        xy = np.concatenate((np.expand_dims(x_mean,2),np.expand_dims(y_mean,2)),axis=2)
        curves_num = xy.shape[0]
        curves_list = np.split(xy,curves_num,axis=0)

        xy_std = np.concatenate((np.expand_dims(x_mean,2),np.expand_dims(y_std,2)),axis=2)
        curves_std_list = np.split(xy_std,curves_num,axis=0)

        per_value['value'] = [np.squeeze(cl).tolist() for cl in curves_list]
        per_value['interval'] = [np.squeeze(cl).tolist() for cl in curves_std_list]
      elif per_value['type'] == 'MATRIX':
        # shape (N,rows,cols)
        scalar_array = np.array(per_value['value'])
        assert(len(scalar_array.shape) == 3)
        m_mean = np.mean(scalar_array,axis=0)
        m_std = np.std(scalar_array,axis=0)

        per_value['value'] = m_mean
        per_value['interval'] = m_std

  multi_statistics['measure'] = series_measures
  ant_statistic_warp = {}
  ant_statistic_warp[title] = multi_statistics
  return ant_statistic_warp

if __name__ == '__main__':
    # experiment 1
    experiment_1_statis = {}
    experiment_1_statis['aa'] = {}
    experiment_1_statis['aa']['time'] = {}
    experiment_1_statis['aa']['time']['elapsed_time'] = 1.2
    experiment_1_statis['aa']['cpu'] = {}
    experiment_1_statis['aa']['cpu']['mem_mean_usage'] = 23
    experiment_1_statis['aa']['cpu']['mem_median_usage'] = 12
    experiment_1_statis['aa']['cpu']['mem_max_usage'] = 44
    experiment_1_statis['aa']['cpu']['cpu_mean_usage'] = 55
    experiment_1_statis['aa']['cpu']['cpu_median_usage'] = 11
    experiment_1_statis['aa']['cpu']['cpu_max_usage'] = 22
    experiment_1_statis['aa']['cpu']['cpu_model'] = 'aabbcc'

    voc_measure = {'statistic':{'name':'voc',
                             'value':[{'name':'MAP','value':[23.0,11.0,12.0],'type':'SCALAR','x':'class','y':'Mean Average Precision'},
                                      {'name':'Mean-MAP','value':0.23,'type':'SCALAR'}]}}

    roc_auc_measure = {'statistic':{'name':'roc_auc',
                             'value':[{'name':'ROC','value':[(np.ones((3,2)) * 3).tolist(),(np.ones((3,2))*2).tolist()],'type':'CURVE','x':'FP','y':'TP'},
                                      {'name':'AUC','value':[0.3,0.4],'type':'SCALAR','x':'class','y':'AUC'}]}}

    pr_f1_measure = {'statistic':{'name':'pr_f1',
                             'value':[{'name':'Precision-Recall','value':[(np.ones((4,2))*4).tolist(),(np.ones((4,2))*3).tolist()],'type':'CURVE','x':'precision','y':'recall'},
                                      {'name':'F1','value':[1.0,2.0],'type':'SCALAR','x':'class','y':'F1'}]}}

    confusion_m = {'statistic': {'name': 'cm',
                              'value': [{'name':'ccmm','value':(np.ones((3,4))*3).tolist(),'type':'MATRIX','x':'class','y':'class'}]}}

    experiment_1_statis['aa']['measure'] = [voc_measure,roc_auc_measure,pr_f1_measure,confusion_m]
    # experiment 2
    experiment_2_statis = {}
    experiment_2_statis['aa'] = {}
    experiment_2_statis['aa']['time'] = {}
    experiment_2_statis['aa']['time']['elapsed_time'] = 1.2
    experiment_2_statis['aa']['cpu'] = {}
    experiment_2_statis['aa']['cpu']['mem_mean_usage'] = 23
    experiment_2_statis['aa']['cpu']['mem_median_usage'] = 12
    experiment_2_statis['aa']['cpu']['mem_max_usage'] = 44
    experiment_2_statis['aa']['cpu']['cpu_mean_usage'] = 55
    experiment_2_statis['aa']['cpu']['cpu_median_usage'] = 11
    experiment_2_statis['aa']['cpu']['cpu_max_usage'] = 22
    experiment_2_statis['aa']['cpu']['cpu_model'] = 'aabbcc'

    voc_measure = {'statistic': {'name': 'voc',
                                 'value': [{'name': 'MAP', 'value': [2.0, 1.0, 2.0], 'type': 'SCALAR', 'x': 'class',
                                            'y': 'Mean Average Precision'},
                                           {'name': 'Mean-MAP', 'value': 0.12, 'type': 'SCALAR'}]}}

    roc_auc_measure = {'statistic': {'name': 'roc_auc',
                                     'value': [
                                         {'name': 'ROC', 'value': [(np.ones((3, 2))*4).tolist(), (np.ones((3, 2)) * 1).tolist()], 'type': 'CURVE',
                                          'x': 'FP', 'y': 'TP'},
                                         {'name': 'AUC', 'value': [0.12, 0.34], 'type': 'SCALAR', 'x': 'class',
                                          'y': 'AUC'}]}}

    pr_f1_measure = {'statistic': {'name': 'pr_f1',
                                   'value': [
                                       {'name': 'Precision-Recall', 'value': [(np.ones((4, 2)) * 1).tolist(), (np.ones((4, 2)) * 2).tolist()],
                                        'type': 'CURVE', 'x': 'precision', 'y': 'recall'},
                                       {'name': 'F1', 'value': [1.4, 1.0], 'type': 'SCALAR', 'x': 'class', 'y': 'F1'}]}}

    confusion_m = {'statistic': {'name': 'cm',
                                 'value': [{'name': 'ccmm', 'value': (np.ones((3, 4)) * 8).tolist(), 'type': 'MATRIX',
                                            'x': 'class', 'y': 'class'}]}}

    experiment_2_statis['aa']['measure'] = [voc_measure, roc_auc_measure, pr_f1_measure,confusion_m]

    ss = multi_repeats_measures_statistic([experiment_1_statis,experiment_2_statis])
    everything_to_html(ss,'/home/mi')
    print(ss)