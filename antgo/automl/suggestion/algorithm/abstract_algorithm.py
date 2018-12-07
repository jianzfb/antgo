import abc


class AbstractSuggestionAlgorithm(object):

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_new_suggestions(self, study_name, trials=[], number=1):
    """
    The study's study_configuration is like this.
    {
          "goal": "MAXIMIZE",
          "maxTrials": 5,
          "maxParallelTrials": 1,
          "params": [
              {
                  "parameterName": "hidden1",
                  "type": "INTEGER",
                  "minValue": 40,
                  "maxValue": 400,
                  "scalingType": "LINEAR"
              },
              {
                  "parameterName": "searchs_block0_branch0_conv2d",
                  "type": "COMPLEX_DISCRETE",
                  "feasiblePoints": "1,1",
                  "superscript": "kernel_size_h,kernel_size_w",
                  "num": 2,
              }
          ],
      }
    
    The trial's parameter_values_json should be like this.
    {
          "params":{
              "hidden1": 40
          }
    }
    
    Args:
      study_name: The study name.
      trials: The all trials of this study.
      number: The number of trial to return. 
    Returns:
      The array of trial objects.
    """
    raise NotImplementedError
