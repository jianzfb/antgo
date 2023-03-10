import os
import json
import copy

class SampleGTTemplate(object):
    def __init__(self) -> None:
        me_file_path = os.path.realpath(__file__)
        parent_folder = os.path.dirname(os.path.dirname(me_file_path))
        with open(os.path.join(parent_folder, 'resource', 'templates', 'sample_gt.json'), 'r') as fp:
            self.dummy_sample = json.load(fp)

    def get(self):
        return copy.deepcopy(self.dummy_sample)