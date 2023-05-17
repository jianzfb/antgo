import os
from antgo.pipeline.extent import op

for op_file_name in os.listdir(os.path.dirname(__file__)):
    if op_file_name[0] == '.':
        continue
    if op_file_name.startswith('_'):
        continue

    op.load(op_file_name, os.path.dirname(__file__))