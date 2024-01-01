import os
import logging
import yaml
import json
import pandas as pd


def table_show(data_list, head_list):
    display_info_format = {}
    for head in head_list:
        display_info_format[head] = []

    for info in data_list:
        for head in head_list:
            display_info_format[head].append(info[head])

    df = pd.DataFrame(display_info_format)
    print(df.to_string(index=False))
