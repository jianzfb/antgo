import os
import logging
import json

def filter_by_tag(src_file, tgt_file, tags):
    # use browser filter
    # src check
    if not os.path.exists(src_file):
        logging.error(f"{src_file} not existed")
        return
    if not src_file.endswith('.json'):
        logging.error(f"{src_file} must be json file")
        return
    
    assert(tags is not None)
    tags = set(tags.split(','))
    tags_text = '_'.join(tags)
    
    with open(src_file, 'r') as fp:
        sample_list = json.load(fp)
    
    with open(tgt_file, 'r') as fp:
        filter_condition = json.load(fp)
    
    remain_sample_list = []
    for _, v in filter_condition.items():   
        tag_set = set(v[0]['tag'])
        if not(tag_set == tags):
            continue
        sample = sample_list[int(v[0]['id'])]
        remain_sample_list.append(sample)
    
    src_folder = os.path.dirname(src_file)
    src_name = src_file.split('/')[-1]
    with open(os.path.join(src_folder, f'{src_name}_filter_{tags_text}.json'), 'w') as fp:
        json.dump(remain_sample_list,fp)






