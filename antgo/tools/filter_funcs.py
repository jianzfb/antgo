import os
import logging
import json

def filter_by_tags(src_file, tgt_file, tags, no_tags):
    # use browser filter
    # src check
    if not os.path.exists(src_file):
        logging.error(f"{src_file} not existed")
        return
    if not src_file.endswith('.json'):
        logging.error(f"{src_file} must be json file")
        return
    
    if tags is None and no_tags is None:
        logging.error("Must set tags or no-tags")
        return
    
    tags_text = None
    no_tags_text = None
    if tags is not None:
        tags = set(tags.split(','))
        tags_text = '_'.join(tags)
    if no_tags is not None:
        no_tags = set(no_tags.split(','))
        no_tags_text = '_'.join(no_tags)
    
    with open(src_file, 'r') as fp:
        sample_list = json.load(fp)
    
    with open(tgt_file, 'r') as fp:
        filter_condition = json.load(fp)
    
    remain_sample_list = []
    for _, v in filter_condition.items():   
        tag_set = set(v[0]['tag'])
        if tags is not None:
            if not(tag_set == tags):
                continue
        if no_tags is not None:
            if tag_set == no_tags:
                continue
            
        sample = sample_list[int(v[0]['id'])]
        remain_sample_list.append(sample)
    
    src_folder = os.path.dirname(src_file)
    src_name = src_file.split('/')[-1].split('.')[0]
    
    prefix = f'{src_name}_filter'
    if tags is not None:
        prefix = f'{prefix}_tags_{tags_text}'
    if no_tags is not None:
        prefix = f'{prefix}_no_tags_{no_tags_text}'
    
    with open(os.path.join(src_folder, f'{prefix}.json'), 'w') as fp:
        json.dump(remain_sample_list,fp)




