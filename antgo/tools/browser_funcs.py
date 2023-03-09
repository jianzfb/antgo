import os
import logging
import json
from antgo.interactcontext import InteractContext

def browser_images(src_file, tags, white_users_str, feedback=True):
    # src check
    if not os.path.exists(src_file):
        logging.error(f"{src_file} not existed")
        return
    if not src_file.endswith('.json'):
        logging.error(f"{src_file} must be json file")
        return
    
    title = src_file.split('/')[-1].split('.')[0]

    if tags is not None:
        tags = tags.split(',')
    else:
        tags = []
    white_users = {}
    if white_users_str is not None:
        for t in white_users_str.split(','):
            user_name, password = t.split(':')
            white_users.update({
                user_name: {'password': password}
            })
    ctx = InteractContext()
    ctx.browser.start(f"{title}", config = {
            'tags': tags,
            'white_users': white_users,
        }, json_file=src_file)
    
    logging.info('Waiting data browser stop.')
    ctx.browser.waiting(not feedback)
    
    if feedback:
        content = ctx.browser.download()
        
        src_folder = os.path.dirname(src_file)
        src_name = src_file.split('/')[-1]
        with open(os.path.join(src_folder, f'{src_name}_browser.json'), 'w') as fp:
            json.dump(content, fp)
