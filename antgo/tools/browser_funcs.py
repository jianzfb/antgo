import os
import logging
import json
from antgo.interactcontext import InteractContext

def browser_images(src_file, tgt_folder, tags, white_users_str):
    # src check
    if not os.path.exists(src_file):
        logging.error(f"{src_file} not existed")
        return

    tags = tags.split(',')
    white_users = {}
    if white_users_str is not None:
        for t in white_users_str.split(','):
            user_name, password = t.split(':')
            white_users.update({
                user_name: {'password': password}
            })
    ctx = InteractContext()
    ctx.browser.start("b_exp", config = {
            'tags': tags,
            'white_users': white_users,
        })
    
    logging.info('Waiting data browser stop.')
    ctx.browser.waiting()
    content = ctx.browser.download()
    with open(os.path.join(tgt_folder, 'check.json'), 'w') as fp:
        json.dump(content, fp)
