import os
import logging
from aligo import Aligo
import json
import base64


def share_data_in_aliyun(data_path):
    ali = Aligo()
    file = ali.get_file_by_path(data_path)
    if file is None:
        return None

    data = ali.private_share_file(file.file_id)
    return data.share_url
