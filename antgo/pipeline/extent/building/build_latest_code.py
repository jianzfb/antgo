from collections import defaultdict
import json
import os
import queue
from .build_hash import get_file_hash
from .build_path import get_virtual_dirname
from .build_dependant import get_dependant_graph

UPDATED_FLAG = '__updated_flag__'
HASH_FILENAME = 'antgo_op.hash.cache'
GRAPH_FILENAME = 'antgo_op.graph.cache'
HASH_FILE_BUFFER = defaultdict(dict)
GRAPH_FILE_BUFFER = defaultdict(dict)
FILE_CHANGED_STATE = dict()


def _read_hash_file(fname):
    global HASH_FILE_BUFFER
    '''read the file which stores the hash number of code files
       Json Format:
       dict(
        filename1=hashcode1,
        filename2=hashcode2,
       )
    '''
    data = HASH_FILE_BUFFER.get(fname, None)
    if data is not None:
        return data
    if os.path.exists(fname):
        with open(fname, 'r') as fin:
            data = json.load(fin)
    else:
        data = dict()
    HASH_FILE_BUFFER[fname] = data
    return data


def _write_hash_file(fname, data):
    with open(fname, 'w') as fout:
        json.dump(data, fout)


def _read_graph_file(fname):
    global GRAPH_FILE_BUFFER

    dirname = os.path.join('.temp', 'cache', os.path.basename(fname).split('.')[0])
    os.makedirs(dirname, exist_ok=True)
    fname = os.path.join(dirname, os.path.basename(fname))

    data = GRAPH_FILE_BUFFER.get(fname, None)
    if data is not None:
        return data
    if os.path.exists(fname):
        with open(fname, 'r') as fin:
            data = json.load(fin)
    else:
        data = dict()
    GRAPH_FILE_BUFFER[fname] = data
    return data


def _write_graph_file(fname, data):
    with open(fname, 'w') as fout:
        json.dump(data, fout)


def file_is_changed(fname):
    '''whether the file is changed'''
    global FILE_CHANGED_STATE
    changed = FILE_CHANGED_STATE.get(fname, None)
    if changed is not None:
        return changed
    if not os.path.exists(fname):
        return False
    fname = os.path.abspath(fname)
    new_hash = get_file_hash(fname)
    # dirname = get_virtual_dirname(os.path.dirname(fname))
    dirname = os.path.join('.temp', 'cache', os.path.basename(fname).split('.')[0])
    os.makedirs(dirname, exist_ok=True)
    hash_fname = os.path.join(dirname, HASH_FILENAME)

    def _get_file_old_hash(fname):
        if not os.path.exists(hash_fname):
            return None
        data = _read_hash_file(hash_fname)
        return data.get(fname, None)

    old_hash = _get_file_old_hash(fname)
    changed = new_hash != old_hash
    FILE_CHANGED_STATE[fname] = changed
    HASH_FILE_BUFFER[hash_fname][fname] = new_hash
    if changed:
        HASH_FILE_BUFFER[hash_fname][UPDATED_FLAG] = True
    return changed


def code_need_to_rebuild(source):
    '''whether the code file should be rebuilt'''
    source = os.path.abspath(source)
    dirname = get_virtual_dirname(os.path.dirname(source))
    graph_fname = os.path.join(dirname, GRAPH_FILENAME)
    graph_buf = _read_graph_file(graph_fname)
    graph = graph_buf.get(source, None)
    if graph is None or file_is_changed(source):
        graph = get_dependant_graph(source)
        graph_buf[UPDATED_FLAG] = True
        graph_buf[source] = graph
    # check graph
    change_state = dict()
    # find all changed file
    files = []
    for k, vs in graph.items():
        files.append(k)
        files.extend(vs)
    files = list(set(files))
    q = queue.Queue()
    for fname in files:
        changed = file_is_changed(fname)
        change_state[fname] = changed or None  # True or None
        q.put(fname)
    while not q.empty():
        fname = q.get()
        change_state[fname] = True
        for v in graph[fname]:
            if change_state[v] is None:
                q.put(v)
    for k in change_state.keys():
        if change_state[k] is None:
            change_state[k] = False
    changed = change_state[source]
    return changed


def save_hash_files():
    for fname in HASH_FILE_BUFFER.keys():
        try:
            v = HASH_FILE_BUFFER[fname]
            v.pop(UPDATED_FLAG)
            _write_hash_file(fname, v)
        except KeyError:
            continue


def save_graph_files():
    for fname in GRAPH_FILE_BUFFER.keys():
        try:
            v = GRAPH_FILE_BUFFER[fname]
            v.pop(UPDATED_FLAG)
            _write_graph_file(fname, v)
        except KeyError:
            continue


def save_latest_state():
    save_hash_files()
    save_graph_files()
