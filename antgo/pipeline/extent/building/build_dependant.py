from collections import defaultdict
import re
import os
import queue

_INCLUDE_FILE_REG = re.compile(r'^\s*#include\s*(?:"|<)\s*(.*?)\s*(?:"|>)\s*')

_C_EXTENSIONS = ['.cpp', '.c', '.cu']


def _is_c_file(fname):
    return os.path.splitext(fname)[-1] in _C_EXTENSIONS


def get_include_file(fname):
    '''return a list of all inclued filename'''
    res = []
    for line in open(fname):
        u = _INCLUDE_FILE_REG.search(line)
        if u is not None:
            inc_fname = u.groups()[0]
            res.append(inc_fname)
    return res


def get_dependant_graph(fname):
    '''return a partial dependant graph, a dict of (str, a list of str)
    included file -> code file
    '''
    graph = defaultdict(list)
    vis = set()
    if not _is_c_file(fname):
        return graph
    fname = os.path.abspath(fname)
    q = queue.Queue()
    q.put(fname)
    dirname = os.path.dirname(fname)
    while not q.empty():
        name = q.get()
        if name in vis:
            continue
        vis.add(name)
        for iname in get_include_file(fname):
            iname = os.path.abspath(os.path.join(dirname, iname))
            graph[iname].append(fname)
            q.put(iname)
    return graph
