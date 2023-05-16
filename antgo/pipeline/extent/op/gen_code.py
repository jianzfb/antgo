import copy
import os
import re

REPLACE_MARK_PATTERN = re.compile(r'\$\s*{\s*(.*?)\s*}')


class CodeGenerator:
    def __init__(self, fname):
        self.fname = fname
        with open(fname) as fin:
            self.code = fin.read()
            self.pit = re.finditer(REPLACE_MARK_PATTERN, self.code)

    def __call__(self, **kwargs):
        code = ""
        i = 0
        for m in self.pit:
            span = m.span()
            code += self.code[i:span[0]]
            i = span[1]
            name = m.groups()[0]
            if name not in kwargs:
                raise Exception('There are some variables which are assigned:\n{} in {}'.format(
                    name, self.fname))
            code += str(kwargs[name])
        code += self.code[i:]
        return code


def gen_code(fname):
    return CodeGenerator(fname)


def get_gen_rel_code(path):
    def gen_rel_code(fname):
        return gen_code(os.path.join(path, fname))
    return gen_rel_code
