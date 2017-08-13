# encoding=utf-8
# @Time    : 17-6-22
# @File    : workflow.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import re
import yaml
import imp
import sys
import shutil
import time
import json
from multiprocessing import Queue
from antgo.utils import logger


def _main_context(main_file, source_paths):
    # filter .py
    key_model = main_file
    dot_pos = key_model.rfind(".")
    if dot_pos != -1:
        key_model = key_model[0:dot_pos]

    sys.path.append(source_paths)
    f, p, d = imp.find_module(key_model, [source_paths])
    module = imp.load_module('mm', f, p, d)
    return module.get_global_context()


class BaseWork(object):
    class _LinkPipe(object):
        def __init__(self, nest, link_type='NORMAL'):
            self._nest = nest
            self._link_type = 'NORMAL'
            self._queue = Queue()

        @property
        def nest(self):
            return self._nest

        @property
        def link_type(self):
            return self._link_type
        @link_type.setter
        def link_type(self, val):
            self._link_type = val

        def put(self, sender='', identity='', value=''):
            self._queue.put(json.dumps({'sender':sender,'identity':identity,'value':value}))

        def get(self):
            return self._queue.get(True)

    def __init__(self, name, code_path, code_main_file, config_parameters, port=None):
        self._running_config = None
        self._workspace_base = ''
        self._datasource = ''
        self._workspace = None
        self._dump_dir = None

        self.port = port
        self._output_pipes = []
        self._input_pipes = []

        self._feedback_output = []
        self._feedback_input = []

        self._stop_shortcuts = []

        self._name = name

        self._code_path = code_path
        self._code_main_file = code_main_file

        self.config_parameters = config_parameters
        self._cpu = config_parameters.get('cpu', None)
        self._gpu = config_parameters.get('gpu', None)
        self._nr_cpu = 0
        self._nr_gpu = 0
        self._occupy = config_parameters.get('occupy', 'share')

        self.status = 'DONE'
        self._need_waiting_feedback = False
        self._is_first = True
        self.collected_value = []

        self._is_root = False

    @property
    def name(self):
        return self._name

    @property
    def output(self):
        output_pipe = BaseWork._LinkPipe(self)
        self._output_pipes.append(output_pipe)
        return output_pipe
    @property
    def output_num(self):
        return len(self._output_pipes)
    @property
    def output_feedback_num(self):
        num = 0
        for output_pipe in self._output_pipes:
            if output_pipe.link_type == 'FEEDBACK':
                num += 1

        return num
    @property
    def output_nonfeedback_num(self):
        num = 0
        for output_pipe in self._output_pipes:
            if output_pipe.link_type != 'FEEDBACK':
                num += 1

        return num

    @property
    def input(self):
        return self._input_pipes
    @input.setter
    def input(self, val):
        link_pipe, link_type = val
        link_pipe.link_type = link_type
        self._input_pipes.append(link_pipe)
    @property
    def input_num(self):
        return len(self._input_pipes)

    @property
    def stop_shortcut(self):
        stop_shortcut = BaseWork._LinkPipe(self)
        self._stop_shortcuts.append(stop_shortcut)
        return stop_shortcut

    @property
    def is_root(self):
        return self._is_root
    @is_root.setter
    def is_root(self, val):
        self._is_root = val

    @property
    def workspace_base(self):
        return self._workspace_base
    @workspace_base.setter
    def workspace_base(self, val):
        self._workspace_base = val

    @property
    def workspace(self):
        if self._workspace is None:
            self._workspace = os.path.join(self.workspace_base, self.name)
        return self._workspace
    @workspace.setter
    def workspace(self, val):
        self._workspace = val

    @property
    def datasource(self):
        return self._datasource
    @datasource.setter
    def datasource(self, val):
        self._datasource = val

    @property
    def dump_dir(self):
        return self._dump_dir
    @dump_dir.setter
    def dump_dir(self, val):
        self._dump_dir = val

    @property
    def nr_cpu(self):
        return self._nr_cpu
    @property
    def nr_gpu(self):
        return self._nr_gpu

    @property
    def cpu(self):
        return self._cpu
    @property
    def gpu(self):
        return self._gpu

    @property
    def occupy(self):
        return self._occupy

    @property
    def need_waiting_feedback(self):
        return self._need_waiting_feedback
    @need_waiting_feedback.setter
    def need_waiting_feedback(self, val):
        self._need_waiting_feedback = val

    @property
    def feedback_output(self):
        return self._feedback_output
    @feedback_output.setter
    def feedback_output(self, val):
        self._feedback_output = val

    @property
    def feedback_input(self):
        return self._feedback_input
    @feedback_input.setter
    def feedback_input(self, val):
        self._feedback_input = val

    def set_computing_resource(self, nr_cpu, nr_gpu):
        self._nr_cpu = nr_cpu
        self._nr_gpu = nr_gpu

    def load(self):
        # load ant context
        _context = _main_context(self._code_main_file, self._code_path)
        return _context

    def load_config(self):
        config_file = None
        for ff in os.listdir(self.dump_dir):
            if 'config.yaml' in ff:
                config_file = os.path.join(self.dump_dir, ff)

        if config_file is not None:
            config = yaml.load(open(config_file, 'r'))
            return config

        return {}

    def update_config(self, config_parameters):
        fp = open(os.path.join(self.dump_dir, 'config.yaml'), 'w')
        yaml.dump(config_parameters, fp)
        fp.close()

    def run(self, *args, **kwargs):
        # flag 'CONTINUE', 'DONE', 'FEEDBACK-DONE' and 'STOP'
        pass

    def send(self, value, identity='CONTINUE'):
        if identity == 'DONE' \
                and len(self._feedback_input) > 0 \
                and not self._need_waiting_feedback:
            identity = 'FEEDBACK-DONE'

        # update current status
        self.status = identity

        if len(self._stop_shortcuts) == 0:
            if len(value) == 0 and identity == 'DONE':
                return

        if len(self.feedback_output) > 0:
            if identity == 'DONE' or identity == 'CONTINUE':
                self.collected_value.append(value)
                for output_pipe in self._output_pipes:
                    if output_pipe.link_type == 'FEEDBACK':
                        # collect all computing result
                        output_pipe.put(sender=self._name, value=value, identity=identity)
            else:
                # 'FEEDBACK-DONE' or 'STOP'
                self.collected_value.append(value)
                for output_pipe in self._output_pipes:
                    if output_pipe.link_type != 'FEEDBACK':
                        output_pipe.put(sender=self._name, value=self.collected_value, identity=identity)
                self.collected_value = []
        else:
            for output_pipe in self._output_pipes:
                output_pipe.put(sender=self._name, value=value, identity=identity)

        for ss in self._stop_shortcuts:
            ss.put(sender=self._name, value='', identity='STOP')

    def start(self, acquired_lock=None):
        # finding feedback output
        for output_pipe in self._output_pipes:
            if output_pipe.link_type == 'FEEDBACK':
                self.feedback_output.append(output_pipe)

        for input_pipe in self._input_pipes:
            if input_pipe.link_type == 'FEEDBACK':
                self.feedback_input.append(input_pipe)

        # waiting or running
        self.waiting_and_run(acquired_lock)

    def _unwarp_value(self, value):
        value_obj = json.loads(value)
        sender = value_obj['sender']
        identity = value_obj['identity']
        info = value_obj['value']

        return sender, identity, info

    def _prepare_running_data(self, identity_list, info_list):
        # check is ok
        for identity, info in zip(identity_list, info_list):
            if self._is_root:
                if len(self.feedback_input) > 0 and self.status == 'FEEDBACK-DONE' and identity == 'STOP':
                    self.send('', identity='STOP')
                    return 0

                if len(self.feedback_input) == 0 and self.status == 'DONE' and identity == 'STOP':
                    self.send('', identity='STOP')
                    return 0

                if identity == 'STOP':
                    # ignore
                    return -1

            if identity == 'STOP':
                # for none root, return directly
                self.send('', identity='STOP')
                return 0

        # using time stamp as running dump folder
        now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.dump_dir = os.path.join(self.workspace, now_time)
        while True:
            index_offset = 0
            if os.path.exists(self.dump_dir):
                self.dump_dir = os.path.join(self.workspace, "%s-%d" % (now_time, index_offset))
            else:
                break

        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # collect
        index_offset = 0
        for identity, info in zip(identity_list, info_list):
            # prepare data
            if type(info) != list:
                info = [info]
            # tranverse in info
            for record in info:
                if len(record) == 0:
                    continue
                if os.path.isfile(record):
                    pos = record.rfind('/')
                    file_name = record[pos+1:]
                    if os.path.exists(os.path.join(self.dump_dir, file_name)):
                        dot_p = file_name.rfind('.')
                        pure_name = file_name if dot_p == -1 else file_name[:dot_p]
                        pure_name = '%s%d'%(pure_name,index_offset)
                        file_name = pure_name+file_name[dot_p:]

                    shutil.copyfile(record, os.path.join(self.dump_dir, file_name))
                else:
                    for ff in os.listdir(record):
                        if ff[0] == '.':
                            continue

                        if os.path.exists(os.path.join(self.dump_dir, ff)):
                            ff = '%s%d' % (ff, index_offset)

                        if os.path.isdir(os.path.join(info, ff)):
                            shutil.copytree(os.path.join(info, ff), os.path.join(self.dump_dir, ff))
                        elif os.path.isfile(os.path.join(info, ff)):
                            shutil.copyfile(os.path.join(info, ff), os.path.join(self.dump_dir, ff))

                index_offset += 1

        # normally run
        return 1

    def waiting_and_run(self, acquired_lock=None):
        while True:
            if len(self._feedback_input) > 0:
                if self.status == 'FEEDBACK-DONE' or not self.need_waiting_feedback or self._is_first:
                    if (self.is_root and not self._is_first) or not self.is_root:
                        # only listening non-feedback pipes
                        identity_list = []
                        info_list = []
                        for input_pipe in self._input_pipes:
                            if input_pipe.link_type == 'NORMAL':
                                value = input_pipe.get()
                                sender, identity, info = self._unwarp_value(value)
                                identity_list.append(identity)
                                info_list.append(info)

                        flag = self._prepare_running_data(identity_list, info_list)
                        if flag == 0:
                            return
                        elif flag == -1:
                            continue
                else:
                    # only listening feedback pipes
                    identity_list = []
                    info_list = []
                    for input_pipe in self._input_pipes:
                        if input_pipe.link_type == 'FEEDBACK':
                            value = input_pipe.get()
                            sender, identity, info = self._unwarp_value(value)
                            identity_list.append(identity)
                            info_list.append(info)

                    flag = self._prepare_running_data(identity_list, info_list)
                    if flag == 0:
                        return
                    elif flag == -1:
                        continue
            else:
                if (self.is_root and not self._is_first) or not self.is_root:
                    identity_list = []
                    info_list = []
                    for input_pipe in self._input_pipes:
                        value = input_pipe.get()
                        sender, identity, info = self._unwarp_value(value)
                        identity_list.append(identity)
                        info_list.append(info)

                    if 'FEEDBACK-DONE' in identity_list:
                        is_continue_pass_next = False
                        for id, identity in enumerate(identity_list):
                            if identity == 'FEEDBACK-DONE' and self._input_pipes[id].nest.output_feedback_num == 0:
                                # directly trigger next
                                self.send('', 'FEEDBACK-DONE')
                                is_continue_pass_next = True
                                break
                        if is_continue_pass_next:
                            continue

                    flag = self._prepare_running_data(identity_list, info_list)
                    if flag == 0:
                        return
                    elif flag == -1:
                        continue

            if self.is_root and self._is_first:
                now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                self.dump_dir = os.path.join(self.workspace, now_time)
                if not os.path.exists(self.dump_dir):
                    os.makedirs(self.dump_dir)

            # run until accquiring computing resource
            if acquired_lock is not None:
                with acquired_lock:
                    self.run()
            else:
                self.run()

            #
            self._is_first = False