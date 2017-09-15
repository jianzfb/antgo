# -*- coding: UTF-8 -*-
# @Time    : 17-8-23
# @File    : daemon.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import time
import atexit
from signal import SIGTERM
import psutil
import signal
import tempfile


class Daemon(object):
  """
  A generic daemon class.

  Usage: subclass the Daemon class and override the run() method
  """
  
  def __init__(self, pidfile, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
    self.stdin = stdin
    self.stdout = stdout
    self.stderr = stderr
    self.pidfile = pidfile
  
  def daemonize(self):
    """
    do the UNIX double-fork magic, see Stevens' "Advanced
    Programming in the UNIX Environment" for details (ISBN 0201563177)
    http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
    """
    try:
      pid = os.fork()
      if pid > 0:
        # exit first parent
        return 0  #(parent process return)
    except OSError as e:
      sys.stderr.write("fork #1 failed: %d (%s)\n" % (e.errno, e.strerror))
      sys.exit(1)
    
    # decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)
    
    # do second fork
    try:
      pid = os.fork()
      if pid > 0:
        # exit from second parent
        return -1 #(grandson process return)
    except OSError as e:
      sys.stderr.write("fork #2 failed: %d (%s)\n" % (e.errno, e.strerror))
      sys.exit(1)
      
    # redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    si = open(self.stdin, 'r')
    so = open(self.stdout, 'a+')
    se = open(self.stderr, 'a+')
    os.dup2(si.fileno(), sys.stdin.fileno())
    si.close()
    os.dup2(so.fileno(), sys.stdout.fileno())
    so.close()
    os.dup2(se.fileno(), sys.stderr.fileno())
    se.close()
  
    # write pidfile
    pidfile = os.path.normpath(tempfile.gettempdir() + '/' + self.pidfile)
    with open(pidfile, 'w+') as fp:
      daemon_pid = os.getpid() # pid
      p = psutil.Process(daemon_pid)
      fp.write('%d %f\n' % (daemon_pid, p._create_time))

    # register
    atexit.register(self.delpid)
    signal.signal(signal.SIGTERM, self.delpid)
    signal.signal(signal.SIGINT, self.delpid)
    # (child process return)
    return 1
  
  def delpid(self):
    pidfile = os.path.normpath(tempfile.gettempdir() + '/' + self.pidfile)
    if os.path.exists(pidfile):
      os.remove(pidfile)
  
  def start(self):
    """
    Start the daemon
    """
    # Check for a pidfile to see if the daemon already runs
    pid = None
    try:
      pidfile = os.path.normpath(tempfile.gettempdir() + '/' + self.pidfile)
      with open(pidfile, 'r') as fp:
        pid, pid_create_time = fp.read().split(' ')
        pid = int(pid.strip())
        pid_create_time = float(pid_create_time)
        
        daemon_p = psutil.Process(pid)  # maybe this process not exist
        if daemon_p._create_time != pid_create_time:
          # there exists process, but it's not the same
          raise RuntimeError
    except:
      pid = None

    if pid is not None:
      # if the daemon already runs, leave saliently
      return
    else:
      # clear record
      self.delpid()
    
    # Start the daemon
    id = self.daemonize()
    if id == 1:
      # daemon process
      self.run()
      sys.exit(0)
    elif id == -1:
      # grandson process (ignore)
      sys.exit(0)
    
    # main process continue
  
  def stop(self):
    """
    Stop the daemon
    """
    # Get the pid from the pidfile
    try:
      pidfile = os.path.normpath(tempfile.gettempdir() + '/' + self.pidfile)
      with open(pidfile, 'r') as fp:
        pid, pid_create_time = fp.read().split(' ')
        pid = int(pid.strip())
    except IOError:
      pid = None
    
    if pid is None:
      message = "Daemon not running?\n"
      sys.stderr.write(message % self.pidfile)
      return  # not an error in a restart
    
    # Try killing the daemon process
    try:
      while 1:
        os.kill(pid, SIGTERM)
        time.sleep(0.1)
    except OSError as err:
      err = str(err)
      if err.find("No such process") > 0:
        self.delpid()
  
  def restart(self):
    """
    Restart the daemon
    """
    self.stop()
    self.start()
  
  def run(self):
    """
    You should override this method when you subclass Daemon. It will be called after the process has been
    daemonized by start() or restart().
    """