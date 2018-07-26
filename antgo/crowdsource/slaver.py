#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, unicode_literals, division, absolute_import

from antgo.crowdsource.common_func import *
import argparse

__author__ = "Aploium <i@z.codes>"
__website__ = "https://github.com/aploium/shootback"


class Slaver:
    """
    slaver socket阶段
        连接master->等待->心跳(重复)--->握手-->正式传输数据->退出
    """

    def __init__(self, communicate_addr, target_addr, max_spare_count=5):
        self.communicate_addr = communicate_addr
        self.target_addr = target_addr
        self.max_spare_count = max_spare_count

        self.spare_slaver_pool = {}
        self.working_pool = {}
        self.socket_bridge = SocketBridge()

    def _connect_master(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.communicate_addr)

        self.spare_slaver_pool[sock.getsockname()] = {
            "conn_slaver": sock,
        }

        return sock

    def _connect_target(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.target_addr)

        log.debug("connected to target[{}] at: {}".format(
            sock.getpeername(),
            sock.getsockname(),
        ))

        return sock

    def _response_heartbeat(self, conn_slaver, hb_from_master):
        # assert isinstance(hb_from_master, CtrlPkg)
        # assert isinstance(conn_slaver, socket.SocketType)
        if hb_from_master.prgm_ver < 0x000B:
            # shootback before 2.2.5-r10 use two-way heartbeat
            #   so just send a heart_beat pkg back
            conn_slaver.send(CtrlPkg.pbuild_heart_beat().raw)
            return True
        else:
            # newer version use TCP-like 3-way heartbeat
            #   the older 2-way heartbeat can't only ensure the
            #   master --> slaver pathway is OK, but the reverse
            #   communicate may down. So we need a TCP-like 3-way
            #   heartbeat
            conn_slaver.send(CtrlPkg.pbuild_heart_beat().raw)
            pkg, verify = CtrlPkg.recv(
                conn_slaver,
                expect_ptype=CtrlPkg.PTYPE_HEART_BEAT)  # type: CtrlPkg,bool
            if verify:
                log.debug("heartbeat success {}".format(
                    fmt_addr(conn_slaver.getsockname())))
                return True
            else:
                log.warning(
                    "received a wrong pkg[{}] during heartbeat, {}".format(
                        pkg, conn_slaver.getsockname()
                    ))
                return False

    def _stage_ctrlpkg(self, conn_slaver):
        """
        handling CtrlPkg until handshake

        well, there is only one CtrlPkg: heartbeat, yet

        it ensures:
            1. network is ok, master is alive
            2. master is shootback_master, not bad guy
            3. verify the SECRET_KEY
            4. tell slaver it's time to connect target

        handshake procedure:
            1. master hello --> slaver
            2. slaver verify master's hello
            3. slaver hello --> master
            4. (immediately after 3) slaver connect to target
            4. master verify slaver
            5. enter real data transfer
        """
        while True:  # 可能会有一段时间的心跳包

            # recv master --> slaver
            # timeout is set to `SPARE_SLAVER_TTL`
            # which means if not receive pkg from master in SPARE_SLAVER_TTL seconds,
            #   this connection would expire and re-connect
            pkg, verify = CtrlPkg.recv(conn_slaver, SPARE_SLAVER_TTL)  # type: CtrlPkg,bool

            if not verify:
                return False

            log.debug("CtrlPkg from {}: {}".format(conn_slaver.getpeername(), pkg))

            if pkg.pkg_type == CtrlPkg.PTYPE_HEART_BEAT:
                # if the pkg is heartbeat pkg, enter handshake procedure
                if not self._response_heartbeat(conn_slaver, pkg):
                    return False

            elif pkg.pkg_type == CtrlPkg.PTYPE_HS_M2S:
                # 拿到了开始传输的握手包, 进入工作阶段

                break

        # send slaver hello --> master
        conn_slaver.send(CtrlPkg.pbuild_hs_s2m().raw)

        return True

    def _transfer_complete(self, addr_slaver):
        """a callback for SocketBridge, do some cleanup jobs"""
        del self.working_pool[addr_slaver]
        log.info("slaver complete: {}".format(addr_slaver))

    def _slaver_working(self, conn_slaver):
        addr_slaver = conn_slaver.getsockname()
        addr_master = conn_slaver.getpeername()

        # --------- handling CtrlPkg until handshake -------------
        try:
            hs = self._stage_ctrlpkg(conn_slaver)
        except Exception as e:
            log.warning("slaver{} waiting handshake failed {}".format(
                fmt_addr(addr_slaver), e))
            log.debug(traceback.print_exc())
            hs = False
        else:
            if not hs:
                log.warning("bad handshake or timeout between: {} and {}".format(
                    fmt_addr(addr_master), fmt_addr(addr_slaver)))

        if not hs:
            # handshake failed or timeout
            del self.spare_slaver_pool[addr_slaver]
            try_close(conn_slaver)

            log.warning("a slaver[{}] abort due to handshake error or timeout".format(
                fmt_addr(addr_slaver)))
            return
        else:
            log.info("Success master handshake from: {} to {}".format(
                fmt_addr(addr_master), fmt_addr(addr_slaver)))

        # ----------- slaver activated! ------------
        # move self from spare_slaver_pool to working_pool
        self.working_pool[addr_slaver] = self.spare_slaver_pool.pop(addr_slaver)

        # ----------- connecting to target ----------
        try:
            conn_target = self._connect_target()
        except:
            log.error("unable to connect target")
            try_close(conn_slaver)

            del self.working_pool[addr_slaver]
            return
        self.working_pool[addr_slaver]["conn_target"] = conn_target

        # ----------- all preparation finished -----------
        # pass two sockets to SocketBridge, and let it do the
        #   real data exchange task
        self.socket_bridge.add_conn_pair(
            conn_slaver, conn_target,
            functools.partial(
                # 这个回调用来在传输完成后删除工作池中对应记录
                self._transfer_complete, addr_slaver
            )
        )

        # this slaver thread exits here
        return

    def serve_forever(self):
        self.socket_bridge.start_as_daemon()  # hi, don't ignore me

        # sleep between two retries if exception occurs
        #   eg: master down or network temporary failed
        # err_delay would increase if err occurs repeatedly
        #   until `max_err_delay`
        # would immediately decrease to 0 after a success connection
        err_delay = 0
        max_err_delay = 15
        # spare_delay is sleep cycle if we are full of spare slaver
        #   would immediately decrease to 0 after a slaver lack
        spare_delay = 0.08
        default_spare_delay = 0.08

        while True:
            if len(self.spare_slaver_pool) >= self.max_spare_count:
                time.sleep(spare_delay)
                spare_delay = (spare_delay + default_spare_delay) / 2.0
                continue
            else:
                spare_delay = 0.0

            try:
                conn_slaver = self._connect_master()
            except Exception as e:
                log.warning("unable to connect master {}".format(e))
                log.debug(traceback.format_exc())
                time.sleep(err_delay)
                if err_delay < max_err_delay:
                    err_delay += 1
                continue

            try:
                t = threading.Thread(target=self._slaver_working,
                                     args=(conn_slaver,)
                                     )
                t.daemon = True
                t.start()

                log.info("connected to master[{}] at {} total: {}".format(
                    fmt_addr(conn_slaver.getpeername()),
                    fmt_addr(conn_slaver.getsockname()),
                    len(self.spare_slaver_pool),
                ))
            except Exception as e:
                log.error("unable create Thread: {}".format(e))
                log.debug(traceback.format_exc())
                time.sleep(err_delay)

                if err_delay < max_err_delay:
                    err_delay += 1
                continue

            # set err_delay if everything is ok
            err_delay = 0


def run_slaver(communicate_addr, target_addr, max_spare_count=5):
    log.info("running as slaver, master addr: {} target: {}".format(
        fmt_addr(communicate_addr), fmt_addr(target_addr)
    ))

    Slaver(communicate_addr, target_addr, max_spare_count=max_spare_count).serve_forever()


def launch_slaver_proxy(master, target):
    global SPARE_SLAVER_TTL
    global SECRET_KEY

    # args = argparse_slaver()
    #
    # if args.verbose and args.quiet:
    #     print("-v and -q should not appear together")
    #     exit(1)

    communicate_addr = split_host(master)
    target_addr = split_host(target)

    SECRET_KEY = 'shootback'
    CtrlPkg.recalc_crc32()

    SPARE_SLAVER_TTL = None
    max_spare_count = 1
    configure_logging(logging.INFO)

    log.info("shootback {} slaver running".format(version_info()))
    log.info("author: {}  site: {}".format(__author__, __website__))
    log.info("Master: {}".format(fmt_addr(communicate_addr)))
    log.info("Target: {}".format(fmt_addr(target_addr)))

    # communicate_addr = ("localhost", 12345)
    # target_addr = ("93.184.216.34", 80)  # www.example.com

    run_slaver(communicate_addr, target_addr, max_spare_count=1)


if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument("-m", "--master", help="proxy master")
  # parser.add_argument("-t", "--target", help="local target")
  #
  # args = parser.parse_args()
  # master_address = args.master
  # target_address = args.target

  master_address = '59.110.164.237:30000'
  target_address = '127.0.0.1:8000'
  launch_slaver_proxy(master_address, target_address)