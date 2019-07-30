#! /usr/bin/env python3

import argparse
import json
import os
import random
import socket
import socketserver
import struct

class Handler(socketserver.BaseRequestHandler):
    def _real_file (self, fn):
        size = os.stat (fn, follow_symlinks=True).st_size
        _send (self.request, json.dumps ({'done':True,
                                          'err':'',
                                          'fn':fn}))
        with open (fn, 'br') as f: _xfer (self.request, size, f)
        return

    def handle (self):
        msg = json.loads (_recv (self.request))

        if not os.path.exists (msg['fn']):
            _send (self.request,
                   json.dumps ({'done':False,
                                'err':'file "' + msg['fn'] + '" does not exist',
                                'fn':msg['fn']}))
        elif os.path.isdir (msg['fn']) and not msg['recurse']:
            _send (self.request,
                   json.dumps ({'done':False,
                                'err':'file "' + msg['fn'] + '" is a directory and recuse if False',
                                'fn':msg['fn']}))
        else:
            if os.path.isdir (msg['fn']):
                for dp,dns,fns in os.walk (msg['fn'], followlinks=True):
                    for fn in fns: self._real_file (os.path.join (dp, fn))
                    pass
                _send (self.request, json.dumps ({'done':True,
                                                  'err':'All files copied',
                                                  'fn':fn}))
            else: self._real_file (msg['fn'])
            pass
        return
    pass

def _recv (s:socket.socket, dump=None)->str:
    msg = b'no message'
    slen = b''
    while len (slen) < 8: slen += s.recv (8 - len (slen))
    l = struct.unpack ('>Q', slen)[0]

    if dump:
        lr = 0
        while lr < l:
            data = s.recv (min ([1400, l - lr]))
            dump.write (data)
            lr += len (data)
            pass
    else:
        msg = b''
        while len (msg) < l: msg += s.recv (l - len (msg))
        pass
    return msg.decode()

def _send (s:socket.socket, message:str)->None:
    s.sendall (struct.pack ('>Q', len (message)))
    s.sendall (message.encode())
    return

def _xfer (s:socket.socket, size:int, src)->None:
    s.sendall (struct.pack ('>Q', size))
    sl = 0
    while sl < size:
        data = src.read (min ([1400, size - sl]))
        s.sendall (data)
        sl += len (data)
        pass
    return

def client (port:int, recurse:bool, filenames:[str]):
    if len (filenames) < 2:
        print ('need at least one input and a single output')
    elif 2 < len (filenames) and not os.path.isdir (filenames[-1]):
        print ('output directory must be a directory if more than 1 filename')
    else:
        for fn in filenames[:-1]:
            s = socket.socket()
            s.connect (('localhost', port))
            _send (s, json.dumps ({'fn':fn, 'recurse':recurse}))
            response = {'done':False, 'err':'', 'fn':''}
            while (not (response['done'] and response['fn'] == fn) and
                   not response['err']):
                response = json.loads (_recv (s))

                if not response['err']:
                    local = os.path.join (filenames[-1], response['fn'])
                    if not os.path.isdir (os.path.dirname (local)):\
                       os.makedirs (os.path.dirname (local))
                    with open (local, 'bw') as f: _recv (s, f)
                else: print ('Request for "' + fn + '" resulted in: ' +
                             response['err'])
                pass
            pass
        pass
    return

def server (filenames:[str]):
    port = False
    while not port:
        try:
            port = random.randint (5000, 65000)
            ss = socketserver.TCPServer (('localhost', port), Handler)
            for fn in filenames:
                with open (fn, 'rt') as f: text = f.read()
                with open (fn + '.dcp', 'tw') as f:\
                     f.write (text.replace ('DCP_PORT_NUMBER', str(port)))
                print (fn)
                print (text)
                print (text.replace ('DCP_PORT_NUMBER', str(port)))
                pass
            ss.serve_forever()
        except OSError: port = False
    return

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='A cheat tool to copy files into a building container during the run statement')
    ap.add_argument('filenames', default=[], metavar='fns', type=str, nargs='*',
                    help='filenames to work with with last always being the output')
    ap.add_argument ('-r', '--recursive', action='store_true', default=False,
                     help='descend into any directories')
    mutex = ap.add_mutually_exclusive_group()
    mutex.add_argument ('-p', '--port', type=int,
                        help='port number on localhost where the server is listening')
    mutex.add_argument ('-s', '--server', action='store_true', default=False,
                         help='run as a service on the host')
    args = ap.parse_args()

    if args.server: server (args.filenames)
    else: client (args.port, args.recursive, args.filenames)
    pass
