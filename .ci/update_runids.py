#! /usr/bin/env python3

import argparse
import git
import os

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='A simple tool to scan log files for new software changesets and their run ids. Once found, update the MD conent. The MD must contain the comment block HERE BEGINS CHANGESET LOG and HERE ENDS CHANGESET LOG. All content between those two comments must not be altered.')
    ap.add_argument ('-L', '--log-dir', required=True,
                     help='directory to scan for log files')
    ap.add_argument ('-l', '--log-prefix', required=True,
                     help='the start (prefix) of each log file name')
    ap.add_argument ('-m', '--markdown', required=True,
                     help='the markdown file to update')
    args = ap.parse_args()

    for fn in sorted (filter (lambda s,lp=args.log_prefix:s.startswith (lp),
                              os.listdir (args.log_dir))):
        with open (os.path.join (args.log_dir, fn), 'rt') as f:\
             lines = f.readlines()
        for line in filter (lambda s:s.count (' :: ') == 3, lines):
            ts,src,crit,msg = line.split (' :: ')

            if all ([src == 'dawgie.pl.farm',
                     crit == 'CRITICAL',
                     0 < msg.find ('New software changeset')]):
                print (msg)
                cs = msg.split()[-1]
                rid = msg[msg.find ('(')+1:msg.find (')')]
                g = git.cmd.Git(os.path.abspath (os.path.join
                                                 (os.path.dirname (__file__),
                                                  '..')))
                details = [s.strip() for s in g.execute ('git log -n 1 {}'.format (cs).split()).split('\n')]

                if -1 < details[4].find ('(#'):
                    start = details[4].find ('(#')
                    end = details[4].find (')', start)
                    pr = details[4][start+2:end]
                    details[4] = details[4][:start] + '([#{0}](https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/pull/{0})'.format (pr) + details[4][end:]
                print (details)
                text = '### Run ID {0}\n\n__Changeset__: [{1}](https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/commit/{1})\n\n__Date__:  {2}\n\n__Title__:  {3}\n{4}'.format (rid, cs, ts, details[4], '\n'.join (details[5:]))
                print (text)
                pass
            pass
        pass
    pass
