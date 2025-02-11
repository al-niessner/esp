#! /usr/bin/env python3

import argparse
import git
import os


def compress_rid_notation(c2rid: {str: [int]}) -> [(int, str, str)]:
    result = []
    for cs, rids in cs2rid.items():
        if len(rids) == 1:
            srid = str(list(rids)[0])
        else:
            srid = '{0}:{1}'.format(min(rids), max(rids))

        result.append((min(rids), srid, cs))
        pass
    return result


def extract_from_log(lfn: str) -> [(int, str)]:
    result = []
    with open(lfn, 'rt') as f:
        lines = f.readlines()
    for line in filter(lambda s: s.count(' :: ') == 3, lines):
        ts, src, crit, msg = line.split(' :: ')

        if all(
            [
                src == 'dawgie.pl.farm',
                crit == 'CRITICAL',
                0 < msg.find('New software changeset'),
            ]
        ):
            cs = msg.split()[-1]
            rid = msg[msg.find('(') + 1 : msg.find(')')]
            result.append((int(rid), cs))
            pass
        pass
    return result


def fetch_existing_messages(mfn: str) -> {str: str}:
    result = {}
    with open(mfn) as f:
        content = f.read()
    index = content.find('DO NOT REMOVE: start rid')
    while -1 < index:
        index = content.find('rid {', index) + 4
        close = content.find('}', index) + 1
        key = content[index:close]
        index = content.find('[//]: # (DO NOT REMOVE: finish rid)', close)
        result[key] = content[close + 3 : index - 1].strip()
        index = content.find('DO NOT REMOVE: start rid', index)
        pass
    return result


def update(existing: {str: str}, events: [(int, str, str)]):
    for event in events:
        cs = event[2]
        key = str(event).replace('(', '{').replace(')', '}')

        if key not in existing:
            g = git.cmd.Git(
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            )
            details = [
                s.strip()
                for s in g.execute('git log -n 1 {}'.format(cs).split()).split(
                    '\n'
                )
            ]
            ts = details[2][8:]

            if -1 < details[4].find('(#'):
                start = details[4].find('(#')
                end = details[4].find(')', start)
                pr = details[4][start + 2 : end]
                details[4] = (
                    details[4][:start]
                    + '([#{0}](https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/pull/{0})'.format(
                        pr
                    )
                    + details[4][end:]
                )
                pass
            text = '### Run ID {0}\n\n__Changeset__: [{1}](https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/commit/{1})\n\n__Date__:  {2}\n\n__Title__:  {3}\n{4}'.format(
                event[1], event[2], ts, details[4], '\n'.join(details[5:])
            )
            existing[key] = text
            pass
        pass
    return


def write(mfn: str, messages: {str: str}) -> None:
    block = []
    for message in sorted(
        messages, key=lambda m: int(m[1 : m.find(',')]), reverse=True
    ):
        block.append('[//]: # (DO NOT REMOVE: start rid {})'.format(message))
        block.append('')
        block.append(messages[message])
        block.append('')
        block.append('[//]: # (DO NOT REMOVE: finish rid)')
        pass
    block.append('')
    with open(mfn) as f:
        markdown = f.read()
    start = markdown.find('[//]: # (DO NOT REMOVE: start list)')
    start += len('[//]: # (DO NOT REMOVE: start list)') + 1
    end = markdown.find('[//]: # (DO NOT REMOVE: finish list)')
    with open(mfn, 'tw') as f:
        f.write(markdown[:start] + '\n'.join(block) + markdown[end:])
    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='A simple tool to scan log files for new software changesets and their run ids. Once found, update the MD conent. The MD must contain the comment block HERE BEGINS CHANGESET LOG and HERE ENDS CHANGESET LOG. All content between those two comments must not be altered.'
    )
    ap.add_argument(
        '-L', '--log-dir', required=True, help='directory to scan for log files'
    )
    ap.add_argument(
        '-l',
        '--log-prefix',
        required=True,
        help='the start (prefix) of each log file name',
    )
    ap.add_argument(
        '-m', '--markdown', required=True, help='the markdown file to update'
    )
    args = ap.parse_args()

    cs2rid = {}
    for fn in sorted(
        filter(
            lambda s, lp=args.log_prefix: s.startswith(lp),
            os.listdir(args.log_dir),
        )
    ):
        for rid, cs in extract_from_log(os.path.join(args.log_dir, fn)):
            if cs in cs2rid:
                cs2rid[cs].add(rid)
            else:
                cs2rid[cs] = set([rid])
            pass
        pass
    events = compress_rid_notation(cs2rid)
    events.sort(key=lambda t: t[0])
    events.reverse()
    existing = fetch_existing_messages(args.markdown)
    update(existing, events)
    write(args.markdown, existing)
    pass
