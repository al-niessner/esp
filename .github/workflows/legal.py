#! /usr/bin/env python

import argparse
import datetime
import os
import sys

def _format (text, length=79, prefix='', suffix='\n'):
    length = length - len (suffix)
    result = '\n'
    for line in text.split('\n'):
        newline = prefix + line
        while length < len (newline):
           l = newline [:length].rfind (' ')
           result += newline[:length if l < 0 else l].strip()+suffix 
           newline = prefix + newline[length if l < 0 else l:].strip()
           pass
        result += newline.strip() + suffix
        pass
    return result + prefix

def shell_style (text): return _format (text, prefix='# ')

ap = argparse.ArgumentParser()
ap.add_argument ('--repair', action='store_true', default=False)
args = ap.parse_args()
with open ('COPYRIGHT.txt', 'rt') as f: cr = f.read()
with open ('LICENSE.txt', 'rt') as f: lic = f.read()
dl = cr.find ('2015-20')
cr = cr.replace (cr[dl:dl+9], 
                 '2015-{}'.format (datetime.datetime.now().year))
formatters = {'.hamlbars':_format, '.hbs':_format, '.html':_format,
              '.dot':_format, '.js':_format,
              '.py':_format, '.sh':shell_style}
legal = True
for p,dns,fns in os.walk ('.'):
    if any ([p.startswith ('./.{}'.format (d))
             for d in [ 'cache', 'git' ]]) : continue

    for fn in filter (lambda s:any ([s.endswith (suffix) for suffix in
                                    ('.py', '.sh', '.hbs',
                                     '.hamlbars', '.html', '.dot')] +
                                    [s.startswith ('Dockerfile.') and s[-1] != '~']),
                      fns):
       suffix = '.sh' if fn.startswith ('Dockerfile.') else fn[fn.rfind ('.'):]
       fn = os.path.join (p, fn)
       with open (fn, 'rt') as f: text = f.read()
       cprl = text.find ('COPY' + 'RIGHT:')
       licl = text.find ('LIC' + 'ENSE:')
       ntrl = text.find ('NT' + 'R:')

       if cprl < licl < ntrl:
           scr = formatters[suffix] (cr)
           slc = formatters[suffix] (lic)

           if text[licl+8:min (ntrl, licl+8+len(slc))] != slc:
               legal = False
               print ('license requires update', fn)
               text = text[:licl+8] + slc + text[ntrl:]
               pass
           if text[cprl+10:min (licl, cprl+10+len(scr))] != scr:
               legal = False
               print ('copyright requires update', fn)
               text = text[:cprl+10] + scr + text[licl:]
               pass

           if args.repair:
               with open (fn, 'tw') as f: f.write (text)
               pass
       else:
          print ('need to update', fn,
                 'with COPY' + 'RIGHT:, LIC' + 'ENSE:, NT' + 'R:')
          legal = False
          pass
       pass
    pass

if not legal:
    sys.exit(1)

print ('all legal preambles seem to be valid')
