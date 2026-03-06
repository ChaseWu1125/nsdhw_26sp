#!/usr/bin/env python3

import sys
import os.path
import os

python_bin = os.getenv("PYTHON_BIN")
if (python_bin=="python0"):
    sys.stderr.write("exec: python0: not found\n")
    sys.exit(1)
if len(sys.argv) > 1:
    fname = sys.argv[1]
    if os.path.exists(fname):
        with open(fname) as fobj:
            lines = fobj.readlines()
        sys.stdout.write('{} lines in {}\n'.format(len(lines), fname))
    else:
        sys.stdout.write('{} not found\n'.format(fname))
