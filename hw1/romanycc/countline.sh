#!/bin/bash
if [ $# -eq 0 ]; then
    echo "missing file name\n"
    exit 1
fi
if [ $# -gt 1 ]; then
    echo "only one argument is allowed\n"
    exit 1
fi
echo $(wc -l $1)
