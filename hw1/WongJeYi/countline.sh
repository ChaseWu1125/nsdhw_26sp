#!/bin/bash

FILENAME=$1
count=0

if [ -f $FILENAME ]; then
    #read file line by line
    while IFS="\n" read l;do
        ((count = count + 1))
    done< "$FILENAME"
else
    echo "No file named '$FILENAME'."
fi

echo ${count}
