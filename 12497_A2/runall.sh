#!/bin/zsh
xargs --arg-file=jobs.txt \
      --max-procs=4  \
      --replace \
      --verbose \
      /bin/sh -c "{}"

xargs --arg-file=jobs2.txt \
      --max-procs=4  \
      --replace \
      --verbose \
      /bin/sh -c "{}"
