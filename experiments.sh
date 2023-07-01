#!/bin/sh
mkdir -p outputs/transformer6Final2with20peopel/Markdown
bsub -o "outputs/transformer6Final2with20peopel/Markdown/transformer6Final2with20peopel_0.md" -J "transformer6Final2with20peopel_0" -env MYARGS="-name transformer6Final2with20peopel-0 -GPU True -time 360000 -model transformer6 -ID 0" < submit_gpu.sh
