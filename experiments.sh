#!/bin/sh
mkdir -p outputs/transformer6Final/Markdown
bsub -o "outputs/transformer6Final/Markdown/transformer6Final_0.md" -J "transformer6Final_0" -env MYARGS="-name transformer6Final-0 -GPU True -time 360000 -model transformer5 -ID 0" < submit_gpu.sh
