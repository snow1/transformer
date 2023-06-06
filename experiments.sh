#!/bin/sh
mkdir -p outputs/transformer2-1/Markdown
bsub -o "outputs/transformer2-1/Markdown/transformer2-1_0.md" -J "transformer2-1_0" -env MYARGS="-name transformer2-1-0 -GPU False -time 360000 -model transformer2 -ID 0" < submit_cpu.sh
