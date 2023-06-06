#!/bin/sh
mkdir -p outputs/transformer2/Markdown
bsub -o "outputs/transformer2/Markdown/transformer2_0.md" -J "transformer2_0" -env MYARGS="-name transformer2-0 -GPU False -time 360000 -model transformer2 -ID 0" < submit_cpu.sh
bsub -o "outputs/transformer2/Markdown/transformer2_1.md" -J "transformer2_1" -env MYARGS="-name transformer2-1 -GPU False -time 360000 -model transformer2 -ID 1" < submit_cpu.sh
