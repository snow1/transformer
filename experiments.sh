#!/bin/sh
mkdir -p outputs/transformer5To1/Markdown
bsub -o "outputs/transformer5To1/Markdown/transformer5To1_0.md" -J "transformer5To1_0" -env MYARGS="-name transformer5To1-0 -GPU False -time 360000 -model transformer5 -ID 0" < submit_cpu.sh
