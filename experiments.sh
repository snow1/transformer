#!/bin/sh

mkdir -p outputs/transgormerFinal555/Markdown
bsub -o "outputs/transgormerFinal555/Markdown/transformer5_2.md" -J "transformer5_2" -env MYARGS="-name transformer5-2 -GPU False -time 360000 -model transformer5 -ID 0" < submit_cpu.sh