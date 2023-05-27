#!/bin/sh
mkdir -p outputs/Trans2/Markdown
bsub -o "outputs/Trans2/Markdown/Trans2_0.md" -J "Trans2_0" -env MYARGS="-name Trans2-0 -GPU False -time 360000 -model transformer -ID 0" < submit_cpu.sh
bsub -o "outputs/Trans2/Markdown/Trans2_1.md" -J "Trans2_1" -env MYARGS="-name Trans2-1 -GPU False -time 360000 -model transformer -ID 1" < submit_cpu.sh
