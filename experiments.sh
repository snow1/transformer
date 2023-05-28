#!/bin/sh
mkdir -p outputs/ACGAN2/Markdown
bsub -o "outputs/ACGAN2/Markdown/ACGAN2_0.md" -J "ACGAN2_0" -env MYARGS="-name ACGAN2-0 -GPU False -time 360000 -model ACGAN -ID 0" < submit_cpu.sh
