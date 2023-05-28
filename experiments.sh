#!/bin/sh
mkdir -p outputs/ACGAN3/Markdown
bsub -o "outputs/ACGAN3/Markdown/ACGAN3_0.md" -J "ACGAN3_0" -env MYARGS="-name ACGAN3-0 -GPU False -time 360000 -model ACGAN -ID 0" < submit_cpu.sh
