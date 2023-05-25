#!/bin/sh
mkdir -p outputs/Example4/Markdown
bsub -o "outputs/Example4/Markdown/Example4_0.md" -J "Example4_0" -env MYARGS="-name Example4-0 -GPU False -time 360000 -b 2.0 -a 1 -d fd -ID 0" < submit_cpu.sh
