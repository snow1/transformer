#!/bin/sh
mkdir -p outputs/Example3/Markdown
bsub -o "outputs/Example3/Markdown/Example3_0.md" -J "Example3_0" -env MYARGS="-name Example3-0 -GPU False -time 360000 -b 2.0 -a 1 -d fd -ID 0" < submit_cpu.sh
