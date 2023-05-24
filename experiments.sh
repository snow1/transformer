#!/bin/sh
mkdir -p outputs/Example2/Markdown
bsub -o "outputs/Example2/Markdown/Example2_0.md" -J "Example2_0" -env MYARGS="-name Example2-0 -GPU False -time 360000 -b 2.0 -a 1 -d fd -ID 0" < submit_cpu.sh
