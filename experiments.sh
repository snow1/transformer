#!/bin/sh
mkdir -p outputs/Example1/Markdown
bsub -o "outputs/Example1/Markdown/Example1_0.md" -J "Example1_0" -env MYARGS="-name Example1-0 -GPU False -time 3600 -b 2.0 -a 1 -d fd -ID 0" < submit_cpu.sh
