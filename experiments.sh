#!/bin/sh
mkdir -p outputs/transformer2TestforMemory_1/Markdown
bsub -o "outputs/transformer2TestforMemory_1/Markdown/transformer2TestforMemory_1_0.md" -J "transformer2TestforMemory_1_0" -env MYARGS="-name transformer2TestforMemory_1-0 -GPU False -time 360000 -model transformer2 -ID 0" < submit_cpu.sh
