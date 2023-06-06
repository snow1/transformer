#!/bin/sh
mkdir -p outputs/transformer2TestforMemory_2/Markdown
bsub -o "outputs/transformer2TestforMemory_2/Markdown/transformer2TestforMemory_2_0.md" -J "transformer2TestforMemory_2_0" -env MYARGS="-name transformer2TestforMemory_2-0 -GPU False -time 360000 -model transformer2 -ID 0" < submit_cpu.sh
