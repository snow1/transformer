#!/bin/sh
mkdir -p outputs/transformer2TestforMemory/Markdown
bsub -o "outputs/transformer2TestforMemory/Markdown/transformer2TestforMemory_0.md" -J "transformer2TestforMemory_0" -env MYARGS="-name transformer2TestforMemory-0 -GPU False -time 360000 -model transformer2 -ID 0" < submit_cpu.sh
