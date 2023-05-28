#!/bin/sh
mkdir -p outputs/CWGAN2/Markdown
bsub -o "outputs/CWGAN2/Markdown/CWGAN2_0.md" -J "CWGAN2_0" -env MYARGS="-name CWGAN2-0 -GPU False -time 360000 -model CWGAN -ID 0" < submit_cpu.sh
bsub -o "outputs/CWGAN2/Markdown/CWGAN2_1.md" -J "CWGAN2_1" -env MYARGS="-name CWGAN2-1 -GPU False -time 360000 -model CWGAN -ID 1" < submit_cpu.sh
