#!/bin/sh
mkdir -p outputs/CWGAN3/Markdown
bsub -o "outputs/CWGAN3/Markdown/CWGAN3_0.md" -J "CWGAN3_0" -env MYARGS="-name CWGAN3-0 -GPU False -time 360000 -model CWGAN -ID 0" < submit_cpu.sh
bsub -o "outputs/CWGAN3/Markdown/CWGAN3_1.md" -J "CWGAN3_1" -env MYARGS="-name CWGAN3-1 -GPU False -time 360000 -model CWGAN -ID 1" < submit_cpu.sh
