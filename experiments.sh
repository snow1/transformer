#!/bin/sh
mkdir -p outputs/CWGAN/Markdown
bsub -o "outputs/CWGAN/Markdown/CWGAN_0.md" -J "CWGAN_0" -env MYARGS="-name CWGAN-0 -GPU False -time 360000 -model CWGAN -ID 0" < submit_cpu.sh
bsub -o "outputs/CWGAN/Markdown/CWGAN_1.md" -J "CWGAN_1" -env MYARGS="-name CWGAN-1 -GPU False -time 360000 -model CWGAN -ID 1" < submit_cpu.sh
