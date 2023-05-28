#!/bin/sh
mkdir -p outputs/ACGAN/Markdown
bsub -o "outputs/ACGAN/Markdown/ACGAN_0.md" -J "ACGAN_0" -env MYARGS="-name ACGAN-0 -GPU False -time 360000 -model ACGAN -ID 0" < submit_cpu.sh
bsub -o "outputs/ACGAN/Markdown/ACGAN_1.md" -J "ACGAN_1" -env MYARGS="-name ACGAN-1 -GPU False -time 360000 -model ACGAN -ID 1" < submit_cpu.sh
