#!/bin/sh
mkdir -p outputs/ACGAN4with3000epoch/Markdown
bsub -o "outputs/ACGAN4with3000epoch/Markdown/ACGAN4with3000epoch_0.md" -J "ACGAN4with3000epoch_0" -env MYARGS="-name ACGAN4with3000epoch-0 -GPU False -time 360000 -model ACGAN -ID 0" < submit_cpu.sh
bsub -o "outputs/ACGAN4with3000epoch/Markdown/ACGAN4with3000epoch_1.md" -J "ACGAN4with3000epoch_1" -env MYARGS="-name ACGAN4with3000epoch-1 -GPU False -time 360000 -model ACGAN -ID 1" < submit_cpu.sh
