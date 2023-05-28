#!/bin/sh
mkdir -p outputs/CWGAN4with1000epoch/Markdown
bsub -o "outputs/CWGAN4with1000epoch/Markdown/CWGAN4with1000epoch_0.md" -J "CWGAN4with1000epoch_0" -env MYARGS="-name CWGAN4with1000epoch-0 -GPU False -time 360000 -model CWGAN -ID 0" < submit_cpu.sh
bsub -o "outputs/CWGAN4with1000epoch/Markdown/CWGAN4with1000epoch_1.md" -J "CWGAN4with1000epoch_1" -env MYARGS="-name CWGAN4with1000epoch-1 -GPU False -time 360000 -model CWGAN -ID 1" < submit_cpu.sh
