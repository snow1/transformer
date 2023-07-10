#!/bin/sh
mkdir -p outputs/transformer44444/Markdown
bsub -o "outputs/transformer44444/Markdown/transformer44444_0.md" -J "transformer44444_0" -env MYARGS="-name transformer44444-0 -GPU False -time 360000 -model transformer4 -ID 0" < submit_cpu.sh
