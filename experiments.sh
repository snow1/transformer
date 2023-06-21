#!/bin/sh
mkdir -p outputs/transformer4/Markdown
bsub -o "outputs/transformer4/Markdown/transformer4_0.md" -J "transformer4_0" -env MYARGS="-name transformer4-0 -GPU False -time 360000 -model transformer4 -ID 0" < submit_cpu.sh
bsub -o "outputs/transformer4/Markdown/transformer4_1.md" -J "transformer4_1" -env MYARGS="-name transformer4-1 -GPU False -time 360000 -model transformer4 -ID 1" < submit_cpu.sh
mkdir -p outputs/transformer5/Markdown
bsub -o "outputs/transformer5/Markdown/transformer5_0.md" -J "transformer5_0" -env MYARGS="-name transformer5-0 -GPU False -time 360000 -model transformer5 -ID 0" < submit_cpu.sh
bsub -o "outputs/transformer5/Markdown/transformer5_1.md" -J "transformer5_1" -env MYARGS="-name transformer5-1 -GPU False -time 360000 -model transformer5 -ID 1" < submit_cpu.sh
