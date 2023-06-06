#!/bin/sh
mkdir -p outputs/transformer222EMB_size5DEPTH2/Markdown
bsub -o "outputs/transformer222EMB_size5DEPTH2/Markdown/transformer222EMB_size5DEPTH2_0.md" -J "transformer222EMB_size5DEPTH2_0" -env MYARGS="-name transformer222EMB_size5DEPTH2-0 -GPU False -time 360000 -model transformer2 -ID 0" < submit_cpu.sh
bsub -o "outputs/transformer222EMB_size5DEPTH2/Markdown/transformer222EMB_size5DEPTH2_1.md" -J "transformer222EMB_size5DEPTH2_1" -env MYARGS="-name transformer222EMB_size5DEPTH2-1 -GPU False -time 360000 -model transformer2 -ID 1" < submit_cpu.sh
