#!/bin/sh
mkdir -p outputs/transformer3WithBigdata/Markdown
bsub -o "outputs/transformer3WithBigdata/Markdown/transformer3WithBigdata_0.md" -J "transformer3_0" -env MYARGS="-name transformer3-0 -GPU False -time 360000 -model transformer3 -ID 0" < submit_cpu.sh
bsub -o "outputs/transformer3WithBigdata/Markdown/transformer3WithBigdata_1.md" -J "transformer3_1" -env MYARGS="-name transformer3-1 -GPU False -time 360000 -model transformer3 -ID 1" < submit_cpu.sh
