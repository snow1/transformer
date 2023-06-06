#!/bin/sh
mkdir -p outputs/transformer2Test/Markdown
bsub -o "outputs/transformer2Test/Markdown/transformer2Test_0.md" -J "transformer2Test_0" -env MYARGS="-name transformer2Test-0 -GPU False -time 360000 -model transformer2 -ID 0" < submit_cpu.sh
