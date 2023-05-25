#!/bin/sh
mkdir -p outputs/LSTMTest1/Markdown
bsub -o "outputs/LSTMTest1/Markdown/LSTMTest1_0.md" -J "LSTMTest1_0" -env MYARGS="-name LSTMTest1-0 -GPU False -time 360000 -b 2.0 -a 1 -d fd -ID 0" < submit_cpu.sh
