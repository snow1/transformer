#!/bin/sh
mkdir -p outputs/LSTMTest2withSoftmax/Markdown
bsub -o "outputs/LSTMTest2withSoftmax/Markdown/LSTMTest2withSoftmax_0.md" -J "LSTMTest2withSoftmax_0" -env MYARGS="-name LSTMTest2withSoftmax-0 -GPU False -time 360000 -model lstm -ID 0" < submit_cpu.sh
