#!/bin/sh
mkdir -p outputs/LSTMTest2 with Softmax/Markdown
bsub -o "outputs/LSTMTest2 with Softmax/Markdown/LSTMTest2 with Softmax_0.md" -J "LSTMTest2 with Softmax_0" -env MYARGS="-name LSTMTest2 with Softmax-0 -GPU False -time 360000 -model lstm -ID 0" < submit_cpu.sh
mkdir -p outputs/linearTest1/Markdown
bsub -o "outputs/linearTest1/Markdown/linearTest1_0.md" -J "linearTest1_0" -env MYARGS="-name linearTest1-0 -GPU False -time 360000 -model linear -ID 0" < submit_cpu.sh
