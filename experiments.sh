#!/bin/sh
mkdir -p outputs/CNN/Markdown
bsub -o "outputs/CNN/Markdown/CNN_0.md" -J "CNN_0" -env MYARGS="-name CNN-0 -GPU False -time 360000 -model cnn -ID 0" < submit_cpu.sh
bsub -o "outputs/CNN/Markdown/CNN_1.md" -J "CNN_1" -env MYARGS="-name CNN-1 -GPU False -time 360000 -model cnn -ID 1" < submit_cpu.sh
