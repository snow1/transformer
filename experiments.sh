#!/bin/sh
mkdir -p outputs/transformerEMB_size5DEPTH2/Markdown
bsub -o "outputs/transformerEMB_size5DEPTH2/Markdown/transformerEMB_size5DEPTH2_0.md" -J "transformerEMB_size5DEPTH2_0" -env MYARGS="-name transformerEMB_size5DEPTH2-0 -GPU False -time 360000 -model transformer -ID 0" < submit_cpu.sh
bsub -o "outputs/transformerEMB_size5DEPTH2/Markdown/transformerEMB_size5DEPTH2_1.md" -J "transformerEMB_size5DEPTH2_1" -env MYARGS="-name transformerEMB_size5DEPTH2-1 -GPU False -time 360000 -model transformer -ID 1" < submit_cpu.sh
