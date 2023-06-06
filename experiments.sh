#!/bin/sh
mkdir -p outputs/transformerTestforMemory/Markdown
bsub -o "outputs/transformerTestforMemory/Markdown/transformerTestforMemory_0.md" -J "transformerTestforMemory_0" -env MYARGS="-name transformerTestforMemory-0 -GPU False -time 360000 -model transformer -ID 0" < submit_cpu.sh
