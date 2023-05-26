#!/bin/sh
mkdir -p outputs/transformerWithpretrain /Markdown
bsub -o "outputs/transformerWithpretrain /Markdown/transformerWithpretrain _0.md" -J "transformerWithpretrain _0" -env MYARGS="-name transformerWithpretrain -0 -GPU False -time 360000 -model transformer -ID 0" < submit_cpu.sh
bsub -o "outputs/transformerWithpretrain /Markdown/transformerWithpretrain _1.md" -J "transformerWithpretrain _1" -env MYARGS="-name transformerWithpretrain -1 -GPU False -time 360000 -model transformer -ID 1" < submit_cpu.sh
