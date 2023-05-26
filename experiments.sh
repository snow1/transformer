#!/bin/sh
mkdir -p outputs/transformerWith22ChanelAndpretrain/Markdown
bsub -o "outputs/transformerWith22ChanelAndpretrain/Markdown/transformerWith22ChanelAndpretrain_0.md" -J "transformerWith22ChanelAndpretrain_0" -env MYARGS="-name transformerWith22ChanelAndpretrain-0 -GPU False -time 360000 -model transformer -ID 0" < submit_cpu.sh
