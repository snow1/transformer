#!/bin/sh
mkdir -p outputs/transformerWith22Chanel/Markdown
bsub -o "outputs/transformerWith22Chanel/Markdown/transformerWith22Chanel_0.md" -J "transformerWith22Chanel_0" -env MYARGS="-name transformerWith22Chanel-0 -GPU False -time 360000 -model transformer -ID 0" < submit_cpu.sh
