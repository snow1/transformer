#!/bin/sh
mkdir -p outputs/Example5/Markdown
bsub -o "outputs/Example5/Markdown/Example5_0.md" -J "Example5_0" -env MYARGS="-name Example5-0 -GPU True -time 3600 -b 2.0 -a 1 -d fd -ID 0" < submit_gpu.sh
