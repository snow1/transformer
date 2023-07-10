#!/bin/sh
mkdir -p outputs/transformer6Final2with109peopelCnanelfrom0to64/Markdown
bsub -o "outputs/transformer6Final2with109peopelCnanelfrom0to64/Markdown/transformer6Final2with109peopelCnanelfrom0to64_0.md" -J "transformer6Final2with109peopelCnanelfrom0to64_0" -env MYARGS="-name transformer6Final2with109peopelCnanelfrom0to64-0 -GPU True -time 360000 -model transformer6 -ID 0" < submit_gpu.sh
