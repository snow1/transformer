#!/bin/sh

mkdir -p outputs/mainTransformermainTransformer/Markdown
bsub -o "outputs/mainTransformermainTransformer/Markdown/mainTransformer_2.md" -J "mainTransformer_2" -env MYARGS="-name mainTransformer-2 -GPU False -time 360000 -model mainTransformer -ID 0" < submit_cpu.sh