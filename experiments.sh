#!/bin/sh

mkdir -p outputs/mainTransformermainTransformer/Markdown
bsub -o "outputs/mainTransformermainTransformer/Markdown/mainTransformermainTransformer_3.md" -J "mainTransformermainTransformer_3" -env MYARGS="-name mainTransformermainTransformer-3 -GPU True -time 360000 -model maintransformer -ID 0" < submit_cpu.sh