#!/bin/sh
mkdir -p outputs/transformer44444/Markdown
bsub -o "outputs/transformer44444/Markdown/transformer44444_0.md" -J "transformer44444_0" -env MYARGS="-name transformer44444-0 -GPU False -time 360000 -model transformer4 -ID 0" < submit_cpu.sh


mkdir -p outputs/transformer333/Markdown
bsub -o "outputs/transformer333/Markdown/transformer333_0.md" -J "transformer333_0" -env MYARGS="-name transformer333-0 -GPU False -time 360000 -model transformer3 -ID 0" < submit_cpu.sh

mkdir -p outputs/mainTransformermainTransformer/Markdown
bsub -o "outputs/mainTransformermainTransformer/Markdown/mainTransformer.md" -J "mainTransformer" -env MYARGS="-name mainTransformer-0 -GPU False -time 360000 -model mainTransformer -ID 0" < submit_cpu.sh