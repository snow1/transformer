#!/bin/sh

mkdir -p outputs/transformer555/Markdown
bsub -o "outputs/transformer555/Markdown/transformer555_0.md" -J "transformer555_0" -env MYARGS="-name transformer555-0 -GPU False -time 360000 -model transformer5 -ID 0" < submit_cpu.sh

mkdir -p outputs/transformer666/Markdown
bsub -o "outputs/transformer666/Markdown/transformer666_0.md" -J "transformer666_0" -env MYARGS="-name transformer666-0 -GPU False -time 360000 -model transformer6 -ID 0" < submit_cpu.sh

mkdir -p outputs/transformer333/Markdown
bsub -o "outputs/transformer333/Markdown/transformer333_0.md" -J "transformer333_0" -env MYARGS="-name transformer333-0 -GPU False -time 360000 -model transformer3 -ID 0" < submit_cpu.sh

mkdir -p outputs/mainTransformermainTransformer/Markdown
bsub -o "outputs/mainTransformermainTransformer/Markdown/mainTransformer.md" -J "mainTransformer" -env MYARGS="-name mainTransformer-0 -GPU False -time 360000 -model mainTransformer -ID 0" < submit_cpu.sh