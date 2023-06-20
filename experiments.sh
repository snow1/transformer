#!/bin/sh
mkdir -p outputs/maintransformerBigdata2/Markdown
bsub -o "outputs/maintransformerBigdata2/Markdown/maintransformerBigdata2_0.md" -J "maintransformerBigdata2_0" -env MYARGS="-name maintransformerBigdata2-0 -GPU False -time 360000 -model maintransformer -ID 0" < submit_cpu.sh
bsub -o "outputs/maintransformerBigdata2/Markdown/maintransformerBigdata2_1.md" -J "maintransformerBigdata2_1" -env MYARGS="-name maintransformerBigdata2-1 -GPU False -time 360000 -model maintransformer -ID 1" < submit_cpu.sh
