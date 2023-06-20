#!/bin/sh
mkdir -p outputs/maintransformerBigdata/Markdown
bsub -o "outputs/maintransformerBigdata/Markdown/maintransformerBigdata_0.md" -J "maintransformerBigdata_0" -env MYARGS="-name maintransformerBigdata-0 -GPU False -time 360000 -model maintransformer -ID 0" < submit_cpu.sh
bsub -o "outputs/maintransformerBigdata/Markdown/maintransformerBigdata_1.md" -J "maintransformerBigdata_1" -env MYARGS="-name maintransformerBigdata-1 -GPU False -time 360000 -model maintransformer -ID 1" < submit_cpu.sh
