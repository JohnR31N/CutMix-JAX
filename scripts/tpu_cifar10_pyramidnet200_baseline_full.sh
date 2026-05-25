#!/usr/bin/env bash
set -e

python train.py \
  --dataset cifar10 \
  --model pyramidnet \
  --pyramid-depth 200 \
  --pyramid-alpha 240 \
  --pyramid-bottleneck \
  --aug none \
  --batch-size 64 \
  --epochs 300 \
  --lr 0.25 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --lr-schedule multistep \
  --lr-milestones 150,225 \
  --lr-gamma 0.1 \
  --seed 0 \
  --data-dir ./data \
  --output-dir ./outputs
