#!/usr/bin/env bash
set -e

python train.py \
  --dataset cifar10 \
  --model small_cnn \
  --aug cutmix \
  --batch-size 128 \
  --epochs 3 \
  --lr 1e-3 \
  --seed 0 \
  --data-dir ./data \
  --cutmix-alpha 1.0 \
  --cutmix-prob 0.5