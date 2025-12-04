#!/bin/bash
#SBATCH --job-name=wavenet-test
#SBATCH --output=logs/wavenet-test.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2:00:00

set -euo pipefail
mkdir -p logs
which python
ml cuda/12.2.0 cudnn/8.9.2.26-cuda-12.2.0

/fred/oz016/bgao_kn/AIGC/aigc/bin/python preprocess.py ljspeech /fred/oz016/bgao_kn/AIGC/data/LJSpeech-test /fred/oz016/bgao_kn/AIGC/wavenet_vocoder_pretrain/data/ljspeech-test \
  --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json

/fred/oz016/bgao_kn/AIGC/aigc/bin/python synthesis.py --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json \
  --conditional=/fred/oz016/bgao_kn/AIGC/wavenet_vocoder_pretrain/data/ljspeech-test/ljspeech-mel-00001.npy \
  20180510_mixture_lj_checkpoint_step000320000_ema.pth \
  generated
