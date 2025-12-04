#!/bin/bash
#SBATCH --job-name=wavenet-test-real
#SBATCH --output=logs/wavenet-test-real.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00

set -euo pipefail
mkdir -p logs
which python
ml cuda/12.2.0 cudnn/8.9.2.26-cuda-12.2.0

/fred/oz016/bgao_kn/AIGC/aigc/bin/python preprocess.py ljspeech /fred/oz016/bgao_kn/AIGC/data/real-audio /fred/oz016/bgao_kn/AIGC/wavenet_vocoder_pretrain/data/real-audio \
  --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json

/fred/oz016/bgao_kn/AIGC/aigc/bin/python synthesis.py --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json \
  --conditional=/fred/oz016/bgao_kn/AIGC/wavenet_vocoder_pretrain/data/real-audio/ljspeech-mel-00001.npy \
  20180510_mixture_lj_checkpoint_step000320000_ema.pth \
  --file-name-suffix _real-1 \
  generated

/fred/oz016/bgao_kn/AIGC/aigc/bin/python synthesis.py --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json \
  --conditional=/fred/oz016/bgao_kn/AIGC/wavenet_vocoder_pretrain/data/real-audio/ljspeech-mel-00002.npy \
  20180510_mixture_lj_checkpoint_step000320000_ema.pth \
  --file-name-suffix _real-2 \
  generated
