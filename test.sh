#bin/bash


/home/supernova/miniconda3/envs/aigc/bin/python preprocess.py ljspeech /data/LJSpeech-test ./data/ljspeech-test \
  --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json

/home/supernova/miniconda3/envs/aigc/bin/python synthesis.py --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json \
  --conditional=./data/ljspeech-test/ljspeech-mel-00001.npy \
  20180510_mixture_lj_checkpoint_step000320000_ema.pth \
  generated