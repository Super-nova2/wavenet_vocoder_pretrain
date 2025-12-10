#!/bin/bash
#SBATCH --job-name=wavenet-test-real
#SBATCH --partition=gpu          # 按需要修改分区
#SBATCH --gres=gpu:1             # 请求 1 张 GPU
#SBATCH --cpus-per-task=2        # 预处理线程数
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=../../logs/%x.out

set -euo pipefail

#script_dir=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)
script_dir=/fred/oz016/bgao_kn/AIGC/wavenet_vocoder_train/egs/mulaw256
VOC_DIR="${script_dir}/../../"
mkdir -p "${VOC_DIR}/logs"

# 环境加载
source /fred/oz016/bgao_kn/AIGC/aigc/bin/activate
which python
ml cuda/12.2.0 cudnn/8.9.2.26-cuda-12.2.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

# Input/output paths
# Put wav files directly under this directory (not in a nested subfolder).
real_audio_dir="/fred/oz016/bgao_kn/AIGC/data/real-audio/wavs/"
hparams="${script_dir}/conf/mulaw256_wavenet.json"
meanvar="${script_dir}/dump/lj/logmelspectrogram/org/meanvar.joblib"
expdir="${script_dir}/exp/lj_train_no_dev_mulaw256_wavenet"
checkpoint="${expdir}/checkpoint_step000050000_ema.pth"

# Derived temp/output dirs
spk="real"
dump_root="${script_dir}/dump/${spk}/logmelspectrogram"
dump_org_dir="${dump_root}/org/real_audio"
dump_norm_dir="${dump_root}/norm/real_audio"
inference_batch_size=1
num_workers=${SLURM_CPUS_PER_TASK:-2}

mkdir -p "${dump_org_dir}" "${dump_norm_dir}"

echo "[1/3] Preprocess real audio to mel features..."
python "${VOC_DIR}/preprocess.py" wavallin "${real_audio_dir}" "${dump_org_dir}" \
  --hparams="global_gain_scale=0.55" --preset="${hparams}" --num_workers="${num_workers}"

echo "[2/3] Apply mean/variance normalization using training stats..."
python "${VOC_DIR}/preprocess_normalize.py" "${dump_org_dir}" "${dump_norm_dir}" "${meanvar}" \
  --num_workers="${num_workers}"

echo "[3/3] Run WaveNet inference (GPU if available)..."
checkpoint_name=$(basename "${checkpoint%.*}")
out_dir="${VOC_DIR}/generated/${checkpoint_name}/real_audio"
python "${VOC_DIR}/evaluate.py" "${dump_norm_dir}" "${checkpoint}" "${out_dir}" \
  --preset="${hparams}" --hparams="batch_size=${inference_batch_size}"

echo "Done. Generated wavs are in ${out_dir}"
