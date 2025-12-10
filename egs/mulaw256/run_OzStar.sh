#!/bin/bash
#SBATCH --job-name=wavenet-vocoder-train
#SBATCH --partition=gpu           # 按集群分区改
#SBATCH --gres=gpu:1              # 需要的 GPU 数
#SBATCH --cpus-per-task=1         # 数据处理用到的 CPU 线程
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=../../logs/%x.out

set -euo pipefail
source /fred/oz016/bgao_kn/AIGC/aigc/bin/activate
which python
ml cuda/12.2.0 cudnn/8.9.2.26-cuda-12.2.0

script_dir=/fred/oz016/bgao_kn/AIGC/wavenet_vocoder_train/egs/mulaw256
VOC_DIR=$script_dir/../../
echo $VOC_DIR

# 路径按需修改
db_root=/fred/oz016/bgao_kn/AIGC/data/LJSpeech/wavs/
spk="lj"
dumpdir=dump

dev_size=10
eval_size=10
limit=1000000
global_gain_scale=0.55
stage=0
stop_stage=3
hparams=conf/mulaw256_wavenet.json
inference_batch_size=32
eval_checkpoint=
eval_max_num_utt=1000000
tag=""

. $VOC_DIR/utils/parse_options.sh || exit 1;

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)

if [ -z ${tag} ]; then
    expname=${spk}_${train_set}_$(basename ${hparams%.*})
else
    expname=${spk}_${train_set}_${tag}
fi
expdir=exp/$expname
feat_typ="logmelspectrogram"
data_root=data/$spk
dump_org_dir=$dumpdir/$spk/$feat_typ/org
dump_norm_dir=$dumpdir/$spk/$feat_typ/norm

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: train/dev/eval split"
    if [ -z $db_root ]; then
      echo "ERROR: DB ROOT must be specified for train/dev/eval splitting."
      exit 1
    fi
    python $VOC_DIR/mksubset.py $db_root $data_root \
      --train-dev-test-split --dev-size $dev_size --test-size $eval_size \
      --limit=$limit
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    for s in ${datasets[@]}; do
      python $VOC_DIR/preprocess.py wavallin $data_root/$s ${dump_org_dir}/$s \
        --hparams="global_gain_scale=${global_gain_scale}" --preset=$hparams
    done
    find $dump_org_dir/$train_set -type f -name "*feats.npy" > train_list.txt
    python $VOC_DIR/compute-meanvar-stats.py train_list.txt $dump_org_dir/meanvar.joblib
    rm -f train_list.txt
    for s in ${datasets[@]}; do
      python $VOC_DIR/preprocess_normalize.py ${dump_org_dir}/$s $dump_norm_dir/$s \
        $dump_org_dir/meanvar.joblib
    done
    cp -f $dump_org_dir/meanvar.joblib ${dump_norm_dir}/meanvar.joblib
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: WaveNet training"
    python $VOC_DIR/train.py --dump-root $dump_norm_dir --preset $hparams \
      --checkpoint-dir=$expdir --log-event-path=tensorboard/${expname}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Synthesis waveform from WaveNet"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    for s in $dev_set $eval_set; do
      dst_dir=$expdir/generated/$name/$s
      python $VOC_DIR/evaluate.py $dump_norm_dir/$s $eval_checkpoint $dst_dir \
        --preset $hparams --hparams="batch_size=$inference_batch_size" \
        --num-utterances=$eval_max_num_utt
    done
fi
